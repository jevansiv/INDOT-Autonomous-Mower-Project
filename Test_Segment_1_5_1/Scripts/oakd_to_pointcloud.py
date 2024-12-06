#!/bin/bash python
import cv2
import numpy as np
from ultralytics import YOLO
from sensor_msgs.msg import PointCloud2, PointField
import std_msgs.msg
import sensor_msgs_py.point_cloud2 as pc2
import os

# ROS2 Packages
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from builtin_interfaces.msg import Time
from cv_bridge import CvBridge

def edit_image(image, blur_amount=0, grain_intensity=0, grain_size=1):
    """
    Applies Gaussian blur and film grain to an image.

    Parameters:
    - image: The input image to be processed.
    - blur_amount: The amount of Gaussian blur to apply (0 = no blur).
    - grain_intensity: The intensity of the film grain (0 = no grain).
    - grain_size: The size of the film grain (1 = normal, higher = larger grain).

    Returns:
    - The processed image with the applied effects.
    """

    # Apply Gaussian blur if blur_amount is greater than 0
    if blur_amount > 0:
        image = cv2.GaussianBlur(image, (0, 0), blur_amount)

    # Apply film grain if grain_intensity is greater than 0
    if grain_intensity > 0:
        # Generate random noise
        noise = np.random.normal(0, grain_intensity, image.shape).astype(np.uint8)

        # Scale the noise by grain size
        noise = cv2.resize(noise, (0, 0), fx=grain_size, fy=grain_size)
        noise = cv2.resize(noise, (image.shape[1], image.shape[0]))

        # Add noise to the image
        image = cv2.add(image, noise)

    return image

def create_pointcloud2(xyz_array):
    """
    Create a ROS 2 PointCloud2 message from an XYZ array.

    Parameters:
    - xyz_array: A 2D numpy array of shape (N, 3) containing XYZ coordinates.

    Returns:
    - pointcloud2_msg: A sensor_msgs/PointCloud2 message.
    """
    header = std_msgs.msg.Header()
    header.stamp = Time()
    header.frame_id = "camera_frame"  # Change to your frame ID

    fields = [PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
              PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
              PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1)]
    
    # Flatten XYZ array and convert to list
    #flat_xyz = xyz_array.ravel().tolist()
    #flat_xyz = xyz_array.reshape(-1).tolist()  # Reshape to 1D array
    #print(flat_xyz.shape)
    #lat_xyz = [xyz for point in xyz_array for xyz in point]
    grouped_xyz = [tuple(point) for point in xyz_array]

    pointcloud2_msg = pc2.create_cloud(header, fields, grouped_xyz)
    
    return pointcloud2_msg



def depth_to_xyz_segmented(depth_image, segmentation_mask, class_ids, fx, fy, cx, cy, depth_scale=1/100):
    """
    Convert depth image to XYZ coordinates for specific segments/classes.

    Parameters:
    - depth_image: A 2D numpy array containing depth values.
    - segmentation_mask: A 2D numpy array of the same size as depth_image, containing class IDs for each pixel.
    - class_ids: A list of class IDs to include in the output.
    - fx, fy, cx, cy: Camera intrinsic parameters.
    - depth_scale: Scale to convert depth units to meters.

    Returns:
    - xyz: A 2D numpy array of shape (N, 3) containing XYZ coordinates for the selected pixels.
    """
    # Mask to select pixels belonging to the specified classes
    mask = np.isin(segmentation_mask, class_ids)
    
    # Apply mask to depth image
    selected_depth = depth_image[mask]
    
    height, width = depth_image.shape
    u, v = np.meshgrid(np.arange(width), np.arange(height))
    
    # Apply the same mask to u and v coordinates
    selected_u = u[mask]
    selected_v = v[mask]
    
    # Compute XYZ coordinates for selected pixels
    z = selected_depth.astype(np.float32) * depth_scale
    x = (selected_u - (cx/0.7)) * z / fx-0.25
    y = (selected_v - (cy/1.3)) * z / fy
    
    # Combine into a single array
    xyz = np.vstack((z, -x, -y)).transpose()
    print(xyz)
    
    return xyz


def get_seg_mask(model, img, model_img_size=640, show_annotation=False):

    h, w = img.shape[0:2]
    results = model.predict(source=cv2.resize(img, (model_img_size, model_img_size), interpolation = cv2.INTER_AREA), save=False, conf=0.1)#, conf=0.2)

    if show_annotation:
        # 0=sign, 1=cone, 2=person, 3=guardrail, 4=post
        color_list = [(0, 0, 255), (0,255,255), (255,0,0), (0,255,0), (255,255,0)]
        mask_color = np.zeros(img.shape, np.uint8)


    mask = np.zeros(img.shape[0:2], np.uint8)

    for result in results:
        # sloppy way to check if there are masks from model prediction
        segd = False
        try:
            x = result.masks.xy
            segd = True
        except:
            print("no mask")
            continue

        for j, i in enumerate(result.masks.xy):
                
            cc = int(result.boxes.cls[j]) # 0=sign, 1=cone, 2=person, 3=guardrail, 4=post
            
            poly = []
            for n in range(i.shape[0]):
                # sometimes the annotation doesnt reach all the way to the border
                pt = [int(i[n, 0]*w/model_img_size), int(i[n, 1]*h/model_img_size)]
                poly.append(pt)

            if len(poly) > 3:

                cv2.fillPoly(mask, pts = [np.array(poly)], color=cc+1)
                if show_annotation:
                    cv2.fillPoly(mask_color, pts = [np.array(poly)], color=color_list[cc])


    if show_annotation:
        print(mask_color.shape, img.shape)
        img = cv2.addWeighted(mask_color, 0.8, img, 1, 0, img)
        cv2.imshow("img", img)

    return mask


class ImageNode(Node):
    def __init__(self,model):
        super().__init__('image_subscriber')
        self.lock = False

        self.grain_intensity = 0.35
        self.grain_size = 1.5
        self.blur_amount = 0.5

        self.publisher_ = self.create_publisher(PointCloud2, '/pcloud', 10)
        self.left_image_sub = self.create_subscription(
            Image,
            '/rgb_left',  # Replace 'tf_static' with your actual topic name
            self.left_rgb_callback,
            1)
        self.left_image_sub  # prevent unused variable warning
        self.right_image_sub = self.create_subscription(
            Image,
            '/rgb_right',  # Replace 'tf_static' with your actual topic name
            self.right_rgb_callback,
            1)
        self.right_image_sub  # prevent unused variable warning


        self.depth_subscription = self.create_subscription(
            Image,
            '/depth',
            self.depth_callback,
            10
        )
        #self.stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
        self.stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
        self.bridge = CvBridge()
        self.model = model



    def left_rgb_callback(self, msg):
        """
        if self.lock:
            return
        """
        try:
            self.left_rgb_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.left_rgb_image = edit_image(self.left_rgb_image, self.blur_amount, self.grain_intensity, self.grain_size)
            self.left_gray = cv2.cvtColor(self.left_rgb_image, cv2.COLOR_BGR2GRAY)
        except Exception as e:
            self.get_logger().info(str(e))
            self.lock = False

    def right_rgb_callback(self, msg):
        """
        if self.lock:
            return
        """
        try:
            """
            self.lock = True
            depth_image = self.left_rgb_image
            self.lock = False
            """
            self.right_rgb_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.right_rgb_image = edit_image(self.left_rgb_image, self.blur_amount, self.grain_intensity, self.grain_size)
            self.right_gray = cv2.cvtColor(self.right_rgb_image, cv2.COLOR_BGR2GRAY)

            # Perform your CV tasks here

            if hasattr(self, 'left_rgb_image'):
                stereo_image = cv2.hconcat([self.left_rgb_image, self.right_rgb_image])

                #cv2.imshow('Stereo Image', stereo_image)
                #cv2.waitKey(1)
                
                depth_map = self.compute_depth_map(self.left_gray, self.right_gray)
                

                print(depth_map.dtype)
                #cv2.imshow('Depth Map', depth_map)
                #cv2.waitKey(1)
                depth_map = self.latest_depth_image
                depth_image = depth_map
                #print("here")
                if depth_map is not None:
                    # Process depth and RGB together if available
                    self.process_depth_and_rgb(depth_image, self.left_rgb_image)

        except Exception as e:
            self.get_logger().info(str(e))
            self.lock = False
    
    def compute_depth_map(self, left_image, right_image):
        disparity = self.stereo.compute(self.left_gray, self.right_gray)
        f = 320
        B = 0.075
        depth_map = (f*B) / disparity

        # Step 1: Identify the maximum finite depth value
        max_depth_value = np.max(depth_map[np.isfinite(depth_map)])

        # Step 2: Replace infinity with the maximum finite depth value
        depth_fixed = np.where(np.isinf(depth_map), max_depth_value, depth_map)

        # Step 3: Normalize the depth values to a finite range (e.g., [0, 1])
        depth_map = (depth_fixed - np.min(depth_fixed)) / (np.max(depth_fixed) - np.min(depth_fixed))

        
        print(np.min(depth_map))
        print(np.max(depth_map))
        depth_map = (depth_map * 255).astype(np.uint8)
        
        tune1 = 5 #15 good #9 #5
        "Helps smooth the noise"
        depth_map = cv2.GaussianBlur(depth_map.astype(np.uint8), (tune1, tune1), 0)

        median_filter_size = 3 # 3, 5, 7 common larger is more aggressive smoothing
        #depth_map_uint8 = cv2.convertScaleAbs(depth_map)
        "For removing the salt and pepper noise"
        #depth_map = cv2.medianBlur(depth_map.astype(np.uint8), median_filter_size)

        
        
        

        #depth_map = cv2.normalize(depth_map, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        tune2 = 9 #13 #9 diameter of pixel neighborhood
        tune3 = 75 #99 #75
        tune4 = 75 #75 standard deviation in color and standard deviation i nspace
        #depth_map = self.bilateral_filter(depth_map, d=tune2, sigma_color=tune3, sigma_space=tune4)

        median_filter_size = 5 # 3, 5, 7 common larger is more aggressive smoothing
        #depth_map_uint8 = cv2.convertScaleAbs(depth_map)
        #depth_map = cv2.medianBlur(depth_map.astype(np.uint8), median_filter_size)
        tune1 = 5 #15 good #9 #5
        "Helps smooth the noise"
        #depth_map = cv2.GaussianBlur(depth_map, (tune1, tune1), 0)
        #depth_map = cv2.GaussianBlur(depth_map.astype(np.uint8), (tune1, tune1), 0)
        #depth_map = depth_map.astype(np.float64)
        print(np.min(depth_map))
        print(np.max(depth_map))
        
        return depth_map

    def bilateral_filter(self, depth_map, d, sigma_color, sigma_space):
        """
        Apply bilateral filtering to the input image.

        Args:
        - image: Input image (grayscale or color).
        - d: Diameter of each pixel neighborhood.
        - sigma_color: Standard deviation in the color space.
        - sigma_space: Standard deviation in the coordinate space.

        Returns:
        - Filtered image.
        """
        #depth_map = cv2.convertScaleAbs(depth_map)
        filtered_depth_map = cv2.bilateralFilter(depth_map.astype(np.uint8), d, sigma_color, sigma_space)
        return filtered_depth_map

    def depth_callback(self, msg):
        if self.lock:
            return
        try:
            self.lock = True
            self.latest_depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            self.lock = False

        except Exception as e:
            self.get_logger().info(str(e))
            self.lock = False


    def process_depth_and_rgb(self, depth_image, rgb_image):
         # Perform segmentation
        mask = get_seg_mask(self.model, rgb_image, show_annotation=False)
        cv2.waitKey(1)
        fx, fy, cx, cy = 1, 1, 1, 1
        fx = rgb_image.shape[1] / (2 * np.tan(np.radians(90) / 2))
        fy = rgb_image.shape[0] / (2 * np.tan(np.radians(90) / 2))
        cx = rgb_image.shape[0]//2
        cy = rgb_image.shape[1]//2
        class_ids = np.arange(4) + 1  # 4 class ids, starting at 1
        xyz = depth_to_xyz_segmented(depth_image, mask, class_ids, fx, fy, cx, cy)
        pointcloud = create_pointcloud2(xyz)
        self.publish_point_cloud(pointcloud)


    def publish_point_cloud(self, pointcloud_msg):
        self.publisher_.publish(pointcloud_msg)




def main(args=None):
    rclpy.init(args=args)
    #current_directory = os.getcwd()
    current_directory = 'path/to/current/directory'
    pt_file = os.path.join(current_directory,"guardrail_etc_mar22.pt")
    #pt_file = '/home/nathan/Desktop/grain_fill/guardrail_etc_mar1.pt'
    model = YOLO(pt_file)
    image_subscriber = ImageNode(model)
    #depth_subscriber = DepthSubscriber(pt_file)
    rclpy.spin(image_subscriber)
    image_subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    import torch
    import torchvision

    print("PyTorch version:", torch.__version__)
    print("torchvision version:", torchvision.__version__)

    main()
