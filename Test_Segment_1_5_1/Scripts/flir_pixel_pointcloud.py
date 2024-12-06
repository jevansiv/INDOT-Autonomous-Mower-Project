#!/usr/bin/env python

import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2, PointField
import random
import std_msgs.msg
from cv_bridge import CvBridge
import numpy as np
import sensor_msgs_py.point_cloud2 as pc2
from builtin_interfaces.msg import Time

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

class ImageSubscriber(Node):
    def __init__(self):
        super().__init__('image_subscriber')
        self.grain_intensity = 0.05
        self.grain_size = 1.0
        self.blur_amount = 0.1
        self.subscription = self.create_subscription(
            Image,
            '/flir',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning
        self.bridge = CvBridge()
        self.publisher_ = self.create_publisher(PointCloud2, '/pcloud', 10)

    def listener_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            cv_image = edit_image(cv_image, self.blur_amount, self.grain_intensity, self.grain_size)
            
            # Extract the bottom third area of the image
            height = cv_image.shape[0]
            bottom_third = cv_image[int(height * 2 / 3):, :]
            
            # Process the bottom third area to find points with values above 38
            thermal_mask = bottom_third[:, :, 0] > 38  # Assuming thermal values are in the first channel

            # Create point cloud from the thermal mask
            points = self.create_points_from_mask(thermal_mask, int(height * 2 / 3))

            # Create and publish the PointCloud2 message
            pointcloud_msg = self.create_pointcloud2(points)
            self.publisher_.publish(pointcloud_msg)

            # Display the image (for debugging purposes)
            cv2.imshow("Image", cv_image)
            cv2.waitKey(1)
        except Exception as e:
            self.get_logger().info(str(e))

    def create_points_from_mask(self, mask, y_offset):
        """
        Create points from a binary mask where the mask is True, with an offset for the y-coordinates.

        Parameters:
        - mask: A 2D binary numpy array.
        - y_offset: The offset to add to the y-coordinates.

        Returns:
        - points: A 2D numpy array of shape (N, 3) containing XYZ coordinates.
        """
        # Use connected components to find clusters of pixels
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)

        points = []
        for i in range(1, num_labels):  # Start from 1 to skip the background
            if stats[i, cv2.CC_STAT_AREA] >= 5:  # Only consider clusters with at least 5 pixels
                print("Cluster")
                x_center, y_center = centroids[i]
                y_center += y_offset  # Add the offset to the y-coordinates


                y_indices, x_indices = np.where(mask)
                noise = np.random.normal(-0.1,0.1,len(x_indices))
                x_indices_use = np.zeros_like(x_indices) + noise + 0.5
                noise = np.random.normal(-0.2,0.2,len(x_indices))
                y_indices = np.zeros(len(x_indices)) + noise + 0.3
                z_indices = np.zeros_like(x_indices)-0.3  # Place the point cloud 2 meters ahead
                points = np.vstack((x_indices_use, y_indices, z_indices)).transpose()

        return np.array(points)

    def create_pointcloud2(self, points):
        """
        Create a ROS2 PointCloud2 message from a list of points.

        Parameters:
        - points: A 2D numpy array of shape (N, 3) containing XYZ coordinates.

        Returns:
        - pointcloud2_msg: A sensor_msgs/PointCloud2 message.
        """
        header = std_msgs.msg.Header()
        header.stamp = Time()  # self.get_clock().now().to_msg()
        header.frame_id = "camera_frame"  # Change to your frame ID

        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1)
        ]

        grouped_xyz = [tuple(point) for point in points]
        pointcloud2_msg = pc2.create_cloud(header, fields, grouped_xyz)
        return pointcloud2_msg

def main(args=None):
    rclpy.init(args=args)
    image_subscriber = ImageSubscriber()
    rclpy.spin(image_subscriber)
    image_subscriber.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
