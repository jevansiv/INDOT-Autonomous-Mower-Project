import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from oscar_ros_msgs.msg import WheelSpeeds
from tf2_msgs.msg import TFMessage
from geometry_msgs.msg import PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
#from tf_transformations import euler_from_quaternion

import cv2
from ultralytics import YOLO
import numpy as np
from cv_bridge import CvBridge
import math

import torch
import torch.nn as nn
import math
from nav_model_transformer import Nav_Model

class NetCV_Conv(nn.Module):
    def __init__(self):
        super(NetCV_Conv, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        in_size = 512

        self.linear_layers = nn.Sequential(
            nn.Linear(in_features=1, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=128),
            nn.ReLU()
        )

        self.linear_layers2 = nn.Sequential(
            nn.Linear(in_features=in_size+128, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=1),
        )

    def forward(self, img, twist):
        x = self.conv_layers(img)
        x = x.view(x.size(0), -1)
        y = self.linear_layers(twist)
        z = torch.cat((y, x), dim=1)
        z = self.linear_layers2(z)
        return z

class ImageProcessingNode(Node):
    def __init__(self):
        super().__init__('image_processing_node')
        self.subscription = self.create_subscription(
            Image,
            'rgb_left',  # Adjust topic name based on your setup
            self.image_callback,
            10)
        #self.subscription = self.create_subscription(TFMessage, '/tf', self.tf_callback, 10)
        #self.amcl_subscription = self.create_subscription(PoseWithCovarianceStamped,'/amcl_pose',self.amcl_pose_callback,10)
        #self.odom_subscription = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.publisher = self.create_publisher(WheelSpeeds, 'target_wheel_speeds_rl', 10)
        self.subscriber = self.create_subscription(WheelSpeeds, 'measured_local', self.measured_local_callback, 10)

        self.seg_model = YOLO("all_obstacles_june21.pt")
        # self.nav_model = NetCV_Conv()
        # self.nav_model.eval()
        # self.nav_model.load_state_dict(torch.load("obstacle_avoid_cnn.pt", map_location='cpu'))
        self.nav_model = Nav_Model(in_scalars=1, num_outputs=1, max_history=48).to('cpu') # transformer1
        self.nav_model.eval()
        # nav_model = Nav_Model().to(device)
        self.nav_model.load_state_dict(torch.load("obstacle_avoid_transformer_sep11_checkpoint.pt", map_location='cpu'))
        self.bridge = CvBridge()
        self.current_location = [0,0]
        self.heading = 0

    def edit_image(self, image, blur_amount=0, grain_intensity=0, grain_size=1):
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

    def measured_local_callback(self, msg):
       local_details = msg
       x = local_details.fl/100
       y = local_details.fr/100*-1
       self.current_location = [x, y]
       self.heading = local_details.rl*-math.pi/180

    def image_callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        # blur amount, graint intensity, # grain_size
        cv_image = self.edit_image(cv_image,0.5, 0.2, 1.5)
        twist, speed = self.navigate_to_destination(cv_image)
        scale =50 #150
        
        fl_speed = speed * (1 - twist / 2)
        fr_speed = speed * (1 + twist / 2)
        rl_speed = fl_speed
        rr_speed = fr_speed

        wheel_speeds_msg = WheelSpeeds()
        wheel_speeds_msg.fl = fl_speed*scale
        wheel_speeds_msg.fr = fr_speed*scale
        wheel_speeds_msg.rl = rl_speed*scale
        wheel_speeds_msg.rr = rr_speed*scale
        self.publisher.publish(wheel_speeds_msg)

        #self.get_logger().info(f"Published wheel speeds: FL={fl_speed}, FR={fr_speed}, RL={rl_speed}, RR={rr_speed}")

    def get_mask(self, img, detection_size=416):
        h, w = img.shape[0:2]
        mask = np.zeros((detection_size, detection_size, 1), np.uint8)
        results = self.seg_model.predict(source=cv2.resize(img, (detection_size, detection_size), interpolation=cv2.INTER_AREA), save=False, conf=0.15)
        
        for result in results:
            if result.masks is not None:
                for j, i in enumerate(result.masks.xy):
                    if int(result.boxes.cls[j]) == 4:  # Check if this is actually the obstacle class
                        l = [[int(i[n, 0]), int(i[n, 1])] for n in range(i.shape[0])]
                        if len(l) > 0:
                            cv2.fillPoly(mask, [np.array(l)], 255)
        
        cv2.imshow("mask", mask)
        cv2.waitKey(1)
        return mask

    def navigate_to_destination(self, img):
        #current_location = [0, 0]  # Example placeholder values
        #heading = 0  # Example placeholder value
        destination = [0, -30]  # Example placeholder values
        # Convert heading to range [0, 360] degrees
        self.heading = (self.heading * 180 / math.pi) % 360
        self.heading = self.heading * math.pi / 180
        #direct_heading = (direct_heading * 180 / math.pi) % 360
        """
        print("Current Location: " + str(self.current_location[0]) + " , " + str(self.current_location[1]) + 
              "; Heading: " + str(self.heading) + " , " + str(direct_heading))
        """
        direct_heading = math.atan2(destination[1]-self.current_location[1], destination[0] - self.current_location[0]) % (2 * math.pi)
        #direct_heading = (-math.atan2(self.current_location[1] - destination[1], self.current_location[0] - destination[0]) + math.pi / 2) % (2 * math.pi)
        print("Current Location: " + str(self.current_location[0]) + " , " + str(self.current_location[1]) + '; Heading: ' + str(self.heading*180/math.pi) + ' , ' + str(direct_heading*180/math.pi))
        #print("Destination Location: " + current_location[0] + " , " + current_location[1])
        #self.heading %= (2 * math.pi)
        angle_off = self.heading - direct_heading
        angle_off = (angle_off + math.pi) % (2 * math.pi) - math.pi
        direct_twist = -angle_off * 2

        mask = self.get_mask(img)  # Use computer vision to get black/white mask of obstacle
        mask = cv2.resize(mask, (32, 32))
        mask = mask.astype(np.float32) / 255
        inp = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)
        tw = torch.tensor([direct_twist])
        # print(inp.shape, tw)
        twist = self.nav_model(inp, [tw]).item()
        #twist = direct_twist
        speed = min(max(0.05, 1 / max(0.01, abs(twist * 2)) - 0.5), 5)

        print(twist)
        print("speed")
        print(speed)

        return twist, speed

def main(args=None):
    rclpy.init(args=args)
    image_processing_node = ImageProcessingNode()
    rclpy.spin(image_processing_node)
    image_processing_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
