import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from oscar_ros_msgs.msg import WheelSpeeds
from geometry_msgs.msg import PoseWithCovarianceStamped
from nav_msgs.msg import Odometry

import cv2
from ultralytics import YOLO
import numpy as np
from cv_bridge import CvBridge
import math
import torch
import torch.nn as nn
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
        self.publisher = self.create_publisher(WheelSpeeds, 'target_wheel_speeds_rl', 10)
        self.subscriber = self.create_subscription(WheelSpeeds, 'measured_local', self.measured_local_callback, 10)

        self.seg_model = YOLO("all_obstacles_june21.pt")
        self.nav_model = Nav_Model(in_scalars=1, num_outputs=1, max_history=48).to('cpu')  # transformer1
        self.nav_model.eval()
        self.nav_model.load_state_dict(torch.load("obstacle_avoid_transformer_sep11_checkpoint.pt", map_location='cpu'))
        self.bridge = CvBridge()
        self.current_location = [0, 0]
        self.heading = 0

        # Waypoint generation
        self.waypoints = self.generate_waypoints(length=10, width=5, spacing=2.5)
        self.current_waypoint_index = 0
        self.total_waypoints = len(self.waypoints)

    def measured_local_callback(self, msg):
        local_details = msg
        x = local_details.fl / 100
        y = local_details.fr / 100 * -1
        self.current_location = [x, y]
        self.heading = local_details.rl * -math.pi / 180

    def generate_waypoints(self, length, width, spacing):
        waypoints = []
        for i in range(int(length / spacing)):
            for j in range(int(width / spacing)):
                if i % 2 == 0:  # Even rows go right
                    x = i * spacing
                    y = j * spacing
                else:  # Odd rows go left
                    x = i * spacing
                    y = (width - j * spacing)
                waypoints.append((x, y))
        return waypoints

    def image_callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        if self.current_waypoint_index < self.total_waypoints:
            destination = self.waypoints[self.current_waypoint_index]
            twist, speed = self.navigate_to_destination(cv_image, destination)

            # Publish wheel speeds
            scale = 100  # Scale factor
            fl_speed = speed * (1 - twist / 2)
            fr_speed = speed * (1 + twist / 2)
            rl_speed = fl_speed
            rr_speed = fr_speed

            wheel_speeds_msg = WheelSpeeds()
            wheel_speeds_msg.fl = fl_speed * scale
            wheel_speeds_msg.fr = fr_speed * scale
            wheel_speeds_msg.rl = rl_speed * scale
            wheel_speeds_msg.rr = rr_speed * scale
            self.publisher.publish(wheel_speeds_msg)

            # Check for waypoint completion
            if self.is_at_waypoint(destination):
                self.get_logger().info(f"Waypoint {self.current_waypoint_index + 1}/{self.total_waypoints} reached: {destination}")
                self.current_waypoint_index += 1

    def is_at_waypoint(self, waypoint):
        distance = math.sqrt((self.current_location[0] - waypoint[0]) ** 2 + (self.current_location[1] - waypoint[1]) ** 2)
        return distance < 0.5  # Tolerance of 0.5 meters

    def navigate_to_destination(self, img, destination):
        # Update heading and calculate direct heading towards the destination
        self.heading = (self.heading * 180 / math.pi) % 360
        self.heading = self.heading * math.pi / 180
        
        direct_heading = math.atan2(destination[1] - self.current_location[1], destination[0] - self.current_location[0])
        angle_off = self.heading - direct_heading
        if angle_off > math.pi:
            angle_off = 2 * math.pi - angle_off
        elif angle_off < -math.pi:
            angle_off += math.pi
        direct_twist = -angle_off * 2

        # Obtain the speed based on the twist
        speed = min(max(0.05, 1 / max(0.01, abs(direct_twist * 2)) - 0.5), 5)

        # Log current status
        self.get_logger().info(f"Current Location: {self.current_location}, Heading: {self.heading * 180 / math.pi:.2f} degrees")
        self.get_logger().info(f"Navigating to waypoint: {destination}")

        return direct_twist, speed

def main(args=None):
    rclpy.init(args=args)
    image_processing_node = ImageProcessingNode()
    rclpy.spin(image_processing_node)
    image_processing_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
