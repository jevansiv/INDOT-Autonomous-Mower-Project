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
import csv  # Import the CSV module

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

        self.seg_model = YOLO("guardrail_sign_post_sep24.pt")
        #self.nav_model = Nav_Model(in_scalars=1, num_outputs=1, max_history=48).to('cpu')  # transformer1
        self.nav_model = Nav_Model().to('cpu')
        self.nav_model.eval()
        self.nav_model.load_state_dict(torch.load("obstacle_avoid_transformer_guardrail_sep17_checkpoint.pt", map_location='cpu'))
        self.bridge = CvBridge()
        self.current_location = [0, 0]
        self.heading = 0

        # Open CSV file in append mode and write the header (if necessary)
        self.csv_file = open('rl_UE_testing_asset.csv', mode='a', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(['X', 'Y', 'Heading'])  # Writing header

    def log_to_csv(self, x, y, heading):
        """Logs the x, y, and heading to the CSV file."""
        self.csv_writer.writerow([x, y, heading])
        self.csv_file.flush()  # Ensure data is written to the file

    def get_mask(self, img, detection_size=416):
        h, w = img.shape[0:2]
        mask = np.zeros((detection_size, detection_size, 1), np.uint8)
        results = self.seg_model.predict(source=cv2.resize(img, (detection_size, detection_size), interpolation=cv2.INTER_AREA), save=False, conf=0.15)
        
        for result in results:
            if result.masks is not None:
                for j, i in enumerate(result.masks.xy):
                    if int(result.boxes.cls[j]) == 0:  # Check if this is actually the obstacle class
                        l = [[int(i[n, 0]), int(i[n, 1])] for n in range(i.shape[0])]
                        if len(l) > 0:
                            cv2.fillPoly(mask, [np.array(l)], 255)
        
        cv2.imshow("mask", mask)
        cv2.waitKey(1)
        return mask

    def measured_local_callback(self, msg):
        local_details = msg
        x = local_details.fl / 100
        y = local_details.fr / 100 * -1
        self.current_location = [x, y]
        self.heading = local_details.rl * -math.pi / 180

        # Log the current position and heading to the CSV
        self.log_to_csv(self.current_location[0], self.current_location[1], self.heading)

    def image_callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        twist, speed = self.navigate_to_destination(cv_image)
        scale = 50

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

    def navigate_to_destination(self, img):
        destination = [16.256,2.101]
        self.heading = (self.heading * 180 / math.pi) % 360
        self.heading = self.heading * math.pi / 180

        direct_heading = math.atan2(destination[1] - self.current_location[1], destination[0] - self.current_location[0]) % (2 * math.pi)
        angle_off = self.heading - direct_heading
        """
        if angle_off > math.pi:
            angle_off = 2 * math.pi - angle_off
        elif angle_off < -math.pi:
            angle_off += math.pi
        direct_twist = -angle_off * 2
        """

        angle_off = (angle_off + math.pi) % (2 * math.pi) - math.pi
        direct_twist = -angle_off * 2

        mask = self.get_mask(img)
        mask = cv2.resize(mask, (32, 32))
        mask = mask.astype(np.float32) / 255
        inp = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)
        tw = torch.tensor([direct_twist])

        twist = self.nav_model(inp, [-tw]).item()*-1
        #twist = direct_twist
        speed = min(max(0.05, 1 / max(0.01, abs(twist * 2)) - 0.5), 5)

        return twist, speed

    def __del__(self):
        """Close the CSV file when the node is destroyed."""
        self.csv_file.close()

def main(args=None):
    rclpy.init(args=args)
    image_processing_node = ImageProcessingNode()
    rclpy.spin(image_processing_node)
    image_processing_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
