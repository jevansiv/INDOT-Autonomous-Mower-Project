import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from oscar_ros_msgs.msg import WheelSpeeds

import cv2
from ultralytics import YOLO
import numpy as np
from cv_bridge import CvBridge, CvBridgeError

import torch
import torch.nn as nn

    

class NetCV_Conv(nn.Module):
    def __init__(self):
        super(NetCV_Conv, self).__init__()
        num_classes = 2
        # Define the convolutional layers for image processing
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

        in_size = 3200
        in_size = 1152

        self.linear_layers = nn.Sequential(
            nn.Linear(in_features=in_size, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=1),
        )

    def forward(self, x):
        x = x.unsqueeze(0)
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        # print(x.shape)
        x = self.linear_layers(x)
        return x

class ImageProcessingNode(Node):
    def __init__(self):
        super().__init__('image_processing_node')
        self.subscription = self.create_subscription(
            Image,
            'rgb_left',  # Adjust topic name based on your setup
            self.image_callback,
            10)
        self.publisher = self.create_publisher(WheelSpeeds, 'target_wheel_speeds', 10)

        # Load YOLO model for object detection
        self.model = YOLO("path/to/weights/guardrail_seg_only_april15.pt")

        # Load navigation model for speed calculation
        self.nav_model = NetCV_Conv()
        self.nav_model.eval()
        self.nav_model.zero_grad()
        self.nav_model.load_state_dict(torch.load("path/to/nav_model/simple_nav_model_reinforcement_convolutional_low_close.pt", map_location='cpu'))
        self.bridge = CvBridge()


    def image_callback(self, msg):
        # Convert ROS Image message to OpenCV image format (BGR)
        fl_speed,rl_speed,fr_speed,rr_speed = self.ros_image_to_cv(msg)

        # Create WheelSpeeds message and publish
        wheel_speeds_msg = WheelSpeeds()
        wheel_speeds_msg.fl = fl_speed
        wheel_speeds_msg.fr = fr_speed
        wheel_speeds_msg.rl = rl_speed
        wheel_speeds_msg.rr = rr_speed
        self.publisher.publish(wheel_speeds_msg)

        self.get_logger().info(f"Published wheel speeds: FL={fl_speed}, FR={fr_speed}, RL={rl_speed}, RR={rr_speed}")


    def ros_image_to_cv(self, ros_image):
        cv_image = self.bridge.imgmsg_to_cv2(ros_image, desired_encoding='bgr8')
        mask = self.detect(cv_image)
        scale = 150
        if np.max(mask)!=0:


            img = cv2.resize(mask, (50, 50))
            # cv2.imshow("img", img)
            # cv2.waitKey(10)
            inp = torch.tensor(img/255, dtype=torch.float32)
            inp = inp.unsqueeze(0)

            with torch.set_grad_enabled(False):
                speeds = self.nav_model(inp)
            s1 = -float(speeds[0])
            s2 = 1.0#float(speeds[1]) / 2

            # left and right array for wheels speeds
            speeds = (1-s1/2, 1+s1/2)
            # publish the speeds to the 4 wheels using message type
            fl = speeds[0]*scale
            rl = speeds[0]*scale
            fr = speeds[1]*scale
            rr = speeds[1]*scale
        else:
            fl = 0.0
            rl = 0.0
            fr = 0.0
            rr = 0.0
        return fl,rl,fr,rr

    def detect(self, img, imageSize=416):
        print(img.shape)
        h, w = img.shape[0:2]
        mask = np.zeros((imageSize, imageSize, 1), np.uint8)
        results = self.model.predict(source=cv2.resize(img, (imageSize, imageSize), interpolation = cv2.INTER_AREA), save=False, conf=0.2)#, conf=0.2)
        res = []
        for result in results:
            if result.masks is not None:
                for j, i in enumerate(result.masks.xy):
                    l= [[int(i[n,0]), int(i[n,1])] for n in range(i.shape[0])]
                    if len(l)>0:
                        cv2.fillPoly(mask, [np.array(l)], 255)
        cv2.imshow("mask", mask)
        cv2.waitKey(1)
        return mask




def main(args=None):
    rclpy.init(args=args)
    image_processing_node = ImageProcessingNode()
    rclpy.spin(image_processing_node)
    image_processing_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()