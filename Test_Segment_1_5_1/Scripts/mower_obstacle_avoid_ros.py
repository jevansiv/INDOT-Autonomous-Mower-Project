import time
import math
import numpy as np
import cv2
import pyproj
from ultralytics import YOLO
import torch
import torch.nn as nn
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
from oscar_ros_msgs.msg import WheelSpeeds
from geometry_msgs.msg import PoseWithCovarianceStamped
from sensor_msgs.msg import Image, NavSatFix
from tf2_msgs.msg import TFMessage
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge

import torch
import torch.nn as nn
import math
from nav_model_transformer import Nav_Model

# Load the segmentation model
seg_model = YOLO("all_obstacles_june21.pt")
device = torch.device("cpu")  # use "cuda" if torch.cuda.is_available() else "cpu"

class NetCV_Conv(nn.Module):
    def __init__(self):
        super(NetCV_Conv, self).__init__()
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

        in_size = 512

        self.linear_layers = nn.Sequential(
            nn.Linear(in_features=1, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=128),
            nn.ReLU()
        )

        self.linear_layers2 = nn.Sequential(
            nn.Linear(in_features=in_size + 128, out_features=256),
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


class Mower(Node):
    def __init__(self):
        super().__init__('mower_node')
        self.subscription = self.create_subscription(
            Image,
            'rgb_left',  # Adjust topic name based on your setup
            self.image_callback,
            10)
        self.publisher = self.create_publisher(WheelSpeeds, 'target_wheel_speeds_rl', 10)
        self.subscriber = self.create_subscription(WheelSpeeds, 'measured_local', self.measured_local_callback, 10)
        self.gps_subriber = self.create_subscription(NavSatFix,'/gps',self.gps_callback, 10)

        # Local variables to store position and heading
        self.current_location = [0.0, 0.0]  # Local coordinates (x, y)
        self.latitude = 0
        self.longitude = 0
        self.heading = 0.0  # Heading in radians
        self.mower_width = 1.1  # meters
        self.last_error = 0
        self.last_destination = [-1, -1]
        self.at_start = False
        self.nav_model = NetCV_Conv().to('cpu')
        self.nav_model.load_state_dict(torch.load("obstacle_avoid_cnn.pt", map_location='cpu'))
        self.nav_model.eval()
        self.bridge = CvBridge()
        northwest = [40.4688786, -86.9907383]
        southwest = [40.4687571, -86.9907369]
        southeast = [40.4687559, -86.9905444]
        northeast = [40.4688837, -86.9905461]
        points = [northwest, southwest, southeast, northeast]
        self.generate_map(points)

    def gps_callback(self, msg):
        # Extract latitude and longitude from the GPS message
        self.latitude = msg.latitude
        self.longitude = msg.longitude

    def get_mask(self, img, detection_size=416):

        h, w = img.shape[0:2]
        mask = np.zeros((detection_size, detection_size, 1), np.uint8)
        return mask
        results = seg_model.predict(source=cv2.resize(img, (detection_size, detection_size), interpolation=cv2.INTER_AREA), save=False, conf=0.15)
        for result in results:
            if result.masks is not None:
                for j, i in enumerate(result.masks.xy):
                    if int(result.boxes.cls[j]) == 4:  # check if this is actually the obstacle class
                        l = [[int(i[n, 0]), int(i[n, 1])] for n in range(i.shape[0])]
                        if len(l) > 0:
                            cv2.fillPoly(mask, [np.array(l)], 255)

        cv2.imshow("mask", mask)
        cv2.waitKey(1)
        return mask

    def measured_local_callback(self, msg):
        local_details = msg
        x = local_details.fl / 100  # Convert to meters
        y = local_details.fr / 100 * -1  # Invert Y coordinate
        self.current_location = [x, y]
        self.heading = local_details.rl * -math.pi / 180  # Convert to radians

    def image_callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        twist, speed = self.navigate(cv_image)

        twist = twist * 15
        # Adjust speeds based on twist and speed
        scale = 10  # Scale for wheel speed
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

    def generate_map(self, points):
        # Local map generation logic based on input points (x, y)
        start_coords = [min(sublist[0] for sublist in points), min(sublist[1] for sublist in points)]
        end_coords = [max(sublist[0] for sublist in points), max(sublist[1] for sublist in points)]

        # Local EPSG projection for consistent map coordinates
        p = pyproj.Proj('epsg:2793')
        self.start_x, self.start_y = p(start_coords[1], start_coords[0])
        self.end_x, self.end_y = p(end_coords[1], end_coords[0])
        self.range_x, self.range_y = abs(self.start_x - self.end_x), abs(self.start_y - self.end_y)

        if self.range_x > self.range_y:
            self.path_covered_img = np.zeros((int(500 * self.range_y / self.range_x), 500), np.uint8)
            self.mower_width_px = int(500 / self.range_x * self.mower_width)
        else:
            self.path_covered_img = np.zeros((500, int(500 * self.range_x / self.range_y)), np.uint8)
            self.mower_width_px = int(500 / self.range_y * self.mower_width)

    def pid_navigate(self, position, heading, destination, img):
        #x, y = position
        # Local EPSG projection for consistent map coordinates
        p = pyproj.Proj('epsg:2793')
        x, y = p(self.longitude, self.latitude)

        correct_heading = (math.atan2(destination[1]-y, destination[0] - x) + math.pi) % (2 * math.pi)
        #correct_heading = (-math.atan2(y-destination[1], x-destination[0])+math.pi/2 + math.pi) % (2*math.pi)

        angle_off = heading - correct_heading
        angle_off = (angle_off + math.pi) % (2 * math.pi) - math.pi

        print(heading, correct_heading, angle_off)

        print("Current Location: " + str(x) + " , " + str(y) + '; Heading: ' + str(self.heading*180/math.pi) + ' , ' + str(correct_heading*180/math.pi))


        close = False

        mask = self.get_mask(img)
        # obstacles here
        """
        if np.max(mask) > 0:
            block_bottom = np.max((np.where(mask!=0))[0])
            if block_bottom/mask.shape[0] > 0.7:
                print("close")
                close = True
        """

        twist = -angle_off*2


        if close:
            mask = cv2.resize(mask, (32, 32))
            mask = mask.astype(np.float32)/255
            inp = torch.tensor(mask, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            twist = self.nav_model(inp.to(device), torch.tensor([[twist]]))[0, 0]
            twist = float(twist.item()) # convert output to float

        speed = min(max(0.05, 1/max(0.01, abs(twist*2))-0.5), 5)


        return twist, speed

    def navigate(self, img):
        print(self.longitude, self.latitude)
        p = pyproj.Proj('epsg:2793')
        x, y = p(self.longitude, self.latitude)
        heading = self.heading

       
        if self.longitude == 0:
            return 0,0
        # position of mower in path_covered_img map
        x_rect = int((x - min(self.start_x, self.end_x)) / self.range_x * self.path_covered_img.shape[1])
        y_rect = self.path_covered_img.shape[0] - int((y - min(self.start_y, self.end_y)) / self.range_y * self.path_covered_img.shape[0])


        if not self.at_start: # have not yet reached initial starting point

            # find closest point to move to. Also find furthest point to move to.

            dest_xy = [self.start_x, self.start_y]
            opposite = [self.end_x, self.end_y]
            if abs(x-self.start_x) > abs(x-self.end_x):
                dest_xy[0] = self.end_x
                opposite[0] = self.start_x
            if abs(y-self.start_y) > abs(y-self.end_y):
                dest_xy[1] = self.end_y
                opposite[1] = self.start_y

            twist, speed = self.pid_navigate([x,y], heading, dest_xy, img)

            print("distance", (x-dest_xy[0])**2 + (y-dest_xy[1])**2)
            if (x-dest_xy[0])**2 + (y-dest_xy[1])**2 < 1.0: # got to start point
                self.at_start = True

                self.side1 = dest_xy[:] # side the robot starts at
                self.side2 = opposite[:] # other side. Switches

                self.lap = 0
                self.turning = True


                if abs(x-opposite[0]) > abs(y-opposite[1]): 
                    self.orientation = 1 # travelling along x-axis
                    print("moving x")

                else: # not actually travelling
                    self.orientation = 0 # travel along y axis
                    print("moving y")


                if self.side1[(self.orientation)%2] < self.side2[(self.orientation)%2]:
                    self.cover_dir = 1
                    print("left to right or bottom to top", x, self.side1[(self.orientation)%2], self.side2[(self.orientation)%2])
                else:
                    self.cover_dir = -1
                    print("right to left or top to bottom")

                self.cover_dir *= self.mower_width*0.8 # this affects how much it shifts over when starting new lap


                if self.orientation == 1:
                    self.turn_dest = [self.side1[0], self.side1[1]+1*self.cover_dir*0]
                else:
                    self.turn_dest = [self.side1[0]+1*self.cover_dir*0, self.side1[1]]

                self.side1[self.orientation] = 0
                self.side2[self.orientation] = 0


            return twist, speed

        else:
            cv2.rectangle(self.path_covered_img, (x_rect, y_rect), (x_rect,y_rect), 255, self.mower_width_px)

            if self.lap % 2 == 0: # go from side1 to side2
                s1 = self.side1
                s2 = self.side2
            else:
                s1 = self.side2
                s2 = self.side1


            if not self.turning:
                if self.orientation == 1: # going along x axis
                    dest = [x, y]

                    if x < s2[0]: # destination should be 1 meter ahead
                        dest[0] += 1
                    else:
                        dest[0] -= 1

                    x_rect_target = int((dest[0] - min(self.start_x, self.end_x)) / self.range_x * self.path_covered_img.shape[1])
                    x_rect_target = min(self.path_covered_img.shape[1]-1, max(0, x_rect_target))

                    mow_edge = (np.where(self.path_covered_img[:, x_rect_target]==0))[0]
                    target_y = 0
                    if len(mow_edge) > 0 and self.cover_dir > 0:
                        target_y = self.path_covered_img.shape[0] - np.max(mow_edge)
                    elif len(mow_edge) > 0 and self.cover_dir < 0:
                        target_y = self.path_covered_img.shape[0] - np.min(mow_edge)
                    elif len(mow_edge) == 0:
                        print("all done")
                        return 0,0

                    target_y = target_y * self.range_y / self.path_covered_img.shape[0] + min(self.start_y, self.end_y)
                    dest[1] = target_y


                    if (s2[0] < x > s1[0] or s2[0] > x < s1[0]):
                        if abs(x-s2[0]) < abs(x-s1[0]):
                            print("reached point x")
                            self.turning = True
                            dest = [x, y]
                            self.turn_dest = [s2[0], y+1*self.cover_dir]
                            self.lap += 1

                            white_line_y = None
                            for y in range(self.path_covered_img.shape[0]):
                                if np.all(self.path_covered_img[y, :] == 255):
                                    white_line_y = y
                                    break

                            if white_line_y is not None:
                                if self.cover_dir > 0:
                                    self.path_covered_img[white_line_y+1:, :] = 128
                                else:
                                    self.path_covered_img[0:white_line_y+1, :] = 128

                else: # going along y axis
                    dest = [x, y]
                    if y < s2[1]:
                        dest[1] += 1
                    else:
                        dest[1] -= 1


                    y_rect_target = self.path_covered_img.shape[0] - int((dest[1] - min(self.start_y, self.end_y)) / self.range_y * self.path_covered_img.shape[0])
                    y_rect_target = min(self.path_covered_img.shape[0]-1, max(0, y_rect_target))

                    mow_edge = (np.where(self.path_covered_img[y_rect_target, :]==0))[0]

                    target_x = 0
                    if len(mow_edge) > 0 and self.cover_dir > 0:
                        target_x = np.min(mow_edge)
                    elif len(mow_edge) > 0 and self.cover_dir < 0:
                        target_x = np.max(mow_edge)
                    elif len(mow_edge) == 0:
                        print("all done")
                        return 0,0

                    target_x = target_x * self.range_x / self.path_covered_img.shape[1] + min(self.start_x, self.end_x)
                    dest[0] = target_x

                    if (s2[1] < y > s1[1] or s2[1] > y < s1[1]):
                        if  abs(y-s2[1]) < abs(y-s1[1]):
                            print("reached point y")
                            self.turning = True
                            dest = [x, y]
                            self.turn_dest = [x+1*self.cover_dir, s2[1]]
                            self.lap += 1

                            white_line_x = None
                            for x in range(self.path_covered_img.shape[1]):
                                if np.all(self.path_covered_img[:, x] == 255):
                                    white_line_x = x
                                    break

                            if white_line_x is not None:
                                if self.cover_dir < 0:
                                    self.path_covered_img[:, white_line_x+1:] = 128
                                else:
                                    self.path_covered_img[:, 0:white_line_x+1] = 128

            else: # turning to start new lap
                dest = self.turn_dest
                if (x-dest[0])**2 + (y-dest[1])**2 < 1:
                    print("done turning")
                    self.turning = False
            cv2.imshow("pth", self.path_covered_img)
            cv2.waitKey(1)

            twist, speed = self.pid_navigate([x,y], heading, dest, img)

            return twist, speed

def main(args=None):
    rclpy.init(args=args)
    mower = Mower()
    rclpy.spin(mower)
    mower.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
