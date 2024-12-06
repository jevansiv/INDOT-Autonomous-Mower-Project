#!/usr/bin/env python

import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import numpy as np

class ImageSubscriber(Node):
    def __init__(self):
        super().__init__('image_subscriber')
        self.subscription = self.create_subscription(
            Image,
            '/flir',
            self.listener_callback,
            10)
        self.publisher = self.create_publisher(Twist, '/target_twist', 10)
        self.bridge = CvBridge()
        self.stop_counter = 0
        self.stop_threshold = 50  # Corresponds to 5 seconds if listener_callback is called at 10Hz
        self.backup_duration = 2  # Duration to back up in seconds
        self.backup_counter = 0
        self.backing_up = False
        self.last_stop_time = None
        self.consecutive_stop_threshold = 20  # Time threshold for consecutive stops in seconds
        self.consecutive_stops = 0
        self.direction = 1

    def listener_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")

            # Convert the image to grayscale
            gray_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

            # Get the dimensions of the image
            height, width = gray_image.shape

            # Define the bottom third of the image
            bottom_third = gray_image[int(height * 2 / 3):, :]
            bottom_third = bottom_third[:,int(width*1/3):int(width*2/3)]
            bottom_third_left = bottom_third[:, :int(width * 1/3)]
            bottom_third_right = bottom_third[:, int(width * 2/3):]


            # Initialize twist message
            twist_msg = Twist()

            if self.backing_up:
                # Back up with specific velocities
                twist_msg.linear.x = -1.0
                twist_msg.angular.z = 1.0*self.direction  # Adjust this value as needed
                self.publisher.publish(twist_msg)

                self.backup_counter += 1
                if self.backup_counter >= self.backup_duration * 10:  # 10Hz * duration
                    # Stop backing up
                    self.backing_up = False
                    self.backup_counter = 0

                return
            threshold = 60
            # Check for values greater than 60 in the bottom third
            if np.any(bottom_third > threshold):
                #left_count =  np.count_nonzero(bottom_third_left_half > threshold)
                #right_count = np.count_nonzero(bottom_third_right_half > threshold)
                #if left_count > right_count:
                #    self.direction = 1
                #else:
                #    self.direction = -1
                self.direction = 1
                self.get_logger().info('stop')
                current_time = self.get_clock().now().to_msg().sec
                if self.last_stop_time is not None and (current_time - self.last_stop_time) <= self.consecutive_stop_threshold:
                    self.consecutive_stops += 1
                else:
                    self.consecutive_stops = 1
                self.last_stop_time = current_time
                self.stop_counter += 1
            else:
                # Check for high values (greater than 38) in the bottom third
                high_values_mask = bottom_third > 38
                left_half_mask = high_values_mask[:, :int(width / 2)]
                right_half_mask = high_values_mask[:, int(width / 2):]

                left_high_value_count = np.count_nonzero(left_half_mask)
                right_high_value_count = np.count_nonzero(right_half_mask)
                """
                if left_high_value_count > right_high_value_count:
                    val_out = "left"
                else:
                    val_out = "right"
                self.get_logger().info(val_out)
                """
                if left_high_value_count > right_high_value_count:
                    self.direction = 1  # Set direction flag for left
                elif right_high_value_count > left_high_value_count:
                    self.direction = -1  # Set direction flag for right

                # Determine the threshold for a large area (e.g., 20% of the bottom third)
                large_area_threshold = 0.2 * bottom_third.size

                if left_high_value_count > large_area_threshold or right_high_value_count > large_area_threshold:
                    self.get_logger().info('stop')
                    self.get_logger().info('38 38 38')
                    current_time = self.get_clock().now().to_msg().sec
                    if self.last_stop_time is not None and (current_time - self.last_stop_time) <= self.consecutive_stop_threshold:
                        self.consecutive_stops += 1
                    else:
                        self.consecutive_stops = 1
                    self.last_stop_time = current_time
                    self.stop_counter += 1
                else:
                    self.get_logger().info('go')
                    twist_msg.linear.x = 1.0  # Assuming 1.0 as the go velocity
                    self.stop_counter = 0

            if self.stop_counter >= self.stop_threshold and self.consecutive_stops >= 2:
                self.get_logger().info('Backing up...')
                self.backing_up = True
                self.stop_counter = 0
                self.consecutive_stops = 0

            # Publish the twist message
            self.publisher.publish(twist_msg)

            # Show the image
            cv2.imshow("Image", cv_image)
            cv2.waitKey(1)
        except Exception as e:
            self.get_logger().info(str(e))

def main(args=None):
    rclpy.init(args=args)
    image_subscriber = ImageSubscriber()
    rclpy.spin(image_subscriber)
    image_subscriber.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
