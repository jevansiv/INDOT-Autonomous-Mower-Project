#!/usr/bin/env python

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os
import time

import cv2
import numpy as np
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
        self.left_subscription = self.create_subscription(Image, '/rgb_left', self.left_image_callback, 10)
        self.right_subscription = self.create_subscription(Image, '/rgb_right', self.right_image_callback, 10)
        self.bridge = CvBridge()
        self.left_index = 1
        self.right_index = 1
        self.save_directory = '/path/to/save/data'  # Specify your desired directory path here

    def left_image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            # Display the left image
            #frame = edit_image(frame, blur_amount=1.0, grain_intensity=0.5, grain_size=2.0)
            #cv2.imshow('Left Image', frame)
            #cv2.waitKey(1)  # Wait for 1 millisecond

            # Capture and save the image
            self.capture_image(frame, 'left')
        except Exception as e:
            self.get_logger().error(f"Error processing left image: {e}")

    def right_image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.capture_image(frame, 'right')
        except Exception as e:
            self.get_logger().error(f"Error processing right image: {e}")

    def capture_image(self, frame, side):
        if side == 'left':
            image_label = f"waterbottle_left_{self.left_index}.png"
            self.left_index += 1
        elif side == 'right':
            image_label = f"waterbottle_right_{self.right_index}.png"
            self.right_index += 1
        else:
            return

        time.sleep(1)
        image_path = os.path.join(self.save_directory, image_label)
        cv2.imwrite(image_path, frame)
        self.get_logger().info(f"Image {image_path} saved.")

def main(args=None):
    rclpy.init(args=args)
    image_subscriber = ImageSubscriber()
    rclpy.spin(image_subscriber)
    image_subscriber.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
