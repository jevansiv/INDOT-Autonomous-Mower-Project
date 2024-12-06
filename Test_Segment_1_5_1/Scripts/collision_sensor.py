import rclpy
from rclpy.node import Node
from example_interfaces.msg import String
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs_py.point_cloud2 as pc2
import numpy as np
import random
import std_msgs.msg
from builtin_interfaces.msg import Time

class CollisionHandler(Node):

    def __init__(self):
        super().__init__('collision_handler')
        
        # Subscribe to the /collision topic
        self.collision_subscriber = self.create_subscription(
            String,
            '/collision',
            self.collision_callback,
            10
        )

        # Publisher for point cloud message
        self.pointcloud_publisher = self.create_publisher(PointCloud2, '/pcloud', 10)

        # Detection probability (adjustable)
        self.hit_detection_probability = 0.80

    def collision_callback(self, msg):
        collision_status = msg.data
        self.get_logger().info(f"Received collision status: {collision_status}")
        
        if collision_status == 'safe':
            # Do nothing if status is 'safe'
            self.get_logger().info("No collision detected, safe.")
        elif collision_status == 'hit':
            # Check with probability if the hit is detected
            if random.random() < self.hit_detection_probability:
                self.get_logger().info("Hit detected with probability condition met. Publishing point cloud...")
                self.publish_pointcloud()
            else:
                self.get_logger().info("Hit detected but did not pass probability threshold.")

    def publish_pointcloud(self):
        # Define a point in 3D space
        #points = np.array([[0.5, 0.3, -0.3]])  # x=0.5, y=0.3, z=-0.3
        x_indices = 0.2
        y_indices = 0.0
        z_indices = -0.3
        points = np.vstack((x_indices,y_indices,z_indices)).transpose()
        # Create a PointCloud2 message from points
        pointcloud2_msg = self.create_pointcloud2(points)

        # Publish the point cloud
        self.pointcloud_publisher.publish(pointcloud2_msg)
        self.get_logger().info(f"Point cloud published with point at (0.5, 0.3, -0.3)")

    def create_pointcloud2(self, points):
        """
        Create a ROS2 PointCloud2 message from a list of points.

        Parameters:
        - points: A 2D numpy array of shape (N, 3) containing XYZ coordinates.

        Returns:
        - pointcloud2_msg: A sensor_msgs/PointCloud2 message.
        """
        # Correctly use header with the frame ID and current time
        header = std_msgs.msg.Header()
        header.stamp = Time() #self.get_clock().now().to_msg()  # Use ROS2 clock to get the current time
        header.frame_id = "camera_frame"  # Specify the correct frame ID

        # Define fields for the PointCloud2 message
        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1)
        ]

        # Convert points to the format needed for PointCloud2
        grouped_xyz = [tuple(point) for point in points]

        # Create and return the PointCloud2 message
        pointcloud2_msg = pc2.create_cloud(header, fields, grouped_xyz)
        return pointcloud2_msg

def main(args=None):
    rclpy.init(args=args)

    # Create the CollisionHandler node
    collision_handler = CollisionHandler()

    # Spin the node to keep it running
    rclpy.spin(collision_handler)

    # Clean up
    collision_handler.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
