#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
import numpy as np

class RadarPlotter(Node):

    def __init__(self):
        super().__init__('radar_plotter')
        self.subscription = self.create_subscription(
            PointCloud2,
            '/radar_raw',
            self.listener_callback,
            10)
        self.publisher = self.create_publisher(PointCloud2, '/pcloud', 10)
        self.subscription  # prevent unused variable warning

    def listener_callback(self, msg):
        # Convert PointCloud2 message to numpy array
        points = self.pointcloud2_to_array(msg)

        # Filter points
        filtered_points = points[(np.linalg.norm(points[:, :3], axis=1) <= 60) & (points[:, 2] >= -0.8)]

        # Create new PointCloud2 message
        new_msg = self.create_pointcloud2(filtered_points, msg.header)


        # Publish the new PointCloud2 message
        self.publisher.publish(new_msg)

    def create_pointcloud2(self, points, header):
        # Create a new PointCloud2 message
        fields = [
            point_cloud2.PointField('x', 0, point_cloud2.PointField.FLOAT32, 1),
            point_cloud2.PointField('y', 4, point_cloud2.PointField.FLOAT32, 1),
            point_cloud2.PointField('z', 8, point_cloud2.PointField.FLOAT32, 1),
            point_cloud2.PointField('intensity', 12, point_cloud2.PointField.FLOAT32, 1)
        ]
        point_cloud_data = [struct.pack('ffff', *point) for point in points]
        point_cloud_data = b''.join(point_cloud_data)

        new_msg = PointCloud2()
        new_msg.header = header
        new_msg.height = 1
        new_msg.width = len(points)
        new_msg.fields = fields
        new_msg.is_bigendian = False
        new_msg.point_step = 16
        new_msg.row_step = new_msg.point_step * new_msg.width
        new_msg.data = point_cloud_data
        new_msg.is_dense = True

        return new_msg

    def pointcloud2_to_array(self, cloud_msg):
        points = []
        for point in self.iter_points(cloud_msg):
            points.append([point[0], point[1], point[2], point[3]])
        return np.array(points)

    def iter_points(self, cloud_msg):
        gen = point_cloud2.read_points(cloud_msg, field_names=("x", "y", "z", "intensity"), skip_nans=True)
        for point in gen:
            yield point

def main(args=None):
    rclpy.init(args=args)
    radar_plotter = RadarPlotter()
    rclpy.spin(radar_plotter)
    radar_plotter.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
