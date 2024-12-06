import rclpy
from rclpy.node import Node
from sensor_msgs.msg import NavSatFix
import csv
import os

class GpsDataExtractor(Node):

    def __init__(self):
        super().__init__('gps_data_extractor_node')

        self.create_subscription(NavSatFix, '/gps/fix', self.callback, 10)

        self.output_csv_filename = "/path/to/output_csv_filename.csv" #update path to csv for output


        # Check if CSV file exists, create it if it doesn't
        if not os.path.isfile(self.output_csv_filename):
            with open(self.output_csv_filename, 'w', newline='') as csv_file:
                fieldnames = ['timestamp', 'latitude', 'longitude', 'altitude']
                writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                writer.writeheader()


        self.csv_file = open(self.output_csv_filename, 'a', newline='')
        self.fieldnames = ['timestamp', 'latitude', 'longitude', 'altitude']
        self.writer = csv.DictWriter(self.csv_file, fieldnames=self.fieldnames)
        #self.writer.writeheader()

    def callback(self, msg):
        timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        latitude = msg.latitude
        longitude = msg.longitude
        altitude = msg.altitude

        self.writer.writerow({
            'timestamp': timestamp,
            'latitude': latitude,
            'longitude': longitude,
            'altitude': altitude
        })

    def on_shutdown(self):
        self.csv_file.close()
        self.get_logger().info('Shutting down')

def main():
    rclpy.init()
    node = GpsDataExtractor()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

