import rclpy
from rclpy.node import Node
from sensor_msgs.msg import NavSatFix
from geometry_msgs.msg import Twist

class GpsTwistPublisher(Node):
    def __init__(self):
        super().__init__('gps_twist_publisher')
        # Subscriber to the /gps topic
        self.subscription = self.create_subscription(
            NavSatFix,
            '/gps',
            self.gps_callback,
            10)
        self.subscription  # prevent unused variable warning
        
        # Publisher to the /target_twist topic
        self.twist_publisher = self.create_publisher(Twist, '/target_twist', 10)

        # Target latitude
        self.target_latitude = # set cut-off latitude or longitudes
        # Latitude tolerance for comparison
        self.latitude_tolerance = 0.0 #00001

    def gps_callback(self, msg):
        # Check if the latitude is close to the target latitude
        #if abs(msg.latitude - self.target_latitude) <= self.latitude_tolerance:
        #    self.publish_twist()
        if msg.latitude < (self.target_latitude + self.latitude_tolerance):
            self.publish_twist()

    def publish_twist(self):
        # Create a Twist message
        twist_msg = Twist()
        # Set the linear z velocity to 1
        twist_msg.linear.z = 1.0
        # Publish the message on the /target_twist topic
        self.twist_publisher.publish(twist_msg)
        self.get_logger().info('Published Twist with linear z velocity = 1.0')

def main(args=None):
    rclpy.init(args=args)
    node = GpsTwistPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
