import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from tf2_msgs.msg import TFMessage
from geometry_msgs.msg import PoseWithCovarianceStamped
from math import sqrt

class WaypointPublisher(Node):

    def __init__(self):
        super().__init__('waypoint_publisher')

        # Use sim time
        #self.declare_parameter('use_sim_time', True)

        amcl_pose_qos = QoSProfile(
          durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
          reliability=QoSReliabilityPolicy.RELIABLE,
          history=QoSHistoryPolicy.KEEP_LAST,
          depth=1)

        # Create publisher and subscription
        self.publisher_ = self.create_publisher(PoseStamped, '/goal_pose', rclpy.qos.qos_profile_system_default)
        self.subscription = self.create_subscription(TFMessage, '/tf', self.tf_callback, 10)
        self.localization_pose_sub = self.create_subscription(PoseWithCovarianceStamped,
                                                              'amcl_pose',
                                                              self._amclPoseCallback,
                                                              amcl_pose_qos)

        # Define waypoints
        self.waypoints = [
            {'x': 5.0, 'y': 5.0, 'z': 0.0, 'qx': 0.0, 'qy': 0.0, 'qz':  0.7560736079329369, 'qw': 0.654486592213524},
            {'x': -5.0, 'y': 5.0, 'z': 0.0, 'qx': 0.0, 'qy': 0.0, 'qz': 0.7560736079329369, 'qw': 0.654486592213524},
            {'x': 5.0, 'y': -5.0, 'z': 0.0, 'qx': 0.0, 'qy': 0.0, 'qz': 0.7560736079329369, 'qw': 0.654486592213524},
            {'x': -5.0, 'y': -5.0, 'z': 0.0, 'qx': 0.0, 'qy': 0.0, 'qz': 0.7560736079329369, 'qw': 0.654486592213524}
        ]

        self.current_waypoint_index = 0
        self.tolerance = 1.0  # Tolerance threshold for waypoint distance
        self.goal_sent = False  # Flag to indicate if a goal has been sent

        # Variable to store the last position for movement check
        self.last_position = None
        
        # Publish the initial goal pose
        self.publish_goal_pose(self.waypoints[self.current_waypoint_index])

        # Create a timer to check robot movement every 3 seconds
        self.timer = self.create_timer(3.0, self.check_movement)

    def _amclPoseCallback(self, msg):
        self.debug('Received amcl pose')
        self.initial_pose_received = True
        return

    def publish_goal_pose(self, waypoint):
        goal_msg = PoseStamped()
        goal_msg.header.frame_id = 'map'
        goal_msg.header.stamp = self.get_clock().now().to_msg()
        goal_msg.pose.position.x = waypoint['x']
        goal_msg.pose.position.y = waypoint['y']
        goal_msg.pose.position.z = waypoint['z']
        goal_msg.pose.orientation.x = waypoint['qx']
        goal_msg.pose.orientation.y = waypoint['qy']
        goal_msg.pose.orientation.z = waypoint['qz']
        goal_msg.pose.orientation.w = waypoint['qw']
        self.publisher_.publish(goal_msg)
        self.publisher_.publish(goal_msg)
        self.publisher_.publish(goal_msg)
        self.get_logger().info(f'Published goal: ({waypoint["x"]}, {waypoint["y"]})')
        self.goal_sent = True

    def tf_callback(self, msg):
        for transform in msg.transforms:
            if transform.child_frame_id == 'base_link':
                current_x = transform.transform.translation.x
                current_y = transform.transform.translation.y
                current_position = (current_x, current_y)
                #self.get_logger().info(f'Current position: ({current_x}, {current_y})')

                # Store the current position for movement check
                self.last_position = current_position

                goal_position = (self.waypoints[self.current_waypoint_index]['x'],
                                 self.waypoints[self.current_waypoint_index]['y'])

                print(current_position)
                if self.is_within_tolerance(current_position, goal_position):
                    self.goal_sent = False
                    self.current_waypoint_index = (self.current_waypoint_index + 1) #% len(self.waypoints)
                    if not self.goal_sent:
                        self.publish_goal_pose(self.waypoints[self.current_waypoint_index])

    def is_within_tolerance(self, current_position, goal_position):
        distance = sqrt((current_position[0] - goal_position[0])**2 + 
                        (current_position[1] - goal_position[1])**2)
        return distance < self.tolerance

    def check_movement(self):
        """Check if the robot is moving towards the goal and republish if not."""
        if self.last_position is None:
            return  # Skip the check if we haven't received any position updates yet

        goal_position = (self.waypoints[self.current_waypoint_index]['x'],
                         self.waypoints[self.current_waypoint_index]['y'])

        # Calculate current distance to the goal
        current_distance = self.calculate_distance(self.last_position, goal_position)

        # If the robot is moving, `self.previous_distance` will have a value
        if hasattr(self, 'previous_distance'):
            if current_distance >= self.previous_distance:
                # Robot is not moving closer to the goal, republish the goal
                self.get_logger().info('Robot is not moving towards the goal, republishing goal.')
                self.publish_goal_pose(self.waypoints[self.current_waypoint_index])

        # Update the previous distance for the next check
        self.previous_distance = current_distance

    def calculate_distance(self, position1, position2):
        """Calculate the Euclidean distance between two positions (x, y)."""
        return sqrt((position1[0] - position2[0])**2 + (position1[1] - position2[1])**2)


def main(args=None):
    rclpy.init(args=args)
    waypoint_publisher = WaypointPublisher()
    rclpy.spin(waypoint_publisher)
    waypoint_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()