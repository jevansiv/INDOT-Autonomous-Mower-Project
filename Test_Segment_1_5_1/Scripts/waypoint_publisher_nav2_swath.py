import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from tf2_msgs.msg import TFMessage
from geometry_msgs.msg import PoseWithCovarianceStamped
from math import sqrt
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy, QoSHistoryPolicy
import tf2_ros

class WaypointPublisher(Node):

    def __init__(self):
        super().__init__('waypoint_publisher')

        amcl_pose_qos = QoSProfile(
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1)

        self.publisher_ = self.create_publisher(PoseStamped, '/goal_pose', rclpy.qos.qos_profile_system_default)
        self.subscription = self.create_subscription(TFMessage, '/tf', self.tf_callback, 10)
        self.localization_pose_sub = self.create_subscription(PoseWithCovarianceStamped,
                                                              'amcl_pose',
                                                              self._amclPoseCallback,
                                                              amcl_pose_qos)

        # Initialize waypoints, tolerance, movement checks, and publish count
        self.waypoints = []
        self.current_waypoint_index = 0
        self.tolerance = 1.0  # Tolerance threshold for waypoint distance
        self.goal_sent = False  # Flag to indicate if a goal has been sent
        self.last_position = None
        self.stuck_timer = None  # Timer to handle non-moving robot
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.initial_pose_received = False
        self.publish_count = 0  # Counter for how many times the current waypoint has been published

        # Publish the first goal pose if available
        if self.waypoints:
            self.publish_goal_pose(self.waypoints[self.current_waypoint_index])

        self.timer = self.create_timer(3.0, self.check_movement)

    def _amclPoseCallback(self, msg):
        self.initial_pose_received = True
        return

    def publish_goal_pose(self, waypoint):
        if waypoint is None:
            self.get_logger().warning("No waypoints available to publish.")
            return

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
        self.get_logger().info(f'Published goal: ({waypoint["x"]}, {waypoint["y"]})')
        self.goal_sent = True
        self.publish_count += 1  # Increment the publish count

    def tf_callback(self, msg):
        for transform in msg.transforms:
            if transform.child_frame_id == 'base_link':
                current_x = transform.transform.translation.x
                current_y = transform.transform.translation.y
                current_position = (current_x, current_y)

                # Transform to map frame
                try:
                    transform = self.tf_buffer.lookup_transform('map', 'base_link', rclpy.time.Time())
                    current_x = transform.transform.translation.x
                    current_y = transform.transform.translation.y
                    current_position = (current_x, current_y)
                except:
                    pass

                # Store the transformed position for movement check
                self.last_position = current_position

                if self.current_waypoint_index >= len(self.waypoints):
                    self.get_logger().info("Done with coverage")
                    return  # Stop processing further waypoints

                goal_position = (self.waypoints[self.current_waypoint_index]['x'],
                                 self.waypoints[self.current_waypoint_index]['y'])

                # Check if the robot is within tolerance
                if self.is_within_tolerance(current_position, goal_position):
                    self.goal_sent = False
                    self.current_waypoint_index += 1  # Move to the next waypoint
                    self.publish_count = 0  # Reset publish count for the next waypoint
                    self.get_logger().info(f'{self.current_waypoint_index}/{len(self.waypoints)} waypoints completed.')
                    if self.current_waypoint_index < len(self.waypoints):
                        self.publish_goal_pose(self.waypoints[self.current_waypoint_index])

    def is_within_tolerance(self, current_position, goal_position):
        distance = sqrt((current_position[0] - goal_position[0])**2 + 
                        (current_position[1] - goal_position[1])**2)
        return distance < self.tolerance

    def check_movement(self):
        """Check if the robot is moving towards the goal and republish if not."""
        if self.last_position is None or self.current_waypoint_index >= len(self.waypoints):
            return  # Skip the check if no position updates or coverage is done

        goal_position = (self.waypoints[self.current_waypoint_index]['x'],
                         self.waypoints[self.current_waypoint_index]['y'])

        current_distance = self.calculate_distance(self.last_position, goal_position)

        # If robot is stuck and has not moved closer to the waypoint
        if hasattr(self, 'previous_distance'):
            if current_distance >= self.previous_distance:
                self.get_logger().info('Robot stuck, republishing goal.')
                self.publish_goal_pose(self.waypoints[self.current_waypoint_index])

        # Check if the publish count has reached 5
        if self.publish_count >= 5:
            self.get_logger().info('Published the same goal 5 times, moving to the next waypoint.')
            self.current_waypoint_index += 1
            self.publish_count = 0  # Reset publish count for the next waypoint
            if self.current_waypoint_index < len(self.waypoints):
                self.publish_goal_pose(self.waypoints[self.current_waypoint_index])

        self.previous_distance = current_distance

    def calculate_distance(self, position1, position2):
        """Calculate the Euclidean distance between two positions (x, y)."""
        return sqrt((position1[0] - position2[0])**2 + (position1[1] - position2[1])**2)

    def define_waypoints(self, length, width, swath_width, point_spacing, reference_corner):
        """Generate waypoints for a swath path with correct reference corner handling."""
        self.waypoints = []  # Clear existing waypoints

        # Set starting position based on reference corner
        if reference_corner == 0:  # Bottom-left corner
            start_x = 0
            start_y = 0
        elif reference_corner == 1:  # Bottom-right corner
            start_x = -length
            start_y = 0
        elif reference_corner == 2:  # Top-left corner
            start_x = 0
            start_y = -width
        elif reference_corner == 3:  # Top-right corner
            start_x = -length
            start_y = -width

        # Generate waypoints with a snake-like pattern
        for y in range(0, int(width / swath_width) + 1):
            for x in range(0, int(length / point_spacing) + 1):
                # Calculate x position based on the row (y) to create a snake pattern
                if y % 2 == 0:  # Even rows go right
                    waypoint_x = start_x + x * point_spacing
                else:  # Odd rows go left
                    waypoint_x = start_x + (length - x * point_spacing)

                waypoint_y = start_y + y * swath_width

                waypoint = {
                    'x': waypoint_x,
                    'y': waypoint_y,
                    'z': 0.0,
                    'qx': 0.0,
                    'qy': 0.0,
                    'qz': 0.7560736079329369,
                    'qw': 0.654486592213524
                }
                self.waypoints.append(waypoint)

        self.get_logger().info(f'Generated {len(self.waypoints)} waypoints.')

def main(args=None):
    rclpy.init(args=args)
    waypoint_publisher = WaypointPublisher()
    # Example usage of define_waypoints
    waypoint_publisher.define_waypoints(length=60.0, width=0.5, swath_width=0.5, point_spacing=2.0, reference_corner=0) #2.0 width
    rclpy.spin(waypoint_publisher)
    waypoint_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
