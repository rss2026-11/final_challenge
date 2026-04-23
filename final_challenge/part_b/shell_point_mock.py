"""Mock basement_point_publisher for local testing.

Publishes a latched PoseArray on /shell_points a short delay after startup.
Disable in launch once the real basement_point_publisher is available.
"""
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile
from geometry_msgs.msg import PoseArray, Pose

DEFAULT_POINTS = [
    (0.0, 0.0),
    (5.0, 0.0),
]


class ShellPointMock(Node):
    def __init__(self):
        super().__init__("shell_point_mock")
        self.declare_parameter("topic", "/shell_points")
        self.declare_parameter("delay_sec", 2.0)
        self.declare_parameter(
            "points",
            [p for pair in DEFAULT_POINTS for p in pair],
        )

        topic = self.get_parameter("topic").get_parameter_value().string_value
        self.delay = self.get_parameter("delay_sec").get_parameter_value().double_value
        flat = list(self.get_parameter("points").get_parameter_value().double_array_value)
        if len(flat) < 4 or len(flat) % 2 != 0:
            self.get_logger().warn(f"bad 'points' param ({flat}); using defaults")
            flat = [p for pair in DEFAULT_POINTS for p in pair]
        self.points = [(flat[i], flat[i + 1]) for i in range(0, len(flat), 2)]

        qos = QoSProfile(depth=1, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)
        self.pub = self.create_publisher(PoseArray, topic, qos)
        self.timer = self.create_timer(self.delay, self._publish_once)
        self.get_logger().info(
            f"shell_point_mock: will publish to {topic} in {self.delay}s -> {self.points}")

    def _publish_once(self):
        msg = PoseArray()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "map"
        for x, y in self.points:
            p = Pose()
            p.position.x = float(x)
            p.position.y = float(y)
            p.position.z = 0.0
            p.orientation.w = 1.0
            msg.poses.append(p)
        self.pub.publish(msg)
        self.get_logger().info(f"published {len(self.points)} shell points (latched)")
        self.timer.cancel()


def main(args=None):
    rclpy.init(args=args)
    rclpy.spin(ShellPointMock())
    rclpy.shutdown()
