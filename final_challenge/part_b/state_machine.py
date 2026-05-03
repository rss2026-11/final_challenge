"""Mission manager for Part B: Mrs. Puff's Boating School.

State flow:
    INIT -> WAIT_GOALS -> NAV_1 -> APPROACH_1 -> PARKED_1
         -> NAV_2 -> APPROACH_2 -> PARKED_2
         -> RETURN -> DONE

Interrupts (override drive from any active-driving state):
    red light visible -> publish stop
    pedestrian close  -> publish stop

Drive mux (we publish to /drive; safety_controller is downstream):
    NAV_*       -> forward /drive/nav  (from trajectory_follower)
    APPROACH_*  -> forward /drive/park (from parking_controller)
    others      -> publish stop

"Parked" is detected by parking_controller's own output: when its speed stays
~0 for parked_stable_sec, it has reached the target. We then hold stop for
park_hold_sec (spec: 5s), firing an image-save trigger in between.
"""
import math
from enum import Enum, auto

import rclpy
from rclpy.node import Node

from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import PoseArray, PoseStamped, PointStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool, String
from vs_msgs.msg import ConeLocationPixel, ConeLocation

class S(Enum):
    INIT = auto()
    WAIT_GOALS = auto()
    NAV_1 = auto()
    APPROACH_1 = auto()
    PARKED_1 = auto()
    BACKUP_1 = auto()
    NAV_2 = auto()
    APPROACH_2 = auto()
    PARKED_2 = auto()
    BACKUP_2 = auto()
    RETURN = auto()
    DONE = auto()


class StateMachine(Node):
    def __init__(self):
        super().__init__("part_b_state_machine")

        self.declare_parameter("drive_topic_out", "/vesc/high_level/input/navigation")
        # self.declare_parameter("shell_points_topic", "/shell_points")
        self.declare_parameter("odom_topic", "/pf/pose/odom")
        self.declare_parameter("nav_input_topic", "/vesc/high_level/input/nav_0")
        self.declare_parameter("park_input_topic", "/vesc/high_level/input/nav_1")
        self.declare_parameter("approach_radius", 3.0)
        self.declare_parameter("parked_stable_sec", 2.0)
        self.declare_parameter("park_hold_sec", 5.0)
        self.declare_parameter("return_to_start", True)
        self.declare_parameter("tick_hz", 20.0)

        self.drive_topic_out = self.get_parameter("drive_topic_out").get_parameter_value().string_value
        # shell_topic = self.get_parameter("shell_points_topic").get_parameter_value().string_value
        odom_topic = self.get_parameter("odom_topic").get_parameter_value().string_value
        nav_in = self.get_parameter("nav_input_topic").get_parameter_value().string_value
        park_in = self.get_parameter("park_input_topic").get_parameter_value().string_value
        self.approach_radius = self.get_parameter("approach_radius").get_parameter_value().double_value
        self.parked_stable = self.get_parameter("parked_stable_sec").get_parameter_value().double_value
        self.park_hold = self.get_parameter("park_hold_sec").get_parameter_value().double_value
        self.return_to_start = self.get_parameter("return_to_start").get_parameter_value().bool_value
        period = 1.0 / self.get_parameter("tick_hz").get_parameter_value().double_value

        self.state = S.INIT
        self.state_entered = self._now()
        self.goals = []
        self.start_pose = None
        self.current_pose = None
        self.latest_nav_drive = None
        self.latest_park_drive = None
        self.red_light = False
        self.was_red_light = False
        self.last_red_time = 0.0
        self.parking_meter_last_seen = 0.0
        self.goal_sent_for = None
        self.trigger_sent_for = None
        self.zero_drive_since = None
        self.backup_start_pose = None
        self.cone_distance = None

        self.drive_pub = self.create_publisher(AckermannDriveStamped, self.drive_topic_out, 1)
        self.safety_pub = self.create_publisher(AckermannDriveStamped, "/vesc/low_level/input/safety", 1)
        self.goal_pub = self.create_publisher(PoseStamped, "/goal_pose", 1)
        self.state_pub = self.create_publisher(String, "/part_b/state", 1)
        self.trigger_pub = self.create_publisher(String, "/part_b/park_trigger", 10)

        # self.create_subscription(PoseArray, shell_topic, self._on_shell_points, 10)
        self.create_subscription(Odometry, odom_topic, self._on_odom, 10)
        self.create_subscription(AckermannDriveStamped, nav_in, self._on_nav, 1)
        self.create_subscription(AckermannDriveStamped, park_in, self._on_park, 1)
        self.create_subscription(Bool, "/detections/traffic_light_is_red", self._on_red, 10)
        self.create_subscription(ConeLocationPixel, "/relative_cone_px", self._on_parking_meter, 10)
        self.create_subscription(ConeLocation, "/relative_cone", self._on_relative_cone, 10)
        self.create_subscription(PointStamped, "/clicked_point", self._on_clicked_point, 10)


        self.create_timer(period, self._tick)
        self.get_logger().info("part_b_state_machine ready")

    def _now(self):
        return self.get_clock().now().nanoseconds / 1e9

    def _in_state_for(self):
        return self._now() - self.state_entered

    def _transition(self, new):
        self.get_logger().info(f"\n====================================\n[STATE SWITCH] {self.state.name} -> {new.name}\n====================================")
        self.state = new
        self.state_entered = self._now()
        self.goal_sent_for = None
        self.trigger_sent_for = None
        self.zero_drive_since = None
        self.backup_start_pose = self.current_pose
        self.cone_distance = None

    def _dist(self, a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def _send_goal(self, xy):
        m = PoseStamped()
        m.header.stamp = self.get_clock().now().to_msg()
        m.header.frame_id = "map"
        m.pose.position.x = float(xy[0])
        m.pose.position.y = float(xy[1])
        m.pose.orientation.w = 1.0
        self.goal_pub.publish(m)
        self.get_logger().info(f"goal -> ({xy[0]:.2f}, {xy[1]:.2f})")
        self.goal_sent_for = self.state

    def _publish_stop(self, emergency=False):
        m = AckermannDriveStamped()
        m.header.stamp = self.get_clock().now().to_msg()
        m.header.frame_id = "base_link"
        m.drive.speed = 0.0
        m.drive.steering_angle = 0.0

        if emergency:
            self.safety_pub.publish(m)
        else:
            self.drive_pub.publish(m)

    def _forward(self, src):
        if src is None:
            self._publish_stop()
            return
        m = AckermannDriveStamped()
        m.header.stamp = self.get_clock().now().to_msg()
        m.header.frame_id = "base_link"
        m.drive = src.drive
        self.drive_pub.publish(m)

    # def _on_shell_points(self, msg):
    #     self.goals = [(p.position.x, p.position.y) for p in msg.poses]
    #     self.get_logger().info(f"shell points: {self.goals}")

    def _on_odom(self, msg):
        self.current_pose = (msg.pose.pose.position.x, msg.pose.pose.position.y)
        if self.start_pose is None:
            self.start_pose = self.current_pose

    def _on_nav(self, msg):
        self.latest_nav_drive = msg

    def _on_park(self, msg):
        self.latest_park_drive = msg
        if abs(msg.drive.speed) < 0.05:
            if self.zero_drive_since is None:
                self.zero_drive_since = self._now()
        else:
            self.zero_drive_since = None

    def _on_red(self, msg):
        current_red = bool(msg.data)

        if current_red:
            self.last_red_time = self._now()
            self.red_light = True
        else:
            # Debounce: only clear the red light if we haven't seen it for 1.0 seconds
            if self._now() - self.last_red_time > 1.0:
                self.red_light = False

        if self.red_light and not self.was_red_light:
            self.get_logger().info("🛑 RED LIGHT DETECTED! Stopping car...")
        elif not self.red_light and self.was_red_light:
            self.get_logger().info("🟢 GREEN LIGHT! Resuming...")
        self.was_red_light = self.red_light

    def _on_parking_meter(self, msg):
        self.parking_meter_last_seen = self._now()

    def _on_relative_cone(self, msg):
        self.cone_distance = math.hypot(msg.x_pos, msg.y_pos)

    def _on_clicked_point(self, msg):
        if self.state in (S.INIT, S.WAIT_GOALS) and len(self.goals) < 2:
            self.goals.append((msg.point.x, msg.point.y))
            self.get_logger().info(f"Received RViz click {len(self.goals)} of 2: ({msg.point.x:.2f}, {msg.point.y:.2f})")

    def _nav_to(self, goal_xy, next_state, final=False):
        if goal_xy is None or self.current_pose is None:
            self._publish_stop()
            return
        if self.goal_sent_for != self.state:
            self._send_goal(goal_xy)
        self._forward(self.latest_nav_drive)
        d = self._dist(self.current_pose, goal_xy)

        if final:
            # Returning to start
            if d < 1.0:
                self._transition(next_state)
        else:
            # Transition to parking controller as soon as the meter is seen
            meter_visible = (self._now() - self.parking_meter_last_seen) < 0.5

            if meter_visible:
                self._transition(next_state)

    def _approach(self, next_state):
        self._forward(self.latest_park_drive)

        # Transition based on distance to cone
        if self.cone_distance is not None and self.cone_distance < 0.9:
            self.get_logger().info(f"Target distance reached: {self.cone_distance:.2f}m. Parking!")
            self._transition(next_state)
            return

        # Fallback: Transition based on 0 speed
        if self._parked_stable_met():
            self.get_logger().info("Target stable stop reached. Parking!")
            self._transition(next_state)

    def _parked(self, label, next_state):
        self._publish_stop()
        held = self._in_state_for()
        if held > 0.5 and self.trigger_sent_for != self.state:
            self.trigger_pub.publish(String(data=label))
            self.trigger_sent_for = self.state
            self.get_logger().info(f"park image trigger: {label}")
        if held >= self.park_hold:
            self._transition(next_state)

    def _parked_stable_met(self):
        return (self.zero_drive_since is not None and
                (self._now() - self.zero_drive_since) >= self.parked_stable)

    def _backup(self, distance, speed, next_state):
        if self.current_pose is None or self.backup_start_pose is None:
            self._publish_stop()
            return

        d = self._dist(self.current_pose, self.backup_start_pose)
        if d >= distance:
            self._transition(next_state)
        else:
            m = AckermannDriveStamped()
            m.header.stamp = self.get_clock().now().to_msg()
            m.header.frame_id = "base_link"
            m.drive.speed = float(speed)
            m.drive.steering_angle = 0.0
            self.drive_pub.publish(m)

    def _tick(self):
        self.state_pub.publish(String(data=self.state.name))

        active = self.state in (
            S.NAV_1, S.APPROACH_1, S.BACKUP_1, S.NAV_2, S.APPROACH_2, S.BACKUP_2, S.RETURN)
        if active and self.red_light:
            self._publish_stop(emergency=True)
            return

        if self.state == S.INIT:
            self._publish_stop()
            if self.current_pose is not None:
                self._transition(S.WAIT_GOALS)

        elif self.state == S.WAIT_GOALS:
            self._publish_stop()
            if len(self.goals) >= 2:
                self._transition(S.NAV_1)

        elif self.state == S.NAV_1:
            self._nav_to(self.goals[0], next_state=S.APPROACH_1)

        elif self.state == S.APPROACH_1:
            self._approach(next_state=S.PARKED_1)

        elif self.state == S.PARKED_1:
            self._parked("location_1", next_state=S.BACKUP_1)

        elif self.state == S.BACKUP_1:
            self._backup(distance=0.5, speed=-1.0, next_state=S.NAV_2)

        elif self.state == S.NAV_2:
            self._nav_to(self.goals[1], next_state=S.APPROACH_2)

        elif self.state == S.APPROACH_2:
            self._approach(next_state=S.PARKED_2)

        elif self.state == S.PARKED_2:
            nxt = S.BACKUP_2 if (self.return_to_start and self.start_pose) else S.DONE
            self._parked("location_2", next_state=nxt)

        elif self.state == S.BACKUP_2:
            self._backup(distance=0.5, speed=-1.0, next_state=S.RETURN)

        elif self.state == S.RETURN:
            self._nav_to(self.start_pose, next_state=S.DONE, final=True)

        elif self.state == S.DONE:
            self._publish_stop()




def main(args=None):
    rclpy.init(args=args)
    rclpy.spin(StateMachine())
    rclpy.shutdown()
