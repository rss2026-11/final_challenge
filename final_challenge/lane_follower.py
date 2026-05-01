#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np
import cv2
from cv_bridge import CvBridge, CvBridgeError
import math

from sensor_msgs.msg import Image
from ackermann_msgs.msg import AckermannDriveStamped

# Homography calibration points (used by both pipelines for control)
PTS_IMAGE_PLANE = [[195, 278],
                   [475, 290],
                   [245, 223],
                   [420, 223]]

PTS_GROUND_PLANE = [[24, 12],
                    [24, -12],
                    [48, 12],
                    [48, -12]]

METERS_PER_INCH = 0.0254

# ── BEV constants ──────────────────────────────────────────────────────────────
# Source trapezoid as fractions of (width, height): top-left, top-right,
# bottom-right, bottom-left.  Tune these so the trapezoid wraps the lane
# region visible directly in front of the car.
BEV_SRC_FRAC = np.float32([
    [0.25, 0.51],   # top-left
    [0.95, 0.51],   # top-right
    [1.00, 0.79],   # bottom-right
    [0.05, 0.79],   # bottom-left
])
BEV_W, BEV_H = 320, 240   # output bird's-eye image size (pixels)
# ──────────────────────────────────────────────────────────────────────────────


class LaneFollower(Node):
    def __init__(self):
        super().__init__("lane_follower")

        self.declare_parameter("speed", 4.0)
        self.declare_parameter("wheelbase_length", 0.32)

        self.speed = self.get_parameter("speed").get_parameter_value().double_value
        self.wheelbase_length = self.get_parameter("wheelbase_length").get_parameter_value().double_value

        self.bridge = CvBridge()

        # Setup homography matrix (image → ground, used for pure pursuit)
        np_pts_ground = np.array(PTS_GROUND_PLANE) * METERS_PER_INCH
        np_pts_ground = np.float32(np_pts_ground[:, np.newaxis, :])

        np_pts_image = np.array(PTS_IMAGE_PLANE) * 1.0
        np_pts_image = np.float32(np_pts_image[:, np.newaxis, :])

        self.h_matrix, err = cv2.findHomography(np_pts_image, np_pts_ground)
        if self.h_matrix is None:
            self.get_logger().error("Homography matrix calculation failed! Check your points.")
        else:
            self.get_logger().info("Homography Transformer Initialized")

        # BEV warp matrices — built lazily on first frame (need image size)
        self.M_bev = None
        self.M_bev_inv = None

        # Last valid lane detections — used when a frame misses a line
        self.last_left  = None   # (s, b) of averaged left line
        self.last_right = None   # (s, b) of averaged right line

        # ROS Publishers and Subscribers
        self.image_sub = self.create_subscription(
            Image,
            "/zed/zed_node/rgb/image_rect_color",
            self.image_callback, 10
        )
        self.drive_pub = self.create_publisher(AckermannDriveStamped, "/vesc/high_level/input/navigation", 10)
        self.debug_pub = self.create_publisher(Image, "/lane_follower/debug", 10)
        self.bev_pub   = self.create_publisher(Image, "/lane_follower/bev_debug", 10)

        self.smooth_target_u = None
        self.alpha = 0.15

        self.get_logger().info("Lane Follower Initialized")

    def transformUvToXy(self, u, v):
        homogeneous_point = np.array([[u], [v], [1]])
        xy = np.dot(self.h_matrix, homogeneous_point)
        scaling_factor = 1.0 / xy[2, 0]
        homogeneous_xy = xy * scaling_factor
        x = homogeneous_xy[0, 0]
        y = homogeneous_xy[1, 0]
        return x, y

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            self.get_logger().error(f"CV Bridge Error: {e}")
            return

        height, width, _ = cv_image.shape

        # Build BEV warp matrices once we know the image size
        if self.M_bev is None:
            src = BEV_SRC_FRAC * np.array([width, height], dtype=np.float32)
            dst = np.float32([[0, 0], [BEV_W, 0], [BEV_W, BEV_H], [0, BEV_H]])
            self.M_bev     = cv2.getPerspectiveTransform(src, dst)
            self.M_bev_inv = cv2.getPerspectiveTransform(dst, src)

        # ── Color mask (shared by both pipelines) ─────────────────────────────
        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        lower_bound = np.array([50, 0, 200])
        upper_bound = np.array([100, 43, 255])
        mask = cv2.inRange(hsv, lower_bound, upper_bound)

        # ROI: lower half only
        roi_top = int(height * 0.5)
        roi_mask = np.zeros_like(mask)
        roi_mask[roi_top:height, :] = 255
        mask = cv2.bitwise_and(mask, roi_mask)
        # ──────────────────────────────────────────────────────────────────────

        # =======================================================================
        # OLD PIPELINE — perspective Hough lines (commented out)
        # =======================================================================
        # edges = cv2.Canny(mask, 50, 150)
        # lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=50, maxLineGap=40)
        #
        # target_u = int(width / 2)
        # target_v = int(height * 0.7)
        #
        # if lines is not None:
        #     left_lines = []
        #     right_lines = []
        #
        #     for line in lines:
        #         x1, y1, x2, y2 = line[0]
        #         if x2 == x1: continue
        #         slope = (y2 - y1) / (x2 - x1)
        #         if abs(slope) < 0.3 or abs(slope) > 10:
        #             continue
        #         x_center = (x1 + x2) / 2
        #         if x_center < width / 2:
        #             left_lines.append(line)
        #         else:
        #             right_lines.append(line)
        #
        #     left_u_avg = None
        #     right_u_avg = None
        #
        #     if left_lines:
        #         left_u_pts = []
        #         for line in left_lines:
        #             x1, y1, x2, y2 = line[0]
        #             cv2.line(cv_image, (x1, y1), (x2, y2), (255, 0, 0), 3)
        #             slope = (y2 - y1) / (x2 - x1)
        #             intercept = y1 - slope * x1
        #             if slope != 0:
        #                 left_u_pts.append((target_v - intercept) / slope)
        #         left_u_avg = np.mean(left_u_pts)
        #
        #     if right_lines:
        #         right_u_pts = []
        #         for line in right_lines:
        #             x1, y1, x2, y2 = line[0]
        #             cv2.line(cv_image, (x1, y1), (x2, y2), (0, 0, 255), 3)
        #             slope = (y2 - y1) / (x2 - x1)
        #             intercept = y1 - slope * x1
        #             if slope != 0:
        #                 right_u_pts.append((target_v - intercept) / slope)
        #         right_u_avg = np.mean(right_u_pts)
        #
        #     if left_u_avg is not None and right_u_avg is not None:
        #         target_u = int((left_u_avg + right_u_avg) / 2)
        #     elif left_u_avg is not None:
        #         lane_width_px = 300
        #         target_u = int(left_u_avg + lane_width_px / 2)
        #     elif right_u_avg is not None:
        #         lane_width_px = 300
        #         target_u = int(right_u_avg - lane_width_px / 2)
        # =======================================================================

        # =======================================================================
        # NEW PIPELINE — trapezoid ROI + Hough lines (hybrid approach)
        # Use the BEV trapezoid as a polygon mask, detect Hough lines inside it
        # in perspective space so lines naturally follow curves.
        # =======================================================================
        trap_pts = (BEV_SRC_FRAC * np.array([width, height], dtype=np.float32)).astype(np.int32)
        poly_mask = np.zeros_like(mask)
        cv2.fillPoly(poly_mask, [trap_pts], 255)
        masked = cv2.bitwise_and(mask, poly_mask)

        edges = cv2.Canny(masked, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=30, minLineLength=30, maxLineGap=40)

        target_u = int(width / 2)
        target_v = int(height * 0.65)

        left_lines = []
        right_lines = []

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if x2 == x1:
                    continue
                slope = (y2 - y1) / (x2 - x1)
                if abs(slope) < 0.3 or abs(slope) > 10:
                    continue
                if (x1 + x2) / 2 < width / 2:
                    left_lines.append(line)
                else:
                    right_lines.append(line)

        def averaged_line(line_group, img_h):
            """Average a group of lines into one and extend it top-to-bottom."""
            slopes, intercepts = [], []
            for line in line_group:
                x1, y1, x2, y2 = line[0]
                if x2 == x1:
                    continue
                s = (y2 - y1) / (x2 - x1)
                slopes.append(s)
                intercepts.append(y1 - s * x1)
            s = np.mean(slopes)
            b = np.mean(intercepts)
            y_bot = img_h
            y_top = int(img_h * 0.4)
            x_bot = int((y_bot - b) / s)
            x_top = int((y_top - b) / s)
            return (x_bot, y_bot), (x_top, y_top), s, b

        s_left = b_left = s_right = b_right = None
        left_u_avg = right_u_avg = None

        if left_lines:
            _, _, s_left, b_left = averaged_line(left_lines, height)
            self.last_left = (s_left, b_left)    # store for next frame
        elif self.last_left is not None:
            s_left, b_left = self.last_left       # reuse last known

        if right_lines:
            _, _, s_right, b_right = averaged_line(right_lines, height)
            self.last_right = (s_right, b_right)
        elif self.last_right is not None:
            s_right, b_right = self.last_right

        if s_left is not None and b_left is not None:
            y_bot, y_top = height, int(height * 0.4)
            x_bot = int((y_bot - b_left) / s_left)
            x_top = int((y_top - b_left) / s_left)
            cv2.line(cv_image, (x_bot, y_bot), (x_top, y_top), (255, 0, 0), 3)
            if s_left != 0:
                left_u_avg = (target_v - b_left) / s_left

        if s_right is not None and b_right is not None:
            y_bot, y_top = height, int(height * 0.4)
            x_bot = int((y_bot - b_right) / s_right)
            x_top = int((y_top - b_right) / s_right)
            cv2.line(cv_image, (x_bot, y_bot), (x_top, y_top), (0, 0, 255), 3)
            if s_right != 0:
                right_u_avg = (target_v - b_right) / s_right

        ASSUMED_LANE_W_PX = 300
        if left_u_avg is not None and right_u_avg is not None:
            target_u = int((left_u_avg + right_u_avg) / 2)
        elif left_u_avg is not None:
            target_u = int(left_u_avg + ASSUMED_LANE_W_PX / 2)
        elif right_u_avg is not None:
            target_u = int(right_u_avg - ASSUMED_LANE_W_PX / 2)

        # Draw trapezoid
        cv2.polylines(cv_image, [trap_pts], isClosed=True, color=(0, 255, 0), thickness=2)

        # Draw magenta center line from the intersection of blue+red downward
        if s_left is not None and s_right is not None and s_left != s_right:
            # Intersection of the two lane lines (the vanishing point)
            x_int = int((b_right - b_left) / (s_left - s_right))
            y_int = int(s_left * x_int + b_left)
            cv2.line(cv_image, (x_int, y_int), (target_u, height), (255, 0, 255), 3)
            # Place the target dot on the magenta line at target_v
            if height != y_int:
                t = (target_v - y_int) / (height - y_int)
                target_u = int(x_int + t * (target_u - x_int))
        else:
            # Fallback when only one lane visible: vertical line
            cv2.line(cv_image, (target_u, int(height * 0.4)), (target_u, height), (255, 0, 255), 3)
        # =======================================================================


        # Draw target dot on original image
        cv2.circle(cv_image, (target_u, target_v), 8, (0, 255, 0), -1)

        # Debug Publishing
        try:
            self.debug_pub.publish(self.bridge.cv2_to_imgmsg(cv_image, "bgr8"))
        except:
            pass

        # ── Homography → ground coordinates ───────────────────────────────────
        x_local, y_local = self.transformUvToXy(target_u, target_v)

        # ── Pure Pursuit ───────────────────────────────────────────────────────
        L_d = math.sqrt(x_local**2 + y_local**2)
        if L_d < 0.05:
            L_d = 0.05

        curvature = (2.0 * y_local) / (L_d**2)
        steering_angle = math.atan(curvature * self.wheelbase_length)

        max_steer = 0.6
        steering_angle = max(min(steering_angle, max_steer), -max_steer)

        drive_msg = AckermannDriveStamped()
        drive_msg.header.stamp = self.get_clock().now().to_msg()
        drive_msg.header.frame_id = "base_link"
        drive_msg.drive.speed = self.speed
        drive_msg.drive.steering_angle = steering_angle
        self.drive_pub.publish(drive_msg)


def main(args=None):
    rclpy.init(args=args)
    lane_follower = LaneFollower()

    try:
        rclpy.spin(lane_follower)
    except KeyboardInterrupt:
        pass

    lane_follower.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
