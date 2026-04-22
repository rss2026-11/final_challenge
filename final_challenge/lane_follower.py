#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np
import cv2
from cv_bridge import CvBridge, CvBridgeError
import math

from sensor_msgs.msg import Image
from ackermann_msgs.msg import AckermannDriveStamped

# Homography calibration points
PTS_IMAGE_PLANE = [[195, 278],
                   [475, 290],
                   [245, 223],
                   [420, 223]]

PTS_GROUND_PLANE = [[24, 12],
                    [24, -12],
                    [48, 12],
                    [48, -12]]

METERS_PER_INCH = 0.0254


class LaneFollower(Node):
    def __init__(self):
        super().__init__("lane_follower")

        self.declare_parameter("speed", 4.0)
        self.declare_parameter("wheelbase_length", 0.32)

        self.speed = self.get_parameter("speed").get_parameter_value().double_value
        self.wheelbase_length = self.get_parameter("wheelbase_length").get_parameter_value().double_value

        self.bridge = CvBridge()

        # Setup homography matrix
        np_pts_ground = np.array(PTS_GROUND_PLANE) * METERS_PER_INCH
        np_pts_ground = np.float32(np_pts_ground[:, np.newaxis, :])

        np_pts_image = np.array(PTS_IMAGE_PLANE) * 1.0
        np_pts_image = np.float32(np_pts_image[:, np.newaxis, :])

        self.h_matrix, err = cv2.findHomography(np_pts_image, np_pts_ground)
        if self.h_matrix is None:
            self.get_logger().error("Homography matrix calculation failed! Check your points.")
        else:
            self.get_logger().info("Homography Transformer Initialized")

        # ROS Publishers and Subscribers
        self.image_sub = self.create_subscription(
            Image,
            "/zed/zed_node/rgb/image_rect_color",
            self.image_callback, 10
        )
        self.drive_pub = self.create_publisher(AckermannDriveStamped, "/drive", 10)
        self.debug_pub = self.create_publisher(Image, "/lane_follower/debug", 10)

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

        # 1. Computer Vision: Find lane center in image (u, v)
        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)

        # Range for yellow / white line
        # This mask often needs to be tuned on race day
        lower_bound = np.array([0, 0, 180])
        upper_bound = np.array([179, 70, 255])
        mask = cv2.inRange(hsv, lower_bound, upper_bound)

        # Create ROI (Region of Interest) - lower half of the screen
        roi_top = int(height * 0.5)
        roi_mask = np.zeros_like(mask)
        roi_mask[roi_top:height, :] = 255
        masked_img = cv2.bitwise_and(mask, roi_mask)

        # Edge detection
        edges = cv2.Canny(masked_img, 50, 150)

        # Hough Transform
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=50, maxLineGap=40)

        target_u = int(width / 2)
        target_v = int(height * 0.7) # Lookahead scanline

        if lines is not None:
            left_lines = []
            right_lines = []

            for line in lines:
                x1, y1, x2, y2 = line[0]
                if x2 == x1: continue
                slope = (y2 - y1) / (x2 - x1)

                # Filter out horizontal/vertical lines
                if abs(slope) < 0.2 or abs(slope) > 10:
                    continue

                if slope < 0:
                    left_lines.append(line)
                else:
                    right_lines.append(line)

            # Draw lines and determine target
            left_u_avg = None
            right_u_avg = None

            if left_lines:
                left_u_pts = []
                for line in left_lines:
                    x1, y1, x2, y2 = line[0]
                    cv2.line(cv_image, (x1, y1), (x2, y2), (255, 0, 0), 3) # Blue for left
                    slope = (y2 - y1) / (x2 - x1)
                    intercept = y1 - slope * x1
                    if slope != 0:
                        left_u_pts.append((target_v - intercept) / slope)
                left_u_avg = np.mean(left_u_pts)

            if right_lines:
                right_u_pts = []
                for line in right_lines:
                    x1, y1, x2, y2 = line[0]
                    cv2.line(cv_image, (x1, y1), (x2, y2), (0, 0, 255), 3) # Red for right
                    slope = (y2 - y1) / (x2 - x1)
                    intercept = y1 - slope * x1
                    if slope != 0:
                        right_u_pts.append((target_v - intercept) / slope)
                right_u_avg = np.mean(right_u_pts)

            # Determine Target Point
            if left_u_avg is not None and right_u_avg is not None:
                # We have both lane lines
                target_u = int((left_u_avg + right_u_avg) / 2)
            elif left_u_avg is not None:
                # We only see left line, estimate right line distance
                lane_width_px = 300 # highly dependent on height/calibration
                target_u = int(left_u_avg + lane_width_px / 2)
            elif right_u_avg is not None:
                lane_width_px = 300
                target_u = int(right_u_avg - lane_width_px / 2)

        # Draw target dot
        cv2.circle(cv_image, (target_u, target_v), 8, (0, 255, 0), -1)

        # Debug Publishing
        try:
            self.debug_pub.publish(self.bridge.cv2_to_imgmsg(cv_image, "bgr8"))
        except:
            pass

        # 2. Homography
        x_local, y_local = self.transformUvToXy(target_u, target_v)

        # 3. Pure Pursuit
        L_d = math.sqrt(x_local**2 + y_local**2)
        if L_d < 0.05:
            L_d = 0.05 # Prevent divide by zero

        curvature = (2.0 * y_local) / (L_d**2)
        steering_angle = math.atan(curvature * self.wheelbase_length)

        # Cap Steering to realistic angles
        max_steer = 0.6
        steering_angle = max(min(steering_angle, max_steer), -max_steer)

        # Publish Drive Command
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
