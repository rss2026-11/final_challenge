"""Saves the latest annotated image when /part_b/park_trigger fires.

Buffers whatever image_topic provides (default: parking_meter_detector's debug
feed). On trigger, writes the most recent image to <save_dir>/<msg.data>.png.
"""
import os
import rclpy
from rclpy.node import Node
import cv2

from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import String


class ImageSaver(Node):
    def __init__(self):
        super().__init__("image_saver")
        self.declare_parameter("image_topic", "/part_b/parking_meter_debug")
        self.declare_parameter("save_dir", "saved_parks")
        image_topic = self.get_parameter("image_topic").get_parameter_value().string_value
        self.save_dir = self.get_parameter("save_dir").get_parameter_value().string_value
        os.makedirs(self.save_dir, exist_ok=True)

        self.bridge = CvBridge()
        self.latest = None

        self.create_subscription(Image, image_topic, self._on_image, 1)
        self.create_subscription(String, "/part_b/park_trigger", self._on_trigger, 10)
        self.get_logger().info(f"image_saver: {image_topic} -> {self.save_dir}/")

    def _on_image(self, msg):
        self.latest = msg

    def _on_trigger(self, msg):
        if self.latest is None:
            self.get_logger().warn("trigger received but no image buffered")
            return
        try:
            bgr = self.bridge.imgmsg_to_cv2(self.latest, "bgr8")
        except Exception as e:
            self.get_logger().error(f"cv_bridge on save: {e}")
            return
        name = msg.data or "park"
        path = os.path.join(self.save_dir, f"{name}.png")
        cv2.imwrite(path, bgr)
        self.get_logger().info(f"saved {path}")


def main(args=None):
    rclpy.init(args=args)
    rclpy.spin(ImageSaver())
    rclpy.shutdown()
