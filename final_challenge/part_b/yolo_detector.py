#!/usr/bin/env python3
"""Consolidated YOLO detector for Part B.

Runs YOLO once per ZED image and derives three outputs in one pass:
  - /relative_cone_px (best 'parking meter' bottom-center pixel)
        -> homography_transformer -> /relative_cone -> parking_controller
  - /detections/traffic_light_is_red (HSV red fraction inside 'traffic light' bbox)
  - /detections/pedestrian_close (bbox height / image height > threshold)
Plus /part_b/debug_image with all bboxes drawn for image_saver.

Adapted from Visual_Servoing/yolo_annotator.py (same class_color_map / Detection /
results_to_detections pattern); only the class list and post-processing changed.
"""

import cv2
import numpy as np
import rclpy
import torch

from cv_bridge import CvBridge
from dataclasses import dataclass
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Bool
from typing import List, Optional
from ultralytics import YOLO

from vs_msgs.msg import ConeLocationPixel


@dataclass(frozen=True)
class Detection:
    class_id: int
    class_name: str
    confidence: float
    x1: int
    y1: int
    x2: int
    y2: int


class YoloDetectorNode(Node):
    def __init__(self) -> None:
        super().__init__("yolo_detector")

        self.model_name = self.declare_parameter(
            "model", "yolo11n.pt").get_parameter_value().string_value
        self.conf_threshold = self.declare_parameter(
            "conf_threshold", 0.3).get_parameter_value().double_value
        self.iou_threshold = self.declare_parameter(
            "iou_threshold", 0.7).get_parameter_value().double_value
        self.image_topic = self.declare_parameter(
            "image_topic", "/zed/zed_node/rgb/image_rect_color"
        ).get_parameter_value().string_value
        self.red_pixel_fraction = self.declare_parameter(
            "red_pixel_fraction", 0.08).get_parameter_value().double_value
        self.traffic_light_min_area = self.declare_parameter(
            "traffic_light_min_area", 200).get_parameter_value().integer_value
        self.close_bbox_height_frac = self.declare_parameter(
            "close_bbox_height_frac", 0.35).get_parameter_value().double_value

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model = YOLO(self.model_name)
        self.model.to(self.device)

        self.class_color_map = self.get_class_color_map()
        self.allowed_cls = [
            i for i, n in self.model.names.items() if n in self.class_color_map]

        self.get_logger().info(f"yolo_detector: {self.model_name} on {self.device}")
        self.get_logger().info(f"keeping class ids: {self.allowed_cls}")

        self.bridge = CvBridge()
        self.sub = self.create_subscription(Image, self.image_topic, self.on_image, 10)
        self.dbg_pub = self.create_publisher(Image, "/part_b/debug_image", 10)
        self.px_pub = self.create_publisher(ConeLocationPixel, "/relative_cone_px", 1)
        self.red_pub = self.create_publisher(Bool, "/detections/traffic_light_is_red", 1)
        self.ped_pub = self.create_publisher(Bool, "/detections/pedestrian_close", 1)

    def get_class_color_map(self) -> dict:
        return {
            "person": (0, 255, 0),          # Green
            "traffic light": (0, 0, 255),   # Red
            "parking meter": (255, 0, 0),   # Blue
        }

    def on_image(self, msg: Image) -> None:
        try:
            bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().error(f"cv_bridge failed: {e}")
            return

        try:
            results = self.model(
                bgr, classes=self.allowed_cls,
                conf=self.conf_threshold, iou=self.iou_threshold, verbose=False)
        except Exception as e:
            self.get_logger().error(f"YOLO inference failed: {e}")
            return

        if not results:
            return

        dets = self.results_to_detections(results[0])
        annotated = self.draw_and_analyze(bgr, dets)

        out = self.bridge.cv2_to_imgmsg(annotated, encoding="bgr8")
        out.header = msg.header
        self.dbg_pub.publish(out)

    def results_to_detections(self, result) -> List[Detection]:
        detections = []
        if result.boxes is None:
            return detections
        xyxy_np = result.boxes.xyxy.detach().cpu().numpy()
        conf_np = result.boxes.conf.detach().cpu().numpy()
        cls_np = result.boxes.cls.detach().cpu().numpy()
        for xys, confs, clss in zip(xyxy_np, conf_np, cls_np):
            detections.append(Detection(
                int(clss), self.model.names[int(clss)], float(confs),
                int(xys[0]), int(xys[1]), int(xys[2]), int(xys[3])))
        return detections

    def draw_and_analyze(self, bgr: np.ndarray, dets: List[Detection]) -> np.ndarray:
        h_img = bgr.shape[0]
        out_image = bgr.copy()

        best_pm: Optional[Detection] = None
        any_red = False
        any_ped_close = False

        for det in dets:
            color = self.class_color_map[det.class_name]

            if det.class_name == "parking meter":
                if best_pm is None or det.confidence > best_pm.confidence:
                    best_pm = det

            elif det.class_name == "traffic light":
                area = (det.x2 - det.x1) * (det.y2 - det.y1)
                if area >= self.traffic_light_min_area:
                    crop = bgr[det.y1:det.y2, det.x1:det.x2]
                    if self._is_red(crop):
                        any_red = True
                        color = (0, 0, 255)

            elif det.class_name == "person":
                frac = (det.y2 - det.y1) / h_img
                if frac > self.close_bbox_height_frac:
                    any_ped_close = True
                    color = (0, 0, 255)

            cv2.rectangle(out_image, (det.x1, det.y1), (det.x2, det.y2), color, 2)
            label = f"{det.class_name} {det.confidence:.2f}"
            ty = max(det.y1 - 10, 10)
            cv2.putText(
                out_image, label, (det.x1, ty),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)

        if best_pm is not None:
            px = ConeLocationPixel()
            px.u = float((best_pm.x1 + best_pm.x2) / 2.0)
            px.v = float(best_pm.y2)
            self.px_pub.publish(px)

        self.red_pub.publish(Bool(data=any_red))
        self.ped_pub.publish(Bool(data=any_ped_close))

        return out_image

    def _is_red(self, crop: np.ndarray) -> bool:
        if crop.size == 0:
            return False
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        m1 = cv2.inRange(hsv, np.array([0, 100, 100]), np.array([10, 255, 255]))
        m2 = cv2.inRange(hsv, np.array([160, 100, 100]), np.array([179, 255, 255]))
        red = m1 | m2
        return (float(np.count_nonzero(red)) / float(red.size)) > self.red_pixel_fraction


def main() -> None:
    rclpy.init()
    node = YoloDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
