#!/usr/bin/env python3
"""Consolidated YOLO detector for Part B.

Runs YOLO once per ZED image and derives two outputs in one pass:
  - /relative_cone_px (best 'parking meter' bottom-center pixel)
        -> homography_transformer -> /relative_cone -> parking_controller
  - /detections/traffic_light_is_red (HSV red fraction inside 'traffic light' bbox)
Plus /part_b/debug_image with all bboxes drawn for image_saver.

Adapted from Visual_Servoing/yolo_annotator.py (same class_color_map / Detection /
results_to_detections pattern); only the class list and post-processing changed.
"""

import cv2
import numpy as np
import rclpy
import torch

from std_msgs.msg import Bool
from vs_msgs.msg import ConeLocationPixel
from typing import List, Optional

from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from dataclasses import dataclass
from rclpy.node import Node
from ultralytics import YOLO


@dataclass
class Detection:
    class_id: int
    class_name: str
    confidence: float
    # Bounding box coordinates in the original image:
    x1: int
    y1: int
    x2: int
    y2: int
    # Red light status (for annotation)
    is_red: bool = False


class YoloAnnotatorNode(Node):
    def __init__(self) -> None:
        super().__init__("yolo_detector")

        self.model_name = (
            self.declare_parameter("model", "yolo11n.pt")
            .get_parameter_value()
            .string_value
        )
        self.conf_threshold = (
            self.declare_parameter("conf_threshold", 0.5)
            .get_parameter_value()
            .double_value
        )
        self.iou_threshold = (
            self.declare_parameter("iou_threshold", 0.7)
            .get_parameter_value()
            .double_value
        )

        self.red_pixel_fraction = self.declare_parameter("red_pixel_fraction", 0.04).get_parameter_value().double_value
        self.traffic_light_min_area = self.declare_parameter("traffic_light_min_area", 200).get_parameter_value().integer_value

        # Fraction of image height to ignore from the top (e.g. 1/3 means skip top third).
        # Detections whose bottom edge falls entirely above this cutoff are dropped.
        self.top_crop_fraction = (
            self.declare_parameter("top_crop_fraction", 1.0 / 3.0)
            .get_parameter_value()
            .double_value
        )

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model = YOLO(self.model_name)
        self.model.to(self.device)

        self.class_color_map = self.get_class_color_map()
        self.allowed_cls = [
            i for i, name in self.model.names.items()
            if name in self.class_color_map
        ]

        self.get_logger().info(f"Model classes: {self.model.names}")
        self.get_logger().info(f"Running {self.model_name} on device {self.device}")
        self.get_logger().info(f"Confidence threshold: {self.conf_threshold}")
        self.get_logger().info(f"Top crop fraction: {self.top_crop_fraction}")
        if self.allowed_cls:
            self.get_logger().info(f"You've chosen to keep these class IDs: {self.allowed_cls}")
        else:
            self.get_logger().warn("No allowed classes matched the model's class list.")

        self.bridge = CvBridge()
        self.sub = self.create_subscription(Image, "/zed/zed_node/rgb/image_rect_color", self.on_image, 10)
        self.pub = self.create_publisher(Image, "/part_b/debug_image", 10)
        self.red_pub = self.create_publisher(Bool, "/detections/traffic_light_is_red", 1)
        self.tl_visible_pub = self.create_publisher(Bool, "/detections/traffic_light_visible", 1)
        self.px_pub = self.create_publisher(ConeLocationPixel, "/relative_cone_px", 1)

    def get_class_color_map(self) -> dict[str, tuple[int, int, int]]:
        return {
            "traffic light": (0, 255, 0),   # Green
            "parking meter": (255, 0, 0),   # Blue
        }

    def on_image(self, msg: Image) -> None:
        try:
            bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().error(f"cv_bridge conversion failed: {e}")
            return

        # Compute the row below which we accept detections. Anything whose bottom edge (y2)
        # sits above this is "fully in the ignored top region" and gets dropped. We don't
        # crop the image itself — keeping coordinates in the original ZED frame means the
        # homography and debug image stay consistent without offset bookkeeping.
        img_h = bgr.shape[0]
        y_cutoff = int(img_h * self.top_crop_fraction)

        try:
            results = self.model(
                bgr,
                classes=self.allowed_cls,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                verbose=False,
            )
        except Exception as e:
            self.get_logger().error(f"YOLO inference failed: {e}")
            return

        if not results:
            return

        all_dets = self.results_to_detections(results[0])

        # Filter out detections fully in the ignored top region. Using y2 (bottom of bbox)
        # means a meter/light that straddles the cutoff still counts — matches the
        # homography logic below which uses y2 as the ground-contact point.
        kept_dets = [d for d in all_dets if d.y2 >= y_cutoff]

        # Reduce N raw bboxes into the two decisions downstream nodes actually consume:
        #   - parking_controller wants ONE target point  -> best_pm (highest-conf parking meter)
        #   - state_machine wants ONE bool               -> any_red (is there a stop-worthy red light)
        best_pm: Optional[Detection] = None
        any_red = False
        any_light = False
        for det in kept_dets:
            if det.class_name == "parking meter":
                # Keep the most confident detection; multiple meters in frame are possible
                # and parking_controller can only chase one.
                if best_pm is None or det.confidence > best_pm.confidence:
                    best_pm = det
            elif det.class_name == "traffic light":
                any_light = True
                # Area gate: tiny far-away specks shouldn't slam the brakes.
                area = (det.x2 - det.x1) * (det.y2 - det.y1)
                if area >= self.traffic_light_min_area:
                    # Crop the ORIGINAL bgr (not the annotated image) — drawing rectangles
                    # over the bbox before HSV-checking would dilute the red-pixel ratio.
                    # YOLO doesn't tell us the light's color, so we have to check it ourselves.
                    crop = bgr[det.y1:det.y2, det.x1:det.x2]
                    if self._is_red(crop):
                        any_red = True
                        det.is_red = True

        if best_pm is not None:
            # Bottom-center pixel, not bbox center: the homography is calibrated on the
            # ground plane, so we want the point where the meter touches the floor.
            # The middle of the bbox would project to a point floating in the air.
            px = ConeLocationPixel()
            px.u = float((best_pm.x1 + best_pm.x2) / 2.0)
            px.v = float(best_pm.y2)
            self.px_pub.publish(px)

        # Publish every frame, including False. state_machine.red_light latches on whatever
        # value arrives, so if we only ever published True the flag would stick on after
        # the first red light and the car would never move again.
        self.red_pub.publish(Bool(data=any_red))
        self.tl_visible_pub.publish(Bool(data=any_light))

        # Drawing is a pure side-effect for the debug image; no decisions live in there.
        # Draw only the kept detections and overlay the cutoff line so it's obvious what
        # region is being ignored.
        annotated = self.draw_detections(bgr, kept_dets)
        cv2.line(annotated, (0, y_cutoff), (annotated.shape[1], y_cutoff), (0, 0, 255), 1, cv2.LINE_AA)
        out_msg = self.bridge.cv2_to_imgmsg(annotated, encoding="bgr8")
        out_msg.header = msg.header
        self.pub.publish(out_msg)

    def results_to_detections(self, result) -> List[Detection]:
        """
        Convert an Ultralytics result into a Detection list.

        YOLOv11 outputs:
          result.boxes.xyxy: (N, 4) tensor
          result.boxes.conf: (N,) tensor
          result.boxes.cls:  (N,) tensor
        """
        detections = []

        if result.boxes is None:
            return detections

        xyxy = result.boxes.xyxy
        conf = result.boxes.conf
        cls = result.boxes.cls

        xyxy_np = xyxy.detach().cpu().numpy() if hasattr(xyxy, "detach") else np.asarray(xyxy)
        conf_np = conf.detach().cpu().numpy() if hasattr(conf, "detach") else np.asarray(conf)
        cls_np = cls.detach().cpu().numpy() if hasattr(cls, "detach") else np.asarray(cls)

        for xys, confs, clss in zip(xyxy_np, conf_np, cls_np):
            detection = Detection(
                int(clss),
                self.model.names[int(clss)],
                float(confs),
                int(xys[0]), int(xys[1]), int(xys[2]), int(xys[3]),
            )
            detections.append(detection)

        return detections

    def _is_red(self, image: np.ndarray) -> bool:
        if image.size == 0:
            return False
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        m1 = cv2.inRange(hsv, np.array([0, 100, 100]), np.array([10, 255, 255]))
        m2 = cv2.inRange(hsv, np.array([160, 100, 100]), np.array([179, 255, 255]))
        red = m1 | m2
        return (float(np.count_nonzero(red)) / float(red.size)) > self.red_pixel_fraction

    def draw_detections(
        self,
        bgr_image: np.ndarray,
        detections: List[Detection],
    ) -> np.ndarray:
        out_image = bgr_image.copy()

        for det in detections:
            top_left = (int(det.x1), int(det.y1))
            bottom_right = (int(det.x2), int(det.y2))
            color = self.class_color_map[det.class_name]
            label = f"{det.class_name} {det.confidence:.2f}"

            if getattr(det, 'is_red', False):
                color = (0, 0, 255)  # Bright Red
                label += " [RED!]"

            cv2.rectangle(out_image, top_left, bottom_right, color, 2)

            text_x = int(det.x1)
            text_y = max(int(det.y1) - 10, 10)
            cv2.putText(
                out_image,
                label,
                (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
                cv2.LINE_AA,
            )

        return out_image


def main() -> None:
    rclpy.init()
    node = YoloAnnotatorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
