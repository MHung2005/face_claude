# ============================================================
# detector.py — FaceDetector (YOLO wrapper)
# ============================================================
# Detect khuôn mặt + keypoints bằng YOLOv12-face.
# Bao gồm logic kiểm tra chất lượng detection.
# ============================================================

import logging
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_DEFAULT_MODEL_PATH = Path(__file__).parent / "models" / "yolov12n-face.pt"


class FaceDetector:
    """Detect khuôn mặt bằng YOLOv12-face (lazy-load model).

    Sử dụng:
        detector = FaceDetector()
        detections = detector.detect(image_bytes)
        # hoặc
        detections = detector.detect_from_ndarray(img_bgr)
    """

    def __init__(
        self,
        model_path=None,
        confidence=0.5,
        min_box_conf=0.25,
        min_head_kpts=2,
        kpt_conf=0.2,
        min_face_size=48,
        high_conf_box_without_kpts=0.65,
    ):
        self._model_path = str(model_path or _DEFAULT_MODEL_PATH)
        self._confidence = confidence
        self._min_box_conf = float(min_box_conf)
        self._min_head_kpts = int(min_head_kpts)
        self._kpt_conf = float(kpt_conf)
        self._min_face_size = int(min_face_size)
        self._high_conf_box_without_kpts = float(high_conf_box_without_kpts)
        self._model = None  # Lazy-loaded

    def _get_model(self):
        if self._model is None:
            from ultralytics import YOLO

            logger.info("Loading YOLO face model from %s ...", self._model_path)
            self._model = YOLO(self._model_path)
            logger.info("YOLO face model loaded successfully.")
        return self._model

    def detect(self, frame_bytes):
        """Detect khuôn mặt từ raw bytes ảnh.

        Returns:
            list[dict]: Mỗi phần tử có keys: box (x1,y1,x2,y2), keypoints (numpy), score.
            Danh sách rỗng nếu không tìm thấy khuôn mặt hợp lệ.
        """
        arr = np.frombuffer(frame_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            return []
        return self.detect_from_ndarray(img)

    def detect_from_ndarray(self, img):
        """Detect khuôn mặt từ numpy BGR image.

        Returns:
            list[dict]: Mỗi phần tử có keys: box, keypoints, score.
        """
        model = self._get_model()
        results = model.predict(img, conf=self._confidence, verbose=False)

        if not results or len(results[0].boxes) == 0:
            return []

        detections_raw = results[0]
        boxes = detections_raw.boxes
        keypoints_data = detections_raw.keypoints

        faces = []
        skip_reasons = []

        for idx in range(len(boxes)):
            is_qualified, reason = self._check_quality(detections_raw, idx)
            if not is_qualified:
                logger.debug("Skipping face #%d: %s", idx, reason)
                skip_reasons.append(reason)
                continue

            box = boxes.xyxy[idx].cpu().numpy().astype(int)
            x1, y1, x2, y2 = box

            kps = None
            if keypoints_data is not None and keypoints_data.xy is not None:
                kps_xy = keypoints_data.xy[idx].cpu().numpy().astype(int)
                if len(kps_xy) >= 2:
                    kps = kps_xy

            score = float(boxes.conf[idx].cpu().numpy())

            faces.append({
                "box": (x1, y1, x2, y2),
                "keypoints": kps,
                "score": score,
            })

        if not faces and skip_reasons:
            logger.info("No face detected. Reasons: %s", "; ".join(skip_reasons))

        return faces

    def _check_quality(self, detections, detection_index):
        """Kiểm tra chất lượng detection (box size, keypoints, confidence)."""
        boxes = detections.boxes
        keypoints = detections.keypoints

        if boxes is None or len(boxes) == 0:
            return False, "skip no bounding box"

        scores = boxes.conf.cpu().numpy() if boxes.conf is not None else None
        if scores is None or len(scores) == 0:
            return False, "skip empty box score"

        best_score = float(scores[detection_index])
        if best_score < self._min_box_conf:
            return False, f"skip low box score ({best_score:.3f})"

        box_xyxy = boxes.xyxy[detection_index].cpu().numpy().astype(int)
        x1, y1, x2, y2 = box_xyxy
        face_w = max(0, x2 - x1)
        face_h = max(0, y2 - y1)
        if min(face_w, face_h) < self._min_face_size:
            return False, f"skip tiny face ({face_w}x{face_h})"

        if keypoints is None or keypoints.xy is None:
            if best_score >= self._high_conf_box_without_kpts:
                return True, f"ok no keypoints but high box score ({best_score:.3f})"
            return False, "skip no keypoints"

        keypoints_xy = keypoints.xy.cpu().numpy()
        if detection_index >= len(keypoints_xy):
            if best_score >= self._high_conf_box_without_kpts:
                return True, f"ok invalid keypoint index but high box score ({best_score:.3f})"
            return False, "skip invalid keypoint index"

        keypoints_conf = keypoints.conf.cpu().numpy() if keypoints.conf is not None else None
        head_indices = [0, 1, 2, 3, 4]
        valid_head = 0
        for idx in head_indices:
            if idx >= keypoints_xy.shape[1]:
                continue

            px = float(keypoints_xy[detection_index, idx, 0])
            py = float(keypoints_xy[detection_index, idx, 1])
            point_conf = float(keypoints_conf[detection_index, idx]) if keypoints_conf is not None else 1.0
            if px > 0 and py > 0 and point_conf >= self._kpt_conf:
                valid_head += 1

        if valid_head < self._min_head_kpts:
            if best_score >= self._high_conf_box_without_kpts:
                return True, (
                    f"ok weak keypoints ({valid_head}) but high box score ({best_score:.3f})"
                )
            return False, f"skip insufficient head keypoints ({valid_head})"

        return True, "ok"
