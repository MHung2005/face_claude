# ============================================================
# quality.py — FaceQualityScorer
# ============================================================
# Đánh giá chất lượng frame cho face enrollment:
# sharpness (Laplacian), brightness, pose bonus.
# ============================================================

import logging
import math

logger = logging.getLogger(__name__)

EXPECTED_POSES = ("front", "left", "right", "up", "down")


class FaceQualityScorer:
    """Đánh giá chất lượng frame cho face enrollment.

    Sử dụng:
        scorer = FaceQualityScorer()
        score = scorer.score(frame_bytes, pose_label="front", seed=0)
    """

    def score(self, frame_bytes, pose_label="unknown", seed=0):
        """Tính điểm chất lượng frame (0.0 - 1.0).

        Args:
            frame_bytes: Raw bytes ảnh.
            pose_label: Nhãn tư thế ("front", "left", "right", "up", "down").
            seed: Index để phân biệt frames khi fallback.

        Returns:
            float: Điểm chất lượng từ 0.0 đến 1.0.
        """
        try:
            import cv2
            import numpy as np

            arr = np.frombuffer(frame_bytes, dtype=np.uint8)
            image = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise ValueError("decode_failed")

            sharpness = min(cv2.Laplacian(image, cv2.CV_64F).var() / 240.0, 1.0)
            brightness = float(image.mean()) / 255.0
            brightness_balance = max(0.0, 1.0 - abs(brightness - 0.55) * 1.8)
            pose_bonus = 0.05 if pose_label in EXPECTED_POSES else 0.0
            score = 0.62 * sharpness + 0.33 * brightness_balance + pose_bonus
            return max(0.0, min(score, 1.0))
        except Exception:
            pose_bonus = 0.05 if pose_label in EXPECTED_POSES else 0.0
            return 0.55 + pose_bonus - (seed * 0.0001)

    @staticmethod
    def cosine_distance(left, right):
        """Tính cosine distance giữa 2 embedding vectors."""
        if len(left) != len(right):
            return math.inf

        left_norm = math.sqrt(sum(value * value for value in left))
        right_norm = math.sqrt(sum(value * value for value in right))
        if left_norm == 0 or right_norm == 0:
            return math.inf

        dot_product = sum(lv * rv for lv, rv in zip(left, right))
        cosine_similarity = dot_product / (left_norm * right_norm)
        return 1 - cosine_similarity
