# ============================================================
# embedding.py - EmbeddingService (Orchestrator)
# ============================================================
# Orchestrator sử dụng AI package (app.ai) để:
#   1. Detect khuôn mặt (FaceDetector)
#   2. Align khuôn mặt (FaceAligner)
#   3. Trích xuất embedding (FaceEmbedder)
# ============================================================

import logging
import time

import cv2
import numpy as np

from ..ai import FaceDetector, FaceEmbedder, FaceAligner, align_face

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Orchestrator: detect + align + embed khuôn mặt.

    Sử dụng AI package bên dưới, cung cấp API cấp cao cho
    RecognitionService và FaceBatchEnrollmentService.
    """

    def __init__(
        self,
        model_path=None,
        yolo_confidence=0.5,
        min_box_conf=0.25,
        min_head_kpts=2,
        kpt_conf=0.2,
        min_face_size=48,
        high_conf_box_without_kpts=0.65,
        insightface_rec_model_path=None,
        insightface_model_name="buffalo_l",
        insightface_det_size=(320, 320),
        insightface_providers=None,
    ):
        self._detector = FaceDetector(
            model_path=model_path,
            confidence=yolo_confidence,
            min_box_conf=min_box_conf,
            min_head_kpts=min_head_kpts,
            kpt_conf=kpt_conf,
            min_face_size=min_face_size,
            high_conf_box_without_kpts=high_conf_box_without_kpts,
        )
        self._embedder = FaceEmbedder(
            rec_model_path=insightface_rec_model_path,
            model_name=insightface_model_name,
            det_size=insightface_det_size,
            providers=insightface_providers,
        )
        self._aligner = FaceAligner()

    def _get_insightface_recognizer(self):
        """Public accessor for pre-warming InsightFace at startup."""
        return self._embedder.get_recognizer()

    # ------------------------------------------------------------------
    # Public API — Luồng full-frame (YOLO detect trên backend)
    # ------------------------------------------------------------------
    def extract_embeddings(self, frame_bytes):
        """Nhận raw bytes ảnh, trả về list các embedding vectors.

        Returns:
            list[list[float]]: Mỗi phần tử là 1 embedding 512 chiều.
            Danh sách rỗng nếu không tìm thấy khuôn mặt.
        """
        arr = np.frombuffer(frame_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            return []

        faces = self._detector.detect_from_ndarray(img)
        if not faces:
            return []

        embeddings = []
        for face in faces:
            x1, y1, x2, y2 = face["box"]
            aligned = self._aligner.align(img, face["keypoints"], (x1, y1, x2, y2))

            if aligned is None or aligned.size == 0:
                continue

            vector = self._embedder.embed(aligned)
            if vector is not None:
                embeddings.append(vector)

        return embeddings

    # ------------------------------------------------------------------
    # Public API — Luồng crop (YOLO ONNX chạy trên browser)
    # ------------------------------------------------------------------
    def extract_embeddings_from_crop(self, crop_bytes, keypoints_list):
        """Nhận ảnh crop + 5 keypoints, trả về embedding + timing.

        Returns:
            tuple(list[float] | None, dict): (embedding, timing_dict).
        """
        t0 = time.perf_counter()
        timing = {}

        arr = np.frombuffer(crop_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            logger.warning("extract_embeddings_from_crop: cannot decode crop image")
            return None, timing
        t1 = time.perf_counter()
        timing["decode_ms"] = round((t1 - t0) * 1000, 1)

        h, w = img.shape[:2]
        if h < 20 or w < 20:
            logger.warning("extract_embeddings_from_crop: crop too small (%dx%d)", w, h)
            return None, timing

        # Parse keypoints
        kps = None
        if keypoints_list and len(keypoints_list) >= 4:
            try:
                flat = [float(v) for v in keypoints_list]
                kps = np.array(flat, dtype=np.int32).reshape(-1, 2)
            except (ValueError, TypeError):
                logger.warning("extract_embeddings_from_crop: invalid keypoints, fallback to raw crop")
                kps = None

        box = (0, 0, w, h)

        t2 = time.perf_counter()
        aligned_face = align_face(img, kps, box)
        t3 = time.perf_counter()
        timing["align_ms"] = round((t3 - t2) * 1000, 1)

        if aligned_face is None or aligned_face.size == 0:
            logger.warning("extract_embeddings_from_crop: alignment produced empty result")
            return None, timing

        t4 = time.perf_counter()
        vector = self._embedder.embed(aligned_face)
        t5 = time.perf_counter()
        timing["get_feat_ms"] = round((t5 - t4) * 1000, 1)
        timing["embed_total_ms"] = round((t5 - t0) * 1000, 1)

        logger.info(
            "[TIMING] decode=%.1fms align=%.1fms get_feat=%.1fms TOTAL=%.1fms",
            timing["decode_ms"], timing["align_ms"], timing["get_feat_ms"],
            timing["embed_total_ms"],
        )

        return vector, timing
