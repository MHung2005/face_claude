# ============================================================
# embedder.py — FaceEmbedder (InsightFace wrapper)
# ============================================================
# Trích xuất face embedding 512 chiều bằng InsightFace (ArcFace).
# Model được lazy-load lần đầu gọi embed().
# ============================================================

import importlib
import logging
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class FaceEmbedder:
    """Trích xuất face embedding 512-dim bằng InsightFace.

    Sử dụng:
        embedder = FaceEmbedder()
        vector = embedder.embed(aligned_face_bgr)
        # vector là list[float] 512 chiều
    """

    def __init__(
        self,
        rec_model_path=None,
        model_name="buffalo_l",
        det_size=(320, 320),
        providers=None,
    ):
        default_rec_model_path = Path.home() / ".insightface" / "models" / "buffalo_l" / "w600k_r50.onnx"
        self._rec_model_path = str(rec_model_path or default_rec_model_path)
        self._model_name = model_name
        self._det_size = tuple(det_size)
        self._providers = list(providers or ["CPUExecutionProvider"])
        self._recognizer = None  # Lazy-loaded

    def get_recognizer(self):
        """Lấy InsightFace recognizer (lazy-load). Public để cho phép pre-warm."""
        if self._recognizer is None:
            ctx_id = 0 if "CUDAExecutionProvider" in self._providers else -1
            model_zoo_module = importlib.import_module("insightface.model_zoo")

            model_file = Path(self._rec_model_path)
            if model_file.is_file():
                try:
                    recognizer = model_zoo_module.get_model(
                        str(model_file),
                        providers=self._providers,
                    )
                    recognizer.prepare(ctx_id=ctx_id)
                    self._recognizer = recognizer
                    return self._recognizer
                except Exception:
                    logger.warning(
                        "Cannot load recognizer from path %s, fallback to FaceAnalysis model name %s",
                        self._rec_model_path,
                        self._model_name,
                        exc_info=True,
                    )

            face_analysis_module = importlib.import_module("insightface.app")
            face_analysis_class = face_analysis_module.FaceAnalysis
            # insightface 0.2.1 không hỗ trợ providers= — chỉ dùng ctx_id trong .prepare()
            app = face_analysis_class(
                name=self._model_name,
            )
            app.prepare(ctx_id=ctx_id, det_size=self._det_size)

            recognizer = None
            models = getattr(app, "models", None)
            if isinstance(models, dict):
                recognizer = models.get("recognition")
                if recognizer is None:
                    for model in models.values():
                        if hasattr(model, "get_feat"):
                            recognizer = model
                            break

            if recognizer is None:
                raise RuntimeError("InsightFace recognition model is unavailable")

            self._recognizer = recognizer
            logger.info("[InsightFace] Model ready (cached).")
            return self._recognizer

        return self._recognizer

    def embed(self, aligned_face):
        """Trích xuất embedding từ ảnh khuôn mặt đã align (numpy BGR).

        Args:
            aligned_face: numpy array BGR, kích thước tùy ý (sẽ resize về 112x112).

        Returns:
            list[float]: Vector embedding 512 chiều, hoặc None nếu thất bại.
        """
        try:
            face_for_recognition = cv2.resize(aligned_face, (112, 112), interpolation=cv2.INTER_AREA)
            recognizer = self.get_recognizer()
            vector = recognizer.get_feat(face_for_recognition)
            if vector is None:
                return None

            vector_np = np.asarray(vector, dtype=np.float32)
            if vector_np.ndim > 1:
                vector_np = vector_np[0]

            return vector_np.reshape(-1).tolist()
        except Exception:
            logger.warning("InsightFace embed failed", exc_info=True)
            return None
