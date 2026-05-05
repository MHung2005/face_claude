# ============================================================
# app.ai — AI Package (Facade Pattern)
# ============================================================
# Gom tất cả tác vụ AI vào một package duy nhất.
# Bất kỳ module nào cần AI chỉ cần:
#
#   from app.ai import FaceDetector, FaceEmbedder, FaceAligner, FaceQualityScorer
#
# ============================================================

from .detector import FaceDetector
from .embedder import FaceEmbedder
from .aligner import FaceAligner, align_face
from .quality import FaceQualityScorer

__all__ = [
    "FaceDetector",
    "FaceEmbedder",
    "FaceAligner",
    "FaceQualityScorer",
    "align_face",
]
