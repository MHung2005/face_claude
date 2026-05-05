# ============================================================
# face_enrollment_service.py — Business Logic cho Face Enrollment
# ============================================================
# Tách logic nghiệp vụ ra khỏi route handler.
# Route chỉ lo validate request + trả response.
# Service lo xử lý data + DB + storage + index.
# ============================================================

import json
import logging

from ..extensions import db
from ..models import FaceEmbedding, FaceSample
from ..services.face_batch_enrollment import FaceBatchEnrollmentService

logger = logging.getLogger(__name__)


class FaceEnrollmentError(Exception):
    """Lỗi nghiệp vụ trong quá trình enrollment."""
    def __init__(self, status, message, payload=None, http_code=400):
        super().__init__(message)
        self.status = status
        self.message = message
        self.payload = payload or {}
        self.http_code = http_code


class FaceEnrollmentService:
    """Xử lý toàn bộ logic enrollment: extract → persist → index.

    Sử dụng:
        service = FaceEnrollmentService(
            embedding_service, storage_service, face_index_service, db
        )
        result = service.enroll_single(employee, images)
        result = service.enroll_batch(employee, frames, metadata, config)
        service.delete_all(employee_id)
    """

    def __init__(self, embedding_service, storage_service, face_index_service):
        self._embedding = embedding_service
        self._storage = storage_service
        self._index = face_index_service

    # ------------------------------------------------------------------
    # Check registration
    # ------------------------------------------------------------------
    @staticmethod
    def has_registration(employee_id):
        """Kiểm tra nhân viên đã có face registration chưa."""
        return (
            FaceSample.query.filter_by(employee_id=employee_id).first() is not None
            or FaceEmbedding.query.filter_by(employee_id=employee_id).first() is not None
        )

    # ------------------------------------------------------------------
    # Single enrollment (5 ảnh tĩnh)
    # ------------------------------------------------------------------
    def enroll_single(self, employee, images, allowed_check_fn=None):
        """Enroll từ danh sách ảnh tĩnh (5 ảnh, mỗi ảnh 1 mặt).

        Returns:
            dict: { prepared_samples, face_sample_count }
        Raises:
            FaceEnrollmentError: Nếu ảnh không hợp lệ.
        """
        saved_paths = []
        prepared_samples = []

        try:
            for sample_index, image in enumerate(images, start=1):
                if image is None or not image.filename:
                    raise FaceEnrollmentError("invalid_request", "images are required")

                if allowed_check_fn and not allowed_check_fn(image.filename):
                    raise FaceEnrollmentError(
                        "invalid_request",
                        "images must be JPEG, PNG, BMP, or WebP format"
                    )

                frame_bytes = image.read()
                if not frame_bytes:
                    raise FaceEnrollmentError("invalid_request", "images are required")

                embeddings = self._embedding.extract_embeddings(frame_bytes)
                if len(embeddings) == 0:
                    raise FaceEnrollmentError(
                        "no_face", "No face detected",
                        {"image_index": sample_index}
                    )
                if len(embeddings) > 1:
                    raise FaceEnrollmentError(
                        "multiple_faces", "Multiple faces detected",
                        {"image_index": sample_index, "faces_detected": len(embeddings)}
                    )

                image_path = self._storage.save_employee_face_sample(
                    employee.id, sample_index, frame_bytes, filename=image.filename
                )
                saved_paths.append(image_path)
                prepared_samples.append(
                    FaceSample(
                        employee_id=employee.id,
                        sample_index=sample_index,
                        image_path=str(image_path),
                        embedding_json=json.dumps(embeddings[0]),
                    )
                )

            db.session.add_all(prepared_samples)
            db.session.commit()
        except FaceEnrollmentError:
            self._storage.remove_employee_face_files(saved_paths)
            raise
        except Exception:
            db.session.rollback()
            self._storage.remove_employee_face_files(saved_paths)
            raise

        # Index embeddings
        for sample in prepared_samples:
            embedding = json.loads(sample.embedding_json)
            self._index.upsert(
                employee_id=employee.id,
                sample_index=sample.sample_index,
                employee_code=employee.employee_code,
                full_name=employee.full_name,
                embedding=embedding,
            )

        return {
            "prepared_samples": prepared_samples,
            "face_sample_count": len(prepared_samples),
        }

    # ------------------------------------------------------------------
    # Batch enrollment (20-30 frames auto-capture)
    # ------------------------------------------------------------------
    def enroll_batch(self, employee, frames, metadata=None, min_frames=8, max_frames=12):
        """Enroll từ batch frames tự động.

        Returns:
            dict: Response data bao gồm preview_samples, embedding counts, etc.
        Raises:
            FaceEnrollmentError: Nếu batch processing thất bại.
        """
        batch_service = FaceBatchEnrollmentService(
            self._embedding,
            min_frames=min_frames,
            max_frames=max_frames,
        )

        batch_result = batch_service.prepare_batch(frames, metadata=metadata)

        saved_paths = []
        try:
            prepared_samples, saved_paths = self._persist_preview_samples(
                employee.id, batch_result["preview_frames"]
            )
            prepared_embeddings = self._persist_embeddings(
                employee.id, batch_result
            )

            db.session.add_all(prepared_samples)
            db.session.add_all(prepared_embeddings)
            db.session.commit()
        except Exception:
            db.session.rollback()
            self._storage.remove_employee_face_files(saved_paths)
            raise

        self._index.refresh()

        return self._build_batch_response(
            employee, prepared_samples, batch_result, prepared_embeddings
        )

    # ------------------------------------------------------------------
    # Replace single sample
    # ------------------------------------------------------------------
    def replace_sample(self, employee, sample_index, image, allowed_check_fn=None):
        """Thay thế 1 ảnh mẫu cụ thể.

        Returns:
            FaceSample: Model đã cập nhật.
        Raises:
            FaceEnrollmentError: Nếu ảnh không hợp lệ.
        """
        if image is None or not image.filename:
            raise FaceEnrollmentError("invalid_request", "image is required")

        if allowed_check_fn and not allowed_check_fn(image.filename):
            raise FaceEnrollmentError("invalid_request", "image must be a JPEG, PNG, BMP, or WebP")

        frame_bytes = image.read()
        if not frame_bytes:
            raise FaceEnrollmentError("invalid_request", "image is required")

        embeddings = self._embedding.extract_embeddings(frame_bytes)
        if len(embeddings) == 0:
            raise FaceEnrollmentError(
                "no_face", "No face detected",
                {"image_index": sample_index}
            )
        if len(embeddings) > 1:
            raise FaceEnrollmentError(
                "multiple_faces", "Multiple faces detected",
                {"image_index": sample_index, "faces_detected": len(embeddings)}
            )

        face_sample = FaceSample.query.filter_by(
            employee_id=employee.id, sample_index=sample_index
        ).first()
        old_image_path = face_sample.image_path if face_sample else None

        new_image_path = self._storage.save_employee_face_sample(
            employee.id, sample_index, frame_bytes, filename=image.filename
        )

        try:
            if face_sample is None:
                face_sample = FaceSample(
                    employee_id=employee.id,
                    sample_index=sample_index,
                    image_path=str(new_image_path),
                    embedding_json=json.dumps(embeddings[0]),
                )
                db.session.add(face_sample)
            else:
                face_sample.image_path = str(new_image_path)
                face_sample.embedding_json = json.dumps(embeddings[0])

            db.session.commit()
        except Exception:
            db.session.rollback()
            self._storage.remove_path(new_image_path)
            raise

        if old_image_path and old_image_path != str(new_image_path):
            self._storage.remove_path(old_image_path)

        self._index.upsert(
            employee_id=employee.id,
            sample_index=sample_index,
            employee_code=employee.employee_code,
            full_name=employee.full_name,
            embedding=embeddings[0],
        )

        return face_sample

    # ------------------------------------------------------------------
    # Delete all face data
    # ------------------------------------------------------------------
    def delete_all(self, employee_id):
        """Xóa toàn bộ face samples + embeddings + index cho employee.

        Returns:
            int: Số lượng samples đã xóa.
        """
        face_samples = (
            FaceSample.query.filter_by(employee_id=employee_id)
            .order_by(FaceSample.sample_index.asc())
            .all()
        )

        for fs in face_samples:
            db.session.delete(fs)
        db.session.commit()

        self._storage.remove_employee_face_files([fs.image_path for fs in face_samples])

        # Also delete embeddings
        face_embeddings = FaceEmbedding.query.filter_by(employee_id=employee_id).all()
        removable_paths = []
        for fe in face_embeddings:
            if fe.image_path:
                removable_paths.append(fe.image_path)
            db.session.delete(fe)
        db.session.commit()

        if removable_paths:
            self._storage.remove_employee_face_files(removable_paths)

        self._index.delete_employee(employee_id)
        return len(face_samples)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _persist_preview_samples(self, employee_id, preview_frames):
        prepared_samples = []
        saved_paths = []

        for preview in preview_frames:
            candidate = preview["candidate"]
            image_path = self._storage.save_employee_face_sample(
                employee_id,
                preview["sample_index"],
                candidate.frame_bytes,
                filename=candidate.filename,
            )
            saved_paths.append(image_path)
            prepared_samples.append(
                FaceSample(
                    employee_id=employee_id,
                    sample_index=preview["sample_index"],
                    image_path=str(image_path),
                    embedding_json=json.dumps(candidate.embedding),
                )
            )

        return prepared_samples, saved_paths

    @staticmethod
    def _persist_embeddings(employee_id, batch_result):
        prepared_embeddings = [
            FaceEmbedding(
                employee_id=employee_id,
                embedding_role="mean",
                pose_label="aggregate",
                quality_score=None,
                image_path=None,
                embedding_json=json.dumps(batch_result["mean_embedding"]),
            )
        ]

        for candidate in batch_result["representative_frames"]:
            prepared_embeddings.append(
                FaceEmbedding(
                    employee_id=employee_id,
                    embedding_role="representative",
                    pose_label=candidate.pose_label,
                    quality_score=candidate.quality_score,
                    image_path=None,
                    embedding_json=json.dumps(candidate.embedding),
                )
            )

        return prepared_embeddings

    @staticmethod
    def _build_batch_response(employee, prepared_samples, batch_result, prepared_embeddings):
        from ..services.auth import serialize_employee
        from .helpers import serialize_face_sample

        preview_samples = []
        pose_by_sample_index = {
            item["sample_index"]: item["pose_label"]
            for item in batch_result["preview_frames"]
        }

        for sample in prepared_samples:
            payload = serialize_face_sample(sample)
            payload["pose_label"] = pose_by_sample_index.get(sample.sample_index, "unknown")
            preview_samples.append(payload)

        representative_count = sum(
            1 for item in prepared_embeddings if item.embedding_role == "representative"
        )
        return {
            "employee": serialize_employee(employee),
            "face_samples": preview_samples,
            "face_sample_count": len(prepared_samples),
            "valid_frame_count": batch_result["valid_frame_count"],
            "rejected_frame_count": batch_result["rejected_frame_count"],
            "selected_frame_count": batch_result["selected_frame_count"],
            "saved_embedding_count": len(prepared_embeddings),
            "representative_embedding_count": representative_count,
            "status": "enrolled_from_batch",
        }
