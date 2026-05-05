import mimetypes
from pathlib import Path

from flask import Blueprint, current_app, jsonify, request, send_file
from sqlalchemy.exc import IntegrityError

from ..models import FaceSample
from ..services.auth import require_manager, serialize_employee
from ..services.face_batch_enrollment import FaceBatchEnrollmentError
from .face_enrollment_service import FaceEnrollmentError, FaceEnrollmentService
from .helpers import (
    get_employee,
    get_service,
    invalid_request,
    serialize_face_sample,
)

ALLOWED_IMAGE_EXTENSIONS = {"jpg", "jpeg", "png", "bmp", "webp"}


def _allowed_image(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_IMAGE_EXTENSIONS


face_enrollment_bp = Blueprint("face_enrollment", __name__)


def _build_capture_config():
    return {
        "min_frames": current_app.config.get("FACE_BATCH_MIN_FRAMES", 8),
        "max_frames": current_app.config.get("FACE_BATCH_MAX_FRAMES", 12),
        "thumbnail_limit": 10,
        "min_capture_gap_ms": current_app.config.get("FACE_CAPTURE_MIN_GAP_MS", 300),
    }


def _get_enrollment_service():
    """Tạo FaceEnrollmentService từ các service đã inject."""
    return FaceEnrollmentService(
        embedding_service=get_service("embedding_service"),
        storage_service=get_service("storage_service"),
        face_index_service=get_service("face_index_service"),
    )


@face_enrollment_bp.get("/manager/employees/<int:employee_id>/face-samples")
def manager_employee_face_samples(employee_id):
    _, error_response = require_manager()
    if error_response is not None:
        return error_response

    employee, error_response = get_employee(employee_id)
    if error_response is not None:
        return error_response

    face_samples = (
        FaceSample.query.filter_by(employee_id=employee.id)
        .order_by(FaceSample.sample_index.asc())
        .all()
    )
    return jsonify(
        {
            "employee": serialize_employee(employee),
            "face_samples": [serialize_face_sample(fs) for fs in face_samples],
            "capture_config": _build_capture_config(),
        }
    )


@face_enrollment_bp.get("/manager/employees/<int:employee_id>/face-samples/<int:sample_index>/image")
def manager_employee_face_sample_image(employee_id, sample_index):
    _, error_response = require_manager()
    if error_response is not None:
        return error_response

    employee, error_response = get_employee(employee_id)
    if error_response is not None:
        return error_response

    face_sample = FaceSample.query.filter_by(employee_id=employee.id, sample_index=sample_index).first()
    if face_sample is None:
        return jsonify({"status": "face_sample_not_found"}), 404

    image_path = Path(face_sample.image_path)
    if not image_path.exists() or not image_path.is_file():
        return jsonify({"status": "face_sample_not_found"}), 404

    mime_type = mimetypes.guess_type(image_path.name)[0] or "application/octet-stream"
    return send_file(image_path, mimetype=mime_type)


@face_enrollment_bp.post("/manager/employees/<int:employee_id>/face-enrollment")
def manager_employee_face_enrollment(employee_id):
    _, error_response = require_manager()
    if error_response is not None:
        return error_response

    employee, error_response = get_employee(employee_id)
    if error_response is not None:
        return error_response

    service = _get_enrollment_service()

    if service.has_registration(employee.id):
        return jsonify({"status": "face_registration_exists"}), 409

    images = request.files.getlist("images")
    expected = current_app.config.get("FACE_SAMPLES_PER_ENROLLMENT", 5)
    if len(images) != expected:
        return invalid_request(f"exactly {expected} images are required")

    try:
        result = service.enroll_single(employee, images, allowed_check_fn=_allowed_image)
    except FaceEnrollmentError as error:
        payload = {"status": error.status, "message": error.message}
        payload.update(error.payload)
        return jsonify(payload), error.http_code
    except IntegrityError:
        if service.has_registration(employee.id):
            return jsonify({"status": "face_registration_exists"}), 409
        raise

    return (
        jsonify(
            {
                "employee": serialize_employee(employee),
                "face_samples": [serialize_face_sample(fs) for fs in result["prepared_samples"]],
                "face_sample_count": result["face_sample_count"],
            }
        ),
        201,
    )


@face_enrollment_bp.post("/manager/employees/<int:employee_id>/face-enrollment/batch")
def manager_employee_face_enrollment_batch(employee_id):
    _, error_response = require_manager()
    if error_response is not None:
        return error_response

    employee, error_response = get_employee(employee_id)
    if error_response is not None:
        return error_response

    service = _get_enrollment_service()

    if service.has_registration(employee.id):
        return jsonify({"status": "face_registration_exists"}), 409

    frames = request.files.getlist("frames")
    for frame in frames:
        if not _allowed_image(frame.filename):
            return invalid_request("all frames must be JPEG, PNG, BMP, or WebP images")

    metadata = request.form.get("metadata")
    min_frames = current_app.config.get("FACE_BATCH_MIN_FRAMES", 8)
    max_frames = current_app.config.get("FACE_BATCH_MAX_FRAMES", 12)

    try:
        response_data = service.enroll_batch(
            employee, frames, metadata=metadata,
            min_frames=min_frames, max_frames=max_frames,
        )
    except FaceBatchEnrollmentError as error:
        payload = {"status": error.status, "message": error.message}
        payload.update(error.payload)
        return jsonify(payload), 400
    except IntegrityError:
        if service.has_registration(employee.id):
            return jsonify({"status": "face_registration_exists"}), 409
        raise

    return jsonify(response_data), 201


@face_enrollment_bp.put("/manager/employees/<int:employee_id>/face-samples/<int:sample_index>")
def manager_employee_face_sample_replace(employee_id, sample_index):
    _, error_response = require_manager()
    if error_response is not None:
        return error_response

    employee, error_response = get_employee(employee_id)
    if error_response is not None:
        return error_response

    max_samples = current_app.config.get("FACE_SAMPLES_PER_ENROLLMENT", 5)
    if sample_index < 1 or sample_index > max_samples:
        return invalid_request(f"sample_index must be between 1 and {max_samples}")

    service = _get_enrollment_service()
    image = request.files.get("image")

    try:
        face_sample = service.replace_sample(
            employee, sample_index, image, allowed_check_fn=_allowed_image
        )
    except FaceEnrollmentError as error:
        payload = {"status": error.status, "message": error.message}
        payload.update(error.payload)
        return jsonify(payload), error.http_code

    return jsonify(
        {
            "employee": serialize_employee(employee),
            "face_sample": serialize_face_sample(face_sample),
            "status": "updated",
        }
    )


@face_enrollment_bp.delete("/manager/employees/<int:employee_id>/face-samples")
def manager_employee_face_samples_delete(employee_id):
    _, error_response = require_manager()
    if error_response is not None:
        return error_response

    employee, error_response = get_employee(employee_id)
    if error_response is not None:
        return error_response

    service = _get_enrollment_service()
    deleted_count = service.delete_all(employee.id)

    return jsonify({"employee_id": employee.id, "deleted_count": deleted_count})
