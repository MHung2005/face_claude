import mimetypes
from datetime import datetime
from pathlib import Path

from flask import Blueprint, current_app, jsonify, request, send_file, url_for
from sqlalchemy.exc import IntegrityError

from ..extensions import db
from ..models import AttendanceEvent, Employee, FaceEmbedding, FaceSample
from ..services.auth import (
    authenticate_manager,
    list_employees,
    login_manager,
    logout_manager,
    require_manager,
    serialize_employee,
    serialize_manager,
)
from .face_enrollment import _delete_face_samples_for_employee
from .helpers import (
    attendance_not_found,
    get_service,
    invalid_request,
    normalize_text,
    snapshot_not_found,
)


manager_bp = Blueprint("manager", __name__)


def _parse_date(value, field_name):
    if value is None:
        return None, None

    normalized = normalize_text(value)
    if not normalized:
        return None, invalid_request(f"{field_name} must be YYYY-MM-DD")

    try:
        return datetime.strptime(normalized, "%Y-%m-%d").date(), None
    except ValueError:
        return None, invalid_request(f"{field_name} must be YYYY-MM-DD")


def _resolve_checkin_snapshot_path(snapshot_path_value):
    checkin_dir = Path(current_app.config["CHECKIN_DIR"]).resolve()
    candidate_path = Path(snapshot_path_value)
    if not candidate_path.is_absolute():
        candidate_path = checkin_dir / candidate_path

    resolved_path = candidate_path.resolve(strict=False)
    try:
        resolved_path.relative_to(checkin_dir)
    except ValueError:
        return None

    return resolved_path


@manager_bp.post("/manager/login")
def manager_login():
    payload = request.get_json(silent=True) or {}
    username = normalize_text(payload.get("username"))
    password = normalize_text(payload.get("password"))
    if not username or not password:
        return invalid_request("username and password are required")

    manager = authenticate_manager(username, password)
    if manager is None:
        return jsonify({"status": "invalid_credentials"}), 401

    login_manager(manager)
    return jsonify({"manager": serialize_manager(manager)})


@manager_bp.post("/manager/logout")
def manager_logout():
    logout_manager()
    return jsonify({"status": "ok"})


@manager_bp.get("/manager/me")
def manager_me():
    manager, error_response = require_manager()
    if error_response is not None:
        return error_response

    return jsonify({"manager": serialize_manager(manager)})


@manager_bp.get("/manager/employees")
def manager_employees():
    _, error_response = require_manager()
    if error_response is not None:
        return error_response

    return jsonify({"employees": list_employees()})


@manager_bp.post("/manager/employees")
def manager_create_employee():
    _, error_response = require_manager()
    if error_response is not None:
        return error_response

    payload = request.get_json(silent=True) or {}
    employee_code = normalize_text(payload.get("employee_code"))
    full_name = normalize_text(payload.get("full_name"))
    department = normalize_text(payload.get("department")) or "Văn phòng"
    position = normalize_text(payload.get("position")) or "Nhân viên"
    if not employee_code or not full_name:
        return invalid_request("employee_code and full_name are required")

    existing_employee = Employee.query.filter_by(employee_code=employee_code).first()
    if existing_employee is not None:
        return jsonify({"status": "duplicate_employee_code"}), 409

    employee = Employee(
        employee_code=employee_code,
        full_name=full_name,
        department=department,
        position=position,
    )
    db.session.add(employee)
    try:
        db.session.commit()
    except IntegrityError:
        db.session.rollback()
        code_conflict = Employee.query.filter_by(employee_code=employee_code).first()
        if code_conflict is not None:
            return jsonify({"status": "duplicate_employee_code"}), 409
        raise

    return jsonify({"employee": serialize_employee(employee)}), 201


@manager_bp.put("/manager/employees/<int:employee_id>")
def manager_update_employee(employee_id):
    _, error_response = require_manager()
    if error_response is not None:
        return error_response

    employee = db.session.get(Employee, employee_id)
    if employee is None:
        return jsonify({"status": "employee_not_found"}), 404

    payload = request.get_json(silent=True) or {}
    employee_code = normalize_text(payload.get("employee_code"))
    full_name = normalize_text(payload.get("full_name"))
    department = normalize_text(payload.get("department")) or "Văn phòng"
    position = normalize_text(payload.get("position")) or "Nhân viên"

    if not employee_code or not full_name:
        return invalid_request("employee_code and full_name are required")

    existing_employee = Employee.query.filter(
        Employee.employee_code == employee_code,
        Employee.id != employee.id,
    ).first()
    if existing_employee is not None:
        return jsonify({"status": "duplicate_employee_code"}), 409

    employee.employee_code = employee_code
    employee.full_name = full_name
    employee.department = department
    employee.position = position
    employee.is_active = bool(payload.get("is_active", employee.is_active))

    try:
        db.session.commit()
    except IntegrityError:
        db.session.rollback()
        code_conflict = Employee.query.filter_by(employee_code=employee_code).first()
        if code_conflict is not None and code_conflict.id != employee.id:
            return jsonify({"status": "duplicate_employee_code"}), 409
        raise

    return jsonify({"employee": serialize_employee(employee)})


@manager_bp.delete("/manager/employees/<int:employee_id>")
def manager_delete_employee(employee_id):
    _, error_response = require_manager()
    if error_response is not None:
        return error_response

    employee = db.session.get(Employee, employee_id)
    if employee is None:
        return jsonify({"status": "employee_not_found"}), 404

    # --- Cascade cleanup (DB side via cascade relationships) ---
    # Delete face-related records and files first (outside transaction for file I/O)
    deleted_face_samples = _delete_face_samples_for_employee(employee.id)

    # --- Hard-delete the employee (cascade deletes FaceSample, FaceEmbedding, AttendanceEvent) ---
    deleted_attendance_count = AttendanceEvent.query.filter_by(employee_id=employee.id).count()
    db.session.delete(employee)
    db.session.commit()

    return jsonify(
        {
            "status": "deleted",
            "employee_id": employee.id,
            "deleted_face_samples": len(deleted_face_samples),
            "deleted_attendance_events": deleted_attendance_count,
        }
    )


@manager_bp.get("/manager/attendance")
def manager_attendance():
    _, error_response = require_manager()
    if error_response is not None:
        return error_response

    from_date, error_response = _parse_date(request.args.get("from"), "from")
    if error_response is not None:
        return error_response

    to_date, error_response = _parse_date(request.args.get("to"), "to")
    if error_response is not None:
        return error_response

    if from_date is None and to_date is None:
        from_date = to_date = datetime.now().date()
    elif from_date is None:
        from_date = to_date
    elif to_date is None:
        to_date = from_date

    if from_date > to_date:
        return invalid_request("from must be less than or equal to to")

    search = normalize_text(request.args.get("search"))
    department = normalize_text(request.args.get("department"))
    position = normalize_text(request.args.get("position"))
    page = request.args.get("page", 1, type=int)
    per_page = request.args.get("per_page", 50, type=int)
    if page < 1:
        page = 1
    if per_page < 1 or per_page > 500:
        per_page = 50

    attendance_service = get_service("attendance_service")
    result = attendance_service.list_attendance_events(
        from_date=from_date,
        to_date=to_date,
        search=search,
        department=department,
        position=position,
        page=page,
        per_page=per_page,
    )

    records = []
    for event, employee in result["records"]:
        records.append(
            {
                "id": event.id,
                "employee_id": event.employee_id,
                "employee_code": employee.employee_code,
                "full_name": employee.full_name,
                "department": employee.department,
                "position": employee.position,
                "checked_in_at": event.checked_in_at.isoformat(),
                "distance": event.distance,
                "snapshot_url": url_for("manager.manager_attendance_snapshot", attendance_id=event.id),
            }
        )

    return jsonify(
        {
            "filters": {
                "from": from_date.isoformat(),
                "to": to_date.isoformat(),
                "search": search or "",
                "department": department or "",
                "position": position or "",
            },
            "pagination": {
                "page": result["page"],
                "per_page": result["per_page"],
                "total": result["total"],
                "pages": result["pages"],
            },
            "records": records,
        }
    )


@manager_bp.get("/manager/dashboard")
def manager_dashboard():
    _, error_response = require_manager()
    if error_response is not None:
        return error_response

    dashboard_summary = get_service("attendance_service").get_dashboard_summary()
    return jsonify(dashboard_summary)


@manager_bp.get("/manager/attendance/<int:attendance_id>/snapshot")
def manager_attendance_snapshot(attendance_id):
    _, error_response = require_manager()
    if error_response is not None:
        return error_response

    attendance_service = get_service("attendance_service")
    attendance_event = attendance_service.get_attendance_event(attendance_id)
    if attendance_event is None:
        return attendance_not_found()

    snapshot_path = _resolve_checkin_snapshot_path(attendance_event.snapshot_path)
    if snapshot_path is None or not snapshot_path.exists() or not snapshot_path.is_file():
        return snapshot_not_found()

    mime_type = mimetypes.guess_type(snapshot_path.name)[0] or "application/octet-stream"
    return send_file(snapshot_path, mimetype=mime_type)
