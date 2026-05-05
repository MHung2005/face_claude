from datetime import datetime

from werkzeug.security import generate_password_hash


try:
    from backend.app.extensions import db
    from backend.app.models import AttendanceEvent, Employee, ManagerUser
    from backend.app.services import attendance as attendance_module
except ModuleNotFoundError:
    from app.extensions import db
    from app.models import AttendanceEvent, Employee, ManagerUser
    from app.services import attendance as attendance_module


def _create_manager(app, username="manager", password="secret123"):
    with app.app_context():
        manager = ManagerUser(
            username=username,
            password_hash=generate_password_hash(password),
        )
        db.session.add(manager)
        db.session.commit()
        return {
            "id": manager.id,
            "username": manager.username,
            "password": password,
        }


def _create_employee(app, employee_code="EMP-001", full_name="Ada Lovelace", department="Văn phòng", position="Nhân viên"):
    with app.app_context():
        employee = Employee(
            employee_code=employee_code,
            full_name=full_name,
            department=department,
            position=position,
        )
        db.session.add(employee)
        db.session.commit()
        return {
            "id": employee.id,
            "employee_code": employee.employee_code,
            "full_name": employee.full_name,
            "department": employee.department,
            "position": employee.position,
            "is_active": employee.is_active,
        }


def _create_attendance_event(app, employee_id, checked_in_at, snapshot_name="snapshot.jpg"):
    snapshot_path = app.config["CHECKIN_DIR"] / checked_in_at.date().isoformat() / snapshot_name
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot_path.write_bytes(b"snapshot-bytes")

    with app.app_context():
        event = AttendanceEvent(
            employee_id=employee_id,
            checked_in_at=checked_in_at,
            checkin_date=checked_in_at.date().isoformat(),
            snapshot_path=str(snapshot_path),
            distance=0.123,
        )
        db.session.add(event)
        db.session.commit()
        return event.id, snapshot_path


def _login_manager(client, manager):
    response = client.post(
        "/api/manager/login",
        json={"username": manager["username"], "password": manager["password"]},
    )
    assert response.status_code == 200


def test_manager_attendance_requires_authentication(client):
    response = client.get("/api/manager/attendance")

    assert response.status_code == 401
    assert response.get_json()["status"] == "unauthorized"


def test_manager_attendance_snapshot_requires_authentication(client):
    response = client.get("/api/manager/attendance/1/snapshot")

    assert response.status_code == 401
    assert response.get_json()["status"] == "unauthorized"


def test_manager_attendance_defaults_to_today_when_filters_missing(app, client, monkeypatch):
    class FixedDateTime:
        @classmethod
        def now(cls):
            return datetime(2026, 4, 3, 10, 0, 0)

    monkeypatch.setattr(attendance_module, "datetime", FixedDateTime)

    manager = _create_manager(app)
    employee_today = _create_employee(app, employee_code="EMP-100", full_name="Ada Lovelace")
    employee_yesterday = _create_employee(app, employee_code="EMP-101", full_name="Grace Hopper")

    _create_attendance_event(app, employee_today["id"], datetime(2026, 4, 3, 8, 15, 0), "today.jpg")
    _create_attendance_event(app, employee_yesterday["id"], datetime(2026, 4, 2, 8, 15, 0), "yesterday.jpg")

    _login_manager(client, manager)

    response = client.get("/api/manager/attendance")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["filters"] == {
        "from": "2026-04-03",
        "to": "2026-04-03",
        "search": "",
        "department": "",
        "position": "",
    }
    assert payload["pagination"]["total"] == 1
    assert payload["records"][0]["employee_code"] == "EMP-100"


def test_manager_attendance_filters_by_date_range_and_search(app, client, monkeypatch):
    class FixedDateTime:
        @classmethod
        def now(cls):
            return datetime(2026, 4, 3, 10, 0, 0)

    monkeypatch.setattr(attendance_module, "datetime", FixedDateTime)

    manager = _create_manager(app)
    employee_a = _create_employee(app, employee_code="EMP-200", full_name="Ada Lovelace")
    employee_b = _create_employee(app, employee_code="EMP-201", full_name="Grace Hopper", department="Kỹ thuật", position="Kỹ sư")
    employee_c = _create_employee(app, employee_code="EMP-202", full_name="Alan Turing", department="Nhân sự", position="Chuyên viên")

    _create_attendance_event(app, employee_a["id"], datetime(2026, 4, 1, 8, 0, 0), "a.jpg")
    _create_attendance_event(app, employee_b["id"], datetime(2026, 4, 2, 8, 0, 0), "b.jpg")
    _create_attendance_event(app, employee_c["id"], datetime(2026, 4, 3, 8, 0, 0), "c.jpg")

    _login_manager(client, manager)

    response = client.get(
        "/api/manager/attendance?from=2026-04-02&to=2026-04-03&search=hopper"
    )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["filters"] == {
        "from": "2026-04-02",
        "to": "2026-04-03",
        "search": "hopper",
        "department": "",
        "position": "",
    }
    assert payload["pagination"]["total"] == 1
    assert payload["records"][0]["employee_code"] == "EMP-201"
    assert payload["records"][0]["full_name"] == "Grace Hopper"
    assert payload["records"][0]["department"] == "Kỹ thuật"
    assert payload["records"][0]["position"] == "Kỹ sư"


def test_manager_attendance_filters_by_department_and_position(app, client, monkeypatch):
    class FixedDateTime:
        @classmethod
        def now(cls):
            return datetime(2026, 4, 3, 10, 0, 0)

    monkeypatch.setattr(attendance_module, "datetime", FixedDateTime)

    manager = _create_manager(app)
    employee_a = _create_employee(app, employee_code="EMP-210", full_name="Ada Lovelace", department="Kỹ thuật", position="Kỹ sư")
    employee_b = _create_employee(app, employee_code="EMP-211", full_name="Grace Hopper", department="Kỹ thuật", position="Trưởng nhóm")
    employee_c = _create_employee(app, employee_code="EMP-212", full_name="Alan Turing", department="Nhân sự", position="Kỹ sư")

    _create_attendance_event(app, employee_a["id"], datetime(2026, 4, 3, 8, 0, 0), "a.jpg")
    _create_attendance_event(app, employee_b["id"], datetime(2026, 4, 3, 8, 10, 0), "b.jpg")
    _create_attendance_event(app, employee_c["id"], datetime(2026, 4, 3, 8, 20, 0), "c.jpg")

    _login_manager(client, manager)

    response = client.get("/api/manager/attendance?department=K%E1%BB%B9%20thu%E1%BA%ADt&position=K%E1%BB%B9%20s%C6%B0")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["filters"] == {
        "from": "2026-04-03",
        "to": "2026-04-03",
        "search": "",
        "department": "Kỹ thuật",
        "position": "Kỹ sư",
    }
    assert payload["pagination"]["total"] == 1
    assert payload["records"][0]["employee_code"] == "EMP-210"


def test_manager_attendance_search_matches_employee_code_case_insensitively(app, client, monkeypatch):
    class FixedDateTime:
        @classmethod
        def now(cls):
            return datetime(2026, 4, 3, 10, 0, 0)

    monkeypatch.setattr(attendance_module, "datetime", FixedDateTime)

    manager = _create_manager(app)
    employee = _create_employee(app, employee_code="EMP-303", full_name="Ada Lovelace")
    _create_attendance_event(app, employee["id"], datetime(2026, 4, 3, 8, 0, 0), "search.jpg")

    _login_manager(client, manager)

    response = client.get("/api/manager/attendance?search=emp-303")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["pagination"]["total"] == 1
    assert payload["records"][0]["employee_code"] == "EMP-303"


def test_manager_attendance_rejects_invalid_date_format(app, client):
    manager = _create_manager(app)
    _login_manager(client, manager)

    response = client.get("/api/manager/attendance?from=2026/04/03")

    assert response.status_code == 400
    assert response.get_json()["status"] == "invalid_request"


def test_manager_attendance_rejects_invalid_date_range(app, client):
    manager = _create_manager(app)
    _login_manager(client, manager)

    response = client.get("/api/manager/attendance?from=2026-04-04&to=2026-04-03")

    assert response.status_code == 400
    assert response.get_json()["status"] == "invalid_request"


def test_manager_attendance_returns_snapshot_url_and_snapshot_bytes(app, client, monkeypatch):
    class FixedDateTime:
        @classmethod
        def now(cls):
            return datetime(2026, 4, 3, 10, 0, 0)

    monkeypatch.setattr(attendance_module, "datetime", FixedDateTime)

    manager = _create_manager(app)
    employee = _create_employee(app, employee_code="EMP-404", full_name="Ada Lovelace")
    event_id, snapshot_path = _create_attendance_event(app, employee["id"], datetime(2026, 4, 3, 8, 0, 0), "snapshot.jpg")

    _login_manager(client, manager)

    list_response = client.get("/api/manager/attendance")

    assert list_response.status_code == 200
    record = list_response.get_json()["records"][0]
    assert record["snapshot_url"] == f"/api/manager/attendance/{event_id}/snapshot"

    snapshot_response = client.get(record["snapshot_url"])

    assert snapshot_response.status_code == 200
    assert snapshot_response.data == snapshot_path.read_bytes()
    assert snapshot_response.content_type == "image/jpeg"


def test_manager_attendance_snapshot_returns_404_for_missing_record(app, client):
    manager = _create_manager(app)
    _login_manager(client, manager)

    response = client.get("/api/manager/attendance/999/snapshot")

    assert response.status_code == 404
    assert response.get_json()["status"] == "attendance_not_found"


def test_manager_attendance_snapshot_returns_404_for_missing_file(app, client):
    manager = _create_manager(app)
    employee = _create_employee(app, employee_code="EMP-505", full_name="Ada Lovelace")

    with app.app_context():
        event = AttendanceEvent(
            employee_id=employee["id"],
            checked_in_at=datetime(2026, 4, 3, 8, 0, 0),
            checkin_date="2026-04-03",
            snapshot_path=str(app.config["CHECKIN_DIR"] / "2026-04-03" / "missing.jpg"),
            distance=0.123,
        )
        db.session.add(event)
        db.session.commit()
        event_id = event.id

    _login_manager(client, manager)

    response = client.get(f"/api/manager/attendance/{event_id}/snapshot")

    assert response.status_code == 404
    assert response.get_json()["status"] == "snapshot_not_found"


def test_manager_attendance_snapshot_rejects_paths_outside_checkin_dir(app, client):
    manager = _create_manager(app)
    employee = _create_employee(app, employee_code="EMP-606", full_name="Ada Lovelace")
    outside_snapshot_path = app.config["CHECKIN_DIR"].parent / "outside-snapshot.jpg"
    outside_snapshot_path.write_bytes(b"outside-bytes")

    with app.app_context():
        event = AttendanceEvent(
            employee_id=employee["id"],
            checked_in_at=datetime(2026, 4, 3, 8, 0, 0),
            checkin_date="2026-04-03",
            snapshot_path=str(outside_snapshot_path),
            distance=0.123,
        )
        db.session.add(event)
        db.session.commit()
        event_id = event.id

    _login_manager(client, manager)

    response = client.get(f"/api/manager/attendance/{event_id}/snapshot")

    assert response.status_code == 404
    assert response.get_json()["status"] == "snapshot_not_found"
