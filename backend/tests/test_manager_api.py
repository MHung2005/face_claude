from pathlib import Path

from werkzeug.security import generate_password_hash


try:
    from backend.app import create_app
    from backend.app.extensions import db
    from backend.app.models import Employee, FaceEmbedding, FaceSample, ManagerUser
except ModuleNotFoundError:
    from app import create_app
    from app.extensions import db
    from app.models import Employee, FaceEmbedding, FaceSample, ManagerUser


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
            "created_at": employee.created_at.isoformat(),
        }


def test_manager_login_requires_username_and_password(client):
    response = client.post("/api/manager/login", json={})

    assert response.status_code == 400
    assert response.get_json()["status"] == "invalid_request"


def test_manager_login_rejects_invalid_credentials(app, client):
    _create_manager(app)

    response = client.post(
        "/api/manager/login",
        json={"username": "manager", "password": "wrong-password"},
    )

    assert response.status_code == 401
    assert response.get_json()["status"] == "invalid_credentials"


def test_manager_login_accepts_configured_secret_override(tmp_path):
    data_dir = tmp_path / "override-data"
    app = create_app(
        {
            "TESTING": True,
            "SECRET_KEY": "override-secret",
            "APP_DB_PATH": data_dir / "app.db",
            "CHECKIN_DIR": data_dir / "checkins",
            "FACES_DIR": data_dir / "faces",
        }
    )
    client = app.test_client()
    manager = _create_manager(app)

    response = client.post(
        "/api/manager/login",
        json={"username": manager["username"], "password": manager["password"]},
    )

    assert response.status_code == 200
    assert response.get_json()["manager"]["username"] == manager["username"]


def test_manager_login_sets_session_and_allows_me_lookup(app, client):
    manager = _create_manager(app)

    response = client.post(
        "/api/manager/login",
        json={"username": manager["username"], "password": manager["password"]},
    )

    assert response.status_code == 200
    assert response.get_json()["manager"] == {
        "id": manager["id"],
        "username": manager["username"],
    }

    me_response = client.get("/api/manager/me")

    assert me_response.status_code == 200
    assert me_response.get_json()["manager"] == {
        "id": manager["id"],
        "username": manager["username"],
    }


def test_manager_me_requires_authentication(client):
    response = client.get("/api/manager/me")

    assert response.status_code == 401
    assert response.get_json()["status"] == "unauthorized"


def test_manager_employee_list_requires_authentication(client):
    response = client.get("/api/manager/employees")

    assert response.status_code == 401
    assert response.get_json()["status"] == "unauthorized"


def test_manager_employee_list_returns_stable_payload_shape(app, client):
    manager = _create_manager(app)
    employee = _create_employee(app, employee_code="EMP-007", full_name="Ada Lovelace")

    login_response = client.post(
        "/api/manager/login",
        json={"username": manager["username"], "password": manager["password"]},
    )
    assert login_response.status_code == 200

    response = client.get("/api/manager/employees")

    assert response.status_code == 200
    assert response.get_json()["employees"] == [employee]


def test_manager_create_employee_requires_authentication(client):
    response = client.post(
        "/api/manager/employees",
        json={"employee_code": "EMP-100", "full_name": "Grace Hopper"},
    )

    assert response.status_code == 401
    assert response.get_json()["status"] == "unauthorized"


def test_manager_create_employee_validates_required_fields(app, client):
    manager = _create_manager(app)

    login_response = client.post(
        "/api/manager/login",
        json={"username": manager["username"], "password": manager["password"]},
    )
    assert login_response.status_code == 200

    response = client.post("/api/manager/employees", json={})

    assert response.status_code == 400
    assert response.get_json()["status"] == "invalid_request"


def test_manager_create_employee_rejects_duplicate_employee_code(app, client):
    manager = _create_manager(app)
    _create_employee(app, employee_code="EMP-100", full_name="Ada Lovelace")

    login_response = client.post(
        "/api/manager/login",
        json={"username": manager["username"], "password": manager["password"]},
    )
    assert login_response.status_code == 200

    response = client.post(
        "/api/manager/employees",
        json={"employee_code": "EMP-100", "full_name": "Grace Hopper"},
    )

    assert response.status_code == 409
    assert response.get_json()["status"] == "duplicate_employee_code"


def test_manager_create_employee_allows_duplicate_full_name_when_code_differs(app, client):
    manager = _create_manager(app)
    _create_employee(app, employee_code="EMP-100", full_name="Ada Lovelace")

    login_response = client.post(
        "/api/manager/login",
        json={"username": manager["username"], "password": manager["password"]},
    )
    assert login_response.status_code == 200

    response = client.post(
        "/api/manager/employees",
        json={"employee_code": "EMP-101", "full_name": "Ada Lovelace"},
    )

    assert response.status_code == 201
    assert response.get_json()["employee"]["employee_code"] == "EMP-101"


def test_manager_create_employee_persists_employee(app, client):
    manager = _create_manager(app)

    login_response = client.post(
        "/api/manager/login",
        json={"username": manager["username"], "password": manager["password"]},
    )
    assert login_response.status_code == 200

    response = client.post(
        "/api/manager/employees",
        json={"employee_code": "EMP-200", "full_name": "Grace Hopper", "department": "Kỹ thuật", "position": "Kỹ sư"},
    )

    assert response.status_code == 201
    payload = response.get_json()["employee"]
    assert payload["employee_code"] == "EMP-200"
    assert payload["full_name"] == "Grace Hopper"
    assert payload["department"] == "Kỹ thuật"
    assert payload["position"] == "Kỹ sư"
    assert payload["is_active"] is True

    with app.app_context():
        employee = Employee.query.filter_by(employee_code="EMP-200").one()
        assert employee.full_name == "Grace Hopper"
        assert employee.department == "Kỹ thuật"
        assert employee.position == "Kỹ sư"


def test_manager_create_employee_rejects_whitespace_only_values(app, client):
    manager = _create_manager(app)

    login_response = client.post(
        "/api/manager/login",
        json={"username": manager["username"], "password": manager["password"]},
    )
    assert login_response.status_code == 200

    response = client.post(
        "/api/manager/employees",
        json={"employee_code": "   ", "full_name": "\t"},
    )

    assert response.status_code == 400
    assert response.get_json()["status"] == "invalid_request"


def test_manager_update_employee_requires_authentication(client):
    response = client.put(
        "/api/manager/employees/1",
        json={"employee_code": "EMP-100", "full_name": "Grace Hopper"},
    )

    assert response.status_code == 401
    assert response.get_json()["status"] == "unauthorized"


def test_manager_update_employee_returns_404_for_missing_employee(app, client):
    manager = _create_manager(app)

    login_response = client.post(
        "/api/manager/login",
        json={"username": manager["username"], "password": manager["password"]},
    )
    assert login_response.status_code == 200

    response = client.put(
        "/api/manager/employees/999",
        json={"employee_code": "EMP-999", "full_name": "Ghost"},
    )

    assert response.status_code == 404
    assert response.get_json()["status"] == "employee_not_found"


def test_manager_update_employee_rejects_duplicate_employee_code(app, client):
    manager = _create_manager(app)
    _create_employee(app, employee_code="EMP-100", full_name="Ada Lovelace")
    target_employee = _create_employee(app, employee_code="EMP-101", full_name="Grace Hopper")

    login_response = client.post(
        "/api/manager/login",
        json={"username": manager["username"], "password": manager["password"]},
    )
    assert login_response.status_code == 200

    response = client.put(
        f"/api/manager/employees/{target_employee['id']}",
        json={"employee_code": "EMP-100", "full_name": "Grace Hopper"},
    )

    assert response.status_code == 409
    assert response.get_json()["status"] == "duplicate_employee_code"


def test_manager_update_employee_persists_fields(app, client):
    manager = _create_manager(app)
    employee = _create_employee(app, employee_code="EMP-200", full_name="Grace Hopper")

    login_response = client.post(
        "/api/manager/login",
        json={"username": manager["username"], "password": manager["password"]},
    )
    assert login_response.status_code == 200

    response = client.put(
        f"/api/manager/employees/{employee['id']}",
        json={
            "employee_code": "EMP-200A",
            "full_name": "Grace Brewster Hopper",
            "department": "Vận hành",
            "position": "Ky su",
            "is_active": False,
        },
    )

    assert response.status_code == 200
    payload = response.get_json()["employee"]
    assert payload["employee_code"] == "EMP-200A"
    assert payload["full_name"] == "Grace Brewster Hopper"
    assert payload["department"] == "Vận hành"
    assert payload["position"] == "Ky su"
    assert payload["is_active"] is False

    with app.app_context():
        updated_employee = db.session.get(Employee, employee["id"])
        assert updated_employee.employee_code == "EMP-200A"
        assert updated_employee.full_name == "Grace Brewster Hopper"
        assert updated_employee.position == "Ky su"
        assert updated_employee.is_active is False


def test_manager_delete_employee_requires_authentication(client):
    response = client.delete("/api/manager/employees/1")

    assert response.status_code == 401
    assert response.get_json()["status"] == "unauthorized"


def test_manager_delete_employee_returns_404_for_missing_employee(app, client):
    manager = _create_manager(app)

    login_response = client.post(
        "/api/manager/login",
        json={"username": manager["username"], "password": manager["password"]},
    )
    assert login_response.status_code == 200

    response = client.delete("/api/manager/employees/999")

    assert response.status_code == 404
    assert response.get_json()["status"] == "employee_not_found"


def test_manager_delete_employee_soft_deletes_and_clears_faces(app, client):
    manager = _create_manager(app)
    employee = _create_employee(app, employee_code="EMP-300", full_name="Delete Me")

    login_response = client.post(
        "/api/manager/login",
        json={"username": manager["username"], "password": manager["password"]},
    )
    assert login_response.status_code == 200

    refresh_calls = {"count": 0}

    class FakeFaceIndexService:
        def refresh(self):
            refresh_calls["count"] += 1

    app.extensions["face_index_service"] = FakeFaceIndexService()

    with app.app_context():
        sample = FaceSample(
            employee_id=employee["id"],
            sample_index=1,
            image_path=str(app.config["FACES_DIR"] / f"employee-{employee['id']}" / "sample-1.jpg"),
            embedding_json="[0.1, 0.2, 0.3]",
        )
        sample_path = Path(sample.image_path)
        sample_path.parent.mkdir(parents=True, exist_ok=True)
        sample_path.write_bytes(b"sample-1")
        db.session.add(sample)
        db.session.add(
            FaceEmbedding(
                employee_id=employee["id"],
                embedding_role="mean",
                pose_label="aggregate",
                quality_score=None,
                image_path=None,
                embedding_json="[0.1, 0.2, 0.3]",
            )
        )
        db.session.commit()

    response = client.delete(f"/api/manager/employees/{employee['id']}")

    assert response.status_code == 200
    assert response.get_json() == {
        "status": "deleted",
        "employee_id": employee["id"],
        "deactivated": True,
        "deleted_face_samples": 1,
    }
    assert refresh_calls["count"] == 1

    with app.app_context():
        deleted_employee = db.session.get(Employee, employee["id"])
        assert deleted_employee.is_active is False
        assert FaceSample.query.filter_by(employee_id=employee["id"]).count() == 0
        assert FaceEmbedding.query.filter_by(employee_id=employee["id"]).count() == 0
        assert sample_path.exists() is False

def test_manager_dashboard_excludes_inactive_employees_from_summary(app, client):
    manager = _create_manager(app)
    active_employee = _create_employee(app, employee_code="EMP-401", full_name="Active Employee")
    inactive_employee = _create_employee(app, employee_code="EMP-402", full_name="Inactive Employee")

    login_response = client.post(
        "/api/manager/login",
        json={"username": manager["username"], "password": manager["password"]},
    )
    assert login_response.status_code == 200

    delete_response = client.delete(f"/api/manager/employees/{inactive_employee['id']}")
    assert delete_response.status_code == 200

    response = client.get("/api/manager/dashboard")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["summary"]["total_employees"] == 1
    assert payload["summary"]["absent_today"] == 1
    assert [item["id"] for item in payload["employee_stats"]] == [active_employee["id"]]
