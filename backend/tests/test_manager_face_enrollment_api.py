from io import BytesIO
from pathlib import Path

import pytest
from werkzeug.datastructures import MultiDict

try:
    from backend.app.extensions import db
    from backend.app.models import Employee, FaceEmbedding, FaceSample
    from backend.tests.test_manager_api import _create_employee, _create_manager
except ModuleNotFoundError:
    from app.extensions import db
    from app.models import Employee, FaceEmbedding, FaceSample
    from tests.test_manager_api import _create_employee, _create_manager


def _login_manager(client, manager):
    response = client.post(
        "/api/manager/login",
        json={"username": manager["username"], "password": manager["password"]},
    )
    assert response.status_code == 200


def _make_enrollment_payload(prefix="face"):
    return MultiDict(
        [
            ("images", (BytesIO(f"{prefix}-1".encode()), "1.jpg")),
            ("images", (BytesIO(f"{prefix}-2".encode()), "2.jpg")),
            ("images", (BytesIO(f"{prefix}-3".encode()), "3.jpg")),
            ("images", (BytesIO(f"{prefix}-4".encode()), "4.jpg")),
            ("images", (BytesIO(f"{prefix}-5".encode()), "5.jpg")),
        ]
    )


def _make_batch_enrollment_payload(frame_count=20):
    poses = ["front", "left", "right", "up", "down"]
    payload = MultiDict()
    metadata_frames = []

    for index in range(frame_count):
        payload.add("frames", (BytesIO(f"frame-{index}".encode()), f"frame-{index}.jpg"))
        metadata_frames.append({"index": index, "hint_pose": poses[index % len(poses)]})

    payload.add("metadata", '{"frames": ' + str(metadata_frames).replace("'", '"') + '}')
    return payload


def test_face_samples_list_requires_authentication(client):
    response = client.get("/api/manager/employees/1/face-samples")

    assert response.status_code == 401
    assert response.get_json()["status"] == "unauthorized"


def test_face_samples_list_returns_404_for_missing_employee(app, client):
    manager = _create_manager(app)
    _login_manager(client, manager)

    response = client.get("/api/manager/employees/999/face-samples")

    assert response.status_code == 404
    assert response.get_json()["status"] == "employee_not_found"


def test_face_samples_list_returns_employee_and_samples(app, client):
    manager = _create_manager(app)
    employee = _create_employee(app, employee_code="EMP-201", full_name="Ada Lovelace")

    with app.app_context():
        sample = FaceSample(
            employee_id=employee["id"],
            sample_index=1,
            image_path="/tmp/faces/EMP-201/1.jpg",
            embedding_json="[0.1, 0.2, 0.3]",
        )
        db.session.add(sample)
        db.session.commit()
        sample_id = sample.id
        created_at = sample.created_at.isoformat()

    _login_manager(client, manager)

    response = client.get(f"/api/manager/employees/{employee['id']}/face-samples")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["employee"]["id"] == employee["id"]
    assert payload["employee"]["employee_code"] == "EMP-201"
    assert payload["employee"]["full_name"] == "Ada Lovelace"
    assert payload["employee"]["is_active"] is True
    assert "created_at" in payload["employee"]
    assert payload["capture_config"] == {
        "min_frames": app.config["FACE_BATCH_MIN_FRAMES"],
        "max_frames": app.config["FACE_BATCH_MAX_FRAMES"],
        "thumbnail_limit": 10,
        "min_capture_gap_ms": app.config["FACE_CAPTURE_MIN_GAP_MS"],
    }
    assert payload["face_samples"] == [
        {
            "id": sample_id,
            "employee_id": employee["id"],
            "sample_index": 1,
            "image_path": "/tmp/faces/EMP-201/1.jpg",
            "image_url": f"/api/manager/employees/{employee['id']}/face-samples/1/image",
            "created_at": created_at,
        }
    ]


def test_face_sample_image_requires_authentication(client):
    response = client.get("/api/manager/employees/1/face-samples/1/image")

    assert response.status_code == 401
    assert response.get_json()["status"] == "unauthorized"


def test_face_sample_image_returns_file_bytes(app, client):
    manager = _create_manager(app)
    employee = _create_employee(app, employee_code="EMP-202", full_name="View Image")

    with app.app_context():
        image_path = tmp_path_sample_path(app, employee["id"], 1)
        image_path.parent.mkdir(parents=True, exist_ok=True)
        image_path.write_bytes(b"face-image")
        db.session.add(
            FaceSample(
                employee_id=employee["id"],
                sample_index=1,
                image_path=str(image_path),
                embedding_json="[0.1, 0.2, 0.3]",
            )
        )
        db.session.commit()

    _login_manager(client, manager)

    response = client.get(f"/api/manager/employees/{employee['id']}/face-samples/1/image")

    assert response.status_code == 200
    assert response.data == b"face-image"


def test_face_enrollment_requires_authentication(client):
    response = client.post(
        "/api/manager/employees/1/face-enrollment",
        data={},
        content_type="multipart/form-data",
    )

    assert response.status_code == 401
    assert response.get_json()["status"] == "unauthorized"


def test_face_enrollment_rejects_missing_employee(app, client):
    manager = _create_manager(app)
    _login_manager(client, manager)

    response = client.post(
        "/api/manager/employees/999/face-enrollment",
        data={"images": (BytesIO(b"face-1"), "1.jpg")},
        content_type="multipart/form-data",
    )

    assert response.status_code == 404
    assert response.get_json()["status"] == "employee_not_found"


def test_face_enrollment_rejects_non_five_image_batches(app, client):
    manager = _create_manager(app)
    employee = _create_employee(app, employee_code="EMP-301", full_name="Grace Hopper")
    _login_manager(client, manager)

    response = client.post(
        f"/api/manager/employees/{employee['id']}/face-enrollment",
        data=MultiDict(
            [
                ("images", (BytesIO(b"face-1"), "1.jpg")),
                ("images", (BytesIO(b"face-2"), "2.jpg")),
                ("images", (BytesIO(b"face-3"), "3.jpg")),
                ("images", (BytesIO(b"face-4"), "4.jpg")),
            ]
        ),
        content_type="multipart/form-data",
    )

    assert response.status_code == 400
    assert response.get_json()["status"] == "invalid_request"


def test_face_enrollment_rejects_existing_registration(app, client):
    manager = _create_manager(app)
    employee = _create_employee(app, employee_code="EMP-302", full_name="Ada Lovelace")

    with app.app_context():
        db.session.add(
            FaceSample(
                employee_id=employee["id"],
                sample_index=1,
                image_path="/tmp/faces/EMP-302/1.jpg",
                embedding_json="[0.1, 0.2, 0.3]",
            )
        )
        db.session.commit()

    _login_manager(client, manager)

    response = client.post(
        f"/api/manager/employees/{employee['id']}/face-enrollment",
        data=_make_enrollment_payload("existing"),
        content_type="multipart/form-data",
    )

    assert response.status_code == 409
    assert response.get_json()["status"] == "face_registration_exists"


def test_face_enrollment_rejects_no_face_and_cleans_files(app, client):
    manager = _create_manager(app)
    employee = _create_employee(app, employee_code="EMP-303", full_name="No Face")
    _login_manager(client, manager)

    class FakeEmbeddingService:
        def extract_embeddings(self, frame_bytes):
            return [] if frame_bytes == b"face-3" else [[0.1, 0.2, 0.3]]

    class FakeFaceIndexService:
        def refresh(self):
            raise AssertionError("refresh should not be called on failure")

    app.extensions["embedding_service"] = FakeEmbeddingService()
    app.extensions["face_index_service"] = FakeFaceIndexService()

    response = client.post(
        f"/api/manager/employees/{employee['id']}/face-enrollment",
        data=_make_enrollment_payload("face"),
        content_type="multipart/form-data",
    )

    assert response.status_code == 400
    assert response.get_json()["status"] == "no_face"
    assert list(Path(app.config["FACES_DIR"]).rglob("*")) == []


def test_face_enrollment_rejects_multiple_faces_and_cleans_files(app, client):
    manager = _create_manager(app)
    employee = _create_employee(app, employee_code="EMP-304", full_name="Multiple Faces")
    _login_manager(client, manager)

    class FakeEmbeddingService:
        def extract_embeddings(self, frame_bytes):
            if frame_bytes == b"face-2":
                return [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
            return [[0.1, 0.2, 0.3]]

    class FakeFaceIndexService:
        def refresh(self):
            raise AssertionError("refresh should not be called on failure")

    app.extensions["embedding_service"] = FakeEmbeddingService()
    app.extensions["face_index_service"] = FakeFaceIndexService()

    response = client.post(
        f"/api/manager/employees/{employee['id']}/face-enrollment",
        data=_make_enrollment_payload("face"),
        content_type="multipart/form-data",
    )

    assert response.status_code == 400
    assert response.get_json()["status"] == "multiple_faces"
    assert list(Path(app.config["FACES_DIR"]).rglob("*")) == []


def test_face_enrollment_persists_five_samples_and_refreshes_index(app, client):
    manager = _create_manager(app)
    employee = _create_employee(app, employee_code="EMP-305", full_name="Grace Hopper")
    _login_manager(client, manager)

    refresh_calls = {"count": 0}

    class FakeEmbeddingService:
        def extract_embeddings(self, frame_bytes):
            suffix = frame_bytes.decode()
            return [[float(ord(suffix[-1])) / 100.0, 0.2, 0.3]]

    class FakeFaceIndexService:
        def refresh(self):
            refresh_calls["count"] += 1

    app.extensions["embedding_service"] = FakeEmbeddingService()
    app.extensions["face_index_service"] = FakeFaceIndexService()

    response = client.post(
        f"/api/manager/employees/{employee['id']}/face-enrollment",
        data=_make_enrollment_payload("sample"),
        content_type="multipart/form-data",
    )

    assert response.status_code == 201
    payload = response.get_json()
    assert payload["employee"]["employee_code"] == "EMP-305"
    assert payload["face_sample_count"] == 5
    assert [sample["sample_index"] for sample in payload["face_samples"]] == [1, 2, 3, 4, 5]
    assert refresh_calls["count"] == 1

    with app.app_context():
        rows = FaceSample.query.filter_by(employee_id=employee["id"]).order_by(FaceSample.sample_index.asc()).all()
        assert len(rows) == 5
        assert [row.sample_index for row in rows] == [1, 2, 3, 4, 5]
        for row in rows:
            assert Path(row.image_path).exists()


def test_face_enrollment_all_or_nothing_cleans_partial_rows_and_files(app, client):
    manager = _create_manager(app)
    employee = _create_employee(app, employee_code="EMP-306", full_name="Partial Fail")
    _login_manager(client, manager)

    class FakeEmbeddingService:
        def extract_embeddings(self, frame_bytes):
            if frame_bytes == b"face-4":
                return []
            return [[0.1, 0.2, 0.3]]

    class FakeFaceIndexService:
        def refresh(self):
            raise AssertionError("refresh should not be called on failure")

    app.extensions["embedding_service"] = FakeEmbeddingService()
    app.extensions["face_index_service"] = FakeFaceIndexService()

    response = client.post(
        f"/api/manager/employees/{employee['id']}/face-enrollment",
        data=_make_enrollment_payload("face"),
        content_type="multipart/form-data",
    )

    assert response.status_code == 400
    assert response.get_json()["status"] == "no_face"

    with app.app_context():
        assert FaceSample.query.filter_by(employee_id=employee["id"]).count() == 0


def test_face_enrollment_cleans_up_on_unexpected_mid_batch_exception(app, client):
    manager = _create_manager(app)
    employee = _create_employee(app, employee_code="EMP-309", full_name="Boom Mid Batch")
    _login_manager(client, manager)

    class BoomEmbeddingService:
        def __init__(self):
            self.calls = 0

        def extract_embeddings(self, frame_bytes):
            self.calls += 1
            if self.calls == 3:
                raise RuntimeError("boom")
            return [[0.1, 0.2, 0.3]]

    class UnexpectedRefreshService:
        def refresh(self):
            raise AssertionError("refresh should not be called on failure")

    embedding_service = BoomEmbeddingService()
    app.extensions["embedding_service"] = embedding_service
    app.extensions["face_index_service"] = UnexpectedRefreshService()

    with pytest.raises(RuntimeError):
        client.post(
            f"/api/manager/employees/{employee['id']}/face-enrollment",
            data=_make_enrollment_payload("boom"),
            content_type="multipart/form-data",
        )

    with app.app_context():
        assert FaceSample.query.filter_by(employee_id=employee["id"]).count() == 0

    assert list(Path(app.config["FACES_DIR"]).rglob("*")) == []


def test_enrolled_faces_are_immediately_visible_to_guest_recognition(app, client):
    manager = _create_manager(app)
    employee = _create_employee(app, employee_code="EMP-310", full_name="Recognition Ready")
    _login_manager(client, manager)

    class FixedEnrollmentEmbeddingService:
        def extract_embeddings(self, frame_bytes):
            return [[0.1, 0.2, 0.3]]

    class FixedGuestEmbeddingService:
        def extract_embeddings(self, frame_bytes):
            return [[0.1, 0.2, 0.3]]

    app.extensions["embedding_service"] = FixedEnrollmentEmbeddingService()

    enrollment_response = client.post(
        f"/api/manager/employees/{employee['id']}/face-enrollment",
        data=_make_enrollment_payload("ready"),
        content_type="multipart/form-data",
    )

    assert enrollment_response.status_code == 201

    app.extensions["recognition_service"].embedding_service = FixedGuestEmbeddingService()

    guest_response = client.post(
        "/api/guest/checkin",
        data={"frame": (BytesIO(b"guest-frame"), "guest.jpg")},
        content_type="multipart/form-data",
    )

    assert guest_response.status_code == 200
    payload = guest_response.get_json()
    assert payload["status"] == "recognized"
    assert payload["employee_id"] == employee["id"]
    assert payload["employee_code"] == "EMP-310"
    assert payload["full_name"] == "Recognition Ready"


def test_face_sample_replace_requires_authentication(client):
    response = client.put(
        "/api/manager/employees/1/face-samples/1",
        data={"image": (BytesIO(b"face"), "1.jpg")},
        content_type="multipart/form-data",
    )

    assert response.status_code == 401
    assert response.get_json()["status"] == "unauthorized"


def test_face_sample_replace_validates_sample_index(app, client):
    manager = _create_manager(app)
    employee = _create_employee(app, employee_code="EMP-311", full_name="Out Of Range")
    _login_manager(client, manager)

    response = client.put(
        f"/api/manager/employees/{employee['id']}/face-samples/6",
        data={"image": (BytesIO(b"face"), "1.jpg")},
        content_type="multipart/form-data",
    )

    assert response.status_code == 400
    assert response.get_json()["status"] == "invalid_request"


def test_face_sample_replace_updates_existing_slot_and_refreshes_index(app, client):
    manager = _create_manager(app)
    employee = _create_employee(app, employee_code="EMP-312", full_name="Replace Me")
    _login_manager(client, manager)

    refresh_calls = {"count": 0}

    class FakeEmbeddingService:
        def extract_embeddings(self, frame_bytes):
            assert frame_bytes == b"replacement-face"
            return [[0.9, 0.2, 0.3]]

    class FakeFaceIndexService:
        def refresh(self):
            refresh_calls["count"] += 1

    app.extensions["embedding_service"] = FakeEmbeddingService()
    app.extensions["face_index_service"] = FakeFaceIndexService()

    with app.app_context():
        image_path = tmp_path_sample_path(app, employee["id"], 3)
        image_path.parent.mkdir(parents=True, exist_ok=True)
        image_path.write_bytes(b"old-face")
        db.session.add(
            FaceSample(
                employee_id=employee["id"],
                sample_index=3,
                image_path=str(image_path),
                embedding_json="[0.1, 0.2, 0.3]",
            )
        )
        db.session.commit()

    response = client.put(
        f"/api/manager/employees/{employee['id']}/face-samples/3",
        data={"image": (BytesIO(b"replacement-face"), "3.jpg")},
        content_type="multipart/form-data",
    )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["status"] == "updated"
    assert payload["face_sample"]["sample_index"] == 3
    assert payload["face_sample"]["employee_id"] == employee["id"]
    assert refresh_calls["count"] == 1

    with app.app_context():
        sample = FaceSample.query.filter_by(employee_id=employee["id"], sample_index=3).one()
        assert Path(sample.image_path).read_bytes() == b"replacement-face"
        assert sample.embedding_json == "[0.9, 0.2, 0.3]"


def test_face_sample_replace_creates_missing_slot(app, client):
    manager = _create_manager(app)
    employee = _create_employee(app, employee_code="EMP-313", full_name="Create Slot")
    _login_manager(client, manager)

    class FakeEmbeddingService:
        def extract_embeddings(self, frame_bytes):
            return [[0.5, 0.2, 0.3]]

    class FakeFaceIndexService:
        def refresh(self):
            return None

    app.extensions["embedding_service"] = FakeEmbeddingService()
    app.extensions["face_index_service"] = FakeFaceIndexService()

    response = client.put(
        f"/api/manager/employees/{employee['id']}/face-samples/2",
        data={"image": (BytesIO(b"new-face"), "2.jpg")},
        content_type="multipart/form-data",
    )

    assert response.status_code == 200
    assert response.get_json()["face_sample"]["sample_index"] == 2

    with app.app_context():
        sample = FaceSample.query.filter_by(employee_id=employee["id"], sample_index=2).one()
        assert Path(sample.image_path).exists()


def test_face_sample_replace_rejects_no_face_without_mutating_existing_slot(app, client):
    manager = _create_manager(app)
    employee = _create_employee(app, employee_code="EMP-314", full_name="No Face Replace")
    _login_manager(client, manager)

    class FakeEmbeddingService:
        def extract_embeddings(self, frame_bytes):
            return []

    class FakeFaceIndexService:
        def refresh(self):
            raise AssertionError("refresh should not be called on failure")

    app.extensions["embedding_service"] = FakeEmbeddingService()
    app.extensions["face_index_service"] = FakeFaceIndexService()

    with app.app_context():
        image_path = tmp_path_sample_path(app, employee["id"], 4)
        image_path.parent.mkdir(parents=True, exist_ok=True)
        image_path.write_bytes(b"old-face")
        db.session.add(
            FaceSample(
                employee_id=employee["id"],
                sample_index=4,
                image_path=str(image_path),
                embedding_json="[0.1, 0.2, 0.3]",
            )
        )
        db.session.commit()

    response = client.put(
        f"/api/manager/employees/{employee['id']}/face-samples/4",
        data={"image": (BytesIO(b"bad-face"), "4.jpg")},
        content_type="multipart/form-data",
    )

    assert response.status_code == 400
    assert response.get_json()["status"] == "no_face"

    with app.app_context():
        sample = FaceSample.query.filter_by(employee_id=employee["id"], sample_index=4).one()
        assert Path(sample.image_path).read_bytes() == b"old-face"
        assert sample.embedding_json == "[0.1, 0.2, 0.3]"


def test_face_deletion_requires_authentication(client):
    response = client.delete("/api/manager/employees/1/face-samples")

    assert response.status_code == 401
    assert response.get_json()["status"] == "unauthorized"


def test_face_deletion_returns_404_for_missing_employee(app, client):
    manager = _create_manager(app)
    _login_manager(client, manager)

    response = client.delete("/api/manager/employees/999/face-samples")

    assert response.status_code == 404
    assert response.get_json()["status"] == "employee_not_found"


def test_face_deletion_allows_zero_samples_and_refreshes_index(app, client):
    manager = _create_manager(app)
    employee = _create_employee(app, employee_code="EMP-307", full_name="Empty Face")
    _login_manager(client, manager)

    refresh_calls = {"count": 0}

    class FakeFaceIndexService:
        def refresh(self):
            refresh_calls["count"] += 1

    app.extensions["face_index_service"] = FakeFaceIndexService()

    response = client.delete(f"/api/manager/employees/{employee['id']}/face-samples")

    assert response.status_code == 200
    assert response.get_json() == {"employee_id": employee["id"], "deleted_count": 0}
    assert refresh_calls["count"] == 1


def test_face_deletion_removes_all_samples_and_files(app, client):
    manager = _create_manager(app)
    employee = _create_employee(app, employee_code="EMP-308", full_name="Delete Me")
    _login_manager(client, manager)

    refresh_calls = {"count": 0}

    class FakeFaceIndexService:
        def refresh(self):
            refresh_calls["count"] += 1

    app.extensions["face_index_service"] = FakeFaceIndexService()

    with app.app_context():
        rows = []
        for sample_index in range(1, 6):
            image_path = tmp_path_sample_path(app, employee["id"], sample_index)
            image_path.parent.mkdir(parents=True, exist_ok=True)
            image_path.write_bytes(f"sample-{sample_index}".encode())
            row = FaceSample(
                employee_id=employee["id"],
                sample_index=sample_index,
                image_path=str(image_path),
                embedding_json="[0.1, 0.2, 0.3]",
            )
            db.session.add(row)
            rows.append(row)
        db.session.commit()

    response = client.delete(f"/api/manager/employees/{employee['id']}/face-samples")

    assert response.status_code == 200
    assert response.get_json() == {"employee_id": employee["id"], "deleted_count": 5}
    assert refresh_calls["count"] == 1

    with app.app_context():
        assert FaceSample.query.filter_by(employee_id=employee["id"]).count() == 0
        for sample_index in range(1, 6):
            assert tmp_path_sample_path(app, employee["id"], sample_index).exists() is False


def tmp_path_sample_path(app, employee_id, sample_index):
    faces_dir = Path(app.config["FACES_DIR"])
    return faces_dir / f"employee-{employee_id}" / f"sample-{sample_index}.jpg"



def test_face_batch_enrollment_rejects_non_configured_frame_batches(app, client):
    manager = _create_manager(app)
    employee = _create_employee(app, employee_code="EMP-315", full_name="Invalid Batch")
    _login_manager(client, manager)
    invalid_count = max(0, app.config["FACE_BATCH_MIN_FRAMES"] - 1)

    response = client.post(
        f"/api/manager/employees/{employee['id']}/face-enrollment/batch",
        data=_make_batch_enrollment_payload(frame_count=invalid_count),
        content_type="multipart/form-data",
    )

    assert response.status_code == 400
    payload = response.get_json()
    assert payload["status"] == "invalid_request"
    assert payload["frame_count"] == invalid_count


def test_face_batch_enrollment_persists_preview_samples_and_embeddings(app, client):
    manager = _create_manager(app)
    employee = _create_employee(app, employee_code="EMP-316", full_name="Batch Enrollment")
    _login_manager(client, manager)

    refresh_calls = {"count": 0}

    class FakeEmbeddingService:
        def extract_embeddings(self, frame_bytes):
            suffix = int(frame_bytes.decode().split("-")[-1])
            return [[0.11 + (suffix * 0.01), 0.2 + (suffix % 3) * 0.01, 0.3]]

    class FakeFaceIndexService:
        def refresh(self):
            refresh_calls["count"] += 1

    app.extensions["embedding_service"] = FakeEmbeddingService()
    app.extensions["face_index_service"] = FakeFaceIndexService()
    frame_count = app.config["FACE_BATCH_MIN_FRAMES"]

    response = client.post(
        f"/api/manager/employees/{employee['id']}/face-enrollment/batch",
        data=_make_batch_enrollment_payload(frame_count=frame_count),
        content_type="multipart/form-data",
    )

    assert response.status_code == 201
    payload = response.get_json()
    assert payload["status"] == "enrolled_from_batch"
    assert payload["face_sample_count"] == 5
    assert payload["valid_frame_count"] == frame_count
    assert payload["selected_frame_count"] == frame_count
    expected_representatives = min(10, frame_count)
    assert payload["saved_embedding_count"] == 1 + expected_representatives
    assert payload["representative_embedding_count"] == expected_representatives
    assert [sample["sample_index"] for sample in payload["face_samples"]] == [1, 2, 3, 4, 5]
    assert [sample["pose_label"] for sample in payload["face_samples"]] == ["front", "left", "right", "up", "down"]
    assert refresh_calls["count"] == 1

    with app.app_context():
        sample_rows = FaceSample.query.filter_by(employee_id=employee["id"]).order_by(FaceSample.sample_index.asc()).all()
        embedding_rows = FaceEmbedding.query.filter_by(employee_id=employee["id"]).order_by(FaceEmbedding.id.asc()).all()
        assert len(sample_rows) == 5
        assert len(embedding_rows) == 1 + expected_representatives
        assert embedding_rows[0].embedding_role == "mean"
        assert sum(1 for row in embedding_rows if row.embedding_role == "representative") == expected_representatives
        for row in sample_rows:
            assert Path(row.image_path).exists()
