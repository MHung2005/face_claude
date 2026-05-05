import importlib
import importlib.util
import sys
import types
from datetime import datetime
from io import BytesIO

from sqlalchemy.exc import IntegrityError


def test_guest_checkin_requires_frame_file(client):
    response = client.post("/api/guest/checkin", data={}, content_type="multipart/form-data")

    assert response.status_code == 400
    assert response.get_json()["status"] == "invalid_request"


def test_guest_checkin_returns_payload_from_recognition_service(app, client):
    expected_payload = {
        "status": "recognized",
        "employee_id": 7,
        "employee_code": "EMP-007",
        "full_name": "Ada Lovelace",
    }

    class FakeRecognitionService:
        def __init__(self):
            self.calls = []

        def process_guest_image(self, frame_bytes, filename=None, content_type=None):
            self.calls.append(
                {
                    "frame_bytes": frame_bytes,
                    "filename": filename,
                    "content_type": content_type,
                }
            )
            return expected_payload

    fake_service = FakeRecognitionService()
    app.extensions["recognition_service"] = fake_service

    response = client.post(
        "/api/guest/checkin",
        data={"frame": (BytesIO(b"frame-bytes"), "guest.jpg")},
        content_type="multipart/form-data",
    )

    assert response.status_code == 200
    assert response.get_json() == expected_payload
    assert fake_service.calls == [
        {
            "frame_bytes": b"frame-bytes",
            "filename": "guest.jpg",
            "content_type": "image/jpeg",
        }
    ]


def test_embedding_service_defers_insightface_import_until_extraction(monkeypatch):
    module_name = "backend.app.services.embedding"
    if importlib.util.find_spec(module_name) is None:
        module_name = "app.services.embedding"

    sys.modules.pop(module_name, None)

    attempted_imports = []
    original_import = __import__

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "insightface" or name.startswith("insightface."):
            attempted_imports.append((name, tuple(fromlist or ())))
            raise AssertionError("insightface should be imported lazily")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr("builtins.__import__", guarded_import)

    importlib.import_module(module_name)

    assert attempted_imports == []


def test_embedding_service_uses_yolo_and_insightface(monkeypatch, tmp_path):
    """Verify the YOLO → align_face → InsightFace pipeline.

    We fake both the YOLO model output and InsightFace app so the test
    runs instantly without real model weights or a real face image.
    """
    import cv2
    import numpy as np

    captured = {}

    class FakeInsightFaceRecognizer:
        def get_feat(self, img):
            captured["img_type"] = type(img).__name__
            captured["img_shape"] = img.shape
            return np.array([0.1, 0.2, 0.3], dtype=np.float32)

    # -- Fake YOLO model that returns one detection with box + keypoints --
    class FakeBoxes:
        def __init__(self, img_h, img_w):
            import torch
            self.xyxy = torch.tensor([[5, 5, img_w - 5, img_h - 5]])
            self.conf = torch.tensor([0.95])

        def __len__(self):
            return 1

    class FakeKeypoints:
        def __init__(self, img_h, img_w):
            import torch
            # 5 keypoints: left_eye, right_eye, nose, left_mouth, right_mouth
            self.xy = torch.tensor([[[15, 15], [img_w - 15, 15], [img_w // 2, img_h // 2],
                                     [15, img_h - 10], [img_w - 15, img_h - 10]]], dtype=torch.float32)
            self.conf = torch.tensor([[0.9, 0.9, 0.9, 0.9, 0.9]], dtype=torch.float32)

    class FakeDetectionResult:
        def __init__(self, img_h, img_w):
            self.boxes = FakeBoxes(img_h, img_w)
            self.keypoints = FakeKeypoints(img_h, img_w)

    class FakeYOLO:
        def __init__(self, *args, **kwargs):
            pass

        def predict(self, img, conf=0.5, verbose=False):
            h, w = img.shape[:2]
            return [FakeDetectionResult(h, w)]

    # Create a small valid image (50x50 so crop is non-empty)
    test_img = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
    _, jpeg_bytes = cv2.imencode(".jpg", test_img)
    frame_bytes = jpeg_bytes.tobytes()

    try:
        from backend.app.services import embedding as embedding_mod
    except ModuleNotFoundError:
        from app.services import embedding as embedding_mod

    service = embedding_mod.EmbeddingService()
    # Inject fake YOLO model directly (skip lazy-load)
    service._yolo_model = FakeYOLO()
    service._insightface_recognizer = FakeInsightFaceRecognizer()

    embeddings = service.extract_embeddings(frame_bytes)

    assert embeddings == [[0.1, 0.2, 0.3]]
    assert captured["img_type"] == "ndarray"
    assert captured["img_shape"] == (112, 112, 3)


def test_storage_service_saves_guest_frame_under_dated_subdirectory(tmp_path, monkeypatch):
    try:
        from backend.app.services import storage as storage_module
    except ModuleNotFoundError:
        from app.services import storage as storage_module

    class FixedDateTime:
        @classmethod
        def now(cls):
            return datetime(2026, 4, 2, 9, 30, 0)

    monkeypatch.setattr(storage_module, "datetime", FixedDateTime)
    service = storage_module.StorageService(tmp_path)

    snapshot_path = service.save_guest_frame(b"frame-bytes", filename="guest.JPG")

    assert snapshot_path.parent == tmp_path / "2026-04-02"
    assert snapshot_path.suffix == ".jpg"
    assert snapshot_path.read_bytes() == b"frame-bytes"


def test_face_index_service_uses_cosine_distance_for_thresholding(monkeypatch):
    try:
        from backend.app.services.face_index import FaceIndexService
    except ModuleNotFoundError:
        from app.services.face_index import FaceIndexService

    service = FaceIndexService(threshold=0.01)
    monkeypatch.setattr(
        service,
        "refresh",
        lambda: service._entries.__setitem__(
            slice(None),
            [
                {
                    "employee_id": 7,
                    "employee_code": "EMP-007",
                    "full_name": "Ada Lovelace",
                    "embedding": [1.0, 0.0],
                }
            ],
        ),
    )

    match = service.find_match([10.0, 0.0])

    assert match is not None
    assert match["employee_id"] == 7
    assert match["distance"] == 0.0


def test_face_index_service_refreshes_on_each_find_match(monkeypatch):
    try:
        from backend.app.services.face_index import FaceIndexService
    except ModuleNotFoundError:
        from app.services.face_index import FaceIndexService

    service = FaceIndexService(threshold=0.01)
    refresh_calls = {"count": 0}

    def fake_refresh():
        refresh_calls["count"] += 1
        employee_id = refresh_calls["count"]
        service._entries = [
            {
                "employee_id": employee_id,
                "employee_code": f"EMP-{employee_id:03d}",
                "full_name": f"Employee {employee_id}",
                "embedding": [1.0, 0.0],
            }
        ]

    monkeypatch.setattr(service, "refresh", fake_refresh)

    first_match = service.find_match([1.0, 0.0])
    second_match = service.find_match([1.0, 0.0])

    assert first_match["employee_id"] == 1
    assert second_match["employee_id"] == 2
    assert refresh_calls["count"] == 2


def test_recognition_service_returns_no_face_when_embedding_list_is_empty():
    try:
        from backend.app.services.recognition import RecognitionService
    except ModuleNotFoundError:
        from app.services.recognition import RecognitionService

    class FakeEmbeddingService:
        def extract_embeddings(self, frame_bytes):
            return []

    class UnexpectedCall:
        def __getattr__(self, name):
            raise AssertionError(f"unexpected call: {name}")

    service = RecognitionService(
        storage_service=UnexpectedCall(),
        embedding_service=FakeEmbeddingService(),
        face_index_service=UnexpectedCall(),
        attendance_service=UnexpectedCall(),
    )

    payload = service.process_guest_image(b"frame-bytes", filename="guest.jpg")

    assert payload == {"status": "no_face"}


def test_recognition_service_returns_multiple_faces_when_more_than_one_embedding():
    try:
        from backend.app.services.recognition import RecognitionService
    except ModuleNotFoundError:
        from app.services.recognition import RecognitionService

    class FakeEmbeddingService:
        def extract_embeddings(self, frame_bytes):
            return [[0.1, 0.2], [0.3, 0.4]]

    class UnexpectedCall:
        def __getattr__(self, name):
            raise AssertionError(f"unexpected call: {name}")

    service = RecognitionService(
        storage_service=UnexpectedCall(),
        embedding_service=FakeEmbeddingService(),
        face_index_service=UnexpectedCall(),
        attendance_service=UnexpectedCall(),
    )

    payload = service.process_guest_image(b"frame-bytes", filename="guest.jpg")

    assert payload == {"status": "multiple_faces", "faces_detected": 2}


def test_recognition_service_cleans_orphan_snapshot_and_reuses_existing_event_metadata(tmp_path):
    try:
        from backend.app.services.recognition import RecognitionService
    except ModuleNotFoundError:
        from app.services.recognition import RecognitionService

    existing_snapshot_path = tmp_path / "persisted.jpg"
    existing_snapshot_path.write_bytes(b"persisted")
    orphan_snapshot_path = tmp_path / "orphan.jpg"
    existing_event = types.SimpleNamespace(
        checked_in_at=datetime(2026, 4, 2, 9, 15, 0),
        snapshot_path=str(existing_snapshot_path),
    )

    class FakeStorageService:
        def save_guest_frame(self, frame_bytes, filename=None):
            orphan_snapshot_path.write_bytes(frame_bytes)
            return orphan_snapshot_path

    class FakeEmbeddingService:
        def extract_embeddings(self, frame_bytes):
            return [[0.1, 0.2, 0.3]]

    class FakeFaceIndexService:
        def find_match(self, embedding):
            return {
                "employee_id": 7,
                "employee_code": "EMP-007",
                "full_name": "Ada Lovelace",
                "distance": 0.12,
            }

    class FakeAttendanceService:
        def record_checkin(self, employee_id, snapshot_path, distance=None, checked_in_at=None):
            return existing_event, False

    service = RecognitionService(
        storage_service=FakeStorageService(),
        embedding_service=FakeEmbeddingService(),
        face_index_service=FakeFaceIndexService(),
        attendance_service=FakeAttendanceService(),
    )

    payload = service.process_guest_image(b"new-frame", filename="guest.jpg")

    assert payload["status"] == "already_checked_in"
    assert payload["snapshot_path"] == str(existing_snapshot_path)
    assert payload["checked_in_at"] == existing_event.checked_in_at.isoformat()
    assert orphan_snapshot_path.exists() is False


def test_attendance_service_returns_existing_event_after_integrity_error(monkeypatch):
    try:
        from backend.app.services import attendance as attendance_module
    except ModuleNotFoundError:
        from app.services import attendance as attendance_module

    class FixedDateTime:
        @classmethod
        def now(cls):
            return datetime(2026, 4, 2, 10, 0, 0)

    existing_event = types.SimpleNamespace(
        employee_id=7,
        checkin_date="2026-04-02",
        checked_in_at=datetime(2026, 4, 2, 9, 5, 0),
        snapshot_path="persisted.jpg",
    )
    rollback_state = {"called": False}
    added_events = []
    find_calls = []

    def fake_find_existing_event(employee_id, checkin_date):
        find_calls.append((employee_id, checkin_date))
        if len(find_calls) == 1:
            return None
        return existing_event

    monkeypatch.setattr(attendance_module, "datetime", FixedDateTime)
    monkeypatch.setattr(attendance_module, "_find_existing_event", fake_find_existing_event)
    monkeypatch.setattr(attendance_module.db.session, "add", lambda event: added_events.append(event))
    monkeypatch.setattr(
        attendance_module.db.session,
        "commit",
        lambda: (_ for _ in ()).throw(IntegrityError("insert", {}, Exception("duplicate"))),
    )
    monkeypatch.setattr(
        attendance_module.db.session,
        "rollback",
        lambda: rollback_state.__setitem__("called", True),
    )

    event, created = attendance_module.AttendanceService().record_checkin(
        employee_id=7,
        snapshot_path="new.jpg",
        distance=0.12,
    )

    assert created is False
    assert event is existing_event
    assert rollback_state["called"] is True
    assert added_events[0].checkin_date == "2026-04-02"
    assert find_calls == [
        (7, "2026-04-02"),
        (7, "2026-04-02"),
    ]
