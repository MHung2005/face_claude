import json
import numpy as np

from .vector_store import VectorStore


class FaceIndexService:
    def __init__(self, vector_store: VectorStore, threshold=0.6):
        self.threshold = threshold
        self._store = vector_store

    @staticmethod
    def _normalize(embedding: list[float]) -> list[float]:  # ← thêm method này
        vec = np.array(embedding, dtype=np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.tolist()

    def setup(self) -> None:
        self._store.setup_index()

    def upsert(self, employee_id, sample_index, employee_code, full_name, embedding):
        self._store.upsert_face_sample(
            employee_id, sample_index, employee_code, full_name,
            self._normalize(embedding)  
        )

    def delete_sample(self, employee_id: int, sample_index: int) -> None:
        self._store.delete_face_sample(employee_id, sample_index)

    def delete_employee(self, employee_id: int) -> None:
        self._store.delete_employee_samples(employee_id)

    def find_match(self, embedding: list[float]) -> dict | None:
        return self._store.find_best_match(embedding, threshold=self.threshold)

    def refresh(self) -> None:
        from ..extensions import db
        from ..models import FaceSample
        from .redis_client import get_redis

        r = get_redis()
        all_face_keys = r.keys("face:*")
        if all_face_keys:
            r.delete(*all_face_keys)

        samples = FaceSample.query.all()
        for sample in samples:
            self.upsert(                                   
                employee_id=sample.employee_id,
                sample_index=sample.sample_index,
                employee_code=sample.employee.employee_code,
                full_name=sample.employee.full_name,
                embedding=list(json.loads(sample.embedding_json)),
            )
