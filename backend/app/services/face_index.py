import json

from .vector_store import VectorStore


class FaceIndexService:
    def __init__(self, vector_store: VectorStore, threshold=0.6):
        self.threshold = threshold
        self._store = vector_store

    def setup(self) -> None:
        self._store.setup_index()

    def upsert(
        self,
        employee_id: int,
        sample_index: int,
        employee_code: str,
        full_name: str,
        embedding: list[float],
    ) -> None:
        self._store.upsert_face_sample(employee_id, sample_index, employee_code, full_name, embedding)

    def delete_sample(self, employee_id: int, sample_index: int) -> None:
        self._store.delete_face_sample(employee_id, sample_index)

    def delete_employee(self, employee_id: int) -> None:
        self._store.delete_employee_samples(employee_id)

    def find_match(self, embedding: list[float]) -> dict | None:
        return self._store.find_best_match(embedding, threshold=self.threshold)

    def refresh(self) -> None:
        """Reload the entire Redis index from database records.

        Deletes all existing entries, then re-inserts every FaceSample
        currently stored in the database. Call after batch enrollment
        or any operation that commits DB records before Redis is updated.
        """
        # Import here to avoid circular imports at module level
        from ..extensions import db
        from ..models import FaceSample

        from .redis_client import get_redis

        # Delete ALL face:* keys from Redis (wildcard pattern, not a real employee_id)
        r = get_redis()
        all_face_keys = r.keys("face:*")
        if all_face_keys:
            r.delete(*all_face_keys)
        samples = FaceSample.query.all()
        for sample in samples:
            self._store.upsert_face_sample(
                employee_id=sample.employee_id,
                sample_index=sample.sample_index,
                employee_code=sample.employee.employee_code,
                full_name=sample.employee.full_name,
                embedding=list(json.loads(sample.embedding_json)),
            )
