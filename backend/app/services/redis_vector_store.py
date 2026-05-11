import logging

import numpy as np
from redis.commands.search.field import NumericField, TextField, VectorField
try:
    from redis.commands.search.index_definition import IndexDefinition, IndexType
except ImportError:
    from redis.commands.search.indexDefinition import IndexDefinition, IndexType
from redis.commands.search.query import Query

from .redis_client import get_redis
from .vector_store import VectorStore

logger = logging.getLogger(__name__)

_INDEX_NAME = "idx:faces"
_KEY_PREFIX = "face:"
_VECTOR_DIM = 512


def _face_key(employee_id: int, sample_index: int) -> str:
    return f"{_KEY_PREFIX}{employee_id}:{sample_index}"


class RedisVectorStore(VectorStore):
    """VectorStore backed by RediSearch VSS (FLAT index, COSINE distance)."""

    def setup_index(self) -> None:
        r = get_redis()
        try:
            r.ft(_INDEX_NAME).info()
            logger.info("RediSearch index '%s' already exists.", _INDEX_NAME)
        except Exception:
            schema = (
                NumericField("employee_id"),
                TextField("employee_code"),
                TextField("full_name"),
                NumericField("sample_index"),
                VectorField(
                    "embedding",
                    "FLAT",
                    {
                        "TYPE": "FLOAT32",
                        "DIM": _VECTOR_DIM,
                        "DISTANCE_METRIC": "COSINE",
                    },
                ),
            )
            r.ft(_INDEX_NAME).create_index(
                schema,
                definition=IndexDefinition(prefix=[_KEY_PREFIX], index_type=IndexType.HASH),
            )
            logger.info("Created RediSearch index '%s'.", _INDEX_NAME)

    def upsert_face_sample(
        self,
        employee_id: int,
        sample_index: int,
        employee_code: str,
        full_name: str,
        embedding: list[float],
    ) -> None:
        r = get_redis()
        vec_bytes = np.array(embedding, dtype=np.float32).tobytes()
        r.hset(
            _face_key(employee_id, sample_index),
            mapping={
                "employee_id": employee_id,
                "sample_index": sample_index,
                "employee_code": employee_code,
                "full_name": full_name,
                "embedding": vec_bytes,
            },
        )

    def delete_face_sample(self, employee_id: int, sample_index: int) -> None:
        get_redis().delete(_face_key(employee_id, sample_index))

    def delete_employee_samples(self, employee_id: int) -> None:
        r = get_redis()
        pattern = f"{_KEY_PREFIX}{employee_id}:*"
        # Use SCAN instead of KEYS to avoid O(N) blocking scan over all Redis keys.
        # SCAN is cursor-based and returns results incrementally without blocking.
        cursor = 0
        while True:
            cursor, keys = r.scan(cursor=cursor, match=pattern, count=100)
            if keys:
                r.delete(*keys)
            if cursor == 0:
                break

    # def find_best_match(
    #     self, embedding: list[float], threshold: float = 0.6
    # ) -> dict | None:
    #     import time
    #     t0 = time.perf_counter()
    #     r = get_redis()
    #     query_vec = np.array(embedding, dtype=np.float32).tobytes()

    #     q = (
    #         Query("*=>[KNN 1 @embedding $vec AS score]")
    #         .sort_by("score")
    #         .return_fields("employee_id", "employee_code", "full_name", "score")
    #         .paging(0, 1)
    #         .dialect(2)
    #     )

    #     try:
    #         results = r.ft(_INDEX_NAME).search(q, query_params={"vec": query_vec})
    #         t1 = time.perf_counter()
    #         logger.info("[TIMING] Redis KNN search: %.1fms", (t1 - t0) * 1000)
    #     except Exception:
    #         logger.exception("RediSearch KNN query failed.")
    #         return None

    #     if not results.docs:
    #         return None

    #     best = results.docs[0]
    #     # RediSearch COSINE distance is in range [0, 2]; score=0 means identical
    #     distance = float(best.score)
    #     if distance > threshold:
    #         return None

    #     return {
    #         "employee_id": int(best.employee_id),
    #         "employee_code": best.employee_code,
    #         "full_name": best.full_name,
    #         "distance": distance,
    #     }
    def find_best_match(
        self, embedding: list[float], threshold: float = 0.45, min_gap: float = 0.1
    ) -> dict | None:
        import time
        t0 = time.perf_counter()
        r = get_redis()
        query_vec = np.array(embedding, dtype=np.float32).tobytes()

        q = (
            Query("*=>[KNN 2 @embedding $vec AS score]")  # lấy top 2
            .sort_by("score")
            .return_fields("employee_id", "employee_code", "full_name", "score")
            .paging(0, 2)
            .dialect(2)
        )

        try:
            results = r.ft(_INDEX_NAME).search(q, query_params={"vec": query_vec})
            t1 = time.perf_counter()
            logger.info("[TIMING] Redis KNN search: %.1fms", (t1 - t0) * 1000)
        except Exception:
            logger.exception("RediSearch KNN query failed.")
            return None

        if not results.docs:
            return None

        best = results.docs[0]
        distance_top1 = float(best.score)

        # Lọc theo threshold
        if distance_top1 > threshold:
            return None

        # Kiểm tra gap giữa top1 và top2
        if len(results.docs) >= 2:
            distance_top2 = float(results.docs[1].score)
            gap = distance_top2 - distance_top1
            if gap < min_gap:
                logger.info(
                    "[MATCH] Rejected: gap %.4f < min_gap %.4f (top1=%.4f, top2=%.4f)",
                    gap, min_gap, distance_top1, distance_top2,
                )
                return None

        return {
            "employee_id": int(best.employee_id),
            "employee_code": best.employee_code,
            "full_name": best.full_name,
            "distance": distance_top1,
        }
