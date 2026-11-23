"""Dynamic inference batching, memory chunking, and prediction caching for tabular serving.

Optimizes high-throughput tabular foundation model inference by coalescing micro-requests,
enforcing maximum partition memory bounds on large table inputs, and caching identical
row prediction digests.
"""
from __future__ import annotations

import hashlib
import json
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple


@dataclass
class BatchInferenceResult:
    predictions: List[Dict[str, Any]]
    batch_size: int
    duration_ms: float
    cache_hits: int
    chunks_processed: int


class TabularInferenceCache:
    """LRU prediction cache keyed by dataset_id + canonical row content digest."""

    def __init__(self, capacity: int = 5000):
        self.capacity = capacity
        self._cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()

    @staticmethod
    def compute_key(dataset_id: str, row: Dict[str, Any]) -> str:
        serialized = json.dumps(row, sort_keys=True, default=str)
        digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]
        return f"{dataset_id}:{digest}"

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        return None

    def put(self, key: str, value: Dict[str, Any]) -> None:
        if key in self._cache:
            self._cache.move_to_end(key)
        self._cache[key] = value
        if len(self._cache) > self.capacity:
            self._cache.popitem(last=False)

    def size(self) -> int:
        return len(self._cache)

    def clear(self) -> None:
        self._cache.clear()


class DynamicInferenceBatcher:
    """Dynamic batching manager that chunks large tables and executes inference with caching."""

    def __init__(
        self,
        max_batch_size: int = 64,
        max_chunk_size: int = 256,
        cache_capacity: int = 5000,
    ):
        self.max_batch_size = max_batch_size
        self.max_chunk_size = max_chunk_size
        self.cache = TabularInferenceCache(capacity=cache_capacity)

    def execute_batched_inference(
        self,
        dataset_id: str,
        rows: List[Dict[str, Any]],
        schema: List[Dict[str, str]],
        fit: Dict[str, Any],
        inference_fn: Callable[[List[Dict[str, Any]], List[Dict[str, str]], Dict[str, Any]], List[Dict[str, Any]]],
    ) -> BatchInferenceResult:
        t0 = time.perf_counter()
        if not rows:
            return BatchInferenceResult(
                predictions=[],
                batch_size=0,
                duration_ms=0.0,
                cache_hits=0,
                chunks_processed=0,
            )

        results: List[Optional[Dict[str, Any]]] = [None] * len(rows)
        uncached_indices: List[int] = []
        uncached_rows: List[Dict[str, Any]] = []
        cache_hits = 0

        # Step 1: Check cache for existing prediction results
        for idx, row in enumerate(rows):
            cache_key = self.cache.compute_key(dataset_id, row)
            cached_val = self.cache.get(cache_key)
            if cached_val is not None:
                results[idx] = cached_val
                cache_hits += 1
            else:
                uncached_indices.append(idx)
                uncached_rows.append(row)

        chunks_processed = 0

        # Step 2: Chunk remaining uncached rows to respect GPU/CPU memory boundaries
        if uncached_rows:
            for start_idx in range(0, len(uncached_rows), self.max_chunk_size):
                chunk = uncached_rows[start_idx : start_idx + self.max_chunk_size]
                chunk_indices = uncached_indices[start_idx : start_idx + self.max_chunk_size]

                chunk_preds = inference_fn(chunk, schema, fit)
                chunks_processed += 1

                for row_idx, pred in zip(chunk_indices, chunk_preds):
                    results[row_idx] = pred
                    c_key = self.cache.compute_key(dataset_id, rows[row_idx])
                    self.cache.put(c_key, pred)

        duration_ms = (time.perf_counter() - t0) * 1000.0

        return BatchInferenceResult(
            predictions=[r for r in results if r is not None],
            batch_size=len(rows),
            duration_ms=round(duration_ms, 3),
            cache_hits=cache_hits,
            chunks_processed=chunks_processed,
        )
