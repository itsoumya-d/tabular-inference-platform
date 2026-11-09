"""In-memory metadata store (Postgres-like boundary), metrics, hash-chained audit log."""
from __future__ import annotations

import hashlib
import json
import math
import threading
import time
from datetime import datetime, timezone
from typing import Any

SLO_BUDGET_MS = 500.0
MAX_BATCH_SIZE = 1000
MAX_UPLOAD_BYTES = 5_000_000
GENESIS_HASH = "0" * 64


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def percentile(samples: list[float], pct: float) -> float:
    if not samples:
        return 0.0
    ordered = sorted(samples)
    if len(ordered) == 1:
        return float(ordered[0])
    rank = math.ceil(pct / 100.0 * len(ordered)) - 1
    rank = max(0, min(rank, len(ordered) - 1))
    return float(ordered[rank])


class Store:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.datasets: dict[str, dict[str, Any]] = {}
        self.jobs: dict[str, dict[str, Any]] = {}
        self.idempotency: dict[str, str] = {}
        self.audit: list[dict[str, Any]] = []
        self.latency_samples: list[float] = []
        self.error_count = 0
        self.request_count = 0
        self.total_rows_processed = 0
        self.total_latency_sec = 0.0

    def clear(self) -> None:
        with self._lock:
            self.datasets.clear()
            self.jobs.clear()
            self.idempotency.clear()
            self.audit.clear()
            self.latency_samples.clear()
            self.error_count = 0
            self.request_count = 0
            self.total_rows_processed = 0
            self.total_latency_sec = 0.0

    # ---- audit (hash-chained) ----
    def add_audit(self, action: str, payload: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            seq = len(self.audit) + 1
            prev_hash = self.audit[-1]["hash"] if self.audit else GENESIS_HASH
            ts = _now_iso()
            payload_str = json.dumps(payload, sort_keys=True, default=str)
            digest = hashlib.sha256(
                (prev_hash + payload_str + str(seq) + ts).encode()
            ).hexdigest()
            entry = {
                "seq": seq,
                "ts": ts,
                "action": action,
                "payload": payload,
                "prev_hash": prev_hash,
                "hash": digest,
            }
            self.audit.append(entry)
            return entry

    def verify_audit_chain(self) -> bool:
        prev = GENESIS_HASH
        for entry in self.audit:
            if entry["prev_hash"] != prev:
                return False
            payload_str = json.dumps(entry["payload"], sort_keys=True, default=str)
            digest = hashlib.sha256(
                (prev + payload_str + str(entry["seq"]) + entry["ts"]).encode()
            ).hexdigest()
            if digest != entry["hash"]:
                return False
            prev = entry["hash"]
        return True

    # ---- metrics ----
    def record_latency(self, latency_ms: float, rows: int = 0, error: bool = False) -> None:
        with self._lock:
            self.request_count += 1
            self.latency_samples.append(float(latency_ms))
            if rows:
                self.total_rows_processed += rows
                self.total_latency_sec += max(latency_ms, 0.0) / 1000.0
            if error:
                self.error_count += 1

    def queue_depth(self) -> int:
        return sum(1 for j in self.jobs.values() if j.get("status") in ("pending", "running"))

    def compute_slo(self, budget_ms: float = SLO_BUDGET_MS) -> dict[str, Any]:
        samples = list(self.latency_samples)
        p50 = percentile(samples, 50)
        p95 = percentile(samples, 95)
        queue = self.queue_depth()
        total = self.request_count
        error_rate = (self.error_count / total) if total else 0.0
        throughput = (
            self.total_rows_processed / self.total_latency_sec
            if self.total_latency_sec > 0
            else 0.0
        )
        breaching = p95 > budget_ms or queue > 10
        return {
            "p50_ms": round(p50, 3),
            "p95_ms": round(p95, 3),
            "slo_budget_ms": budget_ms,
            "slo_status": "breaching" if breaching else "ok",
            "shed_load": bool(breaching),
            "scale_signal": "scale_up" if breaching else "normal",
            "queue_depth": queue,
            "error_rate": round(error_rate, 6),
            "error_count": self.error_count,
            "total_requests": total,
            "throughput_rows_per_sec": round(throughput, 3),
            "total_rows_processed": self.total_rows_processed,
            "latency_samples": len(samples),
        }


store = Store()
