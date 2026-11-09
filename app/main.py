"""FastAPI serving layer: datasets, inference jobs, SLO metrics, audit, health."""
from __future__ import annotations

import threading
import time
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Header, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from . import __version__
from .ingest import (
    compute_column_stats,
    correlation_preview,
    distribution_for,
    infer_schema,
    parse_csv_bytes,
    parse_json_bytes,
    validate_rows,
)
from .inference import fit_dataset, predict_batch
from .store import MAX_BATCH_SIZE, MAX_UPLOAD_BYTES, SLO_BUDGET_MS, store

app = FastAPI(title="Tabular Inference Platform", version=__version__)


class QueryRequest(BaseModel):
    where: dict[str, Any] | None = None
    limit: int = Field(default=100, ge=1, le=1000)
    offset: int = Field(default=0, ge=0)


class JobRequest(BaseModel):
    dataset_id: str
    mode: str = "sync"
    batch_size: int = 128
    timeout_ms: int = 5000
    simulate_delay_ms: int = 0


FRONTEND_INDEX = Path(__file__).resolve().parent.parent / "frontend" / "index.html"


@app.get("/")
def root():
    if FRONTEND_INDEX.exists():
        return FileResponse(FRONTEND_INDEX, media_type="text/html")
    return JSONResponse({"service": "tabular-inference-platform", "version": __version__})


@app.get("/health")
def health():
    return {"status": "ok", "service": "tabular-inference-platform", "version": __version__}


@app.get("/ready")
def ready():
    return {"status": "ready", "service": "tabular-inference-platform", "version": __version__}


@app.post("/datasets/upload")
async def upload_dataset(file: UploadFile = File(...)):
    filename = file.filename or ""
    suffix = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if suffix not in ("csv", "json"):
        raise HTTPException(status_code=400, detail="unsupported_format: only .csv/.json allowed")
    data = await file.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="oversize_upload")
    try:
        raw_rows = parse_csv_bytes(data) if suffix == "csv" else parse_json_bytes(data)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    valid_rows, quarantined = validate_rows(raw_rows)
    if not valid_rows:
        raise HTTPException(status_code=400, detail="no_valid_rows")
    schema = infer_schema(valid_rows)
    stats = compute_column_stats(valid_rows, schema)
    dataset_id = str(uuid.uuid4())
    dataset = {
        "dataset_id": dataset_id,
        "filename": filename,
        "rows": valid_rows,
        "schema": schema,
        "stats": stats,
        "quarantined": quarantined,
        "row_count": len(valid_rows),
        "created_at": time.time(),
    }
    store.datasets[dataset_id] = dataset
    store.add_audit("dataset_upload", {"dataset_id": dataset_id, "rows": len(valid_rows), "filename": filename})
    return {
        "dataset_id": dataset_id,
        "filename": filename,
        "row_count": len(valid_rows),
        "quarantined_count": len(quarantined),
        "quarantined": quarantined,
        "schema": schema,
    }


@app.get("/datasets/{dataset_id}/stats")
def dataset_stats(dataset_id: str):
    ds = store.datasets.get(dataset_id)
    if ds is None:
        raise HTTPException(status_code=404, detail="unknown_dataset")
    return {
        "dataset_id": dataset_id,
        "row_count": ds["row_count"],
        "columns": ds["stats"],
        "schema": ds["schema"],
        "distribution": distribution_for(ds["rows"], ds["schema"]),
        "missing_value_analysis": [
            {"name": s["name"], "null_count": s["null_count"], "null_rate": s["null_rate"]}
            for s in ds["stats"]
        ],
        "correlation_preview": correlation_preview(ds["rows"], ds["schema"]),
        "quarantined_count": len(ds["quarantined"]),
        "quarantined": ds["quarantined"],
    }


def _apply_query(rows, where, limit, offset):
    filtered = rows
    if where:
        filtered = [r for r in rows if all(str(r.get(k)) == str(v) for k, v in where.items())]
    total = len(filtered)
    return total, filtered[offset : offset + limit]


@app.post("/datasets/{dataset_id}/query")
def dataset_query_post(dataset_id: str, body: QueryRequest):
    ds = store.datasets.get(dataset_id)
    if ds is None:
        raise HTTPException(status_code=404, detail="unknown_dataset")
    total, page = _apply_query(ds["rows"], body.where, body.limit, body.offset)
    return {"dataset_id": dataset_id, "total_matched": total, "limit": body.limit, "offset": body.offset, "rows": page}


@app.get("/datasets/{dataset_id}/query")
def dataset_query_get(
    dataset_id: str,
    where_col: str | None = Query(default=None),
    where_value: str | None = Query(default=None),
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
):
    ds = store.datasets.get(dataset_id)
    if ds is None:
        raise HTTPException(status_code=404, detail="unknown_dataset")
    where = {where_col: where_value} if where_col is not None else None
    total, page = _apply_query(ds["rows"], where, limit, offset)
    return {"dataset_id": dataset_id, "total_matched": total, "limit": limit, "offset": offset, "rows": page}


def _run_job(job: dict[str, Any]) -> None:
    ds = store.datasets.get(job["dataset_id"])
    if ds is None:
        job.update({"status": "failed", "error": "unknown_dataset"})
        store.record_latency(0.0, error=True)
        return
    job["status"] = "running"
    start = time.time()
    delay_ms = int(job.get("simulate_delay_ms") or 0)
    if delay_ms:
        time.sleep(delay_ms / 1000.0)
    elapsed_ms = (time.time() - start) * 1000.0
    timeout_ms = int(job.get("timeout_ms") or 5000)
    if delay_ms > timeout_ms:
        job.update(
            {
                "status": "failed",
                "error": "timeout",
                "latency_ms": round(elapsed_ms, 3),
                "predictions": [],
            }
        )
        store.record_latency(elapsed_ms, error=True)
        store.add_audit("job_failed", {"job_id": job["job_id"], "error": "timeout"})
        return
    schema, rows = ds["schema"], ds["rows"]
    fit = fit_dataset(rows, schema)
    batch_size = int(job.get("batch_size") or 128)
    predictions: list[dict[str, Any]] = []
    batches = 0
    for start_idx in range(0, len(rows), batch_size):
        chunk = rows[start_idx : start_idx + batch_size]
        predictions.extend(predict_batch(chunk, schema, fit, start_index=start_idx))
        batches += 1
    elapsed_ms = (time.time() - start) * 1000.0
    elapsed_sec = max(elapsed_ms / 1000.0, 1e-9)
    job.update(
        {
            "status": "done",
            "predictions": predictions,
            "rows_processed": len(predictions),
            "batches": batches,
            "batch_size": batch_size,
            "latency_ms": round(elapsed_ms, 3),
            "throughput_rows_per_sec": round(len(predictions) / elapsed_sec, 3),
            "error": None,
        }
    )
    store.record_latency(elapsed_ms, rows=len(predictions))
    store.add_audit("job_done", {"job_id": job["job_id"], "rows": len(predictions)})


def _public_job(job: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in job.items()}


@app.post("/jobs")
def submit_job(body: JobRequest, idempotency_key: str | None = Header(default=None, alias="Idempotency-Key")):
    if idempotency_key and idempotency_key in store.idempotency:
        existing = store.jobs.get(store.idempotency[idempotency_key])
        if existing is not None:
            return _public_job(existing)
    if body.dataset_id not in store.datasets:
        raise HTTPException(status_code=404, detail="unknown_dataset")
    if body.batch_size <= 0:
        raise HTTPException(status_code=422, detail="invalid_batch_size")
    if body.batch_size > MAX_BATCH_SIZE:
        raise HTTPException(status_code=413, detail="oversize_batch")
    if body.mode not in ("sync", "async"):
        raise HTTPException(status_code=422, detail="invalid_mode")
    job_id = str(uuid.uuid4())
    job: dict[str, Any] = {
        "job_id": job_id,
        "dataset_id": body.dataset_id,
        "mode": body.mode,
        "batch_size": body.batch_size,
        "timeout_ms": body.timeout_ms,
        "simulate_delay_ms": body.simulate_delay_ms,
        "status": "pending",
        "predictions": [],
        "error": None,
        "created_at": time.time(),
    }
    store.jobs[job_id] = job
    if idempotency_key:
        store.idempotency[idempotency_key] = job_id
    store.add_audit("job_submit", {"job_id": job_id, "dataset_id": body.dataset_id, "mode": body.mode})
    if body.mode == "sync":
        _run_job(job)
    else:
        thread = threading.Thread(target=_run_job, args=(job,), daemon=True)
        thread.start()
    return _public_job(job)


@app.get("/jobs/{job_id}")
def get_job(job_id: str):
    job = store.jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="unknown_job")
    return _public_job(job)


@app.get("/jobs")
def list_jobs():
    return {"jobs": [_public_job(j) for j in store.jobs.values()]}


@app.get("/metrics/slo")
def slo():
    return store.compute_slo(budget_ms=SLO_BUDGET_MS)


@app.get("/audit")
def audit():
    return {"entries": store.audit, "verified": store.verify_audit_chain()}
