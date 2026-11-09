import time

from app.store import store


def test_health_and_ready(client):
    assert client.get("/health").json()["status"] == "ok"
    assert client.get("/ready").json()["status"] == "ready"


def test_upload_csv_then_stats(client, csv_dataset):
    did = csv_dataset["dataset_id"]
    assert csv_dataset["row_count"] == 3
    types = {c["name"]: c["type"] for c in csv_dataset["schema"]}
    assert types["age"] == "numeric"
    assert types["city"] == "categorical"
    assert types["joined"] == "datetime"
    stats = client.get(f"/datasets/{did}/stats").json()
    assert stats["row_count"] == 3
    assert "distribution" in stats and "correlation_preview" in stats
    assert "missing_value_analysis" in stats


def test_query_filter_post_and_get(client, csv_dataset):
    did = csv_dataset["dataset_id"]
    posted = client.post(f"/datasets/{did}/query", json={"where": {"city": "NYC"}}).json()
    assert posted["total_matched"] == 2
    got = client.get(f"/datasets/{did}/query", params={"where_col": "city", "where_value": "LA"}).json()
    assert got["total_matched"] == 1


def test_upload_json_quarantine(client):
    payload = b'[{"a": 1}, "bad", {"a": [1]}]'
    r = client.post("/datasets/upload", files={"file": ("x.json", payload, "application/json")})
    assert r.status_code == 200
    body = r.json()
    assert body["row_count"] == 1
    assert body["quarantined_count"] == 2


def test_malformed_and_unsupported_uploads(client):
    bad = client.post("/datasets/upload", files={"file": ("x.csv", b"", "text/csv")})
    assert bad.status_code == 400
    bad_json = client.post("/datasets/upload", files={"file": ("x.json", b"not json", "application/json")})
    assert bad_json.status_code == 400
    unsupported = client.post("/datasets/upload", files={"file": ("x.txt", b"hi", "text/plain")})
    assert unsupported.status_code == 400


def test_unknown_dataset_404(client):
    assert client.get("/datasets/nope/stats").status_code == 404
    assert client.post("/datasets/nope/query", json={}).status_code == 404
    r = client.post("/jobs", json={"dataset_id": "nope", "mode": "sync"})
    assert r.status_code == 404


def test_sync_job_lifecycle(client, csv_dataset):
    job = client.post("/jobs", json={"dataset_id": csv_dataset["dataset_id"], "mode": "sync", "batch_size": 2}).json()
    assert job["status"] == "done"
    assert len(job["predictions"]) == 3
    assert job["batches"] == 2
    assert job["throughput_rows_per_sec"] >= 0
    fetched = client.get(f"/jobs/{job['job_id']}").json()
    assert fetched["job_id"] == job["job_id"]


def test_async_job_lifecycle(client, json_dataset):
    job = client.post("/jobs", json={"dataset_id": json_dataset["dataset_id"], "mode": "async"}).json()
    assert job["status"] in ("pending", "running", "done")
    final = job
    for _ in range(50):
        final = client.get(f"/jobs/{job['job_id']}").json()
        if final["status"] == "done":
            break
        time.sleep(0.05)
    assert final["status"] == "done"
    assert len(final["predictions"]) == 2


def test_idempotent_job_submission(client, csv_dataset):
    headers = {"Idempotency-Key": "key-123"}
    payload = {"dataset_id": csv_dataset["dataset_id"], "mode": "sync"}
    first = client.post("/jobs", json=payload, headers=headers).json()
    second = client.post("/jobs", json=payload, headers=headers).json()
    assert first["job_id"] == second["job_id"]


def test_oversize_batch_rejected(client, csv_dataset):
    r = client.post("/jobs", json={"dataset_id": csv_dataset["dataset_id"], "mode": "sync", "batch_size": 9999})
    assert r.status_code == 413


def test_timeout_job_fails(client, csv_dataset):
    job = client.post(
        "/jobs",
        json={"dataset_id": csv_dataset["dataset_id"], "mode": "sync", "timeout_ms": 1, "simulate_delay_ms": 50},
    ).json()
    assert job["status"] == "failed"
    assert job["error"] == "timeout"


def test_latency_slo_guard_logic():
    store.record_latency(1000.0, rows=10)
    store.record_latency(1200.0, rows=10)
    slo = store.compute_slo(budget_ms=500.0)
    assert slo["p95_ms"] > 500.0
    assert slo["shed_load"] is True
    assert slo["scale_signal"] == "scale_up"
    assert slo["slo_status"] == "breaching"
    assert slo["throughput_rows_per_sec"] > 0


def test_slo_endpoint_fields(client, csv_dataset):
    client.post("/jobs", json={"dataset_id": csv_dataset["dataset_id"], "mode": "sync"})
    slo = client.get("/metrics/slo").json()
    for key in ("p50_ms", "p95_ms", "queue_depth", "error_rate", "throughput_rows_per_sec", "scale_signal"):
        assert key in slo


def test_audit_chain_and_tamper_detection(client, csv_dataset):
    assert store.verify_audit_chain() is True
    assert len(store.audit) >= 1
    audit = client.get("/audit").json()
    assert audit["verified"] is True
    store.audit[0]["payload"] = {"tampered": True}
    assert store.verify_audit_chain() is False


def test_unknown_job_404(client):
    assert client.get("/jobs/nope").status_code == 404


def test_frontend_served(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "Tabular Inference Platform" in r.text
