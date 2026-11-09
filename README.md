# Tabular Inference Platform

![CI](https://github.com/itsoumya-d/tabular-inference-platform/actions/workflows/ci.yml/badge.svg)

Reference implementation of a **tabular foundation-model serving platform** —
tabular = rows × columns. Built as portfolio proof for a **Prior Labs
Full Stack Engineer, ML Platform** role: the stack that role ships in
production is **React/TypeScript frontend + FastAPI backend + Postgres**,
and this repo mirrors that split (TypeScript-styled static dashboard,
FastAPI service, Postgres-like metadata boundary).

> **Boundaries (read first):** no live TabPFN weights, no GPU, no real
> foundation model. Inference is a **deterministic synthetic linear-model
> simulation** (feature scaling/encoding → linear scoring). The metadata
> store is **in-memory**, standing in for Postgres behind a narrow boundary
> (`app/store.py`) so it can be swapped for a real Postgres store without
> touching the API surface.

## Role-fit

| Prior Labs ML Platform surface | This repo |
|---|---|
| Tabular foundation models, rows×columns data | CSV/JSON ingest, schema inference (numeric/categorical/datetime), per-row predictions |
| React/TS + FastAPI + Postgres | `frontend/index.html` (TS-styled, no build) served by FastAPI; in-memory store documented as the Postgres boundary |
| Batch inference, job queues, SLOs | Sync + async jobs, batch-size vs latency, p50/p95 SLO guard with shed/scale signals, throughput |
| Reliability | Idempotency-Key, hash-chained audit log, health/readiness, timeouts, quarantine |

## Architecture

```
frontend/index.html ──▶ FastAPI (app/main.py)
                          ├─ ingest   (app/ingest.py)   CSV/JSON parse, schema inference,
                          │                             validation + quarantine, column stats,
                          │                             distribution, missing-value, correlation
                          ├─ inference(app/inference.py) deterministic scaling/encoding → linear scoring,
                          │                             per-row prediction + uncertainty interval
                          ├─ store    (app/store.py)    in-memory datasets/jobs, metrics,
                          │                             hash-chained audit log (Postgres boundary)
                          └─ jobs     sync inline / async thread, timeout, batch chunking
```

Deterministic model: numeric/datetime features are z-scored against the
dataset mean/std; categoricals are ordinal-encoded; each column gets a
fixed MD5-derived weight in `[-1, 1]`. A row equal to the dataset mean
scores exactly `0.0` — the known-value anchor used in tests.

## Quickstart

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload          # http://127.0.0.1:8000
pytest -q                              # 32 tests
```

## API examples

```bash
# Upload a dataset (CSV or JSON)
curl -F file=@data.csv http://127.0.0.1:8000/datasets/upload

# Stats, distribution, missing-value + correlation preview
curl http://127.0.0.1:8000/datasets/<id>/stats

# Filtered query
curl -X POST http://127.0.0.1:8000/datasets/<id>/query \
  -H 'Content-Type: application/json' -d '{"where":{"city":"NYC"},"limit":10}'

# Batch inference (sync), idempotent async submission
curl -X POST http://127.0.0.1:8000/jobs \
  -H 'Content-Type: application/json' -H 'Idempotency-Key: abc-123' \
  -d '{"dataset_id":"<id>","mode":"async","batch_size":128}'
curl http://127.0.0.1:8000/jobs/<job_id>     # pending → running → done

# Latency / SLO: p50/p95, queue depth, error rate, throughput (rows/sec),
# shed-load + scale-up signal when p95 exceeds the 500 ms budget
curl http://127.0.0.1:8000/metrics/slo
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/ready
```

Endpoints: `GET /`, `GET /health`, `GET /ready`, `POST /datasets/upload`,
`GET /datasets/{id}/stats`, `GET|POST /datasets/{id}/query`,
`POST /jobs`, `GET /jobs/{id}`, `GET /jobs`, `GET /metrics/slo`, `GET /audit`.

Failure paths: malformed CSV/JSON → `400`, unsupported extension → `400`,
unknown dataset/job → `404`, oversize batch (`>1000`) or upload (`>5MB`) → `413`,
job timeout (via `simulate_delay_ms > timeout_ms`) → job status `failed`.

## Tests

`pytest -q` — schema inference (numeric/categorical/datetime/mixed),
validation/quarantine, CSV/JSON parsing, column stats, histogram,
correlation, known-value + deterministic batch inference, uncertainty,
sync/async lifecycle, idempotency, oversize/timeout/unknown failures,
SLO guard, audit chain + tamper detection, frontend serving.

## License

MIT
