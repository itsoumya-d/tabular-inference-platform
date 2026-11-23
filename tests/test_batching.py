import pytest
from app.batching import DynamicInferenceBatcher, TabularInferenceCache
from app.inference import fit_dataset, predict_batch


@pytest.fixture
def sample_dataset():
    schema = [
        {"name": "age", "type": "numeric"},
        {"name": "balance", "type": "numeric"},
        {"name": "country", "type": "categorical"},
    ]
    rows = [
        {"age": 25, "balance": 1000.0, "country": "DE"},
        {"age": 35, "balance": 2500.0, "country": "US"},
        {"age": 45, "balance": 5000.0, "country": "UK"},
        {"age": 55, "balance": 7500.0, "country": "DE"},
    ]
    fit = fit_dataset(rows, schema)
    return schema, rows, fit


def test_lru_cache_capacity_and_eviction():
    cache = TabularInferenceCache(capacity=2)
    cache.put("k1", {"score": 1.0})
    cache.put("k2", {"score": 2.0})
    assert cache.size() == 2
    assert cache.get("k1") == {"score": 1.0}

    # Add k3, k2 should be evicted because k1 was accessed
    cache.put("k3", {"score": 3.0})
    assert cache.size() == 2
    assert cache.get("k1") is not None
    assert cache.get("k2") is None
    assert cache.get("k3") == {"score": 3.0}


def test_dynamic_inference_batching_and_chunking(sample_dataset):
    schema, rows, fit = sample_dataset
    # Test batcher with max_chunk_size=2
    batcher = DynamicInferenceBatcher(max_batch_size=4, max_chunk_size=2, cache_capacity=100)

    res1 = batcher.execute_batched_inference(
        dataset_id="ds_test_01",
        rows=rows,
        schema=schema,
        fit=fit,
        inference_fn=predict_batch,
    )

    assert res1.batch_size == 4
    assert len(res1.predictions) == 4
    assert res1.cache_hits == 0
    assert res1.chunks_processed == 2  # 4 rows in chunks of 2 -> 2 chunks


def test_dynamic_inference_cache_hits_on_repeated_requests(sample_dataset):
    schema, rows, fit = sample_dataset
    batcher = DynamicInferenceBatcher(max_batch_size=4, max_chunk_size=4, cache_capacity=100)

    # First pass: cold cache
    res1 = batcher.execute_batched_inference(
        dataset_id="ds_test_02",
        rows=rows,
        schema=schema,
        fit=fit,
        inference_fn=predict_batch,
    )
    assert res1.cache_hits == 0
    assert res1.chunks_processed == 1

    # Second pass: identical rows should hit cache 100%
    res2 = batcher.execute_batched_inference(
        dataset_id="ds_test_02",
        rows=rows,
        schema=schema,
        fit=fit,
        inference_fn=predict_batch,
    )
    assert res2.cache_hits == 4
    assert res2.chunks_processed == 0
    assert res2.predictions == res1.predictions


def test_empty_rows_returns_empty_batch():
    batcher = DynamicInferenceBatcher()
    res = batcher.execute_batched_inference(
        dataset_id="ds_empty",
        rows=[],
        schema=[],
        fit={},
        inference_fn=lambda r, s, f: [],
    )
    assert res.batch_size == 0
    assert res.predictions == []
    assert res.cache_hits == 0
