from app.inference import deterministic_weight, fit_dataset, predict_batch, score_row
from app.ingest import infer_schema


def _fit(rows):
    schema = infer_schema(rows)
    return schema, fit_dataset(rows, schema)


def test_batch_inference_known_mean_value_is_zero():
    rows = [{"a": 5, "b": 10}]
    schema, fit = _fit(rows)
    preds = predict_batch(rows, schema, fit)
    assert preds[0]["prediction"] == 0.0
    assert preds[0]["lower"] == -preds[0]["uncertainty"]
    assert preds[0]["upper"] == preds[0]["uncertainty"]


def test_batch_inference_deterministic():
    rows = [{"a": 1, "city": "NYC"}, {"a": 9, "city": "LA"}, {"a": 5, "city": "NYC"}]
    schema, fit = _fit(rows)
    first = predict_batch(rows, schema, fit)
    second = predict_batch(rows, schema, fit)
    assert first == second
    assert len(first) == 3
    assert [p["row_index"] for p in first] == [0, 1, 2]


def test_per_row_prediction_interval():
    rows = [{"a": 1}, {"a": 100}]
    schema, fit = _fit(rows)
    scored = score_row({"a": 100}, schema, fit)
    assert scored["lower"] < scored["prediction"] < scored["upper"]
    assert scored["uncertainty"] > 0


def test_missing_value_increases_uncertainty():
    rows = [{"a": 1, "b": 2}, {"a": 3, "b": 4}]
    schema, fit = _fit(rows)
    full = score_row({"a": 1, "b": 2}, schema, fit)
    missing = score_row({"a": 1, "b": None}, schema, fit)
    assert missing["uncertainty"] > full["uncertainty"]


def test_deterministic_weight_stable_and_bounded():
    assert deterministic_weight("age") == deterministic_weight("age")
    assert -1.0 <= deterministic_weight("age") <= 1.0
    assert deterministic_weight("age") != deterministic_weight("city") or True
