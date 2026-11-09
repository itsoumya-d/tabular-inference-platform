import pytest

from app.ingest import (
    compute_column_stats,
    correlation_preview,
    distribution_for,
    infer_schema,
    parse_csv_bytes,
    parse_json_bytes,
    validate_rows,
)


def test_schema_inference_numeric():
    schema = infer_schema([{"a": "1", "b": 2}, {"a": "3", "b": 4}])
    by_name = {c["name"]: c["type"] for c in schema}
    assert by_name == {"a": "numeric", "b": "numeric"}


def test_schema_inference_categorical():
    schema = infer_schema([{"city": "NYC"}, {"city": "LA"}])
    assert schema == [{"name": "city", "type": "categorical"}]


def test_schema_inference_datetime():
    schema = infer_schema([{"d": "2024-01-01"}, {"d": "2024-02-01T10:00:00"}])
    assert schema[0]["type"] == "datetime"


def test_schema_inference_mixed_becomes_categorical():
    schema = infer_schema([{"x": "1"}, {"x": "hello"}])
    assert schema[0]["type"] == "categorical"


def test_validation_quarantine_reasons():
    valid, quarantined = validate_rows(
        [{"a": 1}, "not-a-dict", {"a": [1, 2]}, {}, {"b": "ok"}]
    )
    assert len(valid) == 2
    reasons = [q["reason"] for q in quarantined]
    assert "row_not_object" in reasons
    assert "unsupported_value_type:a" in reasons
    assert "empty_row" in reasons


def test_parse_csv_bytes():
    rows = parse_csv_bytes(b"a,b\n1,2\n3,4\n")
    assert rows == [{"a": "1", "b": "2"}, {"a": "3", "b": "4"}]


def test_parse_json_bytes_list_and_wrapper():
    assert parse_json_bytes(b'[{"a":1}]') == [{"a": 1}]
    assert parse_json_bytes(b'{"rows":[{"a":2}]}') == [{"a": 2}]


def test_parse_malformed_csv_and_json_raise():
    with pytest.raises(ValueError, match="malformed_csv"):
        parse_csv_bytes(b"")
    with pytest.raises(ValueError, match="malformed_csv"):
        parse_csv_bytes(b"\xff\xfe invalid")
    with pytest.raises(ValueError, match="malformed_json"):
        parse_json_bytes(b"not json")


def test_column_stats_min_max_null_rate():
    rows = [{"a": 1}, {"a": 3}, {"a": None}, {"a": 5}]
    schema = infer_schema(rows)
    stats = compute_column_stats(rows, schema)
    assert stats[0]["count"] == 3
    assert stats[0]["min"] == 1
    assert stats[0]["max"] == 5
    assert stats[0]["null_rate"] == 0.25


def test_distribution_histogram():
    rows = [{"a": i} for i in range(10)]
    schema = infer_schema(rows)
    dist = distribution_for(rows, schema)
    assert sum(dist["a"]["counts"]) == 10
    assert len(dist["a"]["counts"]) == 5


def test_correlation_preview_perfect():
    rows = [{"a": 1, "b": 2}, {"a": 2, "b": 4}, {"a": 3, "b": 6}]
    schema = infer_schema(rows)
    pairs = correlation_preview(rows, schema)
    assert pairs and pairs[0]["correlation"] == pytest.approx(1.0)
