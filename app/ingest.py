"""Dataset ingest: CSV/JSON parsing, schema inference, validation/quarantine, stats."""
from __future__ import annotations

import csv
import io
import json
import math
from datetime import datetime
from typing import Any


def is_null(value: Any) -> bool:
    return value is None or (isinstance(value, str) and value.strip() == "")


def is_numeric(value: Any) -> bool:
    if isinstance(value, bool) or value is None:
        return False
    if isinstance(value, (int, float)):
        return True
    if isinstance(value, str):
        try:
            float(value.strip())
            return True
        except ValueError:
            return False
    return False


def parse_numeric(value: Any) -> float:
    return float(value)


def parse_datetime_value(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value
    if not isinstance(value, str) or is_null(value):
        return None
    text = value.strip()
    candidate = text[:-1] + "+00:00" if text.endswith("Z") else text
    try:
        return datetime.fromisoformat(candidate)
    except ValueError:
        return None


def is_datetime(value: Any) -> bool:
    return parse_datetime_value(value) is not None


def infer_schema(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    schema = []
    for col in columns:
        values = [r.get(col) for r in rows if not is_null(r.get(col))]
        if values and all(is_numeric(v) for v in values):
            col_type = "numeric"
        elif values and all(is_datetime(v) for v in values):
            col_type = "datetime"
        else:
            col_type = "categorical"
        schema.append({"name": col, "type": col_type})
    return schema


def validate_rows(raw_rows: list[Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    valid: list[dict[str, Any]] = []
    quarantined: list[dict[str, Any]] = []
    for idx, row in enumerate(raw_rows):
        if not isinstance(row, dict):
            quarantined.append({"index": idx, "reason": "row_not_object", "row": str(row)[:200]})
            continue
        if None in row:
            quarantined.append({"index": idx, "reason": "extra_columns", "row": str(row)[:200]})
            continue
        if len(row) == 0:
            quarantined.append({"index": idx, "reason": "empty_row", "row": "{}"})
            continue
        bad = next((k for k, v in row.items() if isinstance(v, (dict, list))), None)
        if bad is not None:
            quarantined.append(
                {"index": idx, "reason": f"unsupported_value_type:{bad}", "row": str(row)[:200]}
            )
            continue
        valid.append(row)
    return valid, quarantined


def parse_csv_bytes(data: bytes) -> list[Any]:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"malformed_csv: invalid_utf8: {exc}") from exc
    if not text.strip():
        raise ValueError("malformed_csv: empty_file")
    reader = csv.DictReader(io.StringIO(text))
    if not reader.fieldnames or all(f is None or str(f).strip() == "" for f in reader.fieldnames):
        raise ValueError("malformed_csv: missing_header")
    return list(reader)


def parse_json_bytes(data: bytes) -> list[Any]:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"malformed_json: invalid_utf8: {exc}") from exc
    if not text.strip():
        raise ValueError("malformed_json: empty_file")
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"malformed_json: {exc}") from exc
    if isinstance(parsed, dict) and "rows" in parsed:
        if not isinstance(parsed["rows"], list):
            raise ValueError("malformed_json: rows_not_list")
        return parsed["rows"]
    if isinstance(parsed, list):
        return parsed
    raise ValueError("malformed_json: expected_list_or_rows_object")


def _pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den_x = sum((x - mx) ** 2 for x in xs)
    den_y = sum((y - my) ** 2 for y in ys)
    den = math.sqrt(den_x * den_y)
    return round(num / den, 6) if den else 0.0


def compute_column_stats(rows: list[dict[str, Any]], schema: list[dict[str, str]]) -> list[dict[str, Any]]:
    stats = []
    total = len(rows)
    for col in schema:
        name, ctype = col["name"], col["type"]
        raw = [r.get(name) for r in rows]
        non_null = [v for v in raw if not is_null(v)]
        null_count = total - len(non_null)
        entry: dict[str, Any] = {
            "name": name,
            "type": ctype,
            "count": len(non_null),
            "null_count": null_count,
            "null_rate": round(null_count / total, 6) if total else 0.0,
            "distinct_count": len({str(v) for v in non_null}),
        }
        if ctype == "numeric" and non_null:
            nums = [parse_numeric(v) for v in non_null]
            entry.update({"min": min(nums), "max": max(nums), "mean": round(sum(nums) / len(nums), 6)})
        elif ctype == "datetime" and non_null:
            dts = [parse_datetime_value(v) for v in non_null]
            entry.update({"min": min(dts).isoformat(), "max": max(dts).isoformat()})
        else:
            entry.update({"min": None, "max": None})
        stats.append(entry)
    return stats


def distribution_for(rows: list[dict[str, Any]], schema: list[dict[str, str]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for col in schema:
        if col["type"] != "numeric":
            continue
        nums = [parse_numeric(r[col["name"]]) for r in rows if not is_null(r.get(col["name"]))]
        if not nums:
            out[col["name"]] = {"bins": [], "counts": []}
            continue
        lo, hi = min(nums), max(nums)
        if lo == hi:
            out[col["name"]] = {"bins": [lo, hi], "counts": [len(nums)]}
            continue
        bins = [lo + (hi - lo) * i / 5 for i in range(6)]
        counts = [0] * 5
        for v in nums:
            idx = min(int((v - lo) / (hi - lo) * 5) if hi != lo else 0, 4)
            counts[idx] += 1
        out[col["name"]] = {"bins": [round(b, 6) for b in bins], "counts": counts}
    return out


def correlation_preview(rows: list[dict[str, Any]], schema: list[dict[str, str]]) -> list[dict[str, Any]]:
    numeric_cols = [c["name"] for c in schema if c["type"] == "numeric"][:5]
    pairs = []
    for i in range(len(numeric_cols)):
        for j in range(i + 1, len(numeric_cols)):
            a, b = numeric_cols[i], numeric_cols[j]
            xs, ys = [], []
            for r in rows:
                if not is_null(r.get(a)) and not is_null(r.get(b)):
                    xs.append(parse_numeric(r[a]))
                    ys.append(parse_numeric(r[b]))
            pairs.append({"a": a, "b": b, "correlation": _pearson(xs, ys)})
    return pairs
