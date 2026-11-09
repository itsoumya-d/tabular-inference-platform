"""Deterministic synthetic inference: scaling/encoding -> linear-model scoring simulation.

No live TabPFN weights. A row equal to the dataset mean encodes to 0 and
therefore scores exactly 0.0 (used as a known-value test anchor).
"""
from __future__ import annotations

import hashlib
import math
from typing import Any

from .ingest import is_null, parse_datetime_value, parse_numeric


def deterministic_weight(column: str) -> float:
    digest = hashlib.md5(column.encode()).hexdigest()
    return (int(digest[:2], 16) % 21 - 10) / 10.0


def fit_dataset(rows: list[dict[str, Any]], schema: list[dict[str, str]]) -> dict[str, Any]:
    numeric_stats: dict[str, dict[str, float]] = {}
    categories: dict[str, list[str]] = {}
    for col in schema:
        name, ctype = col["name"], col["type"]
        if ctype == "numeric":
            vals = [parse_numeric(r[name]) for r in rows if not is_null(r.get(name))]
            if vals:
                mean = sum(vals) / len(vals)
                var = sum((v - mean) ** 2 for v in vals) / len(vals)
                numeric_stats[name] = {"mean": mean, "std": math.sqrt(var)}
            else:
                numeric_stats[name] = {"mean": 0.0, "std": 0.0}
        elif ctype == "datetime":
            vals = []
            for r in rows:
                dt = parse_datetime_value(r.get(name))
                if dt is not None:
                    vals.append(dt.timestamp())
            if vals:
                mean = sum(vals) / len(vals)
                var = sum((v - mean) ** 2 for v in vals) / len(vals)
                numeric_stats[name] = {"mean": mean, "std": math.sqrt(var)}
            else:
                numeric_stats[name] = {"mean": 0.0, "std": 0.0}
        else:
            categories[name] = sorted({str(r.get(name)) for r in rows if not is_null(r.get(name))})
    return {"numeric": numeric_stats, "categories": categories}


def encode_row(row: dict[str, Any], schema: list[dict[str, str]], fit: dict[str, Any]) -> list[float]:
    encoded = []
    for col in schema:
        name, ctype = col["name"], col["type"]
        value = row.get(name)
        if ctype == "numeric":
            if is_null(value):
                encoded.append(0.0)
            else:
                st = fit["numeric"].get(name, {"mean": 0.0, "std": 0.0})
                std = st["std"] if st["std"] > 0 else 1.0
                encoded.append((parse_numeric(value) - st["mean"]) / std)
        elif ctype == "datetime":
            dt = parse_datetime_value(value)
            if dt is None:
                encoded.append(0.0)
            else:
                st = fit["numeric"].get(name, {"mean": 0.0, "std": 0.0})
                std = st["std"] if st["std"] > 0 else 1.0
                encoded.append((dt.timestamp() - st["mean"]) / std)
        else:
            cats = fit["categories"].get(name, [])
            if is_null(value) or not cats or str(value) not in cats:
                encoded.append(0.0)
            elif len(cats) == 1:
                encoded.append(0.0)
            else:
                encoded.append(cats.index(str(value)) / (len(cats) - 1) - 0.5)
    return encoded


def score_row(row: dict[str, Any], schema: list[dict[str, str]], fit: dict[str, Any]) -> dict[str, Any]:
    encoded = encode_row(row, schema, fit)
    weights = [deterministic_weight(c["name"]) for c in schema]
    prediction = round(sum(w * e for w, e in zip(weights, encoded)), 6)
    missing = sum(1 for c in schema if is_null(row.get(c["name"])))
    missing_fraction = missing / len(schema) if schema else 0.0
    uncertainty = round(0.15 + 0.10 * missing_fraction, 6)
    return {
        "prediction": prediction,
        "lower": round(prediction - uncertainty, 6),
        "upper": round(prediction + uncertainty, 6),
        "uncertainty": uncertainty,
    }


def predict_batch(
    rows: list[dict[str, Any]], schema: list[dict[str, str]], fit: dict[str, Any], start_index: int = 0
) -> list[dict[str, Any]]:
    out = []
    for i, row in enumerate(rows):
        scored = score_row(row, schema, fit)
        out.append({"row_index": start_index + i, **scored})
    return out
