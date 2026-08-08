#!/usr/bin/env python3
"""Shared NDJSON loading and percentile helpers."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable, TextIO


class BenchmarkError(ValueError):
    pass


def _reject_constant(value: str) -> None:
    raise BenchmarkError(f"non-finite JSON constant is not allowed: {value}")


def _finite_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise BenchmarkError(f"non-finite JSON number is not allowed: {value}")
    return parsed


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise BenchmarkError(f"duplicate JSON key is not allowed: {key}")
        result[key] = value
    return result


def strict_json_loads(text: str) -> Any:
    try:
        return json.loads(
            text,
            parse_constant=_reject_constant,
            parse_float=_finite_float,
            object_pairs_hook=_unique_object,
        )
    except json.JSONDecodeError as error:
        raise BenchmarkError(f"invalid JSON: {error.msg}") from error


def load_records(source: TextIO) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(source, 1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            record = strict_json_loads(stripped)
        except BenchmarkError as error:
            raise BenchmarkError(f"line {line_number}: {error}") from error
        if not isinstance(record, dict):
            raise BenchmarkError(f"line {line_number}: record must be an object")
        record["_line"] = line_number
        records.append(record)
    return records


def load_path(path: str) -> list[dict[str, Any]]:
    if path == "-":
        import sys

        return load_records(sys.stdin)
    with Path(path).open(encoding="utf-8") as source:
        return load_records(source)


def clean_record(record: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in record.items() if not key.startswith("_")}


def percentile(values: Iterable[float], percent: int) -> float | None:
    ordered = sorted(values)
    if not ordered:
        return None
    rank = max(1, (percent * len(ordered) + 99) // 100)
    return ordered[rank - 1]
