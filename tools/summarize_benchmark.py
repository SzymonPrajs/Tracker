#!/usr/bin/env python3
"""Derive an honest summary from raw Tracker benchmark frame records."""

from __future__ import annotations

import argparse
import bisect
import json
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from benchmark_common import BenchmarkError, load_path, percentile


def latency(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    return {
        "min": min(values),
        "mean": statistics.fmean(values),
        "p50": percentile(values, 50),
        "p95": percentile(values, 95),
        "p99": percentile(values, 99),
        "max": max(values),
    }


def measurement_window(
    records: list[dict[str, Any]], frames: list[dict[str, Any]]
) -> tuple[float, str, int | None, int | None]:
    summaries = [record for record in records if record.get("type") == "summary"]
    verified = []
    for record in summaries:
        start = record.get("measurement_start_us")
        end = record.get("measurement_end_us")
        duration = record.get("measurement_duration_us")
        if (
            isinstance(start, int)
            and isinstance(end, int)
            and isinstance(duration, (int, float))
            and end > start
            and abs(float(duration) - (end - start)) <= 1.0
        ):
            verified.append((float(duration), start, end))
    if verified:
        duration, start, end = max(verified)
        return duration, "verified_measurement_window", start, end
    durations = [record.get("measurement_duration_us") for record in summaries]
    numeric = [float(value) for value in durations if isinstance(value, (int, float)) and value > 0]
    if numeric:
        return max(numeric), "reported_duration_without_window", None, None
    timestamps = [frame["csi_done_us"] for frame in frames if isinstance(frame.get("csi_done_us"), int)]
    if len(timestamps) >= 2 and max(timestamps) > min(timestamps):
        start = min(timestamps)
        end = max(timestamps)
        return float(end - start), "first_to_last_frame_inferred", start, end
    raise BenchmarkError("cannot determine measurement duration")


def minimum_window_rate(timestamps: list[int], start: int, end: int, window_us: int) -> float | None:
    if end - start < window_us:
        return None
    ordered = sorted(value for value in timestamps if start <= value < end)
    candidates = {start, end - window_us}
    candidates.update(
        min(end - window_us, max(start, value + 1)) for value in ordered if value < end - window_us
    )
    minimum_count = len(ordered)
    for candidate in candidates:
        left = bisect.bisect_left(ordered, candidate)
        right = bisect.bisect_left(ordered, candidate + window_us)
        minimum_count = min(minimum_count, right - left)
    return minimum_count * 1_000_000.0 / window_us


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    runs = [record for record in records if record.get("type") == "run"]
    if len(runs) != 1:
        raise BenchmarkError("exactly one run record is required")
    run = runs[0]
    if any(record.get("run_id") != run.get("run_id") for record in records):
        raise BenchmarkError("records contain more than one run_id")
    frames = [record for record in records if record.get("type") == "frame"]
    source_summaries = [record for record in records if record.get("type") == "summary"]
    telemetry = [record for record in records if record.get("type") == "telemetry"]
    elapsed_us, duration_source, window_start, window_end = measurement_window(records, frames)
    seconds = elapsed_us / 1_000_000.0
    if window_start is not None and window_end is not None:
        frames = [
            frame
            for frame in frames
            if isinstance(frame.get("csi_done_us"), int)
            and window_start <= frame["csi_done_us"] < window_end
        ]
    completed = [frame for frame in frames if frame.get("centroid_us") is not None]
    fresh = [
        frame
        for frame in completed
        if frame.get("drop_reason") is None
        and (
            window_start is None
            or (
                isinstance(frame.get("centroid_us"), int)
                and window_start <= frame["centroid_us"] < window_end
            )
        )
    ]
    drops = Counter(
        str(frame["drop_reason"]) for frame in frames if frame.get("drop_reason") is not None
    )

    end_to_end = [
        float(frame["centroid_us"] - frame["csi_done_us"])
        for frame in completed
        if isinstance(frame.get("centroid_us"), int) and isinstance(frame.get("csi_done_us"), int)
    ]
    ppa = [
        float(frame["ppa_done_us"] - frame["ppa_submit_us"])
        for frame in frames
        if isinstance(frame.get("ppa_done_us"), int) and isinstance(frame.get("ppa_submit_us"), int)
    ]
    inference = [
        float(frame["infer_done_us"] - frame["infer_start_us"])
        for frame in frames
        if isinstance(frame.get("infer_done_us"), int) and isinstance(frame.get("infer_start_us"), int)
    ]

    depth_values = [
        frame["inference_depth"]
        for frame in frames
        if isinstance(frame.get("inference_depth"), int)
    ]
    source_depths = [
        record["max_inference_depth"]
        for record in source_summaries
        if isinstance(record.get("max_inference_depth"), int)
    ]
    telemetry_lost_values = [
        record["telemetry_lost"]
        for record in source_summaries
        if isinstance(record.get("telemetry_lost"), int)
    ]
    telemetry_lost_values.extend(
        record["counters"]["telemetry_lost"]
        for record in telemetry
        if isinstance(record.get("counters"), dict)
        and isinstance(record["counters"].get("telemetry_lost"), int)
    )

    evidence = run.get("evidence_class")
    scope = run.get("timing_scope")
    provenance_ok = evidence == "device_pipeline" and scope == "device_pipeline"
    duration_ok = elapsed_us >= 60_000_000 and duration_source == "verified_measurement_window"
    device_claim = provenance_ok and duration_ok
    result: dict[str, Any] = {
        "schema_version": "benchmark-run-v1",
        "type": "summary",
        "run_id": run.get("run_id"),
        "evidence_class": evidence,
        "timing_scope": scope,
        "device_fps_claim": device_claim,
        "metric": "raw_frame_pipeline_summary",
        "samples": len(frames),
        "measurement_duration_us": elapsed_us,
        "duration_source": duration_source,
        "captured_fps": len(frames) / seconds,
        "fresh_centroid_fps": len(fresh) / seconds,
        "drops": dict(sorted(drops.items())),
        "telemetry_lost": max(telemetry_lost_values) if telemetry_lost_values else None,
        "max_inference_depth": max(depth_values + source_depths) if depth_values or source_depths else None,
        "derived_from_raw": True,
    }
    if window_start is not None and window_end is not None:
        result["measurement_start_us"] = window_start
        result["measurement_end_us"] = window_end
        fresh_timestamps = [
            frame["centroid_us"]
            for frame in fresh
            if isinstance(frame.get("centroid_us"), int)
        ]
        result["minimum_60s_fresh_centroid_fps"] = minimum_window_rate(
            fresh_timestamps, window_start, window_end, 60_000_000
        )
    if end_to_end:
        result["latency_us"] = latency(end_to_end)
    if ppa:
        result["ppa_latency_us"] = latency(ppa)
    if inference:
        result["inference_latency_us"] = latency(inference)
    target_fps = run.get("target_fps")
    if isinstance(target_fps, (int, float)) and target_fps > 0:
        period_us = 1_000_000.0 / float(target_fps)
        result["deadline_misses"] = sum(value > period_us for value in end_to_end)
    if not device_claim:
        reasons = []
        if not provenance_ok:
            reasons.append("requires device_pipeline evidence and timing scope")
        if not duration_ok:
            reasons.append("requires at least 60 seconds with matching start, end, and duration")
        result["device_claim_rejection"] = "; ".join(reasons)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", help="write one NDJSON summary record to this path")
    parser.add_argument("input")
    args = parser.parse_args()
    try:
        result = summarize(load_path(args.input))
    except BenchmarkError as error:
        print(error, file=sys.stderr)
        return 1
    if args.output:
        Path(args.output).write_text(
            json.dumps(result, allow_nan=False, separators=(",", ":"), sort_keys=True) + "\n",
            encoding="utf-8",
        )
    else:
        print(json.dumps(result, allow_nan=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
