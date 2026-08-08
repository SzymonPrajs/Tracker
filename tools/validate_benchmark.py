#!/usr/bin/env python3
"""Validate benchmark NDJSON structure, chronology, provenance, and optional gates."""

from __future__ import annotations

import argparse
import math
import sys
from typing import Any

from benchmark_common import BenchmarkError, load_path

SCHEMA_VERSION = "benchmark-run-v1"
TYPES = {"run", "correctness", "frame", "heap", "telemetry", "summary"}
EVIDENCE = {"host_synthetic", "device_microbenchmark", "device_pipeline"}
SCOPES = {"host_postprocess_only", "device_kernel_only", "device_pipeline"}
FRAME_CLOCKS = (
    "csi_done_us",
    "ppa_submit_us",
    "ppa_done_us",
    "infer_start_us",
    "infer_done_us",
    "centroid_us",
    "actuator_us",
)


def require(record: dict[str, Any], fields: tuple[str, ...], errors: list[str]) -> None:
    for field in fields:
        if field not in record:
            errors.append(f"line {record['_line']}: missing {field}")


def validate_records(records: list[dict[str, Any]], require_gates: bool = False) -> list[str]:
    errors: list[str] = []
    if not records:
        return ["no benchmark records"]
    for record in records:
        require(record, ("schema_version", "type", "run_id"), errors)
        if record.get("schema_version") != SCHEMA_VERSION:
            errors.append(f"line {record['_line']}: wrong schema_version")
        if record.get("type") not in TYPES:
            errors.append(f"line {record['_line']}: unknown record type")

    runs = [record for record in records if record.get("type") == "run"]
    if len(runs) != 1:
        errors.append(f"expected exactly one run record, found {len(runs)}")
        return errors
    run = runs[0]
    require(run, ("evidence_class", "timing_scope", "device_fps_claim", "platform"), errors)
    run_id = run.get("run_id")
    if any(record.get("run_id") != run_id for record in records):
        errors.append("records contain more than one run_id")
    if run.get("evidence_class") not in EVIDENCE:
        errors.append("run has invalid evidence_class")
    if run.get("timing_scope") not in SCOPES:
        errors.append("run has invalid timing_scope")
    expected_scope = {
        "host_synthetic": "host_postprocess_only",
        "device_microbenchmark": "device_kernel_only",
        "device_pipeline": "device_pipeline",
    }.get(run.get("evidence_class"))
    if expected_scope is not None and run.get("timing_scope") != expected_scope:
        errors.append(f"{run.get('evidence_class')} requires {expected_scope} timing scope")
    if run.get("evidence_class") == "host_synthetic" and run.get("device_fps_claim") is not False:
        errors.append("host_synthetic run cannot claim device FPS")
    if run.get("device_fps_claim") is True and (
        run.get("evidence_class") != "device_pipeline" or run.get("timing_scope") != "device_pipeline"
    ):
        errors.append("run device FPS claim lacks device pipeline evidence")
    if not isinstance(run.get("device_fps_claim"), bool):
        errors.append("run device_fps_claim must be boolean")
    if not isinstance(run.get("platform"), str) or not run.get("platform"):
        errors.append("run platform must be a non-empty string")
    if run.get("evidence_class") in {"device_microbenchmark", "device_pipeline"}:
        for field in ("board", "chip_revision", "firmware_sha256", "idf_version", "compiler"):
            if not isinstance(run.get(field), str) or not run.get(field):
                errors.append(f"device evidence requires non-empty {field}")
        if not isinstance(run.get("cpu_mhz"), (int, float)) or run.get("cpu_mhz", 0) <= 0:
            errors.append("device evidence requires positive cpu_mhz")
    if run.get("evidence_class") == "device_pipeline":
        for field in ("camera_mode", "model_sha256"):
            if not isinstance(run.get(field), str) or not run.get(field):
                errors.append(f"device pipeline evidence requires non-empty {field}")
        if not isinstance(run.get("warmup_s"), (int, float)) or run.get("warmup_s", 0) < 10:
            errors.append("device pipeline evidence requires at least 10 seconds warmup")

    previous_sequence: int | None = None
    for frame in (record for record in records if record.get("type") == "frame"):
        require(frame, ("sequence", "csi_done_us"), errors)
        sequence = frame.get("sequence")
        if isinstance(sequence, int) and not isinstance(sequence, bool):
            if previous_sequence is not None and sequence <= previous_sequence:
                errors.append(f"line {frame['_line']}: frame sequence is not strictly increasing")
            previous_sequence = sequence
        else:
            errors.append(f"line {frame['_line']}: frame sequence must be an integer")
        timestamps = [frame.get(field) for field in FRAME_CLOCKS]
        present = [value for value in timestamps if value is not None]
        if any(not isinstance(value, int) or value < 0 for value in present):
            errors.append(f"line {frame['_line']}: frame timestamps must be non-negative integers")
        if present != sorted(present):
            errors.append(f"line {frame['_line']}: frame timestamps are out of order")

    correctness = [record for record in records if record.get("type") == "correctness"]
    for record in correctness:
        require(record, ("name", "passed"), errors)
        if not isinstance(record.get("passed"), bool):
            errors.append(f"line {record['_line']}: correctness passed must be boolean")

    summaries = [record for record in records if record.get("type") == "summary"]
    for summary in summaries:
        require(summary, ("evidence_class", "timing_scope", "device_fps_claim", "metric", "samples"), errors)
        if summary.get("evidence_class") != run.get("evidence_class"):
            errors.append(f"line {summary['_line']}: summary evidence_class differs from run")
        if summary.get("timing_scope") != run.get("timing_scope"):
            errors.append(f"line {summary['_line']}: summary timing_scope differs from run")
        if not isinstance(summary.get("device_fps_claim"), bool):
            errors.append(f"line {summary['_line']}: summary device_fps_claim must be boolean")
        if not isinstance(summary.get("samples"), int) or isinstance(summary.get("samples"), bool):
            errors.append(f"line {summary['_line']}: summary samples must be an integer")
        if summary.get("device_fps_claim"):
            if run.get("evidence_class") != "device_pipeline" or run.get("timing_scope") != "device_pipeline":
                errors.append(f"line {summary['_line']}: device FPS claim lacks device pipeline evidence")
            duration = summary.get("measurement_duration_us", 0)
            if not isinstance(duration, (int, float)) or duration < 60_000_000:
                errors.append(f"line {summary['_line']}: device FPS claim needs at least 60 seconds")
            start = summary.get("measurement_start_us")
            end = summary.get("measurement_end_us")
            if not isinstance(start, int) or not isinstance(end, int) or end <= start:
                errors.append(f"line {summary['_line']}: device FPS claim needs a valid measurement window")
            elif not isinstance(duration, (int, float)) or not math.isclose(
                float(duration), float(end - start), abs_tol=1.0
            ):
                errors.append(f"line {summary['_line']}: duration differs from measurement window")
        latency = summary.get("latency_us") or summary.get("latency_ns")
        if isinstance(latency, dict):
            ordered = [latency.get(key) for key in ("min", "p50", "p95", "p99", "max")]
            numeric = [value for value in ordered if isinstance(value, (int, float))]
            if len(numeric) == len(ordered) and numeric != sorted(numeric):
                errors.append(f"line {summary['_line']}: latency quantiles are out of order")

    if require_gates:
        if not correctness:
            errors.append("acceptance gates require correctness records")
        elif any(record.get("passed") is not True for record in correctness):
            errors.append("one or more correctness gates failed")
        pipeline_summaries = [
            summary for summary in summaries if summary.get("metric") == "raw_frame_pipeline_summary"
        ]
        if not pipeline_summaries:
            errors.append("acceptance gates require a raw_frame_pipeline_summary")
        else:
            summary = pipeline_summaries[-1]
            if summary.get("device_fps_claim") is not True:
                errors.append("acceptance gates require a valid device FPS evidence claim")
            if summary.get("derived_from_raw") is not True:
                errors.append("acceptance gates require a summary derived from raw frame records")
            if summary.get("duration_source") != "verified_measurement_window":
                errors.append("acceptance gates require a verified measurement window")
            target = run.get("target_fps")
            if not isinstance(target, (int, float)) or target <= 0:
                errors.append("acceptance gates require an explicit positive target_fps")
                target = 25
            fresh_fps = summary.get("fresh_centroid_fps")
            if not isinstance(fresh_fps, (int, float)) or fresh_fps < target:
                errors.append(f"fresh centroid rate is below target {target} FPS")
            minimum_window_fps = summary.get("minimum_60s_fresh_centroid_fps")
            if not isinstance(minimum_window_fps, (int, float)) or minimum_window_fps < target:
                errors.append(f"at least one 60-second window is below target {target} FPS")
            period_us = 1_000_000.0 / float(target)
            latency = summary.get("latency_us", {})
            p95 = latency.get("p95") if isinstance(latency, dict) else None
            p99 = latency.get("p99") if isinstance(latency, dict) else None
            if not isinstance(p95, (int, float)) or p95 > period_us:
                errors.append("p95 end-to-end latency exceeds one frame period")
            if not isinstance(p99, (int, float)) or p99 > 1.25 * period_us:
                errors.append("p99 end-to-end latency exceeds 1.25 frame periods")
            if summary.get("telemetry_lost") != 0:
                errors.append("telemetry records were lost")
            if summary.get("max_inference_depth") not in (0, 1):
                errors.append("maximum inference queue depth exceeds one or was not reported")
            start = summary.get("measurement_start_us")
            end = summary.get("measurement_end_us")
            if isinstance(start, int) and isinstance(end, int) and end > start:
                measured_frames = [
                    frame
                    for frame in records
                    if frame.get("type") == "frame"
                    and isinstance(frame.get("csi_done_us"), int)
                    and start <= frame["csi_done_us"] < end
                ]
                if summary.get("samples") != len(measured_frames):
                    errors.append("summary sample count does not match raw frames in measurement window")
                if any(
                    frame.get("drop_reason") is None
                    and frame.get("centroid_us") is not None
                    and not frame.get("output_hash")
                    for frame in measured_frames
                ):
                    errors.append("fresh frame records must include a non-empty output_hash")

        heaps = [record for record in records if record.get("type") == "heap"]
        if not heaps:
            errors.append("acceptance gates require heap snapshots")
        heap_tolerance = run.get("heap_tolerance_bytes", 4096)
        if not isinstance(heap_tolerance, int) or heap_tolerance < 0:
            errors.append("heap_tolerance_bytes must be a non-negative integer")
            heap_tolerance = 4096
        caps = {record.get("caps") for record in heaps}
        for capability in caps:
            baseline = next(
                (record for record in heaps if record.get("caps") == capability and record.get("phase") == "post_warmup"),
                None,
            )
            final = next(
                (record for record in reversed(heaps) if record.get("caps") == capability and record.get("phase") == "final"),
                None,
            )
            if baseline is None or final is None:
                errors.append(f"heap {capability} lacks post_warmup/final snapshots")
                continue
            for field in ("free", "largest"):
                baseline_value = baseline.get(field)
                final_value = final.get(field)
                if not isinstance(baseline_value, int) or not isinstance(final_value, int):
                    errors.append(f"heap {capability} {field} is not an integer")
                elif final_value + heap_tolerance < baseline_value:
                    errors.append(f"heap {capability} {field} declined beyond tolerance")

    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-gates", action="store_true")
    parser.add_argument("input", nargs="+")
    args = parser.parse_args()
    try:
        records = []
        for path in args.input:
            records.extend(load_path(path))
    except BenchmarkError as error:
        print(error, file=sys.stderr)
        return 1
    errors = validate_records(records, args.require_gates)
    for error in errors:
        print(error, file=sys.stderr)
    if errors:
        return 1
    print(f"validated {len(records)} records for run {records[0].get('run_id')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
