#!/usr/bin/env python3

from __future__ import annotations

import unittest

from benchmark_common import BenchmarkError, strict_json_loads
from summarize_benchmark import summarize
from validate_benchmark import validate_records


def with_lines(records: list[dict]) -> list[dict]:
    return [dict(record, _line=index) for index, record in enumerate(records, 1)]


class BenchmarkToolsTest(unittest.TestCase):
    def test_strict_json_rejects_ambiguous_numbers_and_keys(self) -> None:
        with self.assertRaises(BenchmarkError):
            strict_json_loads('{"type":"run","type":"summary"}')
        with self.assertRaises(BenchmarkError):
            strict_json_loads('{"value":NaN}')
        with self.assertRaises(BenchmarkError):
            strict_json_loads('{"value":1e9999}')

    def test_host_cannot_claim_device_fps(self) -> None:
        records = with_lines(
            [
                {
                    "schema_version": "benchmark-run-v1",
                    "type": "run",
                    "run_id": "host",
                    "evidence_class": "host_synthetic",
                    "timing_scope": "host_postprocess_only",
                    "device_fps_claim": True,
                    "platform": "host",
                }
            ]
        )
        self.assertTrue(any("cannot claim" in error for error in validate_records(records)))

    def test_frame_chronology(self) -> None:
        records = with_lines(
            [
                {
                    "schema_version": "benchmark-run-v1",
                    "type": "run",
                    "run_id": "device",
                    "evidence_class": "device_pipeline",
                    "timing_scope": "device_pipeline",
                    "device_fps_claim": False,
                    "platform": "esp32p4",
                },
                {
                    "schema_version": "benchmark-run-v1",
                    "type": "frame",
                    "run_id": "device",
                    "sequence": 1,
                    "csi_done_us": 100,
                    "ppa_done_us": 90,
                },
            ]
        )
        self.assertTrue(any("out of order" in error for error in validate_records(records)))

    def test_summary_uses_reported_duration(self) -> None:
        records = with_lines(
            [
                {
                    "schema_version": "benchmark-run-v1",
                    "type": "run",
                    "run_id": "device",
                    "evidence_class": "device_pipeline",
                    "timing_scope": "device_pipeline",
                    "device_fps_claim": False,
                    "platform": "esp32p4",
                },
                {
                    "schema_version": "benchmark-run-v1",
                    "type": "frame",
                    "run_id": "device",
                    "sequence": 1,
                    "csi_done_us": 100,
                    "centroid_us": 120,
                    "drop_reason": None,
                },
                {
                    "schema_version": "benchmark-run-v1",
                    "type": "summary",
                    "run_id": "device",
                    "evidence_class": "device_pipeline",
                    "timing_scope": "device_pipeline",
                    "device_fps_claim": False,
                    "metric": "firmware_window",
                    "samples": 1,
                    "measurement_duration_us": 1_000_000,
                },
            ]
        )
        result = summarize(records)
        self.assertEqual(result["duration_source"], "reported_duration_without_window")
        self.assertEqual(result["fresh_centroid_fps"], 1.0)
        self.assertFalse(result["device_fps_claim"])
        self.assertIn("60 seconds", result["device_claim_rejection"])

    def test_long_device_run_has_device_provenance(self) -> None:
        records = with_lines(
            [
                {
                    "schema_version": "benchmark-run-v1",
                    "type": "run",
                    "run_id": "device-long",
                    "evidence_class": "device_pipeline",
                    "timing_scope": "device_pipeline",
                    "device_fps_claim": False,
                    "platform": "esp32p4",
                    "target_fps": 25,
                },
                {
                    "schema_version": "benchmark-run-v1",
                    "type": "frame",
                    "run_id": "device-long",
                    "sequence": 1,
                    "csi_done_us": 100,
                    "centroid_us": 30_000,
                    "inference_depth": 1,
                    "drop_reason": None,
                },
                {
                    "schema_version": "benchmark-run-v1",
                    "type": "summary",
                    "run_id": "device-long",
                    "evidence_class": "device_pipeline",
                    "timing_scope": "device_pipeline",
                    "device_fps_claim": False,
                    "metric": "firmware_window",
                    "samples": 1,
                    "measurement_duration_us": 60_000_000,
                    "measurement_start_us": 0,
                    "measurement_end_us": 60_000_000,
                    "telemetry_lost": 0,
                },
            ]
        )
        result = summarize(records)
        self.assertTrue(result["device_fps_claim"])
        self.assertEqual(result["telemetry_lost"], 0)
        self.assertEqual(result["max_inference_depth"], 1)

    def test_complete_device_gate_fixture_passes(self) -> None:
        run_id = "device-gate"
        run = {
            "schema_version": "benchmark-run-v1",
            "type": "run",
            "run_id": run_id,
            "evidence_class": "device_pipeline",
            "timing_scope": "device_pipeline",
            "device_fps_claim": False,
            "platform": "esp32p4",
            "target_fps": 25,
            "warmup_s": 10,
            "board": "test-board",
            "chip_revision": "v3.0",
            "cpu_mhz": 360,
            "firmware_sha256": "a" * 64,
            "model_sha256": "b" * 64,
            "idf_version": "test",
            "compiler": "test",
            "camera_mode": "synthetic-test-window",
        }
        frames = [
            {
                "schema_version": "benchmark-run-v1",
                "type": "frame",
                "run_id": run_id,
                "sequence": index,
                "csi_done_us": index * 40_000,
                "centroid_us": index * 40_000 + 10_000,
                "inference_depth": 1,
                "output_hash": f"{index:08x}",
                "drop_reason": None,
            }
            for index in range(1500)
        ]
        firmware_window = {
            "schema_version": "benchmark-run-v1",
            "type": "summary",
            "run_id": run_id,
            "evidence_class": "device_pipeline",
            "timing_scope": "device_pipeline",
            "device_fps_claim": False,
            "metric": "firmware_window",
            "samples": 1500,
            "measurement_duration_us": 60_000_000,
            "measurement_start_us": 0,
            "measurement_end_us": 60_000_000,
            "telemetry_lost": 0,
            "max_inference_depth": 1,
        }
        correctness = {
            "schema_version": "benchmark-run-v1",
            "type": "correctness",
            "run_id": run_id,
            "name": "known-fixture",
            "passed": True,
        }
        heaps = [
            {
                "schema_version": "benchmark-run-v1",
                "type": "heap",
                "run_id": run_id,
                "at_us": at_us,
                "caps": "SPIRAM",
                "free": 1_000_000,
                "minimum_free": 900_000,
                "largest": 800_000,
                "phase": phase,
            }
            for at_us, phase in ((0, "post_warmup"), (60_000_000, "final"))
        ]
        raw = [run, correctness, *frames, *heaps, firmware_window]
        derived = summarize(with_lines(raw))
        errors = validate_records(with_lines([*raw, derived]), require_gates=True)
        self.assertEqual(errors, [])


if __name__ == "__main__":
    unittest.main()
