#!/usr/bin/env python3
"""Generate deterministic little-endian uint16 heatmap fixtures."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "bench" / "fixtures"
GENERATOR_VERSION = 1


def centroid(values: list[int], width: int, height: int) -> dict[str, int | bool]:
    weight = sum(values)
    if weight == 0:
        return {"valid": False, "weight": 0, "x_q16": 0, "y_q16": 0}
    weighted_x = sum(value * (index % width) for index, value in enumerate(values))
    weighted_y = sum(value * (index // width) for index, value in enumerate(values))
    return {
        "valid": True,
        "weight": weight,
        "x_q16": (weighted_x << 16) // weight,
        "y_q16": (weighted_y << 16) // weight,
    }


def lcg_values(count: int, seed: int) -> list[int]:
    state = seed
    values: list[int] = []
    for _ in range(count):
        state = (1664525 * state + 1013904223) & 0xFFFFFFFF
        values.append((state >> 16) & 0xFFFF)
    return values


def fixture_specs() -> list[tuple[str, int, int, list[int]]]:
    width, height = 72, 40
    zeros = [0] * (width * height)

    single = zeros.copy()
    single[17 * width + 43] = 65535

    dual = zeros.copy()
    dual[4 * width + 5] = 1000
    dual[34 * width + 65] = 3000

    edge_width, edge_height = 7, 5
    edge = [0] * (edge_width * edge_height)
    edge[-1] = 65535

    noise_width, noise_height = 73, 41
    noise = lcg_values(noise_width * noise_height, 0x5EED1234)

    return [
        ("all_zero", width, height, zeros),
        ("single_peak", width, height, single),
        ("weighted_dual_peak", width, height, dual),
        ("odd_shape_edge", edge_width, edge_height, edge),
        ("odd_shape_lcg_noise", noise_width, noise_height, noise),
    ]


def generated_files() -> dict[str, bytes]:
    files: dict[str, bytes] = {}
    records = []
    for name, width, height, values in fixture_specs():
        payload = struct.pack(f"<{len(values)}H", *values)
        filename = f"{name}.u16le.bin"
        files[filename] = payload
        records.append(
            {
                "name": name,
                "file": filename,
                "width": width,
                "height": height,
                "dtype": "uint16",
                "byte_order": "little",
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "expected_centroid": centroid(values, width, height),
            }
        )

    manifest = {
        "fixture_schema": "tracker-centroid-fixtures-v1",
        "generator_version": GENERATOR_VERSION,
        "fixtures": records,
    }
    files["manifest.json"] = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    return files


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="verify checked-in fixtures without writing")
    args = parser.parse_args()
    expected = generated_files()

    if args.check:
        errors = []
        for filename, payload in expected.items():
            path = OUTPUT / filename
            if not path.is_file():
                errors.append(f"missing fixture: {path}")
            elif path.read_bytes() != payload:
                errors.append(f"stale fixture: {path}")
        unexpected = sorted(
            path.name for path in OUTPUT.glob("*") if path.is_file() and path.name not in expected
        )
        errors.extend(f"unexpected fixture: {OUTPUT / filename}" for filename in unexpected)
        for error in errors:
            print(error, file=sys.stderr)
        if errors:
            return 1
        print(f"verified {len(expected) - 1} fixtures and manifest")
        return 0

    OUTPUT.mkdir(parents=True, exist_ok=True)
    for filename, payload in expected.items():
        (OUTPUT / filename).write_bytes(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
