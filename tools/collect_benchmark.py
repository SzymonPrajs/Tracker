#!/usr/bin/env python3
"""Collect benchmark JSON objects from a stream while preserving optional raw logs."""

from __future__ import annotations

import argparse
import json
import sys
import time
from contextlib import closing, nullcontext
from pathlib import Path
from typing import BinaryIO, ContextManager, TextIO

from benchmark_common import BenchmarkError, strict_json_loads


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", help="text input path, or - for stdin")
    source.add_argument("--port", help="serial port; requires pyserial")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--duration", type=float, default=0, help="seconds; zero means until EOF")
    parser.add_argument("--output", required=True)
    parser.add_argument("--raw-log")
    return parser.parse_args()


def input_context(args: argparse.Namespace) -> ContextManager[TextIO]:
    if args.input == "-":
        return nullcontext(sys.stdin)
    if args.input:
        return Path(args.input).open(encoding="utf-8", errors="replace")
    try:
        import serial  # type: ignore[import-not-found]
    except ImportError as error:
        raise SystemExit("--port requires the optional pyserial package") from error
    device: BinaryIO = serial.Serial(args.port, args.baud, timeout=0.25)
    return closing(_SerialText(device))


class _SerialText:
    def __init__(self, source: BinaryIO) -> None:
        self.source = source

    def __iter__(self) -> "_SerialText":
        return self

    def __next__(self) -> str:
        line = self.source.readline()
        if line:
            return line.decode("utf-8", errors="replace")
        return ""

    def close(self) -> None:
        self.source.close()


def main() -> int:
    args = arguments()
    if args.baud <= 0:
        raise SystemExit("--baud must be positive")
    if args.duration < 0:
        raise SystemExit("--duration must be non-negative")
    output_path = Path(args.output).resolve()
    protected_paths = [Path(value).resolve() for value in (args.input, args.raw_log) if value and value != "-"]
    if output_path in protected_paths:
        raise SystemExit("--output must differ from --input and --raw-log")
    if args.raw_log and args.input and args.input != "-" and Path(args.raw_log).resolve() == Path(args.input).resolve():
        raise SystemExit("--raw-log must differ from --input")
    deadline = time.monotonic() + args.duration if args.duration > 0 else None
    accepted = malformed = ignored = 0
    raw = Path(args.raw_log).open("w", encoding="utf-8") if args.raw_log else None
    try:
        with input_context(args) as source, Path(args.output).open("w", encoding="utf-8") as output:
            for line in source:
                if deadline is not None and time.monotonic() >= deadline:
                    break
                if not line:
                    continue
                if raw:
                    raw.write(line)
                stripped = line.strip()
                if not stripped.startswith("{"):
                    ignored += 1
                    continue
                try:
                    record = strict_json_loads(stripped)
                except BenchmarkError:
                    malformed += 1
                    continue
                if not isinstance(record, dict) or record.get("schema_version") != "benchmark-run-v1":
                    ignored += 1
                    continue
                output.write(
                    json.dumps(record, allow_nan=False, separators=(",", ":"), sort_keys=True) + "\n"
                )
                output.flush()
                accepted += 1
    except KeyboardInterrupt:
        pass
    finally:
        if raw:
            raw.close()

    print(
        f"collected={accepted} ignored={ignored} malformed_json={malformed}",
        file=sys.stderr,
    )
    return 1 if malformed or accepted == 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
