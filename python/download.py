#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import tomllib
from pathlib import Path

from datasets import ALL


ROOT = Path(__file__).resolve().parents[1]


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and compact the Tracker datasets."
    )
    parser.add_argument(
        "--config", type=Path, default=ROOT / "config" / "download.toml"
    )
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--only", choices=ALL, help="download just one dataset")
    parser.add_argument(
        "--limit", type=int, help="download at most this many images per dataset"
    )
    parser.add_argument(
        "--force", action="store_true", help="replace an already completed dataset"
    )
    parser.add_argument(
        "--held-out",
        action="store_true",
        help="append only the official validation/test split to existing positives",
    )
    args = parser.parse_args()
    if args.held_out and args.only not in {"wider_face", "crowdhuman", "scut_head"}:
        parser.error("--held-out requires --only wider_face, crowdhuman, or scut_head")
    if args.held_out and (args.force or args.limit is not None):
        parser.error("--held-out cannot be combined with --force or --limit")
    return args


def main() -> int:
    args = arguments()
    with args.config.open("rb") as file:
        config = tomllib.load(file)

    selected = [args.only] if args.only else list(ALL)
    failures = []
    for name in selected:
        output = args.data_dir.resolve() / name
        if (output / "labels.jsonl").exists() and not args.force and not args.held_out:
            print(f"\n{name}: already finished ({output})")
            continue
        if args.force and output.exists():
            shutil.rmtree(output)

        print(f"\n{name}: starting")
        try:
            count = ALL[name](output, config, args.limit, args.held_out)
            print(f"{name}: finished {count:,} images")
        except KeyboardInterrupt:
            print(f"\n{name}: stopped; raw temporary files are being removed")
            raise
        except Exception as error:
            failures.append(name)
            print(f"{name}: failed: {error}")
        finally:
            print(f"{name}: raw temporary files removed")

    if failures:
        print("\nFailed datasets: " + ", ".join(failures))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
