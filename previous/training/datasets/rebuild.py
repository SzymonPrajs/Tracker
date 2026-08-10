#!/usr/bin/env python3
"""Rebuild every local training artifact, then remove reproducible raw inputs."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PIPELINE = ROOT / "training" / "datasets" / "pipeline.py"
PREPROCESS = ROOT / "training" / "datasets" / "preprocess.py"
SOURCES = (
    "scut_head",
    "rpee_heads",
    "r2ppe",
    "open_images_human_head",
    "vgg_hollywood_heads",
    "hollywood_heads",
    "wider_face",
)


def run(*arguments: str) -> None:
    environment = dict(os.environ)
    python_path = str(ROOT / "training")
    environment["PYTHONPATH"] = python_path + os.pathsep + environment.get("PYTHONPATH", "")
    print("+", sys.executable, *arguments, flush=True)
    subprocess.run([sys.executable, *arguments], cwd=ROOT, env=environment, check=True)


def validate_transient_root(path: Path) -> None:
    resolved = path.resolve()
    data_root = (ROOT / "data").resolve()
    if resolved.parent != data_root or resolved.name not in {"raw", "extracted"}:
        raise RuntimeError(f"refusing to prune unexpected path: {resolved}")


def prune_transient_data() -> None:
    for directory in (ROOT / "data" / "raw", ROOT / "data" / "extracted"):
        validate_transient_root(directory)
        directory.mkdir(parents=True, exist_ok=True)
        for item in sorted(directory.iterdir()):
            size = item.stat().st_size if item.is_file() else sum(
                child.stat().st_size for child in item.rglob("*") if child.is_file()
            )
            print(f"removing reproducible transient data: {item} ({size:,} bytes)", flush=True)
            if item.is_dir() and not item.is_symlink():
                shutil.rmtree(item)
            else:
                item.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force-acquire", action="store_true",
        help="download and canonicalize even when the 576x320 cache already validates",
    )
    parser.add_argument(
        "--keep-raw", action="store_true",
        help="retain raw archives and extracted source trees after every derived artifact validates",
    )
    args = parser.parse_args()

    for position, source in enumerate(SOURCES, 1):
        print(f"\n=== source {position}/{len(SOURCES)}: {source} ===", flush=True)
        canonical = ROOT / "data" / "processed" / "576x320" / source / "dataset.json"
        if canonical.exists() and not args.force_acquire:
            print(f"reusing canonical cache: {canonical}", flush=True)
            run(str(PIPELINE), "validate", source)
        else:
            # prepare is itself resumable and prints curl/gdown progress one file at a time.
            run(str(PIPELINE), "prepare", source)
        run(str(PREPROCESS), "materialize", source)
        run(str(PREPROCESS), "validate", source)

    for split in ("train", "val", "test"):
        print(f"\n=== fixed-shape pack: {split} ===", flush=True)
        run(str(PREPROCESS), "pack", split)
        run(str(PREPROCESS), "validate-pack", split)

    if args.keep_raw:
        print("all derived artifacts validated; retaining transient inputs by request", flush=True)
    else:
        print("all derived artifacts validated; pruning raw downloads and extracted trees", flush=True)
        prune_transient_data()
    print("dataset rebuild complete", flush=True)


if __name__ == "__main__":
    main()
