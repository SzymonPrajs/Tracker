#!/usr/bin/env python3
"""Build deployment-sized caches and fixed-shape memory-mapped training packs."""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from datasets.pipeline import ROOT, probability_heatmap


PROCESSED = ROOT / "data" / "processed"
PACKED = ROOT / "data" / "packed"
SOURCE_PRESET = "576x320"
TARGET_PRESET = "288x160"
TARGET_WIDTH = 288
TARGET_HEIGHT = 160
OUTPUT_STRIDE = 4
EDGE_PROBABILITY = 0.05
HEAD_SOURCES = (
    "scut_head",
    "rpee_heads",
    "r2ppe",
    "open_images_human_head",
    "vgg_hollywood_heads",
    "hollywood_heads",
)
ALL_SOURCES = HEAD_SOURCES + ("wider_face",)


def _scaled_box(box: dict, scale_x: float, scale_y: float) -> dict:
    result = dict(box)
    x, y, width, height = box["bbox_cache_xywh"]
    result["bbox_cache_xywh"] = [
        round(x * scale_x, 4),
        round(y * scale_y, 4),
        round(width * scale_x, 4),
        round(height * scale_y, 4),
    ]
    return result


def materialize_source(source: str) -> None:
    """Resize one canonical source and regenerate exact stride-four targets."""
    source_dir = PROCESSED / SOURCE_PRESET / source
    source_annotations = source_dir / "annotations.jsonl"
    source_summary = source_dir / "dataset.json"
    if not source_annotations.exists() or not source_summary.exists():
        raise RuntimeError(f"missing canonical source cache: {source_dir}")

    destination = PROCESSED / TARGET_PRESET / source
    image_dir = destination / "images"
    heatmap_dir = destination / "heatmaps"
    face_heatmap_dir = destination / "face_heatmaps"
    for directory in (image_dir, heatmap_dir, face_heatmap_dir):
        directory.mkdir(parents=True, exist_ok=True)

    annotation_tmp = destination / "annotations.jsonl.tmp"
    started = time.monotonic()
    count = 0
    with source_annotations.open() as incoming, annotation_tmp.open("w") as outgoing:
        for line in incoming:
            row = json.loads(line)
            source_width, source_height = row["cache_size"]
            scale_x = TARGET_WIDTH / source_width
            scale_y = TARGET_HEIGHT / source_height
            image_path = image_dir / f"{row['image_id']}.jpg"
            heatmap_path = heatmap_dir / f"{row['image_id']}.png"
            face_heatmap_path = face_heatmap_dir / f"{row['image_id']}.png"

            if not image_path.exists():
                with Image.open(ROOT / row["cache_image"]) as image:
                    resized = image.convert("RGB").resize(
                        (TARGET_WIDTH, TARGET_HEIGHT), Image.Resampling.LANCZOS
                    )
                    resized.save(image_path, "JPEG", quality=90, subsampling=2)

            heads = [_scaled_box(box, scale_x, scale_y) for box in row["heads"]]
            faces = [_scaled_box(box, scale_x, scale_y) for box in row.get("faces", [])]
            head_boxes = [box["bbox_cache_xywh"] for box in heads if not box["ignore"]]
            face_boxes = [box["bbox_cache_xywh"] for box in faces if not box["ignore"]]
            Image.fromarray(probability_heatmap(
                head_boxes, TARGET_WIDTH, TARGET_HEIGHT, OUTPUT_STRIDE, EDGE_PROBABILITY
            )).save(heatmap_path, "PNG")
            if faces:
                Image.fromarray(probability_heatmap(
                    face_boxes, TARGET_WIDTH, TARGET_HEIGHT, OUTPUT_STRIDE, EDGE_PROBABILITY
                )).save(face_heatmap_path, "PNG")

            transformed = dict(row)
            transformed["schema_version"] = 2
            transformed["parent_cache_image"] = row["cache_image"]
            transformed["cache_image"] = str(image_path.relative_to(ROOT))
            transformed["cache_size"] = [TARGET_WIDTH, TARGET_HEIGHT]
            transformed["heads"] = heads
            transformed["faces"] = faces
            transformed["letterbox"] = {
                "scale": round(row["letterbox"]["scale"] * scale_x, 8),
                "left": round(row["letterbox"]["left"] * scale_x, 4),
                "top": round(row["letterbox"]["top"] * scale_y, 4),
            }
            transformed["heatmap"] = {
                "path": str(heatmap_path.relative_to(ROOT)),
                "size": [TARGET_WIDTH // OUTPUT_STRIDE, TARGET_HEIGHT // OUTPUT_STRIDE],
                "dtype": "uint16",
                "scale": 65535,
                "output_stride": OUTPUT_STRIDE,
                "edge_probability": EDGE_PROBABILITY,
                "overlap": "probabilistic_union",
            }
            transformed.pop("face_heatmap", None)
            if faces:
                transformed["face_heatmap"] = {
                    **transformed["heatmap"],
                    "path": str(face_heatmap_path.relative_to(ROOT)),
                }
            outgoing.write(json.dumps(transformed, separators=(",", ":")) + "\n")
            count += 1
            if count % 1000 == 0:
                rate = count / max(time.monotonic() - started, 0.001)
                print(f"{source}: {count} deployment images ({rate:.1f}/s)", flush=True)

    annotation_path = destination / "annotations.jsonl"
    annotation_tmp.replace(annotation_path)
    original = json.loads(source_summary.read_text())
    summary = dict(original)
    summary["schema_version"] = 2
    summary["parent_cache"] = original["cache"]
    summary["cache"] = {
        "width": TARGET_WIDTH,
        "height": TARGET_HEIGHT,
        "output_stride": OUTPUT_STRIDE,
        "edge_probability": EDGE_PROBABILITY,
        "overlap": "probabilistic_union",
        "image_format": "jpeg",
        "jpeg_quality": 90,
        "heatmap_format": "uint16_png",
    }
    summary["annotations"] = str(annotation_path.relative_to(ROOT))
    summary["derived_from"] = str(source_annotations.relative_to(ROOT))
    (destination / "dataset.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"materialized {source}: {count} images", flush=True)


def validate_source(source: str) -> None:
    directory = PROCESSED / TARGET_PRESET / source
    summary = json.loads((directory / "dataset.json").read_text())
    expected = summary["counts"]["images"]
    counts = Counter()
    seen = set()
    with (directory / "annotations.jsonl").open() as stream:
        for line_number, line in enumerate(stream, 1):
            row = json.loads(line)
            if row["image_id"] in seen:
                raise RuntimeError(f"duplicate image id at line {line_number}")
            seen.add(row["image_id"])
            with Image.open(ROOT / row["cache_image"]) as image:
                if image.size != (TARGET_WIDTH, TARGET_HEIGHT):
                    raise RuntimeError(f"bad image shape: {row['cache_image']}")
            with Image.open(ROOT / row["heatmap"]["path"]) as image:
                heatmap = np.asarray(image, dtype=np.uint16)
            if heatmap.shape != (TARGET_HEIGHT // OUTPUT_STRIDE, TARGET_WIDTH // OUTPUT_STRIDE):
                raise RuntimeError(f"bad heatmap shape: {row['heatmap']['path']}")
            for head in row["heads"]:
                x, y, width, height = head["bbox_cache_xywh"]
                if x < 0 or y < 0 or width <= 0 or height <= 0:
                    raise RuntimeError(f"bad head box at line {line_number}")
                if x + width > TARGET_WIDTH + 1e-3 or y + height > TARGET_HEIGHT + 1e-3:
                    raise RuntimeError(f"unclipped head box at line {line_number}")
                if not head["ignore"]:
                    ix = min(71, int((x + width / 2) // OUTPUT_STRIDE))
                    iy = min(39, int((y + height / 2) // OUTPUT_STRIDE))
                    if heatmap[iy, ix] != 65535:
                        raise RuntimeError(f"non-unit head center at line {line_number}")
                    counts["heads"] += 1
            if row.get("faces"):
                with Image.open(ROOT / row["face_heatmap"]["path"]) as image:
                    face_heatmap = np.asarray(image, dtype=np.uint16)
                for face in row["faces"]:
                    if not face["ignore"]:
                        x, y, width, height = face["bbox_cache_xywh"]
                        ix = min(71, int((x + width / 2) // OUTPUT_STRIDE))
                        iy = min(39, int((y + height / 2) // OUTPUT_STRIDE))
                        if face_heatmap[iy, ix] != 65535:
                            raise RuntimeError(f"non-unit face center at line {line_number}")
                        counts["faces"] += 1
            counts["images"] += 1
    if counts["images"] != expected:
        raise RuntimeError(f"image count mismatch: {counts['images']} != {expected}")
    if counts["heads"] != summary["counts"]["heads"]:
        raise RuntimeError(f"head count mismatch: {counts['heads']}")
    if counts["faces"] != summary["counts"].get("faces", 0):
        raise RuntimeError(f"face count mismatch: {counts['faces']}")
    print(f"validated {source}: {counts['images']} images, {counts['heads']} heads, {counts['faces']} faces")


def _head_rows(split: str) -> list[dict]:
    rows = []
    root = PROCESSED / TARGET_PRESET
    for source in HEAD_SOURCES:
        annotations = root / source / "annotations.jsonl"
        if not annotations.exists():
            raise RuntimeError(f"missing deployment annotations: {annotations}")
        with annotations.open() as stream:
            rows.extend(
                row for row in map(json.loads, stream)
                if row["split"] == split and row.get("primary_target_kind", "head") == "head"
            )
    return rows


def pack_split(split: str) -> None:
    rows = _head_rows(split)
    destination = PACKED / TARGET_PRESET / split
    destination.mkdir(parents=True, exist_ok=True)
    image_tmp = destination / "images.uint8.tmp"
    heatmap_tmp = destination / "heatmaps.uint16.tmp"
    regression_tmp = destination / "regression.float16.tmp"
    mask_tmp = destination / "mask.uint8.tmp"
    index_tmp = destination / "index.jsonl.tmp"
    images = np.memmap(
        image_tmp, mode="w+", dtype=np.uint8,
        shape=(len(rows), TARGET_HEIGHT, TARGET_WIDTH, 3),
    )
    heatmaps = np.memmap(
        heatmap_tmp, mode="w+", dtype=np.uint16,
        shape=(len(rows), TARGET_HEIGHT // OUTPUT_STRIDE, TARGET_WIDTH // OUTPUT_STRIDE),
    )
    regression = np.memmap(
        regression_tmp, mode="w+", dtype=np.float16,
        shape=(len(rows), 4, TARGET_HEIGHT // OUTPUT_STRIDE, TARGET_WIDTH // OUTPUT_STRIDE),
    )
    masks = np.memmap(
        mask_tmp, mode="w+", dtype=np.uint8,
        shape=(len(rows), 1, TARGET_HEIGHT // OUTPUT_STRIDE, TARGET_WIDTH // OUTPUT_STRIDE),
    )
    regression[:] = 0
    masks[:] = 0
    started = time.monotonic()
    with index_tmp.open("w") as index_stream:
        for index, row in enumerate(rows):
            with Image.open(ROOT / row["cache_image"]) as image:
                images[index] = np.asarray(image.convert("RGB"), dtype=np.uint8)
            with Image.open(ROOT / row["heatmap"]["path"]) as image:
                heatmaps[index] = np.asarray(image, dtype=np.uint16)
            for head in row["heads"]:
                if head["ignore"]:
                    continue
                x, y, width, height = head["bbox_cache_xywh"]
                u = (x + width / 2) / OUTPUT_STRIDE
                v = (y + height / 2) / OUTPUT_STRIDE
                ix = min(71, max(0, int(u)))
                iy = min(39, max(0, int(v)))
                if not masks[index, 0, iy, ix]:
                    regression[index, 0, iy, ix] = u - ix - 0.5
                    regression[index, 1, iy, ix] = v - iy - 0.5
                    regression[index, 2, iy, ix] = width / TARGET_WIDTH
                    regression[index, 3, iy, ix] = height / TARGET_HEIGHT
                    masks[index, 0, iy, ix] = 1
            packed_row = {
                "source": row["source"],
                "image_id": row["image_id"],
                "split": row["split"],
                "source_size": row["source_size"],
                "heads": row["heads"],
                "canonical_annotation": str(
                    (PROCESSED / TARGET_PRESET / row["source"] / "annotations.jsonl").relative_to(ROOT)
                ),
            }
            index_stream.write(json.dumps(packed_row, separators=(",", ":")) + "\n")
            if (index + 1) % 1000 == 0:
                rate = (index + 1) / max(time.monotonic() - started, 0.001)
                print(f"pack {split}: {index + 1}/{len(rows)} ({rate:.1f}/s)", flush=True)
    images.flush()
    heatmaps.flush()
    regression.flush()
    masks.flush()
    del images, heatmaps, regression, masks
    image_path = destination / "images.uint8"
    heatmap_path = destination / "heatmaps.uint16"
    regression_path = destination / "regression.float16"
    mask_path = destination / "mask.uint8"
    index_path = destination / "index.jsonl"
    os.replace(image_tmp, image_path)
    os.replace(heatmap_tmp, heatmap_path)
    os.replace(regression_tmp, regression_path)
    os.replace(mask_tmp, mask_path)
    os.replace(index_tmp, index_path)
    sources = Counter(row["source"] for row in rows)
    metadata = {
        "schema_version": 1,
        "split": split,
        "count": len(rows),
        "image": {"path": image_path.name, "shape": [len(rows), TARGET_HEIGHT, TARGET_WIDTH, 3], "dtype": "uint8"},
        "heatmap": {"path": heatmap_path.name, "shape": [len(rows), 40, 72], "dtype": "uint16", "scale": 65535},
        "regression": {"path": regression_path.name, "shape": [len(rows), 4, 40, 72], "dtype": "float16", "channels": ["offset_x", "offset_y", "width", "height"]},
        "mask": {"path": mask_path.name, "shape": [len(rows), 1, 40, 72], "dtype": "uint8"},
        "index": index_path.name,
        "source_counts": dict(sorted(sources.items())),
    }
    (destination / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"packed {split}: {len(rows)} images")


def validate_pack(split: str) -> None:
    directory = PACKED / TARGET_PRESET / split
    metadata = json.loads((directory / "metadata.json").read_text())
    count = metadata["count"]
    expected_image_bytes = count * TARGET_HEIGHT * TARGET_WIDTH * 3
    expected_heatmap_bytes = count * 40 * 72 * 2
    expected_regression_bytes = count * 4 * 40 * 72 * 2
    expected_mask_bytes = count * 40 * 72
    image_path = directory / metadata["image"]["path"]
    heatmap_path = directory / metadata["heatmap"]["path"]
    regression_path = directory / metadata["regression"]["path"]
    mask_path = directory / metadata["mask"]["path"]
    if image_path.stat().st_size != expected_image_bytes:
        raise RuntimeError("packed image byte count mismatch")
    if heatmap_path.stat().st_size != expected_heatmap_bytes:
        raise RuntimeError("packed heatmap byte count mismatch")
    if regression_path.stat().st_size != expected_regression_bytes:
        raise RuntimeError("packed regression byte count mismatch")
    if mask_path.stat().st_size != expected_mask_bytes:
        raise RuntimeError("packed mask byte count mismatch")
    with (directory / metadata["index"]).open() as stream:
        rows = [json.loads(line) for line in stream]
    if len(rows) != count:
        raise RuntimeError("packed index count mismatch")
    heatmaps = np.memmap(heatmap_path, mode="r", dtype=np.uint16, shape=(count, 40, 72))
    regression = np.memmap(regression_path, mode="r", dtype=np.float16, shape=(count, 4, 40, 72))
    masks = np.memmap(mask_path, mode="r", dtype=np.uint8, shape=(count, 1, 40, 72))
    for index in np.linspace(0, count - 1, min(count, 97), dtype=int):
        if not np.isfinite(heatmaps[index]).all():
            raise RuntimeError(f"invalid packed heatmap at {index}")
        for head in rows[index]["heads"]:
            if head["ignore"]:
                continue
            x, y, width, height = head["bbox_cache_xywh"]
            ix = min(71, int((x + width / 2) // OUTPUT_STRIDE))
            iy = min(39, int((y + height / 2) // OUTPUT_STRIDE))
            if heatmaps[index, iy, ix] != 65535:
                raise RuntimeError(f"packed center mismatch at {index}")
            if not masks[index, 0, iy, ix]:
                raise RuntimeError(f"packed regression mask mismatch at {index}")
            if not np.isfinite(regression[index, :, iy, ix]).all():
                raise RuntimeError(f"packed regression value mismatch at {index}")
    packed_bytes = sum(path.stat().st_size for path in (image_path, heatmap_path, regression_path, mask_path))
    print(f"validated pack {split}: {count} rows, {packed_bytes:,} bytes")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("materialize", "validate"):
        child = subparsers.add_parser(command)
        child.add_argument("source", choices=ALL_SOURCES)
    for command in ("pack", "validate-pack"):
        child = subparsers.add_parser(command)
        child.add_argument("split", choices=("train", "val", "test"))
    subparsers.add_parser("all")
    args = parser.parse_args()
    if args.command == "materialize":
        materialize_source(args.source)
    elif args.command == "validate":
        validate_source(args.source)
    elif args.command == "pack":
        pack_split(args.split)
    elif args.command == "validate-pack":
        validate_pack(args.split)
    else:
        for source in ALL_SOURCES:
            materialize_source(source)
            validate_source(source)
        for split in ("train", "val", "test"):
            pack_split(split)
            validate_pack(split)


if __name__ == "__main__":
    main()
