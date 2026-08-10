#!/usr/bin/env python3
"""Reproducible acquisition and canonicalization of head-box datasets."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shutil
import stat
import subprocess
import sys
import time
import zipfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable
from xml.etree import ElementTree

import numpy as np
import requests
from PIL import Image, ImageOps


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RAW = DATA / "raw"
EXTRACTED = DATA / "extracted"
PROCESSED = DATA / "processed"
STATE = DATA / "state"
MANIFEST_PATH = Path(__file__).with_name("sources.json")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text())


def source_config(name: str) -> dict:
    sources = load_manifest()["sources"]
    if name not in sources:
        raise SystemExit(f"unknown source {name!r}; choose from: {', '.join(sources)}")
    return sources[name]


def ensure_layout() -> None:
    for path in (RAW, EXTRACTED, PROCESSED, STATE):
        path.mkdir(parents=True, exist_ok=True)


def human_bytes(value: int) -> str:
    amount = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if amount < 1024 or unit == "TiB":
            return f"{amount:.1f} {unit}"
        amount /= 1024
    raise AssertionError


def digest(path: Path, algorithm: str) -> str:
    result = hashlib.new(algorithm)
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            result.update(chunk)
    return result.hexdigest()


def verify(path: Path, spec: dict) -> None:
    if "size" in spec and path.stat().st_size != spec["size"]:
        raise RuntimeError(
            f"size mismatch for {path}: got {path.stat().st_size}, expected {spec['size']}"
        )
    for algorithm in ("md5", "sha256"):
        expected = spec.get(algorithm)
        if expected:
            print(f"verifying {algorithm}: {path.name}", flush=True)
            actual = digest(path, algorithm)
            if actual.lower() != expected.lower():
                raise RuntimeError(f"{algorithm} mismatch for {path}: {actual} != {expected}")


def selected_files(config: dict, only: str | None) -> list[dict]:
    files = config.get("files", [])
    if only is None:
        return files
    selected = [spec for spec in files if spec["name"] == only]
    if not selected:
        raise SystemExit(f"unknown file {only!r}; choose from: {', '.join(spec['name'] for spec in files)}")
    return selected


def download(source: str, only: str | None = None) -> None:
    ensure_layout()
    config = source_config(source)
    if not config.get("enabled", False):
        raise SystemExit(config.get("blocked", f"source {source} is disabled"))
    destination = RAW / source
    destination.mkdir(parents=True, exist_ok=True)
    files = selected_files(config, only)
    for index, spec in enumerate(files, 1):
        final = destination / spec["filename"]
        partial = final.with_suffix(final.suffix + ".part")
        print(
            f"[{index}/{len(files)}] {source}/{final.name}",
            flush=True,
        )
        if final.exists():
            verify(final, spec)
            print(f"already complete: {human_bytes(final.stat().st_size)}", flush=True)
            continue
        if spec["transport"] == "http":
            command = [
                "curl", "--location", "--fail", "--retry", str(spec.get("retries", 8)),
                "--retry-all-errors", "--progress-bar",
                "--connect-timeout", str(spec.get("connect_timeout", 30)),
                "--speed-time", str(spec.get("speed_time", 120)),
                "--speed-limit", str(spec.get("speed_limit", 1024)),
                "--output", str(partial), spec["url"],
            ]
            if spec.get("resume", True):
                command[command.index("--progress-bar"):command.index("--progress-bar")] = [
                    "--continue-at", "-"
                ]
            subprocess.run(command, check=True)
        elif spec["transport"] == "gdrive":
            import gdown

            result = gdown.download(
                id=spec["id"], output=str(partial), quiet=False, resume=True
            )
            if result is None:
                raise RuntimeError(f"Google Drive download failed for {spec['id']}")
        else:
            raise RuntimeError(f"unsupported transport: {spec['transport']}")
        verify(partial, spec)
        partial.replace(final)
        print(f"complete: {human_bytes(final.stat().st_size)}", flush=True)


def safe_members(archive: zipfile.ZipFile, target: Path):
    base = target.resolve()
    for member in archive.infolist():
        # One official VGG release contains a zero-byte DOS directory entry
        # named exactly "/". It carries no path or payload and is safe to omit;
        # every other absolute member remains rejected below.
        if member.filename == "/" and member.is_dir() and member.file_size == 0:
            continue
        resolved = (target / member.filename).resolve()
        if resolved != base and base not in resolved.parents:
            raise RuntimeError(f"unsafe path in {archive.filename}: {member.filename}")
        mode = member.external_attr >> 16
        if stat.S_ISLNK(mode):
            raise RuntimeError(f"symlink rejected in {archive.filename}: {member.filename}")
        yield member


def extract(source: str, only: str | None = None) -> None:
    ensure_layout()
    config = source_config(source)
    for spec in selected_files(config, only):
        if not spec.get("extract", False):
            continue
        archive_path = RAW / source / spec["filename"]
        if not archive_path.exists():
            raise SystemExit(f"missing {archive_path}; download {source} first")
        verify(archive_path, spec)
        target = EXTRACTED / source / spec["name"]
        marker = target / ".complete.json"
        if marker.exists():
            print(f"already extracted: {target}", flush=True)
            continue
        target.mkdir(parents=True, exist_ok=True)
        print(f"extracting {archive_path.name} -> {target}", flush=True)
        with zipfile.ZipFile(archive_path) as archive:
            members = list(safe_members(archive, target))
            total = sum(member.file_size for member in members)
            done = 0
            last_report = 0.0
            for number, member in enumerate(members, 1):
                archive.extract(member, target)
                done += member.file_size
                now = time.monotonic()
                if now - last_report > 3 or number == len(members):
                    percent = 100 * done / max(total, 1)
                    print(
                        f"  {number}/{len(members)} {percent:5.1f}% "
                        f"({human_bytes(done)}/{human_bytes(total)})",
                        flush=True,
                    )
                    last_report = now
        marker.write_text(json.dumps({
            "archive": str(archive_path.relative_to(ROOT)),
            "archive_size": archive_path.stat().st_size,
            "members": len(members),
            "uncompressed_bytes": total,
        }, indent=2) + "\n")


def image_index(root: Path) -> tuple[dict[str, list[Path]], dict[str, list[Path]]]:
    by_name: dict[str, list[Path]] = defaultdict(list)
    by_stem: dict[str, list[Path]] = defaultdict(list)
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            by_name[path.name.lower()].append(path)
            by_stem[path.stem.lower()].append(path)
    return by_name, by_stem


def unique_image(indexes, name: str, context: Path | None = None) -> Path:
    by_name, by_stem = indexes
    candidates = by_name.get(Path(name).name.lower(), [])
    if not candidates:
        candidates = by_stem.get(Path(name).stem.lower(), [])
    if len(candidates) > 1 and context:
        tokens = set(part.lower() for part in context.parts)
        scored = sorted(
            ((len(tokens & set(x.lower() for x in path.parts)), path) for path in candidates),
            reverse=True,
        )
        if len(scored) == 1 or scored[0][0] > scored[1][0]:
            ranked = [path for _, path in scored]
            return ranked[0]
    if not candidates:
        raise FileNotFoundError(f"no image found for {name}")
    if len(candidates) > 1:
        raise RuntimeError(f"ambiguous image {name}: {candidates[:4]}")
    return candidates[0]


def split_from_path(path: Path) -> str | None:
    lower = "/".join(path.parts).lower()
    for split in ("train", "validation", "val", "test"):
        if re.search(rf"(^|[/_.-]){split}([/_.-]|$)", lower):
            return "val" if split == "validation" else split
    return None


def balanced_group_splits(items: list[dict]) -> dict[str, str]:
    """Assign whole sequence groups while targeting an 80/10/10 image ratio."""
    sizes: dict[str, int] = defaultdict(int)
    for item in items:
        sizes[item["sequence"]] += 1
    targets = {"train": len(items) * 0.8, "val": len(items) * 0.1, "test": len(items) * 0.1}
    totals = {name: 0 for name in targets}
    assignments = {}
    ordered = sorted(
        sizes.items(),
        key=lambda pair: (-pair[1], hashlib.sha1(pair[0].encode()).hexdigest()),
    )
    for group, size in ordered:
        split = min(
            targets,
            key=lambda name: (totals[name] / max(targets[name], 1), name),
        )
        assignments[group] = split
        totals[split] += size
    return assignments


def source_splits(items: list[dict]) -> dict[str, str]:
    assignments = {}
    for item in items:
        split = item.get("source_split")
        if split not in {"train", "val", "test"}:
            raise RuntimeError(f"source split is unavailable for {item['image']}")
        previous = assignments.setdefault(item["sequence"], split)
        if previous != split:
            raise RuntimeError(f"source sequence occurs in both {previous} and {split}: {item['sequence']}")
    return assignments


def frame_sequence(value: str) -> str:
    stem = Path(value).stem
    sequence = re.sub(r"(?:[_-](?:frame[_-]?)?\d+)$", "", stem, flags=re.IGNORECASE)
    return sequence or stem


def open_images_select(source: str) -> None:
    """Select room-scale Human head scenes while retaining paired face labels."""
    config = source_config(source)
    selection = config["selection"]
    output = EXTRACTED / source / "annotations"
    output.mkdir(parents=True, exist_ok=True)
    image_list_tmp = output / "images.txt.tmp"
    summary = {"images": 0, "heads": 0, "faces": 0, "splits": {}}
    with image_list_tmp.open("w") as image_list:
        for spec in config["files"]:
            annotation = RAW / source / spec["filename"]
            if not annotation.exists():
                raise SystemExit(f"missing {annotation}; download {source} first")
            split = "val" if spec["name"].startswith("validation") else "train"
            oi_split = "validation" if split == "val" else split
            selected_tmp = output / f"{split}.jsonl.tmp"
            split_counts = {"images": 0, "heads": 0, "faces": 0}

            def serialise(row: dict, kind: str) -> dict:
                xmin, xmax = float(row["XMin"]), float(row["XMax"])
                ymin, ymax = float(row["YMin"]), float(row["YMax"])
                return {
                    "kind": kind,
                    "bbox_normalized_xywh": [xmin, ymin, xmax - xmin, ymax - ymin],
                    "ignore": bool(int(row.get("IsGroupOf", "0"))),
                    "attributes": {
                        "occluded": int(row.get("IsOccluded", "0")),
                        "truncated": int(row.get("IsTruncated", "0")),
                        "group_of": int(row.get("IsGroupOf", "0")),
                        "depiction": int(row.get("IsDepiction", "0")),
                        "source": row.get("Source"),
                    },
                }

            with annotation.open(newline="") as stream, selected_tmp.open("w") as selected:
                reader = csv.DictReader(stream)
                current_id = None
                heads: list[dict] = []
                faces: list[dict] = []

                def emit() -> None:
                    nonlocal heads, faces
                    if current_id is None:
                        return
                    usable = [
                        item for item in heads
                        if not item["attributes"]["group_of"] and not item["attributes"]["depiction"]
                    ]
                    large = [
                        item for item in usable
                        if item["bbox_normalized_xywh"][3]
                        >= selection["minimum_head_height_fraction"]
                    ]
                    if not large or len(usable) > selection["maximum_heads_per_image"]:
                        heads, faces = [], []
                        return
                    kept_faces = [item for item in faces if not item["attributes"]["depiction"]]
                    selected.write(json.dumps({
                        "image_id": current_id,
                        "split": split,
                        "heads": usable,
                        "faces": kept_faces,
                    }, separators=(",", ":")) + "\n")
                    image_list.write(f"{oi_split}/{current_id}\n")
                    split_counts["images"] += 1
                    split_counts["heads"] += len(usable)
                    split_counts["faces"] += len(kept_faces)
                    heads, faces = [], []

                for row in reader:
                    image_id = row["ImageID"]
                    if current_id != image_id:
                        emit()
                        current_id = image_id
                    if row["LabelName"] == selection["head_mid"]:
                        heads.append(serialise(row, "head"))
                    elif row["LabelName"] == selection["face_mid"]:
                        faces.append(serialise(row, "face"))
                emit()
            selected_tmp.replace(output / f"{split}.jsonl")
            summary["splits"][split] = split_counts
            for key in ("images", "heads", "faces"):
                summary[key] += split_counts[key]
            print(f"selected {split}: {split_counts}", flush=True)
    image_list_tmp.replace(output / "images.txt")
    (output / "selection.json").write_text(json.dumps({
        "source": source,
        "policy": selection,
        "counts": summary,
    }, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


def open_images_download_images(source: str, workers: int = 12) -> None:
    """Download the exact selected Open Images IDs from the official public bucket."""
    selected = EXTRACTED / source / "annotations" / "images.txt"
    if not selected.exists():
        raise SystemExit(f"missing {selected}; run select {source} first")
    items = [line.strip() for line in selected.read_text().splitlines() if line.strip()]
    root = EXTRACTED / source / "images"
    root.mkdir(parents=True, exist_ok=True)

    def fetch(item: str) -> tuple[str, int, str]:
        split, image_id = item.split("/", 1)
        destination = root / split / f"{image_id}.jpg"
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() and destination.stat().st_size:
            return item, destination.stat().st_size, "cached"
        partial = destination.with_suffix(".jpg.part")
        url = f"https://open-images-dataset.s3.amazonaws.com/{split}/{image_id}.jpg"
        for attempt in range(8):
            try:
                with requests.get(url, stream=True, timeout=(15, 90)) as response:
                    response.raise_for_status()
                    with partial.open("wb") as stream:
                        for chunk in response.iter_content(1024 * 1024):
                            if chunk:
                                stream.write(chunk)
                if not partial.stat().st_size:
                    raise RuntimeError("empty response")
                partial.replace(destination)
                return item, destination.stat().st_size, "downloaded"
            except Exception:
                if attempt == 7:
                    raise
                time.sleep(min(2 ** attempt, 30))
        raise AssertionError

    complete = bytes_done = downloaded = 0
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fetch, item): item for item in items}
        for future in as_completed(futures):
            item, size, state = future.result()
            complete += 1
            bytes_done += size
            downloaded += state == "downloaded"
            if complete % 100 == 0 or complete == len(items):
                elapsed = max(time.monotonic() - started, 0.001)
                print(
                    f"images {complete}/{len(items)} ({100 * complete / len(items):.1f}%), "
                    f"{human_bytes(bytes_done)}, {complete / elapsed:.1f} images/s, "
                    f"new={downloaded}", flush=True,
                )


def open_images_records(source: str) -> Iterable[dict]:
    root = EXTRACTED / source
    for split in ("train", "val"):
        annotation = root / "annotations" / f"{split}.jsonl"
        if not annotation.exists():
            continue
        image_split = "validation" if split == "val" else split
        with annotation.open() as stream:
            for line in stream:
                row = json.loads(line)
                image = root / "images" / image_split / f"{row['image_id']}.jpg"
                if not image.exists():
                    raise FileNotFoundError(image)
                with Image.open(image) as opened:
                    width, height = opened.size

                def denormalise(item: dict) -> dict:
                    x, y, w, h = item["bbox_normalized_xywh"]
                    return {
                        "bbox": [x * width, y * height, w * width, h * height],
                        "ignore": item["ignore"],
                        "attributes": item["attributes"],
                    }

                yield {
                    "image": image,
                    "annotation": annotation,
                    "boxes": [denormalise(item) for item in row["heads"]],
                    "faces": [denormalise(item) for item in row["faces"]],
                    "source_split": split,
                    "sequence": row["image_id"],
                }


def pascal_records(source: str) -> Iterable[dict]:
    root = EXTRACTED / source
    indexes = image_index(root)
    source_splits = {}
    for split in ("train", "val", "test"):
        for split_file in root.rglob(f"{split}.txt"):
            for stem in split_file.read_text().split():
                source_splits[stem.lower()] = split
    for annotation in sorted(root.rglob("*.xml")):
        tree = ElementTree.parse(annotation).getroot()
        name = tree.findtext("filename") or annotation.with_suffix(".jpg").name
        try:
            image = unique_image(indexes, name, annotation)
        except FileNotFoundError:
            # SCUT Part A XML omits the "PartA_" prefix present on image files.
            image = unique_image(indexes, annotation.stem, annotation)
        boxes = []
        for obj in tree.findall("object"):
            box = obj.find("bndbox")
            if box is None:
                continue
            xmin = float(box.findtext("xmin", "0"))
            ymin = float(box.findtext("ymin", "0"))
            xmax = float(box.findtext("xmax", "0"))
            ymax = float(box.findtext("ymax", "0"))
            boxes.append({"bbox": [xmin, ymin, xmax - xmin, ymax - ymin], "ignore": False})
        yield {
            "image": image, "annotation": annotation, "boxes": boxes,
            "source_split": source_splits.get(annotation.stem.lower()),
            "sequence": image.stem,
        }


def yolo_records(source: str) -> Iterable[dict]:
    root = EXTRACTED / source
    indexes = image_index(root)
    label_files = [path for path in root.rglob("*.txt") if path.stem.lower() not in {"classes", "readme"}]
    for annotation in sorted(label_files):
        try:
            image = unique_image(indexes, annotation.stem, annotation)
        except FileNotFoundError:
            continue
        with Image.open(image) as opened:
            width, height = opened.size
        boxes = []
        for line in annotation.read_text(errors="replace").splitlines():
            fields = line.split()
            if len(fields) < 5:
                continue
            _, cx, cy, bw, bh = map(float, fields[:5])
            boxes.append({
                "bbox": [(cx - bw / 2) * width, (cy - bh / 2) * height, bw * width, bh * height],
                "ignore": False,
            })
        yield {
            "image": image, "annotation": annotation, "boxes": boxes,
            "source_split": split_from_path(annotation),
            "sequence": frame_sequence(image.stem),
        }


def crowdhuman_records(source: str) -> Iterable[dict]:
    root = EXTRACTED / source
    indexes = image_index(root)
    for annotation in sorted((RAW / source).glob("*.odgt")):
        source_split = "val" if "val" in annotation.name.lower() else "train"
        with annotation.open() as stream:
            for line in stream:
                row = json.loads(line)
                boxes = []
                for item in row.get("gtboxes", []):
                    if item.get("tag") != "person" or "hbox" not in item:
                        continue
                    head_attr = item.get("head_attr", {})
                    extra = item.get("extra", {})
                    boxes.append({
                        "bbox": list(map(float, item["hbox"])),
                        "ignore": bool(head_attr.get("ignore", extra.get("ignore", 0))),
                        "attributes": {"occlusion": head_attr.get("occ"), "unsure": head_attr.get("unsure")},
                    })
                image = unique_image(indexes, row["ID"] + ".jpg")
                yield {
                    "image": image, "annotation": annotation, "boxes": boxes,
                    "source_split": source_split, "sequence": row["ID"],
                }


def r2ppe_records(source: str) -> Iterable[dict]:
    root = EXTRACTED / source
    indexes = image_index(root)
    json_files = list(root.rglob("combined.json"))
    if not json_files:
        json_files = [p for p in root.rglob("*.json") if "combined" in p.name.lower()]
    if len(json_files) != 1:
        raise RuntimeError(f"expected one R2PPE combined annotation, found {json_files}")
    annotation = json_files[0]
    data = json.loads(annotation.read_text())
    head_ids = {item["id"] for item in data["categories"] if item["name"].lower() == "head"}
    heads: dict[int, list[dict]] = defaultdict(list)
    for item in data["annotations"]:
        if item["category_id"] in head_ids:
            heads[item["image_id"]].append({
                "bbox": list(map(float, item["bbox"])),
                "ignore": bool(item.get("iscrowd", 0)),
                "attributes": {"source_annotation_id": item.get("id")},
            })
    for item in data["images"]:
        image = unique_image(indexes, item["file_name"], annotation)
        sequence = frame_sequence(item["file_name"])
        yield {
            "image": image, "annotation": annotation, "boxes": heads.get(item["id"], []),
            "source_split": split_from_path(Path(item["file_name"])),
            "sequence": sequence,
        }


def wider_annotation_blocks(lines: list[str]) -> Iterable[tuple[str, list[str]]]:
    index = 0
    while index < len(lines):
        relative_name = lines[index].strip()
        index += 1
        if not relative_name:
            continue
        count = int(lines[index].strip())
        index += 1
        box_lines = lines[index:index + count]
        index += count
        # The publisher files sometimes declare zero faces and still insert one
        # dummy ten-zero rectangle. Consume exactly that malformed sentinel.
        if count == 0 and index < len(lines):
            fields = lines[index].split()
            if len(fields) == 10 and all(value == "0" for value in fields):
                index += 1
        yield relative_name, box_lines


def wider_face_records(source: str) -> Iterable[dict]:
    root = EXTRACTED / source
    selection = source_config(source)["selection"]
    indexes = image_index(root)
    annotations = sorted(root.rglob("wider_face_*_bbx_gt.txt"))
    if not annotations:
        raise RuntimeError(f"WIDER FACE annotations not found under {root}")
    for annotation in annotations:
        source_split = "val" if "_val_" in annotation.name.lower() else "train"
        lines = annotation.read_text(errors="replace").splitlines()
        for relative_name, box_lines in wider_annotation_blocks(lines):
            faces = []
            for box_line in box_lines:
                fields = [int(value) for value in box_line.split()]
                if len(fields) < 4:
                    continue
                x, y, width, height = fields[:4]
                attributes = {
                    name: fields[position]
                    for position, name in enumerate(
                        ("blur", "expression", "illumination", "invalid", "occlusion", "pose"),
                        start=4,
                    )
                    if position < len(fields)
                }
                faces.append({
                    "bbox": [float(x), float(y), float(width), float(height)],
                    "ignore": bool(attributes.get("invalid", 0)),
                    "attributes": attributes,
                })
            image = unique_image(indexes, relative_name, annotation)
            with Image.open(image) as opened:
                _, image_height = opened.size
            usable = [face for face in faces if not face["ignore"]]
            if (
                not usable
                or len(usable) > selection["maximum_faces_per_image"]
                or max(face["bbox"][3] / image_height for face in usable)
                < selection["minimum_face_height_fraction"]
            ):
                continue
            yield {
                "image": image,
                "annotation": annotation,
                "boxes": [],
                "faces": faces,
                "source_split": source_split,
                "sequence": image.stem,
            }


def vgg_hollywood_records(source: str) -> Iterable[dict]:
    root = EXTRACTED / source
    annotation_files = list(root.rglob("head-stats.annotation"))
    if len(annotation_files) != 1:
        raise RuntimeError(f"expected one VGG head annotation, found {annotation_files}")
    annotation = annotation_files[0]
    indexes = image_index(root)
    image_name = None
    boxes: list[dict] = []
    current: dict | None = None

    def finish_object() -> None:
        nonlocal current
        if current is not None and "bbox" in current:
            attributes = current.pop("attributes")
            current["attributes"] = attributes
            boxes.append(current)
        current = None

    def finish_image() -> dict | None:
        nonlocal boxes
        finish_object()
        if image_name is None:
            return None
        image = unique_image(indexes, image_name, annotation)
        item = {
            "image": image,
            "annotation": annotation,
            "boxes": boxes,
            "source_split": None,
            "sequence": frame_sequence(image.stem),
        }
        boxes = []
        return item

    for raw_line in annotation.read_text(errors="replace").splitlines():
        line = raw_line.strip()
        if line == "########## NEW FILE ##########":
            item = finish_image()
            if item is not None:
                yield item
        elif line.startswith("file:"):
            image_name = line.split(":", 1)[1].strip()
        elif line.startswith("object:"):
            finish_object()
            current = {
                "ignore": False,
                "attributes": {"source_object_id": int(line.split(":", 1)[1])},
            }
        elif current is not None and line.startswith("bbox:"):
            current["bbox"] = [
                float(value) for value in line.split(":", 1)[1].split(",")
            ]
        elif current is not None and ":" in line:
            key, value = line.split(":", 1)
            current["attributes"][key.strip()] = value.strip()
    item = finish_image()
    if item is not None:
        yield item


def hollywood_records(source: str) -> Iterable[dict]:
    root = EXTRACTED / source / "release" / "HollywoodHeads"
    config = source_config(source)
    selection = config["selection"]
    source_splits_by_stem = {}
    for split in ("train", "val", "test"):
        for stem in (root / "Splits" / f"{split}.txt").read_text().split():
            source_splits_by_stem[stem] = split
    last_selected: dict[str, int] = {}
    for annotation in sorted((root / "Annotations").glob("*.xml")):
        stem = annotation.stem
        tree = ElementTree.parse(annotation).getroot()
        size = tree.find("size")
        image_height = float(size.findtext("height"))
        boxes = []
        for obj in tree.findall("object"):
            if obj.findtext("name", "").lower() != "head":
                continue
            box = obj.find("bndbox")
            xmin = float(box.findtext("xmin", "0"))
            ymin = float(box.findtext("ymin", "0"))
            xmax = float(box.findtext("xmax", "0"))
            ymax = float(box.findtext("ymax", "0"))
            difficult = bool(int(obj.findtext("difficult", "0")))
            boxes.append({
                "bbox": [xmin, ymin, xmax - xmin, ymax - ymin],
                "ignore": difficult,
                "attributes": {"difficult": int(difficult)},
            })
        usable = [box for box in boxes if not box["ignore"]]
        if (
            not usable
            or len(usable) > selection["maximum_heads_per_image"]
            or max(box["bbox"][3] / image_height for box in usable)
            < selection["minimum_head_height_fraction"]
        ):
            continue
        movie, frame_text = stem.rsplit("_", 1)
        frame = int(frame_text)
        if frame - last_selected.get(movie, -10**12) < selection["minimum_frame_gap"]:
            continue
        last_selected[movie] = frame
        image = root / "JPEGImages" / f"{stem}.jpeg"
        if not image.exists():
            raise FileNotFoundError(image)
        yield {
            "image": image,
            "annotation": annotation,
            "boxes": boxes,
            "source_split": source_splits_by_stem[stem],
            "sequence": movie,
        }


def records(source: str, format_name: str) -> Iterable[dict]:
    if format_name == "pascal_voc":
        return pascal_records(source)
    if format_name == "yolo":
        return yolo_records(source)
    if format_name == "odgt":
        return crowdhuman_records(source)
    if format_name == "coco" and source == "r2ppe":
        return r2ppe_records(source)
    if format_name == "open_images_csv":
        return open_images_records(source)
    if format_name == "wider_face":
        return wider_face_records(source)
    if format_name == "vgg_hollywood":
        return vgg_hollywood_records(source)
    if format_name == "hollywood_voc":
        return hollywood_records(source)
    raise RuntimeError(f"no converter for {source}/{format_name}")


def letterbox(image: Image.Image, width: int, height: int):
    image = ImageOps.exif_transpose(image).convert("RGB")
    scale = min(width / image.width, height / image.height)
    resized_width = max(1, round(image.width * scale))
    resized_height = max(1, round(image.height * scale))
    resized = image.resize((resized_width, resized_height), Image.Resampling.LANCZOS)
    left = (width - resized_width) // 2
    top = (height - resized_height) // 2
    canvas = Image.new("RGB", (width, height), (114, 114, 114))
    canvas.paste(resized, (left, top))
    return canvas, {"scale": scale, "left": left, "top": top}


def transform_box(box: list[float], transform: dict, width: int, height: int) -> list[float] | None:
    x, y, w, h = box
    x = x * transform["scale"] + transform["left"]
    y = y * transform["scale"] + transform["top"]
    w *= transform["scale"]
    h *= transform["scale"]
    x2, y2 = min(width, x + w), min(height, y + h)
    x, y = max(0.0, x), max(0.0, y)
    if x2 <= x or y2 <= y:
        return None
    return [x, y, x2 - x, y2 - y]


def probability_heatmap(
    boxes: list[list[float]], width: int, height: int, stride: int, edge_probability: float
) -> np.ndarray:
    map_width, map_height = width // stride, height // stride
    result = np.zeros((map_height, map_width), dtype=np.float32)
    cutoff = 1.0 / 65535
    radius_squared = math.log(cutoff) / math.log(edge_probability)
    radius = math.sqrt(radius_squared)
    for x, y, w, h in boxes:
        if w <= 0 or h <= 0:
            continue
        cx, cy = (x + w / 2) / stride, (y + h / 2) / stride
        hx, hy = max(w / (2 * stride), 0.5), max(h / (2 * stride), 0.5)
        x0 = max(0, math.floor(cx - radius * hx))
        x1 = min(map_width, math.ceil(cx + radius * hx) + 1)
        y0 = max(0, math.floor(cy - radius * hy))
        y1 = min(map_height, math.ceil(cy + radius * hy) + 1)
        yy, xx = np.mgrid[y0:y1, x0:x1]
        squared = ((xx - cx) / hx) ** 2 + ((yy - cy) / hy) ** 2
        gaussian = np.exp(math.log(edge_probability) * squared).astype(np.float32)
        # Quantize with floor so integer 2x max-pooling lands in the same cell
        # as center regression at the smaller deployment resolution.
        peak_x = min(map_width - 1, max(0, math.floor(cx)))
        peak_y = min(map_height - 1, max(0, math.floor(cy)))
        if x0 <= peak_x < x1 and y0 <= peak_y < y1:
            gaussian[peak_y - y0, peak_x - x0] = 1.0
        window = result[y0:y1, x0:x1]
        result[y0:y1, x0:x1] = 1 - (1 - window) * (1 - gaussian)
    return np.rint(np.clip(result, 0, 1) * 65535).astype(np.uint16)


def convert(source: str, limit: int | None = None) -> None:
    ensure_layout()
    manifest = load_manifest()
    config = source_config(source)
    cache = manifest["cache"]
    width, height = cache["width"], cache["height"]
    preset = f"{width}x{height}"
    output = PROCESSED / preset / source
    images_dir, heatmaps_dir = output / "images", output / "heatmaps"
    face_heatmaps_dir = output / "face_heatmaps"
    images_dir.mkdir(parents=True, exist_ok=True)
    heatmaps_dir.mkdir(parents=True, exist_ok=True)
    face_heatmaps_dir.mkdir(parents=True, exist_ok=True)
    annotations_tmp = output / "annotations.jsonl.tmp"
    counts = {
        "images": 0, "source_heads": 0, "heads": 0, "ignored_heads": 0,
        "rejected_heads": 0, "faces": 0, "ignored_faces": 0,
        "rejected_faces": 0, "empty_images": 0,
    }
    split_counts: dict[str, int] = defaultdict(int)
    started = time.monotonic()
    source_records = list(records(source, config["annotation_format"]))
    if limit is not None:
        source_records = source_records[:limit]
    split_policy = config.get("split_policy", "sequence_balanced")
    split_assignments = (
        source_splits(source_records)
        if split_policy == "source"
        else balanced_group_splits(source_records)
    )
    with annotations_tmp.open("w") as stream:
        for number, item in enumerate(source_records, 1):
            source_image: Path = item["image"]
            relative_image = source_image.relative_to(ROOT)
            key_hash = hashlib.sha1(str(relative_image).encode()).hexdigest()[:10]
            image_id = f"{source}-{source_image.stem}-{key_hash}"
            image_output = images_dir / f"{image_id}.jpg"
            heatmap_output = heatmaps_dir / f"{image_id}.png"
            face_heatmap_output = face_heatmaps_dir / f"{image_id}.png"
            with Image.open(source_image) as opened:
                source_width, source_height = opened.size
                cached, transform = letterbox(opened, width, height)
            canonical_boxes = []
            heatmap_boxes = []
            for original in item["boxes"]:
                counts["source_heads"] += 1
                mapped = transform_box(original["bbox"], transform, width, height)
                if mapped is None:
                    counts["rejected_heads"] += 1
                    continue
                entry = {
                    "bbox_source_xywh": [round(v, 4) for v in original["bbox"]],
                    "bbox_cache_xywh": [round(v, 4) for v in mapped],
                    "ignore": bool(original.get("ignore", False)),
                }
                if original.get("attributes"):
                    entry["attributes"] = original["attributes"]
                canonical_boxes.append(entry)
                if entry["ignore"]:
                    counts["ignored_heads"] += 1
                else:
                    counts["heads"] += 1
                    heatmap_boxes.append(entry["bbox_cache_xywh"])
            canonical_faces = []
            face_heatmap_boxes = []
            for original in item.get("faces", []):
                mapped = transform_box(original["bbox"], transform, width, height)
                if mapped is None:
                    counts["rejected_faces"] += 1
                    continue
                entry = {
                    "bbox_source_xywh": [round(v, 4) for v in original["bbox"]],
                    "bbox_cache_xywh": [round(v, 4) for v in mapped],
                    "ignore": bool(original.get("ignore", False)),
                    "size_supervision": False,
                }
                if original.get("attributes"):
                    entry["attributes"] = original["attributes"]
                canonical_faces.append(entry)
                if entry["ignore"]:
                    counts["ignored_faces"] += 1
                else:
                    counts["faces"] += 1
                    face_heatmap_boxes.append(entry["bbox_cache_xywh"])
            cached.save(image_output, format="JPEG", quality=cache["jpeg_quality"], subsampling=2)
            heatmap = probability_heatmap(
                heatmap_boxes, width, height, cache["output_stride"], cache["edge_probability"]
            )
            Image.fromarray(heatmap).save(heatmap_output, format="PNG")
            face_heatmap_metadata = None
            if config.get("target_kind") == "face":
                face_heatmap = probability_heatmap(
                    face_heatmap_boxes, width, height,
                    cache["output_stride"], cache["edge_probability"],
                )
                Image.fromarray(face_heatmap).save(face_heatmap_output, format="PNG")
                face_heatmap_metadata = {
                    "path": str(face_heatmap_output.relative_to(ROOT)),
                    "size": [width // cache["output_stride"], height // cache["output_stride"]],
                    "dtype": "uint16",
                    "scale": 65535,
                    "output_stride": cache["output_stride"],
                    "edge_probability": cache["edge_probability"],
                    "overlap": cache["overlap"],
                }
            sequence = f"{source}:{item['sequence']}"
            split = split_assignments[item["sequence"]]
            split_counts[split] += 1
            primary_boxes = face_heatmap_boxes if config.get("target_kind") == "face" else heatmap_boxes
            if not primary_boxes:
                counts["empty_images"] += 1
            row = {
                "schema_version": 1,
                "source": source,
                "image_id": image_id,
                "source_image": str(relative_image),
                "source_annotation": str(Path(item["annotation"]).relative_to(ROOT)),
                "source_size": [source_width, source_height],
                "source_split": item["source_split"],
                "sequence_id": sequence,
                "split": split,
                "cache_image": str(image_output.relative_to(ROOT)),
                "cache_size": [width, height],
                "letterbox": transform,
                "heads": canonical_boxes,
                "faces": canonical_faces,
                "primary_target_kind": config.get("target_kind", "head"),
                "heatmap": {
                    "path": str(heatmap_output.relative_to(ROOT)),
                    "size": [width // cache["output_stride"], height // cache["output_stride"]],
                    "dtype": "uint16",
                    "scale": 65535,
                    "output_stride": cache["output_stride"],
                    "edge_probability": cache["edge_probability"],
                    "overlap": cache["overlap"],
                },
            }
            if face_heatmap_metadata is not None:
                row["face_heatmap"] = face_heatmap_metadata
            stream.write(json.dumps(row, separators=(",", ":")) + "\n")
            counts["images"] += 1
            if number % 100 == 0:
                rate = counts["images"] / max(time.monotonic() - started, 0.001)
                print(f"converted {counts['images']} images, {counts['heads']} heads ({rate:.1f} images/s)", flush=True)
    annotations = output / "annotations.jsonl"
    annotations_tmp.replace(annotations)
    summary = {
        "schema_version": 1,
        "source": source,
        "source_homepage": config["homepage"],
        "license": config["license"],
        "label_semantics": config["label_semantics"],
        "split_policy": split_policy,
        "cache": cache,
        "counts": counts,
        "split_images": dict(sorted(split_counts.items())),
        "annotations": str(annotations.relative_to(ROOT)),
    }
    (output / "dataset.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


def status() -> None:
    ensure_layout()
    manifest = load_manifest()
    print(f"data root: {DATA}")
    for name, config in sorted(
        manifest["sources"].items(), key=lambda item: item[1].get("priority", 999)
    ):
        if not config.get("enabled", False):
            label = config.get("status", "blocked").upper()
            print(f"{name:22} {label:11} {config.get('blocked', 'disabled')}")
            continue
        states = []
        for spec in config.get("files", []):
            path = RAW / name / spec["filename"]
            partial = path.with_suffix(path.suffix + ".part")
            if path.exists():
                states.append(f"{spec['name']}={human_bytes(path.stat().st_size)}")
            elif partial.exists():
                progress = ""
                if spec.get("size"):
                    progress = f"/{human_bytes(spec['size'])} ({100 * partial.stat().st_size / spec['size']:.1f}%)"
                states.append(f"{spec['name']}={human_bytes(partial.stat().st_size)}{progress} partial")
            else:
                states.append(f"{spec['name']}=missing")
        selection_path = EXTRACTED / name / "annotations" / "selection.json"
        if selection_path.exists():
            selection = json.loads(selection_path.read_text())["counts"]
            downloaded_images = sum(1 for _ in (EXTRACTED / name / "images").rglob("*.jpg"))
            states.append(
                f"selected={selection['images']} images/{selection['heads']} heads; "
                f"images={downloaded_images}/{selection['images']}"
            )
        dataset = PROCESSED / f"{manifest['cache']['width']}x{manifest['cache']['height']}" / name / "dataset.json"
        converted = ""
        if dataset.exists():
            count = json.loads(dataset.read_text())["counts"]
            converted = f"; converted={count['images']} images/{count['heads']} heads"
        print(f"{name:22} " + ", ".join(states) + converted)


def validate(source: str) -> None:
    manifest = load_manifest()
    cache = manifest["cache"]
    output = PROCESSED / f"{cache['width']}x{cache['height']}" / source
    annotations = output / "annotations.jsonl"
    summary_path = output / "dataset.json"
    if not annotations.exists() or not summary_path.exists():
        raise SystemExit(f"missing converted dataset for {source}")
    published_counts = json.loads(summary_path.read_text())["counts"]
    seen_ids = set()
    sequence_splits: dict[str, str] = {}
    counts = {
        "images": 0, "source_heads": published_counts["rejected_heads"],
        "heads": 0, "ignored_heads": 0,
        "rejected_heads": published_counts["rejected_heads"],
        "faces": 0, "ignored_faces": 0,
        "rejected_faces": published_counts.get("rejected_faces", 0),
        "empty_images": 0,
    }
    with annotations.open() as stream:
        for line_number, line in enumerate(stream, 1):
            row = json.loads(line)
            image_id = row["image_id"]
            if image_id in seen_ids:
                raise RuntimeError(f"duplicate image_id at line {line_number}: {image_id}")
            seen_ids.add(image_id)
            previous = sequence_splits.setdefault(row["sequence_id"], row["split"])
            if previous != row["split"]:
                raise RuntimeError(f"sequence leakage: {row['sequence_id']} in {previous} and {row['split']}")
            image_path = ROOT / row["cache_image"]
            heatmap_path = ROOT / row["heatmap"]["path"]
            with Image.open(image_path) as image:
                if image.size != tuple(row["cache_size"]):
                    raise RuntimeError(f"image size mismatch: {image_path}")
            with Image.open(heatmap_path) as image:
                heatmap = np.asarray(image, dtype=np.uint16)
            if heatmap.shape != tuple(reversed(row["heatmap"]["size"])):
                raise RuntimeError(f"heatmap size mismatch: {heatmap_path}")
            valid = 0
            for head in row["heads"]:
                counts["source_heads"] += 1
                x, y, width, height = head["bbox_cache_xywh"]
                if not (0 <= x < cache["width"] and 0 <= y < cache["height"] and width > 0 and height > 0):
                    raise RuntimeError(f"invalid box at line {line_number}: {head}")
                if x + width > cache["width"] + 1e-3 or y + height > cache["height"] + 1e-3:
                    raise RuntimeError(f"unclipped box at line {line_number}: {head}")
                if head["ignore"]:
                    counts["ignored_heads"] += 1
                    continue
                counts["heads"] += 1
                valid += 1
                ix = min(heatmap.shape[1] - 1, math.floor((x + width / 2) / cache["output_stride"]))
                iy = min(heatmap.shape[0] - 1, math.floor((y + height / 2) / cache["output_stride"]))
                if heatmap[iy, ix] != 65535:
                    raise RuntimeError(f"head center is not unit probability: {heatmap_path} ({ix}, {iy})")
            faces = row.get("faces", [])
            face_heatmap = None
            if faces and "face_heatmap" in row:
                face_heatmap_path = ROOT / row["face_heatmap"]["path"]
                with Image.open(face_heatmap_path) as image:
                    face_heatmap = np.asarray(image, dtype=np.uint16)
                if face_heatmap.shape != tuple(reversed(row["face_heatmap"]["size"])):
                    raise RuntimeError(f"face heatmap size mismatch: {face_heatmap_path}")
            valid_faces = 0
            for face in faces:
                x, y, width, height = face["bbox_cache_xywh"]
                if not (0 <= x < cache["width"] and 0 <= y < cache["height"] and width > 0 and height > 0):
                    raise RuntimeError(f"invalid face box at line {line_number}: {face}")
                if face["ignore"]:
                    counts["ignored_faces"] += 1
                    continue
                counts["faces"] += 1
                valid_faces += 1
                if face_heatmap is not None:
                    ix = min(face_heatmap.shape[1] - 1, math.floor((x + width / 2) / cache["output_stride"]))
                    iy = min(face_heatmap.shape[0] - 1, math.floor((y + height / 2) / cache["output_stride"]))
                    if face_heatmap[iy, ix] != 65535:
                        raise RuntimeError(f"face center is not unit probability: {face_heatmap_path} ({ix}, {iy})")
            primary_valid = valid_faces if row.get("primary_target_kind") == "face" else valid
            if primary_valid == 0:
                counts["empty_images"] += 1
            counts["images"] += 1
    summary_counts = published_counts
    keys = ["images", "source_heads", "heads", "ignored_heads", "empty_images"]
    if "faces" in summary_counts:
        keys.extend(("faces", "ignored_faces", "rejected_faces"))
    for key in keys:
        if counts[key] != summary_counts[key]:
            raise RuntimeError(f"summary mismatch for {key}: {counts[key]} != {summary_counts[key]}")
    print(
        f"validated {source}: {counts['images']} images, {counts['heads']} usable heads, "
        f"{counts['rejected_heads']} rejected, {len(sequence_splits)} isolated sequences",
        flush=True,
    )


def audit(source: str) -> None:
    manifest = load_manifest()
    cache = manifest["cache"]
    output = PROCESSED / f"{cache['width']}x{cache['height']}" / source
    annotations = output / "annotations.jsonl"
    if not annotations.exists():
        raise SystemExit(f"missing converted dataset for {source}")
    head_heights = []
    face_heights = []
    image_head_counts = []
    image_face_counts = []
    with annotations.open() as stream:
        for line in stream:
            row = json.loads(line)
            source_height = row["source_size"][1]
            heads = [head for head in row["heads"] if not head["ignore"]]
            faces = [face for face in row.get("faces", []) if not face["ignore"]]
            image_head_counts.append(len(heads))
            image_face_counts.append(len(faces))
            head_heights.extend(head["bbox_source_xywh"][3] / source_height for head in heads)
            face_heights.extend(face["bbox_source_xywh"][3] / source_height for face in faces)

    def distribution(values: list[float]) -> dict:
        if not values:
            return {"count": 0}
        array = np.asarray(values)
        return {
            "count": len(values),
            "height_fraction_percentiles": {
                str(percentile): round(float(np.percentile(array, percentile)), 6)
                for percentile in (1, 10, 25, 50, 75, 90, 99)
            },
            "boxes_at_least_fraction": {
                str(threshold): int((array >= threshold).sum())
                for threshold in (0.05, 0.10, 0.15, 0.20, 0.30, 0.50)
            },
        }

    head_counts = np.asarray(image_head_counts)
    result = {
        "source": source,
        "images": len(image_head_counts),
        "heads": distribution(head_heights),
        "faces": distribution(face_heights),
        "images_by_usable_head_count": {
            "empty": int((head_counts == 0).sum()),
            "one_to_four": int(((head_counts >= 1) & (head_counts <= 4)).sum()),
            "five_to_eight": int(((head_counts >= 5) & (head_counts <= 8)).sum()),
            "more_than_eight": int((head_counts > 8).sum()),
        },
    }
    (output / "audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("download", "extract"):
        child = subparsers.add_parser(command)
        child.add_argument("source")
        child.add_argument("--file", help="process one named file from the source manifest")
    child = subparsers.add_parser("validate")
    child.add_argument("source")
    child = subparsers.add_parser("audit")
    child.add_argument("source")
    child = subparsers.add_parser("prepare")
    child.add_argument("source")
    child = subparsers.add_parser("convert")
    child.add_argument("source")
    child.add_argument("--limit", type=int)
    child = subparsers.add_parser("select")
    child.add_argument("source")
    child = subparsers.add_parser("download-images")
    child.add_argument("source")
    child.add_argument("--workers", type=int, default=12)
    subparsers.add_parser("status")
    args = parser.parse_args()
    if args.command == "download":
        download(args.source, args.file)
    elif args.command == "extract":
        extract(args.source, args.file)
    elif args.command == "convert":
        convert(args.source, args.limit)
    elif args.command == "validate":
        validate(args.source)
    elif args.command == "audit":
        audit(args.source)
    elif args.command == "select":
        if args.source != "open_images_human_head":
            raise SystemExit("select is currently implemented for open_images_human_head")
        open_images_select(args.source)
    elif args.command == "download-images":
        if args.source != "open_images_human_head":
            raise SystemExit("download-images is currently implemented for open_images_human_head")
        open_images_download_images(args.source, args.workers)
    elif args.command == "prepare":
        download(args.source)
        extract(args.source)
        if args.source == "open_images_human_head":
            open_images_select(args.source)
            open_images_download_images(args.source)
        convert(args.source)
        validate(args.source)
        audit(args.source)
    else:
        status()


if __name__ == "__main__":
    main()
