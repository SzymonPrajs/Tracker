from __future__ import annotations

import io
import hashlib
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import requests
from PIL import Image
from tqdm import tqdm


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def clip_labels(
    labels: list[dict[str, Any]], width: int, height: int
) -> list[dict[str, Any]]:
    clipped = []
    for label in labels:
        x, y, box_width, box_height = label["box"]
        x1 = max(0.0, x)
        y1 = max(0.0, y)
        x2 = min(float(width), x + box_width)
        y2 = min(float(height), y + box_height)
        if x2 <= x1 or y2 <= y1:
            continue
        result = dict(label)
        result["box"] = [
            round(x1, 3),
            round(y1, 3),
            round(x2 - x1, 3),
            round(y2 - y1, 3),
        ]
        if any(
            abs(before - after) > 5e-4
            for before, after in zip(
                (x, y, x + box_width, y + box_height), (x1, y1, x2, y2)
            )
        ):
            result["truncated"] = True
        clipped.append(result)
    return clipped


def _open(item: dict[str, Any]) -> Image.Image:
    if "path" in item:
        return Image.open(item["path"])
    for attempt in range(3):
        try:
            with requests.get(item["url"], timeout=60) as response:
                response.raise_for_status()
                return Image.open(io.BytesIO(response.content))
        except requests.RequestException:
            if attempt == 2:
                raise
            time.sleep(attempt + 1)
    raise RuntimeError("unreachable")


def _compact(
    item: dict[str, Any], output: Path, width: int, height: int
) -> dict[str, Any]:
    with _open(item) as source:
        image = source.convert("RGB")
        old_width, old_height = image.size
        image.thumbnail((width, height), Image.Resampling.LANCZOS)
        new_width, new_height = image.size
        filename = f"{safe_name(str(item['id']))}.webp"
        image_path = output / "images" / filename
        image.save(image_path, "WEBP", quality=80, method=4)
        content_hash = hashlib.sha256(image_path.read_bytes()).hexdigest()

    x_scale = new_width / old_width
    y_scale = new_height / old_height
    labels = []
    for label in item.get("labels", []):
        x, y, box_width, box_height = label["box"]
        scaled = dict(label)
        scaled["box"] = [
            round(x * x_scale, 3),
            round(y * y_scale, 3),
            round(box_width * x_scale, 3),
            round(box_height * y_scale, 3),
        ]
        labels.append(scaled)
    labels = clip_labels(labels, new_width, new_height)

    return {
        "image": f"images/{filename}",
        "source": item["source"],
        "source_id": str(item["id"]),
        "split": item.get("split", "train"),
        "group": item.get("group"),
        "content_hash": content_hash,
        "width": new_width,
        "height": new_height,
        "labels": labels,
        "negative_for": item.get("negative_for", []),
    }


def compact_images(
    items: list[dict[str, Any]],
    output: Path,
    width: int,
    height: int,
    workers: int,
    label: str,
) -> list[dict[str, Any]]:
    def convert(item: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
        try:
            return _compact(item, output, width, height), None
        except Exception as error:
            return None, f"{item['id']}: {error}"

    records = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = tqdm(
            pool.map(convert, items),
            total=len(items),
            desc=f"{label}: images",
            unit="image",
        )
        for record, error in results:
            if error:
                tqdm.write(f"{label}: skipped {error}")
            elif record:
                records.append(record)
    if items and not records:
        raise RuntimeError(f"{label}: every image failed")
    return records


def write_labels(output: Path, records: list[dict[str, Any]]) -> None:
    temporary = output / "labels.jsonl.tmp"
    with temporary.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, separators=(",", ":")) + "\n")
    temporary.replace(output / "labels.jsonl")
