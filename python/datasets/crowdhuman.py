from __future__ import annotations

import json
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

from common.files import google_drive, start_output, unzip
from common.images import compact_images, write_labels


ANNOTATIONS = "1UUTea5mYqvlUObsC1Z8CFldHJAtLtMX3"
PARTS = [
    "134QOvaatwKdy0iIeNqA_p-xkAhkV4F8Y",
    "17evzPh7gc1JBNvnW1ENXLy5Kr4Q_Nnla",
    "1tdp0UCgxrqy1B6p8LkR-Iy0aIJ8l4fJW",
]


def _read_annotations(path: Path) -> dict[str, dict]:
    records = {}
    with path.open(encoding="utf-8") as file:
        for line in file:
            record = json.loads(line)
            records[record["ID"]] = record
    return records


def _items(root: Path, annotations: dict[str, dict]) -> list[dict]:
    items = []
    for image in sorted(root.rglob("*.jpg")):
        source_id = image.stem
        annotation = annotations.get(source_id)
        if annotation is None:
            continue
        labels = []
        for person in annotation["gtboxes"]:
            ignored = bool(person.get("extra", {}).get("ignore", 0))
            head_ignored = bool(person.get("head_attr", {}).get("ignore", 0))
            if "hbox" in person:
                labels.append(
                    {
                        "kind": "head",
                        "box": person["hbox"],
                        "ignore": ignored or head_ignored,
                    }
                )
            if "vbox" in person:
                labels.append(
                    {"kind": "person_visible", "box": person["vbox"], "ignore": ignored}
                )
            if "fbox" in person:
                labels.append(
                    {"kind": "person_full", "box": person["fbox"], "ignore": ignored}
                )
        items.append(
            {
                "source": "crowdhuman",
                "id": source_id,
                "split": "train",
                "path": image,
                "labels": labels,
            }
        )
    return items


def run(output: Path, config: dict, limit: int | None = None) -> int:
    target = min(
        limit or config["datasets"]["crowdhuman"], config["datasets"]["crowdhuman"]
    )
    start_output(output)
    records = []
    with TemporaryDirectory(prefix="tracker_crowdhuman_") as temporary:
        raw = Path(temporary)
        annotation_path = google_drive(
            ANNOTATIONS, raw / "annotation_train.odgt", "CrowdHuman annotations"
        )
        annotations = _read_annotations(annotation_path)
        for number, file_id in enumerate(PARTS, 1):
            if len(records) >= target:
                break
            archive = google_drive(
                file_id, raw / f"train_{number}.zip", f"CrowdHuman part {number}/3"
            )
            extracted = unzip(
                archive, raw / f"part_{number}", f"CrowdHuman part {number}/3"
            )
            items = _items(extracted, annotations)[: target - len(records)]
            records.extend(
                compact_images(
                    items,
                    output,
                    config["image_width"],
                    config["image_height"],
                    config["workers"],
                    f"CrowdHuman part {number}/3",
                )
            )
            archive.unlink()
            shutil.rmtree(extracted)
        write_labels(output, records)
    return len(records)
