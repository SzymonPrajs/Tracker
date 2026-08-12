from __future__ import annotations

import json
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

from common.files import download, start_output, unzip
from common.images import compact_images, write_labels


BASE = "https://huggingface.co/datasets/yongAgain/CrowdHuman/resolve/main/"
ANNOTATIONS = {
    "train": BASE + "annotation_train.odgt",
    "validation": BASE + "annotation_val.odgt",
}
PARTS = {
    "train": [
        BASE + "CrowdHuman_train01.zip",
        BASE + "CrowdHuman_train02.zip",
        BASE + "CrowdHuman_train03.zip",
    ],
    "validation": [BASE + "CrowdHuman_val.zip"],
}


def _read_annotations(path: Path) -> dict[str, dict]:
    records = {}
    with path.open(encoding="utf-8") as file:
        for line in file:
            record = json.loads(line)
            records[record["ID"]] = record
    return records


def _items(root: Path, annotations: dict[str, dict], split: str) -> list[dict]:
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
        items.append(
            {
                "source": "crowdhuman",
                "id": source_id,
                "split": split,
                "path": image,
                "labels": labels,
            }
        )
    return items


def run(
    output: Path,
    config: dict,
    limit: int | None = None,
    held_out: bool = False,
) -> int:
    targets = {
        "train": config["datasets"]["crowdhuman_train"],
        "validation": config["datasets"]["crowdhuman_validation"],
    }
    if limit is not None:
        targets = {"train": max(1, limit * 4 // 5), "validation": max(1, limit // 5)}
    records = []
    if held_out:
        labels_path = output / "labels.jsonl"
        if not labels_path.exists():
            raise FileNotFoundError("download the training split before --held-out")
        records = [json.loads(line) for line in labels_path.read_text().splitlines()]
        records = [record for record in records if record.get("split") != "validation"]
        for record in records:
            record["labels"] = [
                label for label in record["labels"] if label["kind"] == "head"
            ]
    start_output(output)
    with TemporaryDirectory(prefix="tracker_crowdhuman_") as temporary:
        raw = Path(temporary)
        for split in ("train", "validation"):
            if held_out and split != "validation":
                continue
            annotation_path = download(
                ANNOTATIONS[split],
                raw / f"annotation_{split}.odgt",
                f"CrowdHuman: {split} annotations",
            )
            annotations = _read_annotations(annotation_path)
            split_records = []
            for number, url in enumerate(PARTS[split], 1):
                if len(split_records) >= targets[split]:
                    break
                label = f"CrowdHuman: {split} part {number}/{len(PARTS[split])}"
                archive = download(url, raw / f"{split}_{number}.zip", label)
                extracted = unzip(archive, raw / f"{split}_{number}", label)
                items = _items(extracted, annotations, split)[
                    : targets[split] - len(split_records)
                ]
                split_records.extend(
                    compact_images(
                        items,
                        output,
                        config["image_width"],
                        config["image_height"],
                        config["workers"],
                        label,
                    )
                )
                archive.unlink()
                shutil.rmtree(extracted)
            records.extend(split_records)
        write_labels(output, records)
    return len(records)
