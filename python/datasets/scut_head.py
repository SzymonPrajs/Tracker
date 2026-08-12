from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path
from tempfile import TemporaryDirectory

from common.files import google_drive, start_output, unzip
from common.images import compact_images, write_labels


PARTS = {
    "A": "1yaOF9os5wPVNNG4GVzNyULLVe74vdrBE",
    "B": "1LZ_KlTPStDEcqycfqUkDkqQ-aNMMC3cl",
}


def _read_part(root: Path, part: str) -> list[dict]:
    images: dict[str, list[Path]] = {}
    for path in root.rglob("*"):
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            images.setdefault(path.stem, []).append(path)
    train_ids = set(
        next(root.rglob("ImageSets/Main/trainval.txt"))
        .read_text(encoding="utf-8")
        .split()
    )
    validation_ids = set(
        next(root.rglob("ImageSets/Main/test.txt"))
        .read_text(encoding="utf-8")
        .split()
    )
    items = []
    for annotation in sorted(root.rglob("*.xml")):
        if annotation.stem in train_ids:
            split = "train"
        elif annotation.stem in validation_ids:
            split = "validation"
        else:
            continue
        tree = ET.parse(annotation).getroot()
        candidates = images.get(annotation.stem, [])
        image = candidates[0] if len(candidates) == 1 else None
        if image is None:
            continue
        boxes = []
        for person in tree.findall("object"):
            box = person.find("bndbox")
            if box is None:
                continue
            xmin = float(box.findtext("xmin", "0"))
            ymin = float(box.findtext("ymin", "0"))
            xmax = float(box.findtext("xmax", "0"))
            ymax = float(box.findtext("ymax", "0"))
            boxes.append(
                {"kind": "head", "box": [xmin, ymin, xmax - xmin, ymax - ymin]}
            )
        items.append(
            {
                "source": "scut_head",
                "id": f"part_{part}_{annotation.stem}",
                "split": split,
                "path": image,
                "labels": boxes,
            }
        )
    return items


def run(
    output: Path,
    config: dict,
    limit: int | None = None,
    held_out: bool = False,
) -> int:
    records = []
    if held_out:
        labels_path = output / "labels.jsonl"
        if not labels_path.exists():
            raise FileNotFoundError("download the training split before --held-out")
        records = [json.loads(line) for line in labels_path.read_text().splitlines()]
        records = [record for record in records if record.get("split") != "validation"]
    start_output(output)
    with TemporaryDirectory(prefix="tracker_scut_head_") as temporary:
        raw = Path(temporary)
        for part, file_id in PARTS.items():
            archive = google_drive(
                file_id, raw / f"part_{part}.zip", f"SCUT-HEAD part {part}"
            )
            extracted = unzip(archive, raw / f"part_{part}", f"SCUT-HEAD part {part}")
            items = _read_part(extracted, part)
            if held_out:
                items = [item for item in items if item["split"] == "validation"]
            records.extend(
                compact_images(
                    items,
                    output,
                    config["image_width"],
                    config["image_height"],
                    config["workers"],
                    f"SCUT-HEAD part {part}",
                )
            )
            archive.unlink()
        if limit is not None:
            training = [record for record in records if record["split"] == "train"]
            validation = [
                record for record in records if record["split"] == "validation"
            ]
            records = (
                training[: max(1, limit * 4 // 5)] + validation[: max(1, limit // 5)]
            )
        else:
            training = [record for record in records if record["split"] == "train"]
            validation = [
                record for record in records if record["split"] == "validation"
            ]
            records = (
                training[: config["datasets"]["scut_head_train"]]
                + validation[: config["datasets"]["scut_head_validation"]]
            )
        write_labels(output, records)
    return len(records)
