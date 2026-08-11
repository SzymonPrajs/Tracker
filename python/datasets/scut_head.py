from __future__ import annotations

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
    images = {
        path.stem: path
        for path in root.rglob("*")
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    }
    items = []
    for annotation in sorted(root.rglob("*.xml")):
        lower_path = str(annotation).lower()
        if "test" in lower_path:
            continue
        tree = ET.parse(annotation).getroot()
        image = images.get(annotation.stem)
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
                "split": "train",
                "path": image,
                "labels": boxes,
            }
        )
    return items


def run(output: Path, config: dict, limit: int | None = None) -> int:
    target = min(
        limit or config["datasets"]["scut_head"], config["datasets"]["scut_head"]
    )
    start_output(output)
    records = []
    with TemporaryDirectory(prefix="tracker_scut_head_") as temporary:
        raw = Path(temporary)
        for part, file_id in PARTS.items():
            if len(records) >= target:
                break
            archive = google_drive(
                file_id, raw / f"part_{part}.zip", f"SCUT-HEAD part {part}"
            )
            extracted = unzip(archive, raw / f"part_{part}", f"SCUT-HEAD part {part}")
            items = _read_part(extracted, part)[: target - len(records)]
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
        write_labels(output, records)
    return len(records)
