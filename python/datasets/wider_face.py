from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from common.files import download, start_output, unzip
from common.images import compact_images, write_labels


IMAGES = {
    "train": "https://huggingface.co/datasets/wider_face/resolve/main/data/WIDER_train.zip",
    "validation": "https://huggingface.co/datasets/wider_face/resolve/main/data/WIDER_val.zip",
}
ANNOTATIONS = (
    "https://huggingface.co/datasets/wider_face/resolve/main/data/wider_face_split.zip"
)


def _annotations(path: Path, image_root: Path, split: str) -> list[dict]:
    lines = path.read_text(encoding="utf-8").splitlines()
    items = []
    line = 0
    while line < len(lines):
        relative = lines[line].strip()
        count = int(lines[line + 1])
        boxes = []
        for raw in lines[line + 2 : line + 2 + count]:
            values = [int(value) for value in raw.split()]
            x, y, width, height = values[:4]
            boxes.append(
                {
                    "kind": "face",
                    "box": [x, y, width, height],
                    "ignore": bool(values[7]),
                    "blur": values[4],
                    "occlusion": values[8],
                    "pose": values[9],
                }
            )
        items.append(
            {
                "source": "wider_face",
                "id": relative.removesuffix(".jpg"),
                "split": split,
                "path": image_root / relative,
                "labels": boxes,
            }
        )
        # WIDER writes one all-zero placeholder row after a zero box count.
        line += 3 if count == 0 else 2 + count
    return items


def run(
    output: Path,
    config: dict,
    limit: int | None = None,
    held_out: bool = False,
) -> int:
    targets = {
        "train": config["datasets"]["wider_face_train"],
        "validation": config["datasets"]["wider_face_validation"],
    }
    if limit is not None:
        targets = {"train": max(1, limit * 4 // 5), "validation": max(1, limit // 5)}
    existing = []
    if held_out:
        labels_path = output / "labels.jsonl"
        if not labels_path.exists():
            raise FileNotFoundError("download the training split before --held-out")
        existing = [json.loads(line) for line in labels_path.read_text().splitlines()]
        existing = [
            record for record in existing if record.get("split") != "validation"
        ]
    start_output(output)
    with TemporaryDirectory(prefix="tracker_wider_face_") as temporary:
        raw = Path(temporary)
        download(ANNOTATIONS, raw / "annotations.zip", "WIDER FACE: annotations")
        unzip(raw / "annotations.zip", raw / "annotations", "WIDER FACE")
        records = existing
        for split, url in IMAGES.items():
            if held_out and split != "validation":
                continue
            archive = download(url, raw / f"{split}.zip", f"WIDER FACE: {split}")
            extracted = unzip(archive, raw / split, f"WIDER FACE: {split}")
            annotation_name = "wider_face_train_bbx_gt.txt"
            image_folder = "WIDER_train"
            if split == "validation":
                annotation_name = "wider_face_val_bbx_gt.txt"
                image_folder = "WIDER_val"
            items = _annotations(
                next((raw / "annotations").rglob(annotation_name)),
                next(extracted.rglob(image_folder)) / "images",
                split,
            )[: targets[split]]
            records.extend(
                compact_images(
                    items,
                    output,
                    config["image_width"],
                    config["image_height"],
                    config["workers"],
                    f"WIDER FACE: {split}",
                )
            )
        write_labels(output, records)
    return len(records)
