from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

from common.files import download, start_output, unzip
from common.images import compact_images, write_labels


IMAGES = "https://huggingface.co/datasets/wider_face/resolve/main/data/WIDER_train.zip"
ANNOTATIONS = (
    "https://huggingface.co/datasets/wider_face/resolve/main/data/wider_face_split.zip"
)


def _annotations(path: Path, image_root: Path) -> list[dict]:
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
                    "ignore": bool(values[9]),
                    "blur": values[4],
                    "occlusion": values[7],
                }
            )
        items.append(
            {
                "source": "wider_face",
                "id": relative.removesuffix(".jpg"),
                "split": "train",
                "path": image_root / relative,
                "labels": boxes,
            }
        )
        # WIDER writes one all-zero placeholder row after a zero box count.
        line += 3 if count == 0 else 2 + count
    return items


def run(output: Path, config: dict, limit: int | None = None) -> int:
    target = min(
        limit or config["datasets"]["wider_face"], config["datasets"]["wider_face"]
    )
    start_output(output)
    with TemporaryDirectory(prefix="tracker_wider_face_") as temporary:
        raw = Path(temporary)
        download(ANNOTATIONS, raw / "annotations.zip", "WIDER FACE: annotations")
        unzip(raw / "annotations.zip", raw / "annotations", "WIDER FACE")
        download(IMAGES, raw / "images.zip", "WIDER FACE: images")
        unzip(raw / "images.zip", raw / "images", "WIDER FACE")
        items = _annotations(
            next((raw / "annotations").rglob("wider_face_train_bbx_gt.txt")),
            next((raw / "images").rglob("WIDER_train")) / "images",
        )[:target]
        records = compact_images(
            items,
            output,
            config["image_width"],
            config["image_height"],
            config["workers"],
            "WIDER FACE",
        )
        write_labels(output, records)
    return len(records)
