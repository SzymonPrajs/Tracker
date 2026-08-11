from __future__ import annotations

import random
import zipfile
from collections import defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory

import ijson

from common.files import download, start_output
from common.images import compact_images, write_labels


ANNOTATIONS = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
IMAGE = "http://images.cocodataset.org/train2017/{filename}"


def _select(path: Path, positives: int, negatives: int) -> list[dict]:
    images = {}
    with path.open("rb") as file:
        for image in ijson.items(file, "images.item"):
            images[image["id"]] = image["file_name"]

    boxes = defaultdict(list)
    person_images = set()
    with path.open("rb") as file:
        for annotation in ijson.items(file, "annotations.item"):
            if annotation["category_id"] != 1:
                continue
            image_id = annotation["image_id"]
            person_images.add(image_id)
            boxes[image_id].append(
                {
                    "kind": "person",
                    "box": [float(value) for value in annotation["bbox"]],
                    "ignore": bool(annotation.get("iscrowd", 0)),
                }
            )

    rng = random.Random(42)
    positive_ids = list(person_images)
    negative_ids = list(images.keys() - person_images)
    rng.shuffle(positive_ids)
    rng.shuffle(negative_ids)

    items = []
    for image_id in positive_ids[:positives]:
        filename = images[image_id]
        items.append(
            {
                "source": "coco",
                "id": image_id,
                "split": "train",
                "url": IMAGE.format(filename=filename),
                "labels": boxes[image_id],
            }
        )
    for image_id in negative_ids[:negatives]:
        filename = images[image_id]
        items.append(
            {
                "source": "coco",
                "id": image_id,
                "split": "train",
                "url": IMAGE.format(filename=filename),
                "labels": [],
                "negative_for": ["person"],
            }
        )
    rng.shuffle(items)
    return items


def run(output: Path, config: dict, limit: int | None = None) -> int:
    positives = config["datasets"]["coco_people"]
    negatives = config["datasets"]["coco_negatives"]
    if limit is not None:
        positives = (limit + 1) // 2
        negatives = limit // 2

    start_output(output)
    with TemporaryDirectory(prefix="tracker_coco_") as temporary:
        raw = Path(temporary)
        archive = download(ANNOTATIONS, raw / "annotations.zip", "COCO: annotations")
        with zipfile.ZipFile(archive) as source:
            source.extract("annotations/instances_train2017.json", raw)
        items = _select(
            raw / "annotations" / "instances_train2017.json", positives, negatives
        )
        records = compact_images(
            items,
            output,
            config["image_width"],
            config["image_height"],
            config["workers"],
            "COCO",
        )
        write_labels(output, records)
    return len(records)
