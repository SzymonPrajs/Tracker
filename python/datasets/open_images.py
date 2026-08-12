from __future__ import annotations

import csv
import random
from pathlib import Path
from tempfile import TemporaryDirectory

from common.files import download, start_output
from common.images import compact_images, write_labels


CLASSES = {"/m/0dzct", "/m/04hgtk"}  # Human face, Human head
METADATA = {
    "train": "https://storage.googleapis.com/openimages/2018_04/train/train-annotations-human-imagelabels-boxable.csv",
    "validation": "https://storage.googleapis.com/openimages/2018_04/validation/validation-annotations-human-imagelabels-boxable.csv",
}
IMAGE = "https://open-images-dataset.s3.amazonaws.com/{split}/{image_id}.jpg"


def _joint_negatives(path: Path) -> list[str]:
    labels: dict[str, dict[str, float]] = {}
    with path.open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            label = row["LabelName"]
            if label in CLASSES:
                labels.setdefault(row["ImageID"], {})[label] = float(row["Confidence"])
    return sorted(
        image_id
        for image_id, states in labels.items()
        if states.keys() >= CLASSES and all(states[label] == 0 for label in CLASSES)
    )


def run(
    output: Path,
    config: dict,
    limit: int | None = None,
    held_out: bool = False,
) -> int:
    if held_out:
        raise ValueError("Open Images already downloads train and validation negatives")
    targets = {
        "train": config["datasets"]["open_images_train_negatives"],
        "validation": config["datasets"]["open_images_validation_negatives"],
    }
    if limit is not None:
        targets = {"train": max(1, limit * 4 // 5), "validation": max(1, limit // 5)}

    start_output(output)
    items = []
    with TemporaryDirectory(prefix="tracker_open_images_") as temporary:
        raw = Path(temporary)
        for split, url in METADATA.items():
            metadata = download(
                url, raw / f"{split}.csv", f"Open Images: {split} labels"
            )
            image_ids = _joint_negatives(metadata)
            random.Random(42).shuffle(image_ids)
            if len(image_ids) < targets[split]:
                raise RuntimeError(
                    f"Open Images has {len(image_ids)} verified {split} negatives, "
                    f"but {targets[split]} were requested"
                )
            for image_id in image_ids[: targets[split]]:
                items.append(
                    {
                        "source": "open_images",
                        "id": image_id,
                        "split": split,
                        "url": IMAGE.format(split=split, image_id=image_id),
                        "labels": [],
                        "negative_for": ["head_center"],
                    }
                )

        records = compact_images(
            items,
            output,
            config["image_width"],
            config["image_height"],
            config["workers"],
            "Open Images negatives",
        )
        write_labels(output, records)
    return len(records)
