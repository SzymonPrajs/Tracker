import json
import numpy as np
import torch
import zipfile

import pytest

from datasets.pipeline import (
    balanced_group_splits,
    probability_heatmap,
    safe_members,
    transform_box,
    wider_annotation_blocks,
)
from datasets.preprocess import _scaled_box
from tracker_training.data import HeadDataset, PackedHeadDataset


def probability(values):
    return values.astype(np.float32) / 65535


def test_heatmap_has_unit_center_and_configured_box_edge():
    heatmap = probability_heatmap([[32, 16, 32, 16]], 96, 64, 4, 0.05)
    values = probability(heatmap)
    assert values.shape == (16, 24)
    assert values[6, 12] == 1.0
    assert abs(values[6, 16] - 0.05) < 2 / 65535
    assert abs(values[8, 12] - 0.05) < 2 / 65535


def test_overlapping_heads_use_probability_union():
    one = probability(probability_heatmap([[32, 16, 32, 16]], 96, 64, 4, 0.05))
    two = probability(probability_heatmap(
        [[32, 16, 32, 16], [32, 16, 32, 16]], 96, 64, 4, 0.05
    ))
    expected = 1 - (1 - one[6, 14]) ** 2
    assert abs(two[6, 14] - expected) < 2 / 65535
    assert two[6, 14] > one[6, 14]


def test_box_transform_scales_offsets_and_clips():
    transform = {"scale": 2.0, "left": 4, "top": 8}
    assert transform_box([10, 5, 20, 10], transform, 100, 80) == [24.0, 18.0, 40.0, 20.0]
    assert transform_box([-10, -10, 12, 12], transform, 100, 80) == [0.0, 0.0, 8.0, 12.0]


def test_balanced_splits_never_split_a_sequence():
    items = [
        {"sequence": f"sequence-{sequence}"}
        for sequence in range(10)
        for _ in range(10)
    ]
    assignments = balanced_group_splits(items)
    counts = {name: 0 for name in ("train", "val", "test")}
    for item in items:
        counts[assignments[item["sequence"]]] += 1
    assert counts == {"train": 80, "val": 10, "test": 10}


def test_archive_extraction_rejects_path_traversal(tmp_path):
    archive_path = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("../escape.txt", "no")
    with zipfile.ZipFile(archive_path) as archive:
        with pytest.raises(RuntimeError, match="unsafe path"):
            list(safe_members(archive, tmp_path / "output"))


def test_archive_extraction_skips_only_empty_root_marker(tmp_path):
    archive_path = tmp_path / "root-marker.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        marker = zipfile.ZipInfo("/")
        marker.external_attr = 0x10
        archive.writestr(marker, b"")
        archive.writestr("images/example.jpg", b"image")
    with zipfile.ZipFile(archive_path) as archive:
        members = list(safe_members(archive, tmp_path / "output"))
    assert [member.filename for member in members] == ["images/example.jpg"]


def test_wider_zero_face_dummy_line_does_not_desynchronise_parser():
    lines = [
        "event/empty.jpg", "0", "0 0 0 0 0 0 0 0 0 0",
        "event/face.jpg", "1", "10 20 30 40 0 0 0 0 0 0",
    ]
    assert list(wider_annotation_blocks(lines)) == [
        ("event/empty.jpg", []),
        ("event/face.jpg", ["10 20 30 40 0 0 0 0 0 0"]),
    ]


def test_room_sampler_preserves_source_mass_and_downweights_dense_scenes():
    dataset = HeadDataset.__new__(HeadDataset)
    dataset.rows = [
        {"source": "room", "heads": [{"ignore": False}]},
        {"source": "room", "heads": [{"ignore": False}] * 12},
        {"source": "crowd", "heads": [{"ignore": False}]},
    ]
    mix = {
        "source_probability_mass": {"room": 0.8, "crowd": 0.2},
        "head_count_weight": {
            "empty": 0.35, "one_to_four": 1.0,
            "five_to_eight": 0.5, "more_than_eight": 0.15,
        },
    }
    weights = dataset.sampling_weights(mix)
    assert abs(float(weights[:2].sum()) - 0.8) < 1e-9
    assert abs(float(weights[2]) - 0.2) < 1e-9
    assert weights[0] > weights[1]


def test_deployment_box_preserves_source_truth_and_scales_cache_coordinates():
    box = {"bbox_source_xywh": [2, 4, 8, 10], "bbox_cache_xywh": [4, 8, 16, 20], "ignore": False}
    scaled = _scaled_box(box, 0.5, 0.5)
    assert scaled["bbox_source_xywh"] == box["bbox_source_xywh"]
    assert scaled["bbox_cache_xywh"] == [2.0, 4.0, 8.0, 10.0]


def test_packed_dataset_returns_saved_fixed_shape_targets(tmp_path):
    split = tmp_path / "train"
    split.mkdir()
    np.zeros((1, 160, 288, 3), dtype=np.uint8).tofile(split / "images.uint8")
    heatmap = np.zeros((1, 40, 72), dtype=np.uint16)
    heatmap[0, 5, 7] = 65535
    heatmap.tofile(split / "heatmaps.uint16")
    regression = np.zeros((1, 4, 40, 72), dtype=np.float16)
    regression[0, :, 5, 7] = [0.1, -0.2, 0.3, 0.4]
    regression.tofile(split / "regression.float16")
    mask = np.zeros((1, 1, 40, 72), dtype=np.uint8)
    mask[0, 0, 5, 7] = 1
    mask.tofile(split / "mask.uint8")
    row = {"source": "fixture", "image_id": "one", "split": "train", "heads": [{"ignore": False}]}
    (split / "index.jsonl").write_text(json.dumps(row) + "\n")
    metadata = {
        "split": "train", "count": 1, "index": "index.jsonl",
        "image": {"path": "images.uint8", "shape": [1, 160, 288, 3]},
        "heatmap": {"path": "heatmaps.uint16", "shape": [1, 40, 72]},
        "regression": {"path": "regression.float16", "shape": [1, 4, 40, 72]},
        "mask": {"path": "mask.uint8", "shape": [1, 1, 40, 72]},
    }
    (split / "metadata.json").write_text(json.dumps(metadata))

    image, (loaded_heatmap, offset, size, loaded_mask) = PackedHeadDataset(tmp_path)[0]
    assert image.shape == (3, 160, 288) and image.dtype == torch.uint8
    assert loaded_heatmap[0, 5, 7] == 1
    assert loaded_mask[0, 5, 7]
    assert np.allclose(offset[:, 5, 7].numpy(), [0.1, -0.2], atol=2e-4)
    assert np.allclose(size[:, 5, 7].numpy(), [0.3, 0.4], atol=2e-4)
