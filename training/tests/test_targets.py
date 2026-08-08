import pytest
import torch

from tracker_training.synthetic import SyntheticSceneDataset, make_synthetic_scene
from tracker_training.targets import HeadTarget, encode_targets


def test_integer_gaussian_and_cell_centre_offset_convention():
    targets = encode_targets([HeadTarget(11.0, 10.0, 32.0, 40.0)])
    assert torch.argmax(targets["heatmap"]).item() == 2 * 72 + 2
    assert targets["heatmap"][0, 2, 2].item() == 1.0
    assert targets["offset"][:, 2, 2].tolist() == [0.25, 0.0]
    assert targets["size"][:, 2, 2].tolist() == pytest.approx([32.0 / 288.0, 0.25])


def test_multi_face_collision_keeps_highest_visibility_owner():
    heads = [
        HeadTarget(9.0, 9.0, 40.0, 40.0, visibility=0.2, track_id="low"),
        HeadTarget(10.0, 10.0, 20.0, 20.0, visibility=0.9, track_id="high"),
        HeadTarget(80.0, 40.0, 24.0, 30.0, visibility=1.0, track_id="other"),
    ]
    targets = encode_targets(heads)
    assert targets["collision_count"].item() == 1
    assert targets["collision_mask"][0, 2, 2]
    assert targets["owner_index"][2, 2].item() == 1
    assert targets["reg_mask"].sum().item() == 2
    assert targets["heatmap"][0, 10, 20].item() == 1.0


def test_synthetic_scene_is_deterministic_and_multiface():
    first = make_synthetic_scene(7, seed=123, head_count=4)
    second = make_synthetic_scene(7, seed=123, head_count=4)
    assert torch.equal(first.image, second.image)
    assert first.heads == second.heads
    assert len(first.heads) == 4
    assert first.image.shape == (3, 160, 288)


def test_synthetic_dataset_varies_counts_and_guarantees_empty_negatives():
    dataset = SyntheticSceneDataset(5, seed=0, head_count=None, max_heads=4)
    assert [len(dataset[index].heads) for index in range(len(dataset))] == [0, 1, 2, 3, 4]
    assert torch.equal(dataset[0].image, make_synthetic_scene(0, seed=0, head_count=0).image)
