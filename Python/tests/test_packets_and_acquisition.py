from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from conftest import negative_candidate, write_manifest
from PIL import Image

from data_pipeline import PacketReader, acquire, plan_acquisition, validate_packet
from data_pipeline.errors import PipelineError
from data_pipeline.images import orient_normalized_point


def _positive_candidate(image_id: str, image: Path) -> dict[str, object]:
    return {
        "source_image_id": image_id,
        "image_url": image.as_uri(),
        "source_split": "train",
        "instances": [
            {
                "source_instance_id": "head-1",
                "semantic": "head_visible",
                "geometry_kind": "bbox",
                "coordinates": [0.25, 0.25, 0.75, 0.75],
                "coordinate_space": "normalized",
                "quality": "exact",
            }
        ],
        "coverage_records": [
            {
                "exact_class_or_mid": "head",
                "status": "positive_exhaustive",
                "evidence_origin": "fixture annotation",
            }
        ],
        "duplicate_group": "scene-1",
    }


def test_end_to_end_packet_is_minimized_readable_and_cleaned(
    tmp_path: Path, make_image, make_config
) -> None:
    large = make_image(tmp_path / "large.png", (800, 400), (100, 20, 30))
    small = make_image(tmp_path / "small.png", (100, 50), (20, 100, 30))
    manifest = write_manifest(
        tmp_path / "manifest.json",
        [_positive_candidate("large", large), negative_candidate("small", small)],
    )
    config = make_config(manifest, max_images=2)
    result = acquire(config)

    assert result.cleanup_verified
    assert validate_packet(result.packet_path)["images"] == 2
    records = {
        record.source_image_id: record for record in PacketReader(result.packet_path).records()
    }
    assert (records["large"].stored_width, records["large"].stored_height) == (200, 100)
    assert (records["small"].stored_width, records["small"].stored_height) == (100, 50)
    assert records["large"].instances[0].coordinates == (50.0, 25.0, 150.0, 75.0)
    assert records["large"].native_instances[0].coordinates == (0.25, 0.25, 0.75, 0.75)
    staging_entries = {path.name for path in config.paths.staging_root.iterdir()}
    assert staging_entries == {".tracker-data-staging"}
    report = json.loads(result.run_report_path.read_text())
    assert report["cleanup_verified"] is True
    assert report["staging_directory_exists"] is False
    assert not any(path.suffix == ".source" for path in tmp_path.rglob("*"))


def test_packet_validator_detects_tampering(tmp_path: Path, make_image, make_config) -> None:
    image = make_image(tmp_path / "image.png")
    manifest = write_manifest(tmp_path / "manifest.json", [negative_candidate("negative", image)])
    result = acquire(make_config(manifest, max_images=1))
    stored_image = next((result.packet_path / "images").iterdir())
    stored_image.write_bytes(stored_image.read_bytes() + b"tamper")
    with pytest.raises(PipelineError, match="checksum mismatch"):
        validate_packet(result.packet_path)


def test_packet_validator_rejects_symlinks(tmp_path: Path, make_image, make_config) -> None:
    image = make_image(tmp_path / "image.png")
    manifest = write_manifest(tmp_path / "manifest.json", [negative_candidate("one", image)])
    result = acquire(make_config(manifest, max_images=1))
    os.symlink(result.packet_path / "README.md", result.packet_path / "linked-readme")
    with pytest.raises(PipelineError, match="symlink"):
        validate_packet(result.packet_path)


def test_corrupt_source_fails_and_staging_is_cleaned(tmp_path: Path, make_config) -> None:
    corrupt = tmp_path / "corrupt.jpg"
    corrupt.write_bytes(b"not an image")
    manifest = write_manifest(tmp_path / "manifest.json", [negative_candidate("corrupt", corrupt)])
    config = make_config(manifest, max_images=1)
    with pytest.raises(PipelineError, match="cannot decode"):
        acquire(config)
    assert {path.name for path in config.paths.staging_root.iterdir()} == {".tracker-data-staging"}
    reports = list(config.paths.reports_root.glob("*.json"))
    assert len(reports) == 1
    assert json.loads(reports[0].read_text())["cleanup_verified"] is True


def test_oversize_transfer_fails_before_packet_promotion(tmp_path: Path, make_config) -> None:
    large = tmp_path / "large.bin"
    large.write_bytes(b"x" * 10_000)
    manifest = write_manifest(tmp_path / "manifest.json", [negative_candidate("large", large)])
    config = make_config(
        manifest,
        max_images=1,
        max_download_bytes=2_000,
        max_temporary_bytes=2_000,
    )
    with pytest.raises(PipelineError, match="byte budget"):
        acquire(config)
    assert not any(config.paths.packets_root.rglob("packet.json"))
    assert {path.name for path in config.paths.staging_root.iterdir()} == {".tracker-data-staging"}


def test_packet_destination_is_immutable(tmp_path: Path, make_image, make_config) -> None:
    image = make_image(tmp_path / "image.png")
    manifest = write_manifest(tmp_path / "manifest.json", [negative_candidate("one", image)])
    config = make_config(manifest, max_images=1)
    acquire(config)
    with pytest.raises(PipelineError, match="already exists"):
        acquire(config)


def test_sequence_group_never_crosses_internal_split(
    tmp_path: Path, make_image, make_config
) -> None:
    first = make_image(tmp_path / "first.png")
    second = make_image(tmp_path / "second.png")
    candidates = [negative_candidate("first", first), negative_candidate("second", second)]
    for candidate in candidates:
        candidate["sequence_id"] = "recording-7"
        candidate["duplicate_group"] = candidate["source_image_id"]
    manifest = write_manifest(tmp_path / "manifest.json", candidates)
    result = acquire(make_config(manifest, max_images=2))
    splits = {record.internal_split for record in PacketReader(result.packet_path).records()}
    assert len(splits) == 1


def test_dry_run_has_no_filesystem_side_effects(tmp_path: Path, make_image, make_config) -> None:
    image = make_image(tmp_path / "image.png")
    manifest = write_manifest(tmp_path / "manifest.json", [negative_candidate("one", image)])
    config = make_config(manifest, max_images=1)
    plan = plan_acquisition(config)
    assert plan["network_access"] is False
    assert plan["filesystem_changes"] is False
    assert not config.paths.staging_root.exists()
    assert not config.paths.packets_root.exists()


def test_unmanaged_staging_is_never_deleted(tmp_path: Path, make_image, make_config) -> None:
    image = make_image(tmp_path / "image.png")
    manifest = write_manifest(tmp_path / "manifest.json", [negative_candidate("one", image)])
    config = make_config(manifest, max_images=1)
    config.paths.staging_root.mkdir(parents=True)
    precious = config.paths.staging_root / "do-not-delete.txt"
    precious.write_text("user data")
    with pytest.raises(PipelineError, match="non-empty unmarked"):
        acquire(config)
    assert precious.read_text() == "user data"


def test_orientation_mapping_covers_all_exif_variants() -> None:
    expected = {
        1: (0.2, 0.3),
        2: (0.8, 0.3),
        3: (0.8, 0.7),
        4: (0.2, 0.7),
        5: (0.3, 0.2),
        6: (0.7, 0.2),
        7: (0.7, 0.8),
        8: (0.3, 0.8),
    }
    for orientation, point in expected.items():
        assert orient_normalized_point(0.2, 0.3, orientation) == pytest.approx(point)


def test_saved_packet_contains_no_raw_archive_or_target_cache(
    tmp_path: Path, make_image, make_config
) -> None:
    image = make_image(tmp_path / "image.png")
    manifest = write_manifest(tmp_path / "manifest.json", [negative_candidate("one", image)])
    result = acquire(make_config(manifest, max_images=1))
    relative = {
        path.relative_to(result.packet_path).as_posix()
        for path in result.packet_path.rglob("*")
        if path.is_file()
    }
    assert not any(Path(name).suffix in {".zip", ".tar", ".npy", ".npz"} for name in relative)
    assert not any("heatmap" in name or "target" in name for name in relative)


def test_images_are_decodable_after_reader_validation(
    tmp_path: Path, make_image, make_config
) -> None:
    image = make_image(tmp_path / "image.png")
    manifest = write_manifest(tmp_path / "manifest.json", [negative_candidate("one", image)])
    reader = PacketReader(acquire(make_config(manifest, max_images=1)).packet_path)
    record = next(reader.records())
    with Image.open(reader.image_path(record)) as stored:
        stored.load()
        assert stored.size == (200, 100)
