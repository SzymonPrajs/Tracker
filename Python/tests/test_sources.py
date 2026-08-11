from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from data_pipeline.sources.manifest import ManifestAdapter
from data_pipeline.sources.open_images import OpenImagesAdapter


def _write_csv(path: Path, fields: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_open_images_keeps_head_face_and_negative_evidence_distinct(tmp_path: Path) -> None:
    classes = tmp_path / "classes.csv"
    classes.write_text(
        "/m/head,Human head\n/m/head-alt,Alternate head\n/m/face,Human face\n",
        encoding="utf-8",
    )
    boxes = tmp_path / "boxes.csv"
    fields = [
        "ImageID",
        "LabelName",
        "XMin",
        "XMax",
        "YMin",
        "YMax",
        "IsOccluded",
        "IsTruncated",
        "IsGroupOf",
        "IsDepiction",
        "IsInside",
    ]
    _write_csv(
        boxes,
        fields,
        [
            {
                "ImageID": "positive",
                "LabelName": "/m/head",
                "XMin": 0.1,
                "XMax": 0.3,
                "YMin": 0.2,
                "YMax": 0.5,
                "IsOccluded": 0,
                "IsTruncated": 0,
                "IsGroupOf": 0,
                "IsDepiction": 0,
                "IsInside": 0,
            }
        ],
    )
    labels = tmp_path / "labels.csv"
    _write_csv(
        labels,
        ["ImageID", "Source", "LabelName", "Confidence"],
        [
            {
                "ImageID": "negative",
                "Source": "verification",
                "LabelName": "/m/head",
                "Confidence": 0,
            }
        ],
    )
    adapter = OpenImagesAdapter(
        {
            "split": "validation",
            "boxes_url": boxes.as_uri(),
            "image_labels_url": labels.as_uri(),
            "class_descriptions_url": classes.as_uri(),
            "class_semantics": {
                "/m/head": "head_visible",
                "/m/head-alt": "head_visible",
                "/m/face": "face_visible",
            },
            "verified_negative_mids": ["/m/head"],
            "negative_fraction": 0.5,
        }
    )
    result = adapter.discover(
        staging_dir=tmp_path / "staging",
        max_candidates=2,
        seed=7,
        max_metadata_bytes=100_000,
        timeout_seconds=5,
    )
    assert {candidate.source_image_id for candidate in result.candidates} == {
        "positive",
        "negative",
    }
    positive = next(item for item in result.candidates if item.source_image_id == "positive")
    assert positive.instances[0].semantic.value == "head_visible"
    positive_coverage = {
        record.exact_class_or_mid: record.status.value for record in positive.coverage_records
    }
    assert positive_coverage["/m/head"] == "positive_exhaustive"
    assert positive_coverage["/m/head-alt"] == "unknown"
    negative = next(item for item in result.candidates if item.source_image_id == "negative")
    assert any(record.status.value == "verified_absent" for record in negative.coverage_records)
    assert any(record.status.value == "unknown" for record in negative.coverage_records)


def test_manifest_selection_is_deterministic(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    candidates = [
        {
            "source_image_id": f"image-{index}",
            "image_url": (tmp_path / f"image-{index}.png").as_uri(),
            "source_split": "train",
            "coverage_records": [
                {
                    "exact_class_or_mid": "head",
                    "status": "verified_absent",
                    "evidence_origin": "fixture",
                }
            ],
        }
        for index in range(5)
    ]
    manifest.write_text(
        json.dumps({"schema_version": "tracker.source-manifest.v1", "candidates": candidates})
    )
    adapter = ManifestAdapter({"manifest_url": manifest.as_uri()})
    first = adapter.discover(
        staging_dir=tmp_path / "a",
        max_candidates=3,
        seed=19,
        max_metadata_bytes=100_000,
        timeout_seconds=5,
    )
    second = adapter.discover(
        staging_dir=tmp_path / "b",
        max_candidates=3,
        seed=19,
        max_metadata_bytes=100_000,
        timeout_seconds=5,
    )
    assert first.selection["selected_ids"] == second.selection["selected_ids"]


def test_adapter_rejects_unknown_settings() -> None:
    with pytest.raises(ValidationError, match="surprise"):
        ManifestAdapter({"manifest_url": "file:///x", "surprise": True})
