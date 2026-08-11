from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pytest
from PIL import Image

from data_pipeline.config import PipelineConfig


@pytest.fixture
def make_image() -> Callable[[Path, tuple[int, int], tuple[int, int, int]], Path]:
    def create(
        path: Path,
        size: tuple[int, int] = (640, 320),
        color: tuple[int, int, int] = (40, 80, 120),
    ) -> Path:
        Image.new("RGB", size, color).save(path)
        return path

    return create


@pytest.fixture
def make_config(tmp_path: Path) -> Callable[..., PipelineConfig]:
    def create(
        manifest: Path,
        *,
        max_images: int = 8,
        max_download_bytes: int = 4_000_000,
        max_temporary_bytes: int = 4_000_000,
        packet_version: str = "fixture-v1",
    ) -> PipelineConfig:
        return PipelineConfig.model_validate(
            {
                "schema_version": "tracker.data.v1",
                "research_authorization_id": "test-fixture",
                "source": {
                    "kind": "manifest",
                    "name": "fixture",
                    "version": packet_version,
                    "source_url": "https://example.invalid/fixture",
                    "settings": {"manifest_url": manifest.as_uri()},
                },
                "storage": {
                    "max_width": 200,
                    "max_height": 200,
                    "codec": "png",
                    "quality": 92,
                    "lossless": True,
                },
                "limits": {
                    "max_selected_images": max_images,
                    "max_download_bytes": max_download_bytes,
                    "max_temporary_bytes": max_temporary_bytes,
                    "max_selected_packet_bytes": 4_000_000,
                    "max_corpus_bytes": 20_000_000,
                    "min_free_bytes": 0,
                },
                "paths": {
                    "staging_root": str(tmp_path / "data" / "staging"),
                    "packets_root": str(tmp_path / "data" / "packets"),
                    "reports_root": str(tmp_path / "data" / "reports"),
                },
                "selection_seed": 17,
                "internal_split_seed": 23,
                "internal_validation_fraction": 0.2,
                "request_timeout_seconds": 5,
            }
        )

    return create


def write_manifest(path: Path, candidates: list[dict[str, object]]) -> Path:
    path.write_text(
        json.dumps({"schema_version": "tracker.source-manifest.v1", "candidates": candidates}),
        encoding="utf-8",
    )
    return path


def negative_candidate(image_id: str, image: Path) -> dict[str, object]:
    return {
        "source_image_id": image_id,
        "image_url": image.as_uri(),
        "source_split": "train",
        "coverage_records": [
            {
                "exact_class_or_mid": "head",
                "status": "verified_absent",
                "evidence_origin": "hand-checked fixture",
            }
        ],
    }
