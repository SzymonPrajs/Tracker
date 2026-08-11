import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from data_pipeline.config import PipelineConfig, load_config
from data_pipeline.records import (
    CandidateInstance,
    CoordinateSpace,
    GeometryKind,
    GeometryQuality,
    Semantic,
    SourceCandidate,
)


def test_bbox_requires_positive_area() -> None:
    with pytest.raises(ValidationError, match="positive area"):
        CandidateInstance(
            source_instance_id="box",
            semantic=Semantic.HEAD_VISIBLE,
            geometry_kind=GeometryKind.BBOX,
            coordinates=(0.5, 0.2, 0.5, 0.8),
            coordinate_space=CoordinateSpace.NORMALIZED,
            quality=GeometryQuality.EXACT,
        )


def test_source_id_cannot_be_a_path() -> None:
    with pytest.raises(ValidationError, match="opaque ID"):
        SourceCandidate(
            source_image_id="../../escape",
            image_url="file:///tmp/image.png",
            source_split="train",
            instances=(
                CandidateInstance(
                    source_instance_id="box",
                    semantic=Semantic.HEAD_VISIBLE,
                    geometry_kind=GeometryKind.BBOX,
                    coordinates=(0.1, 0.1, 0.2, 0.2),
                    coordinate_space=CoordinateSpace.NORMALIZED,
                    quality=GeometryQuality.EXACT,
                ),
            ),
        )


def test_config_rejects_unknown_fields(make_config, tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text('{"schema_version":"tracker.source-manifest.v1","candidates":[]}')
    raw = make_config(manifest).model_dump(mode="json")
    raw["surprise"] = True
    with pytest.raises(ValidationError, match="surprise"):
        PipelineConfig.model_validate(raw)


def test_relative_paths_resolve_from_config_file(tmp_path: Path) -> None:
    config_path = tmp_path / "config" / "pipeline.json"
    config_path.parent.mkdir()
    config_path.write_text(
        json.dumps(
            {
                "schema_version": "tracker.data.v1",
                "research_authorization_id": "test",
                "source": {
                    "kind": "manifest",
                    "name": "x",
                    "version": "1",
                    "source_url": "x",
                    "settings": {"manifest_url": "file:///x"},
                },
                "storage": {"max_width": 10, "max_height": 10},
                "limits": {
                    "max_selected_images": 1,
                    "max_download_bytes": 10,
                    "max_temporary_bytes": 10,
                    "max_selected_packet_bytes": 10,
                    "max_corpus_bytes": 10,
                    "min_free_bytes": 0,
                },
                "paths": {
                    "staging_root": "../../staging",
                    "packets_root": "../../packets",
                    "reports_root": "../../reports",
                },
                "selection_seed": 1,
                "internal_split_seed": 2,
                "internal_validation_fraction": 0,
            }
        ),
        encoding="utf-8",
    )
    config = load_config(config_path)
    assert config.paths.staging_root == tmp_path.parent / "staging"
