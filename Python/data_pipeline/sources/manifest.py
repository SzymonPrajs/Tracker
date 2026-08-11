from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from data_pipeline.records import SourceCandidate
from data_pipeline.sources.base import DiscoveryResult
from data_pipeline.transfer import download_metadata


class ManifestSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    manifest_url: str


class ManifestDocument(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str
    candidates: tuple[SourceCandidate, ...]


def _rank(seed: int, source_image_id: str) -> str:
    return hashlib.sha256(f"{seed}:{source_image_id}".encode()).hexdigest()


class ManifestAdapter:
    """Read a strict prebuilt source manifest without interpreting its semantics."""

    def __init__(self, settings: dict[str, object]) -> None:
        self.settings = ManifestSettings.model_validate(settings)

    def discover(
        self,
        *,
        staging_dir: Path,
        max_candidates: int,
        seed: int,
        max_metadata_bytes: int,
        timeout_seconds: float,
    ) -> DiscoveryResult:
        path = staging_dir / "metadata" / "source-manifest.json"
        transfer = download_metadata(
            self.settings.manifest_url,
            path,
            byte_limit=max_metadata_bytes,
            timeout_seconds=timeout_seconds,
        )
        with path.open("r", encoding="utf-8") as handle:
            document = ManifestDocument.model_validate(json.load(handle))
        if document.schema_version != "tracker.source-manifest.v1":
            raise ValueError(f"unsupported source manifest schema {document.schema_version!r}")
        identifiers = [candidate.source_image_id for candidate in document.candidates]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("source manifest contains duplicate source_image_id values")
        selected = tuple(
            sorted(document.candidates, key=lambda item: _rank(seed, item.source_image_id))[
                :max_candidates
            ]
        )
        return DiscoveryResult(
            candidates=selected,
            metadata_files=(path,),
            metadata_sha256={path.name: transfer.sha256},
            source_counts={"manifest_candidates": len(document.candidates)},
            selection={
                "algorithm": "sha256(seed:source_image_id)",
                "seed": seed,
                "available": len(document.candidates),
                "selected": len(selected),
                "selected_ids": [item.source_image_id for item in selected],
            },
        )
