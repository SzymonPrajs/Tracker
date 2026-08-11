from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from data_pipeline.records import SourceCandidate


@dataclass(frozen=True)
class DiscoveryResult:
    candidates: tuple[SourceCandidate, ...]
    metadata_files: tuple[Path, ...]
    metadata_sha256: dict[str, str]
    source_counts: dict[str, int]
    selection: dict[str, object]


class SourceAdapter(Protocol):
    def discover(
        self,
        *,
        staging_dir: Path,
        max_candidates: int,
        seed: int,
        max_metadata_bytes: int,
        timeout_seconds: float,
    ) -> DiscoveryResult: ...
