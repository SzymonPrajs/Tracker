from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import PIL

from data_pipeline.config import SCHEMA_VERSION, PipelineConfig
from data_pipeline.errors import PipelineError
from data_pipeline.images import canonicalize_image
from data_pipeline.packets.validate import sha256_file, validate_packet
from data_pipeline.records import ImageRecord, SourceCandidate
from data_pipeline.sources.base import DiscoveryResult
from data_pipeline.transfer import TransferResult


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")


def stable_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def packet_identity(config: PipelineConfig) -> tuple[str, str]:
    config_hash = stable_hash(config.logical_dict())
    return f"{config.source.version}-{config_hash[:12]}", config_hash


def _git_revision() -> dict[str, object]:
    repository = Path(__file__).resolve().parents[3]
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain", "--untracked-files=no"],
                cwd=repository,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        return {"git_commit": commit, "tracked_tree_dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"git_commit": None, "tracked_tree_dirty": None}


def _internal_split(candidate: SourceCandidate, config: PipelineConfig) -> str:
    if candidate.source_split != "train" or config.internal_validation_fraction == 0:
        return candidate.source_split
    group = (
        candidate.sequence_id
        or candidate.camera_id
        or candidate.scene_id
        or candidate.duplicate_group
        or candidate.source_image_id
    )
    digest = hashlib.sha256(f"{config.internal_split_seed}:{group}".encode()).digest()
    fraction = int.from_bytes(digest[:8], "big") / 2**64
    return (
        "internal_validation"
        if fraction < config.internal_validation_fraction
        else "internal_train"
    )


class PacketBuilder:
    """Build one unpromoted packet; acquisition owns staging and atomic promotion."""

    def __init__(
        self,
        partial_root: Path,
        config: PipelineConfig,
        discovery: DiscoveryResult,
    ) -> None:
        self.root = partial_root
        self.config = config
        self.discovery = discovery
        self.packet_id, self.config_hash = packet_identity(config)
        self.records: list[ImageRecord] = []
        self.root.mkdir(parents=True, exist_ok=False)

    def byte_count(self) -> int:
        return sum(path.stat().st_size for path in self.root.rglob("*") if path.is_file())

    def _check_size(self) -> None:
        size = self.byte_count()
        if size > self.config.limits.max_selected_packet_bytes:
            raise PipelineError(
                "packet_limit",
                "canonical packet exceeded its configured byte cap",
                bytes=size,
                limit=self.config.limits.max_selected_packet_bytes,
            )

    def add(
        self,
        candidate: SourceCandidate,
        downloaded_path: Path,
        transfer: TransferResult,
    ) -> None:
        canonical = canonicalize_image(
            downloaded_path,
            self.root / "images",
            candidate.source_image_id,
            candidate.instances,
            self.config.storage,
        )
        record = ImageRecord(
            source=self.config.source.name,
            source_version=self.config.source.version,
            source_url=transfer.final_url,
            source_image_id=candidate.source_image_id,
            source_sha256=transfer.sha256,
            source_width=canonical.source_width,
            source_height=canonical.source_height,
            oriented_width=canonical.oriented_width,
            oriented_height=canonical.oriented_height,
            orientation=canonical.orientation,
            stored_path=canonical.relative_path,
            stored_sha256=canonical.stored_sha256,
            stored_width=canonical.stored_width,
            stored_height=canonical.stored_height,
            storage_profile=self.config.storage.profile_id,
            sequence_id=candidate.sequence_id,
            camera_id=candidate.camera_id,
            scene_id=candidate.scene_id,
            duplicate_group=candidate.duplicate_group
            or f"{self.config.source.name}:{candidate.source_image_id}",
            source_split=candidate.source_split,
            internal_split=_internal_split(candidate, self.config),
            research_authorization_id=self.config.research_authorization_id,
            acquisition_checksum=transfer.sha256,
            coverage_records=candidate.coverage_records,
            native_instances=candidate.instances,
            instances=canonical.instances,
        )
        self.records.append(record)
        self._check_size()

    def finalize(self) -> dict[str, object]:
        if not self.records:
            raise PipelineError("empty_selection", "source adapter selected no usable images")
        records_path = self.root / "records.jsonl"
        with records_path.open("w", encoding="utf-8") as handle:
            for record in self.records:
                handle.write(record.model_dump_json())
                handle.write("\n")
        selection = dict(self.discovery.selection)
        selection["selected_ids"] = [record.source_image_id for record in self.records]
        write_json(self.root / "selection.json", selection)
        write_json(
            self.root / "reports" / "validation.json",
            {
                "status": "passed",
                "checks": [
                    "source IDs unique and path-safe",
                    "images decoded and fit without crop or upscale",
                    "vector geometry transformed with orientation and resize",
                    "packet and download caps enforced",
                    "record and selection counts reconciled",
                ],
                "images": len(self.records),
                "instances": sum(len(record.instances) for record in self.records),
            },
        )
        (self.root / "README.md").write_text(
            f"# {self.config.source.name} packet {self.packet_id}\n\n"
            "This immutable packet stores one canonical image per selected source image and "
            "provenance-preserving vector labels. It contains no raw archive, extraction tree, "
            "heatmap, augmented copy, or model-sized target cache.\n\n"
            "Validate it with `python -m data_pipeline validate PATH_TO_THIS_PACKET`.\n",
            encoding="utf-8",
        )
        created_files = {
            "README.md",
            "packet.json",
            "selection.json",
            "records.jsonl",
            "reports/validation.json",
            "checksums.sha256",
            *(record.stored_path for record in self.records),
        }
        packet = {
            "schema_version": SCHEMA_VERSION,
            "status": "complete",
            "packet_id": self.packet_id,
            "producer": {
                "name": "data_pipeline",
                "version": "0.1.0",
                "pillow_version": PIL.__version__,
                **_git_revision(),
            },
            "source": {
                "name": self.config.source.name,
                "version": self.config.source.version,
                "url": self.config.source.source_url,
                "adapter": self.config.source.kind,
                "metadata_sha256": self.discovery.metadata_sha256,
                "source_counts": self.discovery.source_counts,
            },
            "resolved_logical_config": self.config.logical_dict(),
            "config_sha256": self.config_hash,
            "storage": self.config.storage.model_dump(mode="json"),
            "counts": {
                "images": len(self.records),
                "instances": sum(len(record.instances) for record in self.records),
                "native_instances": sum(len(record.native_instances) for record in self.records),
                "coverage_records": sum(len(record.coverage_records) for record in self.records),
            },
            "created_files": sorted(created_files),
        }
        write_json(self.root / "packet.json", packet)
        checksum_paths = sorted(created_files - {"checksums.sha256"})
        with (self.root / "checksums.sha256").open("w", encoding="utf-8") as handle:
            for relative in checksum_paths:
                handle.write(f"{sha256_file(self.root / relative)}  {relative}\n")
        self._check_size()
        return validate_packet(self.root)
