from __future__ import annotations

import json
import os
import shutil
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from data_pipeline.config import PipelineConfig
from data_pipeline.errors import PipelineError
from data_pipeline.packets import PacketBuilder, packet_identity, validate_packet
from data_pipeline.packets.build import write_json
from data_pipeline.sources import create_adapter
from data_pipeline.transfer import download_image

STAGING_MARKER = ".tracker-data-staging"
ACTIVE_LOCK = ".active.lock"


@dataclass(frozen=True)
class AcquisitionResult:
    packet_path: Path
    run_report_path: Path
    validation: dict[str, object]
    cleanup_verified: bool

    def as_dict(self) -> dict[str, object]:
        return {
            "status": "passed",
            "packet_path": str(self.packet_path),
            "run_report_path": str(self.run_report_path),
            "validation": self.validation,
            "cleanup_verified": self.cleanup_verified,
        }


def plan_acquisition(config: PipelineConfig) -> dict[str, object]:
    create_adapter(config.source.kind, config.source.settings)
    packet_id, config_hash = packet_identity(config)
    destination = config.paths.packets_root / config.source.name / packet_id
    return {
        "status": "dry-run",
        "source": config.source.name,
        "adapter": config.source.kind,
        "packet_id": packet_id,
        "config_sha256": config_hash,
        "destination": str(destination),
        "max_selected_images": config.limits.max_selected_images,
        "storage_profile": config.storage.profile_id,
        "network_access": False,
        "filesystem_changes": False,
    }


def _ensure_plain_directory(path: Path) -> None:
    if path.exists() and path.is_symlink():
        raise PipelineError("root_symlink", f"managed root cannot be a symlink: {path}")
    path.mkdir(parents=True, exist_ok=True)
    if not path.is_dir():
        raise PipelineError("root_type", f"managed root is not a directory: {path}")


def _prepare_staging(root: Path, run_id: str) -> tuple[Path, Path]:
    _ensure_plain_directory(root)
    marker = root / STAGING_MARKER
    existing = {entry.name for entry in root.iterdir()}
    if not marker.exists():
        if existing:
            raise PipelineError(
                "staging_unmanaged",
                "refusing to adopt a non-empty unmarked staging directory",
                entries=sorted(existing),
            )
        marker.write_text("Tracker data staging root v1\n", encoding="utf-8")
    elif (
        marker.is_symlink()
        or marker.read_text(encoding="utf-8") != "Tracker data staging root v1\n"
    ):
        raise PipelineError("staging_marker", "staging marker is missing or invalid")
    unexpected = {entry.name for entry in root.iterdir()} - {STAGING_MARKER}
    if unexpected:
        raise PipelineError(
            "staging_not_empty",
            "prior raw staging must be empty before another source starts",
            entries=sorted(unexpected),
        )
    lock = root / ACTIVE_LOCK
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as error:
        raise PipelineError(
            "staging_locked", "another acquisition owns the staging root"
        ) from error
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(json.dumps({"pid": os.getpid(), "run_id": run_id}) + "\n")
    stage = root / f"{run_id}--source"
    try:
        stage.mkdir()
    except Exception:
        lock.unlink(missing_ok=True)
        raise
    return stage, lock


def _tree_bytes(root: Path) -> int:
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file())


def _guarded_remove(child: Path, expected_parent: Path, name_fragment: str) -> None:
    if not child.exists():
        return
    if child.is_symlink() or child.resolve().parent != expected_parent.resolve():
        raise PipelineError("cleanup_scope", f"refusing unsafe cleanup target: {child}")
    if name_fragment not in child.name:
        raise PipelineError("cleanup_scope", f"cleanup target lacks run identity: {child}")
    shutil.rmtree(child)


def _write_run_report(path: Path, report: dict[str, Any]) -> None:
    write_json(path, report)


def acquire(config: PipelineConfig) -> AcquisitionResult:
    adapter = create_adapter(config.source.kind, config.source.settings)
    packet_id, config_hash = packet_identity(config)
    run_id = f"{config.source.name}-{uuid.uuid4().hex}"
    report_path = config.paths.reports_root / f"{run_id}.json"
    started = time.time()
    for root in (config.paths.packets_root, config.paths.reports_root):
        _ensure_plain_directory(root)
    destination_parent = config.paths.packets_root / config.source.name
    _ensure_plain_directory(destination_parent)
    destination = destination_parent / packet_id
    if destination.exists():
        raise PipelineError(
            "packet_immutable",
            "packet destination already exists; change selection or version instead of mutating it",
            destination=str(destination),
        )
    corpus_bytes = _tree_bytes(config.paths.packets_root)
    if corpus_bytes + config.limits.max_selected_packet_bytes > config.limits.max_corpus_bytes:
        raise PipelineError(
            "corpus_limit",
            "configured packet could exceed the total corpus cap",
            corpus_bytes=corpus_bytes,
            packet_cap=config.limits.max_selected_packet_bytes,
            corpus_cap=config.limits.max_corpus_bytes,
        )
    _ensure_plain_directory(config.paths.staging_root)
    free_bytes = shutil.disk_usage(config.paths.staging_root).free
    required_free = (
        config.limits.min_free_bytes
        + config.limits.max_temporary_bytes
        + config.limits.max_selected_packet_bytes
    )
    if free_bytes < required_free:
        raise PipelineError(
            "free_space",
            "insufficient free space for declared staging, packet, and reserve caps",
            free_bytes=free_bytes,
            required_free_bytes=required_free,
        )

    stage: Path | None = None
    lock: Path | None = None
    partial = destination_parent / f".{packet_id}.{run_id}.partial"
    failure: PipelineError | None = None
    cleanup_verified = False
    validation: dict[str, object] | None = None
    total_downloaded = 0
    selected_count = 0
    try:
        stage, lock = _prepare_staging(config.paths.staging_root, run_id)
        discovery = adapter.discover(
            staging_dir=stage,
            max_candidates=config.limits.max_selected_images,
            seed=config.selection_seed,
            max_metadata_bytes=config.limits.max_download_bytes,
            timeout_seconds=config.request_timeout_seconds,
        )
        selected_count = len(discovery.candidates)
        total_downloaded = sum(path.stat().st_size for path in discovery.metadata_files)
        if total_downloaded > config.limits.max_download_bytes:
            raise PipelineError("download_limit", "metadata consumed the complete download budget")
        if _tree_bytes(stage) > config.limits.max_temporary_bytes:
            raise PipelineError("temporary_limit", "metadata exceeded the temporary storage cap")
        builder = PacketBuilder(partial, config, discovery)
        for index, candidate in enumerate(discovery.candidates):
            raw_path = stage / "pixels" / f"selected-{index:08d}.source"
            try:
                transfer = download_image(
                    candidate.image_url,
                    raw_path,
                    byte_limit=config.limits.max_download_bytes - total_downloaded,
                    timeout_seconds=config.request_timeout_seconds,
                    expected_sha256=candidate.expected_sha256,
                )
                total_downloaded += transfer.byte_count
                if _tree_bytes(stage) > config.limits.max_temporary_bytes:
                    raise PipelineError(
                        "temporary_limit", "active source staging exceeded its byte cap"
                    )
                builder.add(candidate, raw_path, transfer)
            finally:
                raw_path.unlink(missing_ok=True)
        validation = builder.finalize()
    except PipelineError as error:
        failure = error
    except Exception as error:
        failure = PipelineError("unexpected", f"{type(error).__name__}: {error}")
    finally:
        cleanup_error: PipelineError | None = None
        try:
            if stage is not None:
                _guarded_remove(stage, config.paths.staging_root, run_id)
            cleanup_verified = stage is not None and not stage.exists()
        except PipelineError as error:
            cleanup_error = error
        if lock is not None:
            lock.unlink(missing_ok=True)
        if cleanup_error is not None:
            failure = cleanup_error

    if failure is not None or validation is None or not cleanup_verified:
        partial_cleanup_error: PipelineError | None = None
        try:
            _guarded_remove(partial, destination_parent, run_id)
        except PipelineError as error:
            partial_cleanup_error = error
        if partial_cleanup_error is not None:
            failure = partial_cleanup_error
        report = {
            "status": "failed",
            "run_id": run_id,
            "source": config.source.name,
            "packet_id": packet_id,
            "config_sha256": config_hash,
            "selected_images": selected_count,
            "downloaded_bytes": total_downloaded,
            "cleanup_verified": cleanup_verified,
            "staging_directory_exists": bool(stage and stage.exists()),
            "elapsed_seconds": time.time() - started,
            "error": (failure or PipelineError("cleanup", "cleanup was not verified")).as_dict(),
        }
        _write_run_report(report_path, report)
        raise failure or PipelineError("cleanup", "raw staging cleanup was not verified")

    try:
        os.replace(partial, destination)
        validation = validate_packet(destination)
    except Exception as error:
        if partial.exists():
            _guarded_remove(partial, destination_parent, run_id)
        failure = (
            error if isinstance(error, PipelineError) else PipelineError("promotion", str(error))
        )
        _write_run_report(
            report_path,
            {
                "status": "failed",
                "run_id": run_id,
                "source": config.source.name,
                "packet_id": packet_id,
                "cleanup_verified": cleanup_verified,
                "error": failure.as_dict(),
            },
        )
        if isinstance(error, PipelineError):
            raise
        raise failure from error

    _write_run_report(
        report_path,
        {
            "status": "passed",
            "run_id": run_id,
            "source": config.source.name,
            "packet_id": packet_id,
            "config_sha256": config_hash,
            "packet_path": str(destination),
            "selected_images": selected_count,
            "downloaded_bytes": total_downloaded,
            "packet_bytes": _tree_bytes(destination),
            "cleanup_verified": True,
            "staging_directory_exists": False,
            "elapsed_seconds": time.time() - started,
            "validation": validation,
        },
    )
    return AcquisitionResult(destination, report_path, validation, True)
