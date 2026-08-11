from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from PIL import Image

from data_pipeline.config import SCHEMA_VERSION
from data_pipeline.errors import PipelineError
from data_pipeline.records import GeometryKind, ImageRecord

FORBIDDEN_SUFFIXES = {
    ".zip",
    ".tar",
    ".tgz",
    ".gz",
    ".bz2",
    ".7z",
    ".rar",
    ".npy",
    ".npz",
}
REQUIRED_FILES = {
    "README.md",
    "packet.json",
    "selection.json",
    "records.jsonl",
    "reports/validation.json",
    "checksums.sha256",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _contained_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for directory, dirnames, filenames in os.walk(root, followlinks=False):
        directory_path = Path(directory)
        for name in dirnames:
            if (directory_path / name).is_symlink():
                raise PipelineError("packet_symlink", f"packet contains symlink directory: {name}")
        for name in filenames:
            path = directory_path / name
            if path.is_symlink():
                raise PipelineError("packet_symlink", f"packet contains symlink file: {path.name}")
            files.append(path)
    return files


def _load_checksums(path: Path) -> dict[str, str]:
    checksums: dict[str, str] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line:
            continue
        try:
            digest, relative = line.split("  ", 1)
        except ValueError as error:
            raise PipelineError(
                "checksum_format", f"invalid checksum line {line_number}"
            ) from error
        relative_path = Path(relative)
        if (
            relative_path.is_absolute()
            or ".." in relative_path.parts
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise PipelineError("checksum_format", f"unsafe checksum line {line_number}")
        if relative in checksums:
            raise PipelineError("checksum_duplicate", f"duplicate checksum path {relative}")
        checksums[relative] = digest
    return checksums


def validate_packet(root: Path) -> dict[str, object]:
    root = root.resolve(strict=True)
    if not root.is_dir():
        raise PipelineError("packet_type", "packet path is not a directory")
    files = _contained_files(root)
    relative_files = {path.relative_to(root).as_posix() for path in files}
    missing = REQUIRED_FILES - relative_files
    if missing:
        raise PipelineError(
            "packet_missing", "packet is missing required files", missing=sorted(missing)
        )
    forbidden = sorted(
        name for name in relative_files if Path(name).suffix.lower() in FORBIDDEN_SUFFIXES
    )
    if forbidden:
        raise PipelineError(
            "packet_raw_artifact",
            "packet contains forbidden raw archives or raster caches",
            files=forbidden,
        )

    checksums = _load_checksums(root / "checksums.sha256")
    expected_checksum_paths = relative_files - {"checksums.sha256"}
    if set(checksums) != expected_checksum_paths:
        raise PipelineError(
            "checksum_inventory",
            "checksum inventory does not exactly match packet files",
            missing=sorted(expected_checksum_paths - set(checksums)),
            extra=sorted(set(checksums) - expected_checksum_paths),
        )
    for relative, expected in checksums.items():
        actual = sha256_file(root / relative)
        if actual != expected:
            raise PipelineError(
                "checksum_mismatch",
                f"checksum mismatch for {relative}",
                expected=expected,
                actual=actual,
            )

    packet = json.loads((root / "packet.json").read_text(encoding="utf-8"))
    if packet.get("schema_version") != SCHEMA_VERSION or packet.get("status") != "complete":
        raise PipelineError("packet_manifest", "packet schema or status is invalid")
    if set(packet.get("created_files", ())) != relative_files:
        raise PipelineError("packet_inventory", "packet.json created_files does not match packet")
    selection = json.loads((root / "selection.json").read_text(encoding="utf-8"))
    build_report = json.loads((root / "reports/validation.json").read_text(encoding="utf-8"))
    if build_report.get("status") != "passed":
        raise PipelineError("build_validation", "builder did not record a passing validation")

    records: list[ImageRecord] = []
    with (root / "records.jsonl").open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                records.append(ImageRecord.model_validate_json(line))
            except Exception as error:
                raise PipelineError(
                    "record_schema", f"invalid record at line {line_number}"
                ) from error
    if len(records) != packet.get("counts", {}).get("images"):
        raise PipelineError("record_count", "record count does not match packet manifest")
    record_ids = [record.source_image_id for record in records]
    if len(record_ids) != len(set(record_ids)):
        raise PipelineError("record_duplicate", "packet contains duplicate source image IDs")
    if record_ids != selection.get("selected_ids"):
        raise PipelineError("selection_mismatch", "records differ from frozen selected ID order")

    stored_paths: set[str] = set()
    instance_count = 0
    for record in records:
        if record.stored_path in stored_paths:
            raise PipelineError("image_duplicate", f"duplicate stored path {record.stored_path}")
        stored_paths.add(record.stored_path)
        image_path = root / record.stored_path
        if not image_path.is_file() or sha256_file(image_path) != record.stored_sha256:
            raise PipelineError(
                "image_checksum", f"stored image differs for {record.source_image_id}"
            )
        with Image.open(image_path) as image:
            image.load()
            if image.size != (record.stored_width, record.stored_height):
                raise PipelineError(
                    "image_dimensions", f"stored dimensions differ for {record.source_image_id}"
                )
        max_width = int(packet["storage"]["max_width"])
        max_height = int(packet["storage"]["max_height"])
        if record.stored_width > max_width or record.stored_height > max_height:
            raise PipelineError(
                "storage_envelope", f"image exceeds storage envelope: {record.source_image_id}"
            )
        if len(record.native_instances) != len(record.instances):
            raise PipelineError(
                "native_reconciliation",
                f"native and canonical instance counts differ: {record.source_image_id}",
            )
        native_ids = [instance.source_instance_id for instance in record.native_instances]
        canonical_ids = [instance.source_instance_id for instance in record.instances]
        if native_ids != canonical_ids:
            raise PipelineError(
                "native_reconciliation",
                f"native and canonical instance identities differ: {record.source_image_id}",
            )
        for instance in record.instances:
            instance_count += 1
            xs = instance.coordinates[::2]
            ys = instance.coordinates[1::2]
            if any(x < 0 or x > record.stored_width for x in xs) or any(
                y < 0 or y > record.stored_height for y in ys
            ):
                raise PipelineError(
                    "geometry_bounds", f"geometry outside image: {record.source_image_id}"
                )
            if instance.geometry_kind == GeometryKind.BBOX:
                x0, y0, x1, y1 = instance.coordinates
                if not (x0 < x1 and y0 < y1):
                    raise PipelineError(
                        "geometry_area", f"zero-area bbox: {record.source_image_id}"
                    )
    if instance_count != packet.get("counts", {}).get("instances"):
        raise PipelineError("instance_count", "instance count differs from packet manifest")
    extra_images = {name for name in relative_files if name.startswith("images/")} - stored_paths
    if extra_images:
        raise PipelineError(
            "image_orphan", "packet contains unreferenced images", files=sorted(extra_images)
        )

    return {
        "status": "passed",
        "schema_version": SCHEMA_VERSION,
        "packet_id": packet["packet_id"],
        "images": len(records),
        "instances": instance_count,
        "files": len(relative_files),
        "packet_manifest_sha256": sha256_file(root / "packet.json"),
    }
