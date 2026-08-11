from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError

from data_pipeline.config import StorageConfig
from data_pipeline.errors import PipelineError
from data_pipeline.records import (
    CandidateInstance,
    CanonicalInstance,
    CoordinateSpace,
    GeometryKind,
)


@dataclass(frozen=True)
class CanonicalImage:
    relative_path: str
    stored_sha256: str
    source_width: int
    source_height: int
    oriented_width: int
    oriented_height: int
    stored_width: int
    stored_height: int
    orientation: str
    instances: tuple[CanonicalInstance, ...]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fit_dimensions(width: int, height: int, max_width: int, max_height: int) -> tuple[int, int]:
    scale = min(1.0, max_width / width, max_height / height)
    return max(1, min(width, round(width * scale))), max(1, min(height, round(height * scale)))


def orient_normalized_point(x: float, y: float, orientation: int) -> tuple[float, float]:
    transforms = {
        1: (x, y),
        2: (1.0 - x, y),
        3: (1.0 - x, 1.0 - y),
        4: (x, 1.0 - y),
        5: (y, x),
        6: (1.0 - y, x),
        7: (1.0 - y, 1.0 - x),
        8: (y, 1.0 - x),
    }
    try:
        return transforms[orientation]
    except KeyError as error:
        raise PipelineError("orientation", f"unsupported EXIF orientation {orientation}") from error


def transform_instance(
    instance: CandidateInstance,
    *,
    original_width: int,
    original_height: int,
    orientation: int,
    stored_width: int,
    stored_height: int,
) -> CanonicalInstance:
    pairs = list(zip(instance.coordinates[::2], instance.coordinates[1::2], strict=True))
    if instance.coordinate_space == CoordinateSpace.PIXELS:
        pairs = [(x / original_width, y / original_height) for x, y in pairs]
    if instance.geometry_kind == GeometryKind.BBOX:
        (x0, y0), (x1, y1) = pairs
        pairs = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    oriented = [orient_normalized_point(x, y, orientation) for x, y in pairs]
    pixels = [
        (
            min(float(stored_width), max(0.0, x * stored_width)),
            min(float(stored_height), max(0.0, y * stored_height)),
        )
        for x, y in oriented
    ]
    if instance.geometry_kind == GeometryKind.BBOX:
        xs = [point[0] for point in pixels]
        ys = [point[1] for point in pixels]
        coordinates = (min(xs), min(ys), max(xs), max(ys))
        if not (coordinates[0] < coordinates[2] and coordinates[1] < coordinates[3]):
            raise PipelineError("geometry_area", "transformed bbox has zero area")
    else:
        coordinates = tuple(value for point in pixels for value in point)
    if not all(math.isfinite(value) for value in coordinates):
        raise PipelineError("geometry_finite", "transformed geometry is not finite")
    return CanonicalInstance(
        source_instance_id=instance.source_instance_id,
        semantic=instance.semantic,
        geometry_kind=instance.geometry_kind,
        coordinates=coordinates,
        quality=instance.quality,
        occluded=instance.occluded,
        truncated=instance.truncated,
        ignored=instance.ignored,
        uncertain=instance.uncertain,
        derivation_rule=instance.derivation_rule,
        parent_instance_id=instance.parent_instance_id,
    )


def _safe_image_name(source_image_id: str, codec: str) -> str:
    readable = re.sub(r"[^A-Za-z0-9._-]+", "-", source_image_id).strip(".-")[:48] or "image"
    suffix = {"jpeg": "jpg", "png": "png", "webp": "webp"}[codec]
    identity = hashlib.sha256(source_image_id.encode()).hexdigest()[:12]
    return f"{readable}-{identity}.{suffix}"


def canonicalize_image(
    source_path: Path,
    output_images: Path,
    source_image_id: str,
    instances: tuple[CandidateInstance, ...],
    storage: StorageConfig,
) -> CanonicalImage:
    try:
        with Image.open(source_path) as raw_image:
            original_width, original_height = raw_image.size
            orientation = int(raw_image.getexif().get(274, 1))
            image = ImageOps.exif_transpose(raw_image)
            image.load()
    except (UnidentifiedImageError, OSError, ValueError) as error:
        raise PipelineError("image_decode", f"cannot decode image {source_image_id}") from error
    oriented_width, oriented_height = image.size
    stored_width, stored_height = fit_dimensions(
        oriented_width, oriented_height, storage.max_width, storage.max_height
    )
    if (stored_width, stored_height) != image.size:
        image = image.resize((stored_width, stored_height), Image.Resampling.LANCZOS)
    image = image.convert("RGB")
    output_images.mkdir(parents=True, exist_ok=True)
    filename = _safe_image_name(source_image_id, storage.codec)
    output_path = output_images / filename
    save_options: dict[str, object]
    if storage.codec == "webp":
        save_options = {
            "format": "WEBP",
            "lossless": storage.lossless,
            "quality": storage.quality,
            "method": 6,
        }
    elif storage.codec == "jpeg":
        save_options = {
            "format": "JPEG",
            "quality": storage.quality,
            "optimize": True,
            "subsampling": 0,
        }
    else:
        save_options = {"format": "PNG", "optimize": True}
    image.save(output_path, **save_options)
    canonical_instances = tuple(
        transform_instance(
            instance,
            original_width=original_width,
            original_height=original_height,
            orientation=orientation,
            stored_width=stored_width,
            stored_height=stored_height,
        )
        for instance in instances
    )
    return CanonicalImage(
        relative_path=f"images/{filename}",
        stored_sha256=sha256_file(output_path),
        source_width=original_width,
        source_height=original_height,
        oriented_width=oriented_width,
        oriented_height=oriented_height,
        stored_width=stored_width,
        stored_height=stored_height,
        orientation=f"exif-{orientation}-normalized",
        instances=canonical_instances,
    )
