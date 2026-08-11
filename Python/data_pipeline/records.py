from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, PositiveInt, field_validator, model_validator


class StrictRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Semantic(StrEnum):
    HEAD_FULL = "head_full"
    HEAD_VISIBLE = "head_visible"
    FACE_VISIBLE = "face_visible"
    PERSON_FULL = "person_full"
    PERSON_VISIBLE = "person_visible"
    PERSON_MASK = "person_mask"
    POSE = "pose"
    HEAD_POINT = "head_point"


class GeometryKind(StrEnum):
    BBOX = "bbox"
    POINT = "point"
    POLYGON = "polygon"
    KEYPOINTS = "keypoints"


class GeometryQuality(StrEnum):
    EXACT = "exact"
    APPROXIMATE = "approximate"
    DERIVED = "derived"
    WEAK = "weak"


class CoverageStatus(StrEnum):
    POSITIVE_EXHAUSTIVE = "positive_exhaustive"
    VERIFIED_ABSENT = "verified_absent"
    PARTIAL = "partial"
    UNKNOWN = "unknown"


class CoordinateSpace(StrEnum):
    NORMALIZED = "normalized"
    PIXELS = "pixels"


FiniteFloat = Annotated[float, Field(allow_inf_nan=False)]


class CoverageRecord(StrictRecord):
    exact_class_or_mid: str = Field(min_length=1)
    status: CoverageStatus
    annotation_size_threshold: str | None = None
    evidence_origin: str = Field(min_length=1)


class CandidateInstance(StrictRecord):
    source_instance_id: str = Field(min_length=1)
    semantic: Semantic
    geometry_kind: GeometryKind
    coordinates: tuple[FiniteFloat, ...]
    coordinate_space: CoordinateSpace
    quality: GeometryQuality
    occluded: bool | None = None
    truncated: bool | None = None
    ignored: bool = False
    uncertain: bool = False
    derivation_rule: str | None = None
    parent_instance_id: str | None = None

    @model_validator(mode="after")
    def validate_geometry(self) -> CandidateInstance:
        expected = {GeometryKind.BBOX: 4, GeometryKind.POINT: 2}
        if self.geometry_kind in expected and len(self.coordinates) != expected[self.geometry_kind]:
            raise ValueError(f"{self.geometry_kind} requires {expected[self.geometry_kind]} values")
        if len(self.coordinates) % 2:
            raise ValueError("geometry must contain x/y pairs")
        if self.geometry_kind == GeometryKind.BBOX:
            x0, y0, x1, y1 = self.coordinates
            if not (x0 < x1 and y0 < y1):
                raise ValueError("bbox must have positive area")
        if self.coordinate_space == CoordinateSpace.NORMALIZED and any(
            value < 0.0 or value > 1.0 for value in self.coordinates
        ):
            raise ValueError("normalized coordinates must be in [0, 1]")
        return self


class SourceCandidate(StrictRecord):
    source_image_id: str = Field(min_length=1)
    image_url: str = Field(min_length=1)
    source_split: str = Field(min_length=1)
    instances: tuple[CandidateInstance, ...] = ()
    coverage_records: tuple[CoverageRecord, ...] = ()
    sequence_id: str | None = None
    camera_id: str | None = None
    scene_id: str | None = None
    duplicate_group: str | None = None
    expected_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    strata: tuple[str, ...] = ()

    @field_validator("source_image_id")
    @classmethod
    def source_id_is_opaque(cls, value: str) -> str:
        if value in {".", ".."} or "/" in value or "\\" in value or "\x00" in value:
            raise ValueError("source_image_id must be an opaque ID, not a path")
        return value

    @model_validator(mode="after")
    def has_evidence(self) -> SourceCandidate:
        if not self.instances and not self.coverage_records:
            raise ValueError(
                "candidate must contain positive, negative, partial, or unknown evidence"
            )
        return self


class CanonicalInstance(StrictRecord):
    source_instance_id: str
    semantic: Semantic
    geometry_kind: GeometryKind
    coordinates: tuple[FiniteFloat, ...]
    quality: GeometryQuality
    occluded: bool | None = None
    truncated: bool | None = None
    ignored: bool = False
    uncertain: bool = False
    derivation_rule: str | None = None
    parent_instance_id: str | None = None


class ImageRecord(StrictRecord):
    source: str
    source_version: str
    source_url: str
    source_image_id: str
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_width: PositiveInt
    source_height: PositiveInt
    oriented_width: PositiveInt
    oriented_height: PositiveInt
    orientation: str
    stored_path: str
    stored_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    stored_width: PositiveInt
    stored_height: PositiveInt
    storage_profile: str
    sequence_id: str | None = None
    camera_id: str | None = None
    scene_id: str | None = None
    duplicate_group: str
    source_split: str
    internal_split: str
    research_authorization_id: str
    acquisition_checksum: str = Field(pattern=r"^[0-9a-f]{64}$")
    coverage_records: tuple[CoverageRecord, ...]
    native_instances: tuple[CandidateInstance, ...]
    instances: tuple[CanonicalInstance, ...]

    @field_validator("stored_path")
    @classmethod
    def stored_path_is_contained(cls, value: str) -> str:
        path = Path(value)
        if path.is_absolute() or ".." in path.parts or not value.startswith("images/"):
            raise ValueError("stored_path must be a contained images/ path")
        return value

    @model_validator(mode="after")
    def no_upscale(self) -> ImageRecord:
        if self.stored_width > self.oriented_width or self.stored_height > self.oriented_height:
            raise ValueError("canonical packets must never upscale source images")
        return self
