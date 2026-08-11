from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, PositiveInt, field_validator, model_validator

SCHEMA_VERSION = "tracker.data.v1"


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class StorageConfig(StrictModel):
    max_width: PositiveInt
    max_height: PositiveInt
    codec: Literal["webp", "png", "jpeg"] = "webp"
    quality: int = Field(default=92, ge=1, le=100)
    lossless: bool = False

    @property
    def profile_id(self) -> str:
        mode = "lossless" if self.lossless else f"q{self.quality}"
        return f"fit-{self.max_width}x{self.max_height}-{self.codec}-{mode}"


class LimitsConfig(StrictModel):
    max_selected_images: PositiveInt
    max_download_bytes: PositiveInt
    max_temporary_bytes: PositiveInt
    max_selected_packet_bytes: PositiveInt
    max_corpus_bytes: PositiveInt
    min_free_bytes: int = Field(ge=0)

    @model_validator(mode="after")
    def coherent_limits(self) -> LimitsConfig:
        if self.max_download_bytes > self.max_temporary_bytes:
            raise ValueError("max_download_bytes cannot exceed max_temporary_bytes")
        if self.max_selected_packet_bytes > self.max_corpus_bytes:
            raise ValueError("packet cap cannot exceed corpus cap")
        return self


class PathsConfig(StrictModel):
    staging_root: Path
    packets_root: Path
    reports_root: Path

    @field_validator("staging_root", "packets_root", "reports_root")
    @classmethod
    def paths_must_be_resolved(cls, value: Path) -> Path:
        if not value.is_absolute():
            raise ValueError("paths must be absolute after configuration resolution")
        return value

    @model_validator(mode="after")
    def roots_are_distinct(self) -> PathsConfig:
        roots = [self.staging_root, self.packets_root, self.reports_root]
        if len(set(roots)) != len(roots):
            raise ValueError("staging, packet, and report roots must be distinct")
        for left in roots:
            for right in roots:
                if left != right and (left in right.parents or right in left.parents):
                    raise ValueError("data roots cannot contain one another")
        return self


class SourceConfig(StrictModel):
    kind: Literal["manifest", "open_images"]
    name: str = Field(min_length=1, pattern=r"^[a-z0-9][a-z0-9_-]*$")
    version: str = Field(min_length=1, pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
    source_url: str = Field(min_length=1)
    settings: dict[str, Any]


class PipelineConfig(StrictModel):
    schema_version: Literal[SCHEMA_VERSION]
    research_authorization_id: str = Field(min_length=1)
    source: SourceConfig
    storage: StorageConfig
    limits: LimitsConfig
    paths: PathsConfig
    selection_seed: int = Field(ge=0, le=2**63 - 1)
    internal_split_seed: int = Field(ge=0, le=2**63 - 1)
    internal_validation_fraction: float = Field(ge=0.0, lt=1.0)
    request_timeout_seconds: float = Field(default=30.0, gt=0.0, le=600.0)

    @field_validator("internal_validation_fraction", "request_timeout_seconds")
    @classmethod
    def finite_number(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("configuration numbers must be finite")
        return value

    def logical_dict(self) -> dict[str, object]:
        data = self.model_dump(mode="json")
        data.pop("paths")
        return data


def load_config(path: Path) -> PipelineConfig:
    path = path.resolve(strict=True)
    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    paths = raw.get("paths")
    if isinstance(paths, dict):
        for key in ("staging_root", "packets_root", "reports_root"):
            value = paths.get(key)
            if isinstance(value, str):
                candidate = Path(value).expanduser()
                paths[key] = str(
                    candidate.resolve()
                    if candidate.is_absolute()
                    else (path.parent / candidate).resolve()
                )
    return PipelineConfig.model_validate(raw)
