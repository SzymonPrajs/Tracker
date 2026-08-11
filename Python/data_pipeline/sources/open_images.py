from __future__ import annotations

import csv
import hashlib
from collections import defaultdict
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from data_pipeline.records import (
    CandidateInstance,
    CoordinateSpace,
    CoverageRecord,
    CoverageStatus,
    GeometryKind,
    GeometryQuality,
    Semantic,
    SourceCandidate,
)
from data_pipeline.sources.base import DiscoveryResult
from data_pipeline.transfer import download_metadata


class OpenImagesSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    split: str = Field(pattern=r"^(train|validation|test)$")
    boxes_url: str
    image_labels_url: str | None = None
    class_descriptions_url: str
    image_url_template: str = "https://open-images-dataset.s3.amazonaws.com/{split}/{image_id}.jpg"
    class_semantics: dict[str, Semantic]
    verified_negative_mids: tuple[str, ...] = ()
    negative_fraction: float = Field(default=0.2, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def negatives_are_explicit(self) -> OpenImagesSettings:
        if self.verified_negative_mids and self.image_labels_url is None:
            raise ValueError("verified negatives require image_labels_url")
        missing = set(self.verified_negative_mids) - set(self.class_semantics)
        if missing:
            raise ValueError(f"negative MIDs have no semantic mapping: {sorted(missing)}")
        return self


def _require_columns(reader: csv.DictReader[str], required: set[str], name: str) -> None:
    missing = required - set(reader.fieldnames or ())
    if missing:
        raise ValueError(f"{name} is missing required columns: {sorted(missing)}")


def _flag(value: str) -> bool | None:
    return True if value == "1" else False if value == "0" else None


def _rank(seed: int, kind: str, image_id: str) -> str:
    return hashlib.sha256(f"{seed}:{kind}:{image_id}".encode()).hexdigest()


class OpenImagesAdapter:
    """Parse a bounded class-scoped subset of official Open Images metadata."""

    def __init__(self, settings: dict[str, Any]) -> None:
        self.settings = OpenImagesSettings.model_validate(settings)

    def _fetch(
        self, staging_dir: Path, max_bytes: int, timeout_seconds: float
    ) -> tuple[dict[str, Path], dict[str, str]]:
        urls = {
            "boxes.csv": self.settings.boxes_url,
            "classes.csv": self.settings.class_descriptions_url,
        }
        if self.settings.image_labels_url is not None:
            urls["image-labels.csv"] = self.settings.image_labels_url
        paths: dict[str, Path] = {}
        checksums: dict[str, str] = {}
        used = 0
        for name, url in urls.items():
            path = staging_dir / "metadata" / name
            transfer = download_metadata(
                url,
                path,
                byte_limit=max_bytes - used,
                timeout_seconds=timeout_seconds,
            )
            used += transfer.byte_count
            paths[name] = path
            checksums[name] = transfer.sha256
        return paths, checksums

    def _class_names(self, path: Path) -> dict[str, str]:
        names: dict[str, str] = {}
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.reader(handle):
                if len(row) != 2:
                    raise ValueError("Open Images class descriptions need exactly two columns")
                names[row[0]] = row[1]
        missing = set(self.settings.class_semantics) - set(names)
        if missing:
            raise ValueError(f"unknown Open Images MIDs: {sorted(missing)}")
        return names

    def _boxes(
        self, path: Path
    ) -> tuple[
        dict[str, list[CandidateInstance]],
        dict[str, dict[str, set[str]]],
        int,
    ]:
        by_image: dict[str, list[CandidateInstance]] = defaultdict(list)
        mid_evidence: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
        native_count = 0
        required = {
            "ImageID",
            "LabelName",
            "XMin",
            "XMax",
            "YMin",
            "YMax",
            "IsOccluded",
            "IsTruncated",
            "IsGroupOf",
            "IsDepiction",
            "IsInside",
        }
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            _require_columns(reader, required, "Open Images boxes")
            for row_number, row in enumerate(reader, start=2):
                mid = row["LabelName"]
                if mid not in self.settings.class_semantics:
                    continue
                native_count += 1
                ignored = any(
                    _flag(row[name]) is True for name in ("IsGroupOf", "IsDepiction", "IsInside")
                )
                mid_evidence[row["ImageID"]][mid].add("ignored" if ignored else "usable")
                by_image[row["ImageID"]].append(
                    CandidateInstance(
                        source_instance_id=f"oi-box-{row_number}",
                        semantic=self.settings.class_semantics[mid],
                        geometry_kind=GeometryKind.BBOX,
                        coordinates=(
                            float(row["XMin"]),
                            float(row["YMin"]),
                            float(row["XMax"]),
                            float(row["YMax"]),
                        ),
                        coordinate_space=CoordinateSpace.NORMALIZED,
                        quality=GeometryQuality.EXACT,
                        occluded=_flag(row["IsOccluded"]),
                        truncated=_flag(row["IsTruncated"]),
                        ignored=ignored,
                        uncertain=ignored,
                        derivation_rule=f"Open Images native box MID {mid}",
                    )
                )
        return (
            dict(by_image),
            {
                image_id: {mid: set(states) for mid, states in mids.items()}
                for image_id, mids in mid_evidence.items()
            },
            native_count,
        )

    def _verified_negatives(self, path: Path | None) -> dict[str, set[str]]:
        if path is None:
            return {}
        absent: dict[str, set[str]] = defaultdict(set)
        wanted = set(self.settings.verified_negative_mids)
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            _require_columns(
                reader,
                {"ImageID", "Source", "LabelName", "Confidence"},
                "Open Images labels",
            )
            for row in reader:
                if (
                    row["LabelName"] in wanted
                    and row["Confidence"] == "0"
                    and row["Source"] in {"verification", "crowdsource-verification"}
                ):
                    absent[row["ImageID"]].add(row["LabelName"])
        return dict(absent)

    def discover(
        self,
        *,
        staging_dir: Path,
        max_candidates: int,
        seed: int,
        max_metadata_bytes: int,
        timeout_seconds: float,
    ) -> DiscoveryResult:
        paths, checksums = self._fetch(staging_dir, max_metadata_bytes, timeout_seconds)
        names = self._class_names(paths["classes.csv"])
        boxes, mid_evidence, native_box_count = self._boxes(paths["boxes.csv"])
        absent = self._verified_negatives(paths.get("image-labels.csv"))

        negative_ids = [image_id for image_id in absent if image_id not in boxes]
        negative_cap = min(
            round(max_candidates * self.settings.negative_fraction), len(negative_ids)
        )
        positive_cap = min(max_candidates - negative_cap, len(boxes))
        ranked_positive = sorted(boxes, key=lambda value: _rank(seed, "positive", value))
        selected_positive = ranked_positive[:positive_cap]
        selected_negative = sorted(negative_ids, key=lambda value: _rank(seed, "negative", value))[
            :negative_cap
        ]
        remaining = max_candidates - len(selected_positive) - len(selected_negative)
        if remaining:
            selected_positive.extend(ranked_positive[positive_cap : positive_cap + remaining])

        candidates: list[SourceCandidate] = []
        for image_id in selected_positive + selected_negative:
            instances = tuple(boxes.get(image_id, ()))
            coverage: list[CoverageRecord] = []
            for mid in self.settings.class_semantics:
                states = mid_evidence.get(image_id, {}).get(mid, set())
                if states == {"usable"}:
                    status = CoverageStatus.POSITIVE_EXHAUSTIVE
                    origin = f"Open Images {self.settings.split} native boxes ({names[mid]})"
                elif states:
                    status = CoverageStatus.PARTIAL
                    origin = (
                        f"Open Images {self.settings.split} group/depiction/inside boxes "
                        f"require ignore handling ({names[mid]})"
                    )
                elif mid in absent.get(image_id, set()):
                    status = CoverageStatus.VERIFIED_ABSENT
                    origin = f"Open Images human verification ({names[mid]})"
                else:
                    status = CoverageStatus.UNKNOWN
                    origin = "Open Images class not verified for this image"
                coverage.append(
                    CoverageRecord(
                        exact_class_or_mid=mid,
                        status=status,
                        evidence_origin=origin,
                    )
                )
            candidates.append(
                SourceCandidate(
                    source_image_id=image_id,
                    image_url=self.settings.image_url_template.format(
                        split=self.settings.split, image_id=image_id
                    ),
                    source_split=self.settings.split,
                    instances=instances,
                    coverage_records=tuple(coverage),
                    duplicate_group=f"open-images:{image_id}",
                    strata=("positive" if instances else "verified-negative",),
                )
            )

        return DiscoveryResult(
            candidates=tuple(candidates),
            metadata_files=tuple(paths.values()),
            metadata_sha256=checksums,
            source_counts={
                "mapped_native_boxes": native_box_count,
                "positive_images": len(boxes),
                "verified_negative_images": len(absent),
            },
            selection={
                "algorithm": "separate sha256(seed:positive|negative:image_id) ranking",
                "seed": seed,
                "negative_fraction": self.settings.negative_fraction,
                "selected_positive": len(selected_positive),
                "selected_negative": len(selected_negative),
                "selected_ids": [candidate.source_image_id for candidate in candidates],
                "class_semantics": {
                    mid: semantic.value for mid, semantic in self.settings.class_semantics.items()
                },
            },
        )
