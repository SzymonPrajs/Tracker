from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from data_pipeline.packets.validate import validate_packet
from data_pipeline.records import ImageRecord


class PacketReader:
    def __init__(self, packet_root: Path, *, validate: bool = True) -> None:
        self.packet_root = packet_root.resolve(strict=True)
        if validate:
            validate_packet(self.packet_root)

    def records(self) -> Iterator[ImageRecord]:
        with (self.packet_root / "records.jsonl").open("r", encoding="utf-8") as handle:
            for line in handle:
                yield ImageRecord.model_validate_json(line)

    def image_path(self, record: ImageRecord) -> Path:
        path = (self.packet_root / record.stored_path).resolve(strict=True)
        if self.packet_root not in path.parents:
            raise ValueError("record image escaped packet root")
        return path
