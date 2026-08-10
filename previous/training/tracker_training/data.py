"""Load canonical head scenes and their saved probability maps."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F
from torch.utils.data import Dataset


REPOSITORY = Path(__file__).resolve().parents[2]


class HeadDataset(Dataset):
    """Model-sized images plus heatmap, offset, and head-size targets."""

    def __init__(
        self,
        data_root: Path,
        split: str = "train",
        input_size: tuple[int, int] = (288, 160),
        output_stride: int = 4,
    ):
        self.input_width, self.input_height = input_size
        self.output_stride = output_stride
        self.rows = []
        for annotation in sorted(data_root.glob("*/annotations.jsonl")):
            with annotation.open() as stream:
                for line in stream:
                    row = json.loads(line)
                    # Face-only sources are retained for auxiliary/future training but
                    # must not silently become full-head size supervision.
                    if row["split"] == split and row.get("primary_target_kind", "head") == "head":
                        self.rows.append(row)
        if not self.rows:
            raise RuntimeError(f"no {split!r} records found under {data_root}")

    def __len__(self):
        return len(self.rows)

    def sampling_weights(self, mix: dict) -> torch.Tensor:
        """Give each source its configured epoch mass, then favour room-like counts."""
        available = Counter(row["source"] for row in self.rows)
        configured = mix["source_probability_mass"]
        missing_mass = sum(configured.get(source, 0.0) for source in configured if source not in available)
        masses = {source: configured.get(source, 0.0) for source in available}
        unconfigured = [source for source, mass in masses.items() if mass == 0]
        if unconfigured:
            fallback = max(0.01, missing_mass / len(unconfigured))
            for source in unconfigured:
                masses[source] = fallback

        count_weight = mix["head_count_weight"]
        within = defaultdict(float)
        row_factors = []
        for row in self.rows:
            count = sum(not head["ignore"] for head in row["heads"])
            if count == 0:
                factor = count_weight["empty"]
            elif count <= 4:
                factor = count_weight["one_to_four"]
            elif count <= 8:
                factor = count_weight["five_to_eight"]
            else:
                factor = count_weight["more_than_eight"]
            row_factors.append(factor)
            within[row["source"]] += factor
        weights = [
            masses[row["source"]] * factor / within[row["source"]]
            for row, factor in zip(self.rows, row_factors, strict=True)
        ]
        return torch.tensor(weights, dtype=torch.double)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(REPOSITORY / row["cache_image"]) as opened:
            image = opened.convert("RGB").resize(
                (self.input_width, self.input_height), Image.Resampling.BILINEAR
            )
            image_tensor = torch.from_numpy(np.asarray(image, dtype=np.uint8).copy())
        image_tensor = image_tensor.permute(2, 0, 1).float()
        image_tensor = (image_tensor - 128) / 128

        with Image.open(REPOSITORY / row["heatmap"]["path"]) as opened:
            heatmap = torch.from_numpy(np.asarray(opened, dtype=np.uint16).copy()).float()
        heatmap = heatmap[None] / row["heatmap"]["scale"]
        output_width = self.input_width // self.output_stride
        output_height = self.input_height // self.output_stride
        factor_x = heatmap.shape[2] // output_width
        factor_y = heatmap.shape[1] // output_height
        if factor_x < 1 or factor_y < 1 or heatmap.shape[2] % output_width or heatmap.shape[1] % output_height:
            raise RuntimeError(f"heatmap cannot be reduced exactly: {heatmap.shape} -> {(output_height, output_width)}")
        if factor_x > 1 or factor_y > 1:
            heatmap = F.max_pool2d(heatmap, kernel_size=(factor_y, factor_x))

        offset = torch.zeros(2, output_height, output_width)
        size = torch.zeros(2, output_height, output_width)
        mask = torch.zeros(1, output_height, output_width, dtype=torch.bool)
        cache_width, cache_height = row["cache_size"]
        for head in row["heads"]:
            if head["ignore"]:
                continue
            x, y, width, height = head["bbox_cache_xywh"]
            cx = (x + width / 2) * self.input_width / cache_width
            cy = (y + height / 2) * self.input_height / cache_height
            u, v = cx / self.output_stride, cy / self.output_stride
            ix = min(output_width - 1, max(0, int(u)))
            iy = min(output_height - 1, max(0, int(v)))
            if not mask[0, iy, ix]:
                offset[:, iy, ix] = torch.tensor([u - ix - 0.5, v - iy - 0.5])
                size[:, iy, ix] = torch.tensor([width / cache_width, height / cache_height])
                mask[0, iy, ix] = True
        return image_tensor, (heatmap, offset, size, mask)


class PackedHeadDataset(Dataset):
    """Fixed-shape memory maps with no JPEG decode, resize, or target building."""

    def __init__(self, data_root: Path, split: str = "train"):
        self.directory = Path(data_root) / split
        metadata_path = self.directory / "metadata.json"
        if not metadata_path.exists():
            raise RuntimeError(f"missing packed split metadata: {metadata_path}")
        self.metadata = json.loads(metadata_path.read_text())
        if self.metadata["split"] != split:
            raise RuntimeError(f"packed split mismatch in {metadata_path}")
        with (self.directory / self.metadata["index"]).open() as stream:
            self.rows = [json.loads(line) for line in stream]
        if len(self.rows) != self.metadata["count"]:
            raise RuntimeError(f"packed index count mismatch in {self.directory}")
        self._arrays = None

    def __len__(self):
        return len(self.rows)

    def sampling_weights(self, mix: dict) -> torch.Tensor:
        return HeadDataset.sampling_weights(self, mix)

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_arrays"] = None
        return state

    def _open_arrays(self):
        if self._arrays is not None:
            return self._arrays
        count = len(self)
        image = self.metadata["image"]
        heatmap = self.metadata["heatmap"]
        regression = self.metadata["regression"]
        mask = self.metadata["mask"]
        self._arrays = (
            np.memmap(
                self.directory / image["path"], mode="c", dtype=np.uint8,
                shape=tuple(image["shape"]),
            ),
            np.memmap(
                self.directory / heatmap["path"], mode="c", dtype=np.uint16,
                shape=tuple(heatmap["shape"]),
            ),
            np.memmap(
                self.directory / regression["path"], mode="c", dtype=np.float16,
                shape=tuple(regression["shape"]),
            ),
            np.memmap(
                self.directory / mask["path"], mode="c", dtype=np.uint8,
                shape=tuple(mask["shape"]),
            ),
        )
        assert all(array.shape[0] == count for array in self._arrays)
        return self._arrays

    def __getitem__(self, index):
        images, heatmaps, regression, masks = self._open_arrays()
        # Keep packed images uint8 through collation and host-to-device transfer;
        # the training loop normalizes a whole batch on the accelerator.
        image = torch.from_numpy(images[index]).permute(2, 0, 1)
        heatmap = torch.from_numpy(heatmaps[index])[None].float() / 65535
        dense = torch.from_numpy(regression[index]).float()
        mask = torch.from_numpy(masks[index]).bool()
        return image, (heatmap, dense[:2], dense[2:], mask)
