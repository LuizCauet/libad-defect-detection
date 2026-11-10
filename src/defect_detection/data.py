"""Paired visible-light data loading for the supervised LIBAD classifier."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import tifffile
import torch
from torch import Tensor
from torch.nn import functional as torch_functional
from torch.utils.data import Dataset


VALID_SPLITS = {"train", "validation", "test"}


@dataclass(frozen=True)
class VisibleSample:
    sample_key: str
    sample_name: str
    category: str
    source: str
    label: int
    side_a_path: Path
    side_b_path: Path


def _resolve_project_path(project_root: Path, relative_path: str) -> Path:
    root = project_root.resolve()
    resolved = (root / relative_path).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError(f"Manifest path escapes the project directory: {relative_path}") from error
    return resolved


def read_visible_manifest(
    manifest_path: Path, project_root: Path, split: str
) -> list[VisibleSample]:
    if split not in VALID_SPLITS:
        raise ValueError(f"Split must be one of {sorted(VALID_SPLITS)}, received {split!r}")

    with manifest_path.open(encoding="utf-8", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["split"] == split]
    if not rows:
        raise ValueError(f"No {split!r} samples found in {manifest_path}")

    samples: list[VisibleSample] = []
    for row in rows:
        label = int(row["label"])
        if label not in (0, 1):
            raise ValueError(f"Invalid binary label for {row['sample_key']}: {label}")
        expected_label = 1 if row["source"] == "anomaly" else 0
        if label != expected_label:
            raise ValueError(f"Label/source mismatch for {row['sample_key']}")

        samples.append(
            VisibleSample(
                sample_key=row["sample_key"],
                sample_name=row["sample_name"],
                category=row["category"],
                source=row["source"],
                label=label,
                side_a_path=_resolve_project_path(project_root, row["side_a_path"]),
                side_b_path=_resolve_project_path(project_root, row["side_b_path"]),
            )
        )
    return samples


class VisiblePairDataset(Dataset[dict[str, Tensor | str]]):
    """Load A/B images as an ordered two-channel float32 tensor."""

    def __init__(
        self,
        manifest_path: Path,
        project_root: Path,
        split: str,
        image_size: tuple[int, int] = (640, 512),
        channel_mean: Sequence[float] | None = None,
        channel_std: Sequence[float] | None = None,
    ) -> None:
        self.samples = read_visible_manifest(manifest_path, project_root, split)
        self.split = split
        self.image_size = image_size

        if (channel_mean is None) != (channel_std is None):
            raise ValueError("channel_mean and channel_std must be provided together")
        if channel_mean is not None and channel_std is not None:
            if len(channel_mean) != 2 or len(channel_std) != 2:
                raise ValueError("Visible A/B normalization requires two means and two standard deviations")
            if any(value <= 0 for value in channel_std):
                raise ValueError("Channel standard deviations must be positive")
            self.channel_mean = torch.tensor(channel_mean, dtype=torch.float32).view(2, 1, 1)
            self.channel_std = torch.tensor(channel_std, dtype=torch.float32).view(2, 1, 1)
        else:
            self.channel_mean = None
            self.channel_std = None

    def __len__(self) -> int:
        return len(self.samples)

    @staticmethod
    def _read_visible(path: Path) -> Tensor:
        pixels = tifffile.imread(path)
        if pixels.ndim != 2:
            raise ValueError(f"Expected one grayscale plane in {path}, found shape {pixels.shape}")
        minimum = int(pixels.min())
        maximum = int(pixels.max())
        if minimum < 0 or maximum > 255:
            raise ValueError(f"Expected 8-bit visible values in {path}, found range {minimum}..{maximum}")
        return torch.from_numpy(np.asarray(pixels, dtype=np.float32)) / 255.0

    def __getitem__(self, index: int) -> dict[str, Tensor | str]:
        sample = self.samples[index]
        side_a = self._read_visible(sample.side_a_path)
        side_b = self._read_visible(sample.side_b_path)
        if side_a.shape != side_b.shape:
            raise ValueError(
                f"A/B shape mismatch for {sample.sample_key}: A={tuple(side_a.shape)}, "
                f"B={tuple(side_b.shape)}"
            )

        image = torch.stack((side_a, side_b), dim=0)
        image = torch_functional.interpolate(
            image.unsqueeze(0),
            size=self.image_size,
            mode="bilinear",
            align_corners=False,
            antialias=True,
        ).squeeze(0)
        if self.channel_mean is not None and self.channel_std is not None:
            image = (image - self.channel_mean) / self.channel_std

        return {
            "image": image,
            "label": torch.tensor(sample.label, dtype=torch.float32),
            "sample_key": sample.sample_key,
            "sample_name": sample.sample_name,
            "category": sample.category,
            "source": sample.source,
        }
