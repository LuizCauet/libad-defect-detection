"""Decode and validate every paired visible-light sample in the fixed manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from defect_detection.data import VisiblePairDataset  # noqa: E402


EXPECTED_COUNTS = {"train": 446, "validation": 149, "test": 149}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "manifests" / "visible_binary_split_v1.csv",
    )
    parser.add_argument(
        "--normalization",
        type=Path,
        default=PROJECT_ROOT / "manifests" / "visible_binary_normalization_v1.json",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    normalization = json.loads(args.normalization.read_text(encoding="utf-8"))
    mean = normalization["mean"]
    std = normalization["std"]
    if normalization["channel_order"] != ["side_a", "side_b"]:
        raise ValueError("Unexpected normalization channel order")

    all_keys: set[str] = set()
    summary: dict[str, object] = {}
    for split, expected_count in EXPECTED_COUNTS.items():
        dataset = VisiblePairDataset(
            manifest_path=args.manifest,
            project_root=PROJECT_ROOT,
            split=split,
            image_size=(640, 512),
            channel_mean=mean,
            channel_std=std,
        )
        if len(dataset) != expected_count:
            raise ValueError(f"Expected {expected_count} {split} samples, found {len(dataset)}")
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )

        channel_sum = torch.zeros(2, dtype=torch.float64)
        channel_squared_sum = torch.zeros(2, dtype=torch.float64)
        pixels_per_channel = 0
        labels = torch.zeros(2, dtype=torch.int64)
        split_keys: list[str] = []
        observed_min = float("inf")
        observed_max = float("-inf")

        for batch in loader:
            images = batch["image"]
            if images.dtype != torch.float32:
                raise TypeError(f"Expected float32 images, found {images.dtype}")
            if images.ndim != 4 or images.shape[1:] != (2, 640, 512):
                raise ValueError(f"Unexpected {split} batch shape: {tuple(images.shape)}")
            if not torch.isfinite(images).all():
                raise ValueError(f"Non-finite image value found in {split}")

            batch_labels = batch["label"]
            if not torch.isin(batch_labels, torch.tensor([0.0, 1.0])).all():
                raise ValueError(f"Non-binary label found in {split}")
            labels += torch.bincount(batch_labels.to(torch.int64), minlength=2)

            double_images = images.to(torch.float64)
            channel_sum += double_images.sum(dim=(0, 2, 3))
            channel_squared_sum += double_images.square().sum(dim=(0, 2, 3))
            pixels_per_channel += images.shape[0] * images.shape[2] * images.shape[3]
            observed_min = min(observed_min, float(images.min()))
            observed_max = max(observed_max, float(images.max()))
            split_keys.extend(batch["sample_key"])

        if len(split_keys) != len(set(split_keys)):
            raise ValueError(f"Duplicate sample key found inside {split}")
        overlap = all_keys.intersection(split_keys)
        if overlap:
            raise ValueError(f"Sample keys cross partitions: {sorted(overlap)[:5]}")
        all_keys.update(split_keys)

        aggregate_mean = channel_sum / pixels_per_channel
        aggregate_std = (
            channel_squared_sum / pixels_per_channel - aggregate_mean.square()
        ).clamp_min(0).sqrt()
        if split == "train":
            if not torch.allclose(aggregate_mean, torch.zeros(2, dtype=torch.float64), atol=1e-6):
                raise ValueError(f"Normalized training means are not zero: {aggregate_mean.tolist()}")
            if not torch.allclose(aggregate_std, torch.ones(2, dtype=torch.float64), atol=1e-6):
                raise ValueError(f"Normalized training standard deviations are not one: {aggregate_std.tolist()}")

        summary[split] = {
            "samples": len(split_keys),
            "normal": int(labels[0]),
            "defective": int(labels[1]),
            "normalized_channel_mean": aggregate_mean.tolist(),
            "normalized_channel_std": aggregate_std.tolist(),
            "observed_value_range": [observed_min, observed_max],
        }

    if len(all_keys) != 744:
        raise ValueError(f"Expected 744 unique samples across partitions, found {len(all_keys)}")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print("Validated all 744 paired samples without cross-partition overlap.")


if __name__ == "__main__":
    main()
