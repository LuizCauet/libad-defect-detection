"""Calculate fixed A/B normalization statistics from the training split only."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from defect_detection.data import VisiblePairDataset  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "manifests" / "visible_binary_split_v1.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "manifests" / "visible_binary_normalization_v1.json",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def calculate_statistics(loader: DataLoader) -> tuple[list[float], list[float], int]:
    channel_sum = torch.zeros(2, dtype=torch.float64)
    channel_squared_sum = torch.zeros(2, dtype=torch.float64)
    pixels_per_channel = 0

    for batch in loader:
        images = batch["image"].to(dtype=torch.float64)
        if images.ndim != 4 or images.shape[1:] != (2, 640, 512):
            raise ValueError(f"Unexpected training batch shape: {tuple(images.shape)}")
        channel_sum += images.sum(dim=(0, 2, 3))
        channel_squared_sum += images.square().sum(dim=(0, 2, 3))
        pixels_per_channel += images.shape[0] * images.shape[2] * images.shape[3]

    mean = channel_sum / pixels_per_channel
    variance = channel_squared_sum / pixels_per_channel - mean.square()
    std = variance.clamp_min(0).sqrt()
    return mean.tolist(), std.tolist(), pixels_per_channel


def write_checked(path: Path, content: bytes, overwrite: bool) -> None:
    if path.exists() and path.read_bytes() != content and not overwrite:
        raise FileExistsError(f"{path} exists with different content; use --overwrite to replace it")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def main() -> None:
    args = parse_args()
    dataset = VisiblePairDataset(
        manifest_path=args.manifest,
        project_root=PROJECT_ROOT,
        split="train",
        image_size=(640, 512),
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    mean, std, pixels_per_channel = calculate_statistics(loader)

    output = {
        "normalization_version": "visible_binary_normalization_v1",
        "manifest": args.manifest.relative_to(PROJECT_ROOT).as_posix(),
        "manifest_sha256": sha256_file(args.manifest),
        "source_split": "train",
        "training_samples": len(dataset),
        "channel_order": ["side_a", "side_b"],
        "resize_height": 640,
        "resize_width": 512,
        "input_scaling": "uint8 divided by 255.0",
        "pixels_per_channel": pixels_per_channel,
        "mean": mean,
        "std": std,
        "variance_definition": "population variance over all resized training pixels",
    }
    content = (json.dumps(output, indent=2, sort_keys=True) + "\n").encode("utf-8")
    write_checked(args.output, content, args.overwrite)

    print(json.dumps(output, indent=2, sort_keys=True))
    print(f"Saved: {args.output.resolve()}")


if __name__ == "__main__":
    main()
