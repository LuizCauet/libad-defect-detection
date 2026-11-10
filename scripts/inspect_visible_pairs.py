"""Create a deterministic montage of paired LIBAD visible-light images."""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tifffile


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Compare visible-light sides A and B for the same LIBAD sample."
    )
    parser.add_argument(
        "--split-csv",
        type=Path,
        default=project_root / "data" / "splits" / "LIBAD_normal70train_noval_seed7345.csv",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=project_root / "data" / "LIBAD",
    )
    parser.add_argument("--count-per-label", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--category")
    parser.add_argument("--source", choices=("normal", "anomaly"))
    parser.add_argument("--split")
    parser.add_argument(
        "--include-xray-h",
        action="store_true",
        help="Add the registered high-resolution X-ray image with percentile contrast.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=project_root / "artifacts" / "visible_pair_inspection.png",
    )
    return parser.parse_args()


def load_rows(split_csv: Path) -> list[dict[str, str]]:
    with split_csv.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def choose_rows(
    rows: list[dict[str, str]], count_per_label: int, seed: int, source: str | None
) -> list[dict[str, str]]:
    rng = random.Random(seed)
    selected: list[dict[str, str]] = []
    for selected_source in ((source,) if source else ("normal", "anomaly")):
        candidates = [row for row in rows if row["source"] == selected_source]
        selected.extend(rng.sample(candidates, min(count_per_label, len(candidates))))
    return selected


def read_visible(path: Path) -> np.ndarray:
    image = tifffile.imread(path).astype(np.float32)
    return image / 255.0


def read_xray_high_contrast(path: Path) -> np.ndarray:
    image = tifffile.imread(path).astype(np.float32)
    lower, upper = np.percentile(image, (1.0, 99.0))
    if upper <= lower:
        return np.zeros_like(image)
    return np.clip((image - lower) / (upper - lower), 0.0, 1.0)


def main() -> None:
    args = parse_args()
    available_rows = load_rows(args.split_csv)
    if args.category:
        available_rows = [
            row
            for row in available_rows
            if row.get("category", row.get("class_name")) == args.category
        ]
    if args.split:
        available_rows = [row for row in available_rows if row["split"] == args.split]
    rows = choose_rows(available_rows, args.count_per_label, args.seed, args.source)
    if not rows:
        raise RuntimeError(f"No samples found in {args.split_csv}")

    column_count = 4 if args.include_xray_h else 3
    figure, axes = plt.subplots(
        len(rows),
        column_count,
        figsize=(4 * column_count, 3.2 * len(rows)),
        constrained_layout=True,
        squeeze=False,
    )
    for row_index, row in enumerate(rows):
        sample_key = row["sample_key"]
        side_a = read_visible(args.data_root / f"{sample_key}A.tiff")
        side_b = read_visible(args.data_root / f"{sample_key}B.tiff")
        if side_a.shape != side_b.shape:
            raise RuntimeError(
                f"Shape mismatch for {sample_key}: A={side_a.shape}, B={side_b.shape}"
            )

        difference = np.abs(side_a - side_b)
        label = "DEFECT" if row["source"] == "anomaly" else "NORMAL"
        titles = [
            f"{sample_key} | {label} | side A",
            "side B",
        ]
        images = [side_a, side_b]
        if args.include_xray_h:
            images.append(read_xray_high_contrast(args.data_root / f"{sample_key}X.tiff"))
            titles.append("high-resolution X-ray (1–99% contrast)")
        images.append(difference)
        titles.append("absolute A/B pixel difference")
        for column, (image, title) in enumerate(zip(images, titles, strict=True)):
            axes[row_index, column].imshow(image, cmap="gray", vmin=0.0, vmax=1.0)
            axes[row_index, column].set_title(title, fontsize=9)
            axes[row_index, column].axis("off")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=160)
    plt.close(figure)
    print(f"Saved {len(rows)} paired samples to: {args.output.resolve()}")
    print("Inspect whether the same boundaries and structures occupy matching coordinates in A and B.")


if __name__ == "__main__":
    main()
