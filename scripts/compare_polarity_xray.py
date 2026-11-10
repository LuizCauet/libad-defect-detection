"""Compare normal and defective polarity X-rays with a shared display scale."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tifffile


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = PROJECT_ROOT / "manifests" / "visible_binary_split_v1.csv"
DATA_ROOT = PROJECT_ROOT / "data" / "LIBAD"
OUTPUT = PROJECT_ROOT / "reports" / "polarity_xray_normal_vs_defect.png"


def main() -> None:
    with MANIFEST.open(encoding="utf-8-sig", newline="") as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if row.get("category", row.get("class_name")) == "7_polarity"
            and row["split"] == "validation"
        ]

    grouped = {
        source: sorted(
            (row for row in rows if row["source"] == source),
            key=lambda row: row["sample_key"],
        )
        for source in ("normal", "anomaly")
    }
    images = {
        row["sample_key"]: tifffile.imread(
            DATA_ROOT / f"{row['sample_key']}X.tiff"
        ).astype(np.float32)
        for row in rows
    }

    # Use one scale for every panel so absolute X-ray intensity remains comparable.
    sampled_pixels = np.concatenate(
        [image[::8, ::8].ravel() for image in images.values()]
    )
    lower, upper = np.percentile(sampled_pixels, (1.0, 99.0))

    figure, axes = plt.subplots(
        2,
        7,
        figsize=(17.5, 7.2),
        constrained_layout=True,
        squeeze=False,
    )
    for row_index, source in enumerate(("normal", "anomaly")):
        for column_index, row in enumerate(grouped[source]):
            sample_key = row["sample_key"]
            axes[row_index, column_index].imshow(
                images[sample_key], cmap="gray", vmin=lower, vmax=upper
            )
            axes[row_index, column_index].set_title(
                sample_key.rsplit("/", 1)[-1], fontsize=11
            )
            axes[row_index, column_index].axis("off")
        axes[row_index, 0].set_ylabel(
            "NORMAL" if source == "normal" else "POLARITY DEFECT",
            fontsize=12,
            fontweight="bold",
        )

    figure.suptitle(
        "Polarity validation samples — high-resolution X-ray\n"
        "Shared 1st–99th percentile scale across all 14 images",
        fontsize=14,
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT, dpi=170)
    plt.close(figure)
    print(f"Saved comparison to: {OUTPUT}")
    print(f"Shared display range: {lower:.3f} to {upper:.3f}")


if __name__ == "__main__":
    main()
