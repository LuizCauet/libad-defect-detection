"""Aggregate the three validation-only visible CNN runs into a report."""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SEEDS = (17, 42, 73)


def mean(values: list[float]) -> float:
    return statistics.fmean(values)


def population_std(values: list[float]) -> float:
    return statistics.pstdev(values)


def percent(value: float) -> str:
    return f"{100 * value:.2f}%"


def main() -> None:
    results = [
        json.loads(
            (PROJECT_ROOT / "artifacts" / "visible_cnn_v1" / f"seed_{seed}" / "results.json").read_text(
                encoding="utf-8"
            )
        )
        for seed in SEEDS
    ]
    if any(result["test_predictions_generated"] for result in results):
        raise RuntimeError("A validation-only run reports final-test predictions")

    seed_rows: list[dict[str, float | int | str]] = []
    for result in results:
        selected = result["selected_operating_point"]
        seed_rows.append(
            {
                "seed": result["seed"],
                "best_epoch": result["best_epoch"],
                "epochs_completed": result["epochs_completed"],
                "auroc": result["auroc"],
                "average_precision": result["average_precision"],
                "selected_threshold": selected["threshold"],
                "selected_recall": selected["recall"],
                "selected_fpr": selected["false_positive_rate"],
                "selected_precision": selected["precision"],
                "selected_true_positive": selected["true_positive"],
                "selected_false_negative": selected["false_negative"],
                "selected_false_positive": selected["false_positive"],
                "selected_true_negative": selected["true_negative"],
                "fpr_at_90_recall": result["fpr_at_90_recall"]["false_positive_rate"],
                "fpr_at_95_recall": result["fpr_at_95_recall"]["false_positive_rate"],
                "elapsed_seconds": result["elapsed_seconds"],
                "peak_cuda_allocated_bytes": result["peak_cuda_allocated_bytes"],
                "mlflow_run_id": result["mlflow_run_id"],
            }
        )

    summary_path = PROJECT_ROOT / "reports" / "visible-cnn-validation-summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=seed_rows[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(seed_rows)

    metric_names = (
        "auroc",
        "average_precision",
        "selected_recall",
        "selected_fpr",
        "selected_precision",
        "fpr_at_90_recall",
        "fpr_at_95_recall",
        "elapsed_seconds",
        "peak_cuda_allocated_bytes",
    )
    aggregates = {
        metric: {
            "mean": mean([float(row[metric]) for row in seed_rows]),
            "population_std": population_std([float(row[metric]) for row in seed_rows]),
            "minimum": min(float(row[metric]) for row in seed_rows),
            "maximum": max(float(row[metric]) for row in seed_rows),
        }
        for metric in metric_names
    }

    category_names = sorted(results[0]["per_category_recall_at_selected_threshold"])
    category_rows = []
    for category in category_names:
        recalls = [
            float(result["per_category_recall_at_selected_threshold"][category]["recall"])
            for result in results
        ]
        category_rows.append(
            {
                "category": category,
                "validation_defects": results[0]["per_category_recall_at_selected_threshold"][category][
                    "defects"
                ],
                "mean_recall": mean(recalls),
                "minimum_recall": min(recalls),
                "maximum_recall": max(recalls),
            }
        )

    machine_summary = {
        "architecture": "visible_cnn_v1",
        "seeds": list(SEEDS),
        "validation_samples": {"normal": 72, "defective": 77},
        "threshold_rule": "maximum validation recall with validation FPR <= 0.10",
        "aggregates": aggregates,
        "per_category": category_rows,
        "test_predictions_generated": False,
    }
    json_path = PROJECT_ROOT / "reports" / "visible-cnn-validation-summary.json"
    json_path.write_text(
        json.dumps(machine_summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    lines = [
        "# Visible CNN validation results",
        "",
        "## Scope",
        "",
        "These are development results from the fixed 149-sample validation partition: 72 normal and 77 defective samples. The sealed final test partition was not scored. Each run used the same architecture and data split with a different initialization and training-shuffle seed.",
        "",
        "The selected operating point maximizes validation recall while keeping validation false-positive rate (FPR) at or below 10%. Each false positive changes validation FPR by approximately 1.39 percentage points because there are 72 normal validation samples.",
        "",
        "## Per-seed results",
        "",
        "| Seed | Best epoch | AUROC | Selected threshold | Recall | FPR | Precision | FPR at ≥90% recall | FPR at ≥95% recall |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in seed_rows:
        lines.append(
            f"| {row['seed']} | {row['best_epoch']} | {float(row['auroc']):.4f} | "
            f"{float(row['selected_threshold']):.4f} | {percent(float(row['selected_recall']))} | "
            f"{percent(float(row['selected_fpr']))} | {percent(float(row['selected_precision']))} | "
            f"{percent(float(row['fpr_at_90_recall']))} | {percent(float(row['fpr_at_95_recall']))} |"
        )

    lines.extend(
        [
            "",
            "## Aggregate across three seeds",
            "",
            "| Measurement | Mean | Population standard deviation | Range |",
            "| --- | ---: | ---: | ---: |",
            f"| AUROC | {aggregates['auroc']['mean']:.4f} | {aggregates['auroc']['population_std']:.4f} | {aggregates['auroc']['minimum']:.4f}–{aggregates['auroc']['maximum']:.4f} |",
            f"| Recall under 10% FPR constraint | {percent(aggregates['selected_recall']['mean'])} | {percent(aggregates['selected_recall']['population_std'])} | {percent(aggregates['selected_recall']['minimum'])}–{percent(aggregates['selected_recall']['maximum'])} |",
            f"| Selected FPR | {percent(aggregates['selected_fpr']['mean'])} | {percent(aggregates['selected_fpr']['population_std'])} | {percent(aggregates['selected_fpr']['minimum'])}–{percent(aggregates['selected_fpr']['maximum'])} |",
            f"| FPR at ≥90% recall | {percent(aggregates['fpr_at_90_recall']['mean'])} | {percent(aggregates['fpr_at_90_recall']['population_std'])} | {percent(aggregates['fpr_at_90_recall']['minimum'])}–{percent(aggregates['fpr_at_90_recall']['maximum'])} |",
            f"| FPR at ≥95% recall | {percent(aggregates['fpr_at_95_recall']['mean'])} | {percent(aggregates['fpr_at_95_recall']['population_std'])} | {percent(aggregates['fpr_at_95_recall']['minimum'])}–{percent(aggregates['fpr_at_95_recall']['maximum'])} |",
            "",
            "## Recall by defect category at each seed's selected threshold",
            "",
            "| Category | Validation defects | Mean recall | Minimum | Maximum |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in category_rows:
        lines.append(
            f"| {row['category']} | {row['validation_defects']} | "
            f"{percent(float(row['mean_recall']))} | {percent(float(row['minimum_recall']))} | "
            f"{percent(float(row['maximum_recall']))} |"
        )

    total_elapsed = sum(float(row["elapsed_seconds"]) for row in seed_rows)
    peak_mib = max(float(row["peak_cuda_allocated_bytes"]) for row in seed_rows) / 1024**2
    lines.extend(
        [
            "",
            "## Findings",
            "",
            f"- Mean validation AUROC was **{aggregates['auroc']['mean']:.4f}**, with a range of {aggregates['auroc']['minimum']:.4f} to {aggregates['auroc']['maximum']:.4f}. Initialization materially affected the result.",
            f"- Under the 10% FPR constraint, mean recall was **{percent(aggregates['selected_recall']['mean'])}** and individual-seed recall ranged from {percent(aggregates['selected_recall']['minimum'])} to {percent(aggregates['selected_recall']['maximum'])}.",
            f"- Reaching at least 95% recall required mean FPR of **{percent(aggregates['fpr_at_95_recall']['mean'])}**, ranging from {percent(aggregates['fpr_at_95_recall']['minimum'])} to {percent(aggregates['fpr_at_95_recall']['maximum'])}.",
            "- Polarity recall was 0% for all three seeds at the selected operating thresholds. This is the clearest category-level failure to investigate.",
            "- Streak and bare-foil defects reached 100% recall for every seed at the selected thresholds, although their validation sample counts remain small.",
            f"- The three runs took {total_elapsed / 60:.1f} minutes in total. Peak PyTorch CUDA allocation was approximately {peak_mib:.0f} MiB.",
            "",
            "## Limits",
            "",
            "- These are validation results used for development; they are not final test estimates.",
            "- Thresholds differ by seed because score calibration differs across trained models.",
            "- Per-category estimates use only 4 to 11 defective validation samples per category.",
            "- The earlier PatchCore result used a different split and cannot support a direct superiority claim.",
            "- No model change has yet been tested against this initial configuration.",
            "",
            "## Reproduce",
            "",
            "```powershell",
            ".\\.venv\\Scripts\\python.exe scripts\\train_visible_cnn.py",
            ".\\.venv\\Scripts\\python.exe scripts\\summarize_visible_cnn_validation.py",
            "```",
            "",
        ]
    )
    report_path = PROJECT_ROOT / "reports" / "visible-cnn-validation.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved: {report_path}")
    print(f"Saved: {summary_path}")
    print(f"Saved: {json_path}")


if __name__ == "__main__":
    main()
