"""Train and validate visible_cnn_v1 without accessing the sealed test split."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import torch
from sklearn.metrics import roc_curve
from torch import nn
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from defect_detection.data import VisiblePairDataset  # noqa: E402
from defect_detection.evaluation import (  # noqa: E402
    operating_point_at_recall,
    ranking_metrics,
    select_fpr_constrained_threshold,
    threshold_curve,
)
from defect_detection.model import VisibleDefectCNN, count_trainable_parameters  # noqa: E402


DEFAULT_SEEDS = (17, 42, 73)


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
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "visible_cnn_v1",
    )
    parser.add_argument("--tracking-db", type=Path, default=PROJECT_ROOT / "mlflow.db")
    parser.add_argument(
        "--artifact-root", type=Path, default=PROJECT_ROOT / "mlartifacts"
    )
    parser.add_argument("--experiment-name", default="libad-visible-binary-cnn")
    parser.add_argument("--run-prefix", default="visible_cnn_v1")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--maximum-fpr", type=float, default=0.10)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def seed_everything(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def create_loaders(
    args: argparse.Namespace, normalization: dict[str, object], seed: int, device: torch.device
) -> tuple[DataLoader, DataLoader]:
    common = {
        "manifest_path": args.manifest,
        "project_root": PROJECT_ROOT,
        "image_size": (640, 512),
        "channel_mean": normalization["mean"],
        "channel_std": normalization["std"],
    }
    train_dataset = VisiblePairDataset(split="train", **common)
    validation_dataset = VisiblePairDataset(split="validation", **common)
    generator = torch.Generator().manual_seed(seed)
    loader_arguments = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        generator=generator,
        drop_last=False,
        **loader_arguments,
    )
    validation_loader = DataLoader(
        validation_dataset,
        shuffle=False,
        drop_last=False,
        **loader_arguments,
    )
    return train_loader, validation_loader


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_function: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    total_samples = 0
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = loss_function(logits, labels)
        if not torch.isfinite(loss):
            raise RuntimeError("Training loss became non-finite")
        loss.backward()
        optimizer.step()
        total_loss += float(loss.detach().cpu()) * images.shape[0]
        total_samples += images.shape[0]
    return total_loss / total_samples


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    loss_function: nn.Module,
    device: torch.device,
) -> tuple[float, list[dict[str, object]]]:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    rows: list[dict[str, object]] = []
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            logits = model(images)
            loss = loss_function(logits, labels)
            scores = logits.sigmoid().cpu().numpy()
            label_values = labels.cpu().to(torch.int64).numpy()
            total_loss += float(loss.cpu()) * images.shape[0]
            total_samples += images.shape[0]
            for index in range(images.shape[0]):
                rows.append(
                    {
                        "sample_key": batch["sample_key"][index],
                        "sample_name": batch["sample_name"][index],
                        "category": batch["category"][index],
                        "source": batch["source"][index],
                        "label": int(label_values[index]),
                        "logit": float(logits[index].cpu()),
                        "score": float(scores[index]),
                    }
                )
    return total_loss / total_samples, rows


def save_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"Cannot save empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def plot_history(path: Path, history: list[dict[str, float | int]]) -> None:
    epochs = [int(row["epoch"]) for row in history]
    figure, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
    axis.plot(epochs, [float(row["train_loss"]) for row in history], label="training")
    axis.plot(epochs, [float(row["validation_loss"]) for row in history], label="validation")
    axis.axvline(int(min(history, key=lambda row: float(row["validation_loss"]))["epoch"]), color="gray", linestyle="--", label="best validation loss")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Binary cross-entropy loss")
    axis.set_title("visible_cnn_v1 learning curves")
    axis.legend()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def plot_tradeoff(path: Path, labels: list[int], scores: list[float], selected: dict[str, float | int]) -> None:
    fpr, recall, _ = roc_curve(labels, scores)
    figure, axis = plt.subplots(figsize=(6, 6), constrained_layout=True)
    axis.plot(fpr, recall, label="validation ROC")
    axis.scatter(
        [float(selected["false_positive_rate"])],
        [float(selected["recall"])],
        color="red",
        label="selected at FPR ≤ 10%",
        zorder=3,
    )
    axis.plot([0, 1], [0, 1], color="gray", linestyle="--", label="random ranking")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.set_xlabel("False-positive rate")
    axis.set_ylabel("Defect recall")
    axis.set_title("Validation threshold tradeoff")
    axis.legend()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def per_category_recall(rows: list[dict[str, object]], threshold: float) -> dict[str, dict[str, float | int]]:
    categories = sorted({str(row["category"]) for row in rows})
    result: dict[str, dict[str, float | int]] = {}
    for category in categories:
        anomalies = [row for row in rows if row["category"] == category and int(row["label"]) == 1]
        caught = sum(float(row["score"]) >= threshold for row in anomalies)
        result[category] = {
            "defects": len(anomalies),
            "caught": caught,
            "recall": caught / len(anomalies),
        }
    return result


def run_seed(
    args: argparse.Namespace,
    normalization: dict[str, object],
    seed: int,
    device: torch.device,
) -> dict[str, object]:
    seed_everything(seed)
    run_dir = args.output_root / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    train_loader, validation_loader = create_loaders(args, normalization, seed, device)
    model = VisibleDefectCNN().to(device)
    loss_function = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    checkpoint_path = run_dir / "best_checkpoint.pt"
    history: list[dict[str, float | int]] = []
    best_validation_loss = math.inf
    best_epoch = 0
    epochs_without_improvement = 0
    stop_reason = "maximum_epochs"
    start = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    with mlflow.start_run(run_name=f"{args.run_prefix}_seed_{seed}") as active_run:
        mlflow.log_params(
            {
                "architecture": "visible_cnn_v1",
                "seed": seed,
                "manifest_sha256": sha256_file(args.manifest),
                "normalization_version": normalization["normalization_version"],
                "train_samples": len(train_loader.dataset),
                "validation_samples": len(validation_loader.dataset),
                "input_channels": 2,
                "input_height": 640,
                "input_width": 512,
                "batch_size": args.batch_size,
                "optimizer": "AdamW",
                "learning_rate": args.learning_rate,
                "weight_decay": args.weight_decay,
                "maximum_epochs": args.max_epochs,
                "patience": args.patience,
                "parameter_count": count_trainable_parameters(model),
                "maximum_validation_fpr": args.maximum_fpr,
                "test_split_accessed": False,
            }
        )

        for epoch in range(1, args.max_epochs + 1):
            train_loss = train_epoch(model, train_loader, optimizer, loss_function, device)
            validation_loss, validation_rows = evaluate(
                model, validation_loader, loss_function, device
            )
            labels = [int(row["label"]) for row in validation_rows]
            scores = [float(row["score"]) for row in validation_rows]
            epoch_auroc = ranking_metrics(labels, scores)["auroc"]
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "validation_loss": validation_loss,
                    "validation_auroc": epoch_auroc,
                }
            )
            mlflow.log_metrics(
                {
                    "train_loss": train_loss,
                    "validation_loss": validation_loss,
                    "validation_auroc": epoch_auroc,
                },
                step=epoch,
            )
            print(
                f"seed={seed} epoch={epoch:03d} train_loss={train_loss:.6f} "
                f"val_loss={validation_loss:.6f} val_auroc={epoch_auroc:.4f}",
                flush=True,
            )

            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                best_epoch = epoch
                epochs_without_improvement = 0
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "epoch": epoch,
                        "validation_loss": validation_loss,
                        "seed": seed,
                        "architecture": "visible_cnn_v1",
                        "normalization": normalization,
                    },
                    checkpoint_path,
                )
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= args.patience:
                    stop_reason = "early_stopping"
                    break

        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])
        final_validation_loss, validation_rows = evaluate(
            model, validation_loader, loss_function, device
        )
        labels = [int(row["label"]) for row in validation_rows]
        scores = [float(row["score"]) for row in validation_rows]
        ranking = ranking_metrics(labels, scores)
        curve = threshold_curve(labels, scores)
        selected = select_fpr_constrained_threshold(curve, args.maximum_fpr)
        at_90 = operating_point_at_recall(curve, 0.90)
        at_95 = operating_point_at_recall(curve, 0.95)
        category_recall = per_category_recall(validation_rows, float(selected["threshold"]))

        elapsed_seconds = time.perf_counter() - start
        peak_cuda_bytes = (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
        )
        results: dict[str, object] = {
            "status": "completed",
            "seed": seed,
            "mlflow_run_id": active_run.info.run_id,
            "best_epoch": best_epoch,
            "epochs_completed": int(history[-1]["epoch"]),
            "stop_reason": stop_reason,
            "best_validation_loss": best_validation_loss,
            "reloaded_checkpoint_validation_loss": final_validation_loss,
            **ranking,
            "selected_operating_point": selected,
            "fpr_at_90_recall": at_90,
            "fpr_at_95_recall": at_95,
            "per_category_recall_at_selected_threshold": category_recall,
            "elapsed_seconds": elapsed_seconds,
            "peak_cuda_allocated_bytes": peak_cuda_bytes,
            "test_predictions_generated": False,
        }

        save_csv(run_dir / "training_history.csv", history)
        save_csv(run_dir / "validation_scores.csv", validation_rows)
        save_csv(run_dir / "validation_threshold_curve.csv", curve)
        plot_history(run_dir / "learning_curves.png", history)
        plot_tradeoff(run_dir / "validation_tradeoff.png", labels, scores, selected)
        (run_dir / "results.json").write_text(
            json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        config = {
            "manifest": args.manifest.relative_to(PROJECT_ROOT).as_posix(),
            "manifest_sha256": sha256_file(args.manifest),
            "normalization": args.normalization.relative_to(PROJECT_ROOT).as_posix(),
            "architecture": "visible_cnn_v1",
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "max_epochs": args.max_epochs,
            "patience": args.patience,
            "seed": seed,
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        }
        (run_dir / "config.json").write_text(
            json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

        mlflow.log_metrics(
            {
                "best_validation_loss": best_validation_loss,
                "validation_auroc_final": ranking["auroc"],
                "validation_average_precision_final": ranking["average_precision"],
                "selected_threshold": float(selected["threshold"]),
                "selected_recall": float(selected["recall"]),
                "selected_fpr": float(selected["false_positive_rate"]),
                "fpr_at_90_recall": float(at_90["false_positive_rate"]),
                "fpr_at_95_recall": float(at_95["false_positive_rate"]),
                "elapsed_seconds": elapsed_seconds,
                "peak_cuda_allocated_bytes": float(peak_cuda_bytes or 0),
            }
        )
        mlflow.set_tags(
            {
                "stage": "validation_only",
                "test_split_accessed": "false",
                "model_family": "supervised_binary_cnn",
            }
        )
        mlflow.log_artifacts(str(run_dir))

    print(
        f"completed seed={seed} best_epoch={best_epoch} auroc={ranking['auroc']:.4f} "
        f"selected_recall={float(selected['recall']):.4f} "
        f"selected_fpr={float(selected['false_positive_rate']):.4f}",
        flush=True,
    )
    return results


def main() -> None:
    args = parse_args()
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    if args.max_epochs <= 0 or args.patience <= 0 or args.batch_size <= 0:
        raise ValueError("max_epochs, patience, and batch_size must be positive")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    normalization = json.loads(args.normalization.read_text(encoding="utf-8"))
    if normalization["manifest_sha256"] != sha256_file(args.manifest):
        raise ValueError("Normalization statistics do not match the requested manifest")

    args.output_root.mkdir(parents=True, exist_ok=True)
    args.tracking_db.parent.mkdir(parents=True, exist_ok=True)
    args.artifact_root.mkdir(parents=True, exist_ok=True)
    tracking_uri = f"sqlite:///{args.tracking_db.resolve().as_posix()}"
    mlflow.set_tracking_uri(tracking_uri)
    if mlflow.get_experiment_by_name(args.experiment_name) is None:
        mlflow.create_experiment(
            args.experiment_name,
            artifact_location=args.artifact_root.resolve().as_uri(),
        )
    mlflow.set_experiment(args.experiment_name)

    results = [run_seed(args, normalization, seed, device) for seed in args.seeds]
    summary = {
        "architecture": "visible_cnn_v1",
        "seeds": args.seeds,
        "validation_auroc_mean": float(np.mean([result["auroc"] for result in results])),
        "validation_auroc_std_population": float(
            np.std([result["auroc"] for result in results], ddof=0)
        ),
        "test_predictions_generated": False,
    }
    (args.output_root / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
