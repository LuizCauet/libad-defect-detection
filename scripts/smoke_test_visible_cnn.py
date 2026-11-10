"""Run end-to-end diagnostic checks for the initial visible-light CNN."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from defect_detection.data import VisiblePairDataset  # noqa: E402
from defect_detection.model import (  # noqa: E402
    VisibleDefectCNN,
    count_trainable_parameters,
    finite_gradient_summary,
)


EXPECTED_PARAMETER_COUNT = 97_425
SMOKE_SEED = 17


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
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "smoke_visible_cnn_v1",
    )
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--target-loss", type=float, default=0.05)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def balanced_indices(dataset: VisiblePairDataset, per_label: int) -> list[int]:
    selected: dict[int, list[int]] = {0: [], 1: []}
    for index, sample in enumerate(dataset.samples):
        if len(selected[sample.label]) < per_label:
            selected[sample.label].append(index)
        if all(len(indices) == per_label for indices in selected.values()):
            break
    if not all(len(indices) == per_label for indices in selected.values()):
        raise RuntimeError("Could not form the balanced smoke-test subset")
    return selected[0] + selected[1]


def load_batch(dataset: VisiblePairDataset, indices: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
    items = [dataset[index] for index in indices]
    images = torch.stack([item["image"] for item in items])
    labels = torch.stack([item["label"] for item in items])
    return images, labels


def write_history(path: Path, history: list[dict[str, float | int]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("step", "loss", "accuracy"))
        writer.writeheader()
        writer.writerows(history)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(SMOKE_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    normalization = json.loads(args.normalization.read_text(encoding="utf-8"))
    dataset_arguments = {
        "manifest_path": args.manifest,
        "project_root": PROJECT_ROOT,
        "image_size": (640, 512),
        "channel_mean": normalization["mean"],
        "channel_std": normalization["std"],
    }
    train_dataset = VisiblePairDataset(split="train", **dataset_arguments)
    validation_dataset = VisiblePairDataset(split="validation", **dataset_arguments)

    train_indices = balanced_indices(train_dataset, per_label=8)
    train_images, train_labels = load_batch(train_dataset, train_indices)
    validation_images, validation_labels = load_batch(validation_dataset, list(range(16)))
    if train_images.shape != (16, 2, 640, 512):
        raise ValueError(f"Unexpected smoke training shape: {tuple(train_images.shape)}")
    if train_images.dtype != torch.float32 or train_labels.dtype != torch.float32:
        raise TypeError("Smoke images and labels must be float32")
    if train_labels.sum().item() != 8:
        raise ValueError("Smoke training subset must contain eight normal and eight defective samples")

    train_images = train_images.to(device)
    train_labels = train_labels.to(device)
    validation_images = validation_images.to(device)
    validation_labels = validation_labels.to(device)

    model = VisibleDefectCNN().to(device)
    parameter_count = count_trainable_parameters(model)
    if parameter_count != EXPECTED_PARAMETER_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_PARAMETER_COUNT:,} parameters, found {parameter_count:,}"
        )
    loss_function = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)

    model.train()
    with torch.no_grad():
        initial_logits = model(train_images)
        initial_loss = float(loss_function(initial_logits, train_labels).cpu())
    if initial_logits.shape != (16,) or not torch.isfinite(initial_logits).all():
        raise ValueError("Initial forward pass did not produce 16 finite logits")

    initial_parameters = {
        name: parameter.detach().clone() for name, parameter in model.named_parameters()
    }
    history: list[dict[str, float | int]] = []
    first_gradient_norms: dict[str, float] | None = None
    start_time = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    passed_overfit = False
    for step in range(1, args.max_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        logits = model(train_images)
        loss = loss_function(logits, train_labels)
        if not torch.isfinite(loss):
            raise ValueError(f"Non-finite loss at step {step}")
        loss.backward()
        gradient_norms = finite_gradient_summary(model)
        if first_gradient_norms is None:
            first_gradient_norms = gradient_norms
        optimizer.step()

        with torch.no_grad():
            accuracy = float(((logits.sigmoid() >= 0.5) == train_labels.bool()).float().mean())
        history.append({"step": step, "loss": float(loss.detach().cpu()), "accuracy": accuracy})
        if accuracy == 1.0 and float(loss.detach().cpu()) <= args.target_loss:
            passed_overfit = True
            break

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed_seconds = time.perf_counter() - start_time
    if not passed_overfit:
        raise RuntimeError(
            f"Model did not overfit the balanced subset within {args.max_steps} steps; "
            f"last result={history[-1]}"
        )

    changed_parameters = [
        name
        for name, parameter in model.named_parameters()
        if not torch.equal(parameter.detach(), initial_parameters[name])
    ]
    if set(changed_parameters) != set(initial_parameters):
        missing = sorted(set(initial_parameters).difference(changed_parameters))
        raise RuntimeError(f"Optimizer did not update every trainable parameter: {missing}")

    model.eval()
    with torch.no_grad():
        validation_logits = model(validation_images)
        validation_loss = loss_function(validation_logits, validation_labels)
    if validation_logits.shape != (16,) or not torch.isfinite(validation_logits).all():
        raise ValueError("Validation inference did not produce 16 finite logits")
    if not torch.isfinite(validation_loss):
        raise ValueError("Validation loss is not finite")

    checkpoint_path = args.output_dir / "checkpoint.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "architecture": "visible_cnn_v1",
            "parameter_count": parameter_count,
            "normalization": normalization,
            "seed": SMOKE_SEED,
            "smoke_train_indices": train_indices,
        },
        checkpoint_path,
    )
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    reloaded_model = VisibleDefectCNN().to(device)
    reloaded_model.load_state_dict(checkpoint["model_state_dict"])
    reloaded_model.eval()
    with torch.no_grad():
        reloaded_logits = reloaded_model(validation_images)
    maximum_reload_difference = float((validation_logits - reloaded_logits).abs().max().cpu())
    if maximum_reload_difference != 0.0:
        raise RuntimeError(
            f"Reloaded logits differ from saved-model logits by {maximum_reload_difference}"
        )

    history_path = args.output_dir / "training_history.csv"
    write_history(history_path, history)
    results = {
        "status": "passed",
        "purpose": "pipeline diagnostic; not a model-quality result",
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "seed": SMOKE_SEED,
        "parameter_count": parameter_count,
        "training_subset_samples": len(train_indices),
        "training_subset_normal": 8,
        "training_subset_defective": 8,
        "initial_loss": initial_loss,
        "final_loss": history[-1]["loss"],
        "final_training_accuracy": history[-1]["accuracy"],
        "steps": history[-1]["step"],
        "target_loss": args.target_loss,
        "validation_batch_loss_diagnostic_only": float(validation_loss.cpu()),
        "first_step_gradient_norms": first_gradient_norms,
        "all_parameters_updated": True,
        "checkpoint_reload_maximum_logit_difference": maximum_reload_difference,
        "elapsed_training_seconds": elapsed_seconds,
        "peak_cuda_allocated_bytes": (
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
        ),
        "final_test_predictions_generated": False,
    }
    results_path = args.output_dir / "results.json"
    results_path.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(json.dumps(results, indent=2, sort_keys=True))
    print(f"Checkpoint: {checkpoint_path.resolve()}")
    print(f"History: {history_path.resolve()}")
    print(f"Results: {results_path.resolve()}")


if __name__ == "__main__":
    main()
