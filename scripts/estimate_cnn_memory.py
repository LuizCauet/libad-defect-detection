"""Estimate and measure training memory for the proposed visible-light CNN."""

from __future__ import annotations

import argparse

import torch
from torch import nn


HEIGHT = 640
WIDTH = 512
CHANNELS = (2, 16, 32, 64, 128)


class VisibleDefectCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        blocks = []
        for input_channels, output_channels in zip(CHANNELS[:-1], CHANNELS[1:]):
            blocks.extend(
                [
                    nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2),
                ]
            )
        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveMaxPool2d(1)
        self.classifier = nn.Linear(128, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.features(inputs)
        pooled = self.pool(features).flatten(1)
        return self.classifier(pooled).squeeze(1)


def mib(byte_count: int | float) -> float:
    return float(byte_count) / (1024**2)


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def analytic_activation_elements() -> list[tuple[str, int]]:
    shapes = [
        ("input", 2, 640, 512),
        ("conv1", 16, 640, 512),
        ("pool1", 16, 320, 256),
        ("conv2", 32, 320, 256),
        ("pool2", 32, 160, 128),
        ("conv3", 64, 160, 128),
        ("pool3", 64, 80, 64),
        ("conv4", 128, 80, 64),
        ("pool4", 128, 40, 32),
        ("global_max_pool", 128, 1, 1),
    ]
    return [(name, channels * height * width) for name, channels, height, width in shapes]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    model = VisibleDefectCNN()
    parameters = parameter_count(model)
    activations = analytic_activation_elements()
    per_sample_bytes = sum(elements * 4 for _, elements in activations)

    print(f"Parameters: {parameters:,}")
    print(f"FP32 parameter weights: {mib(parameters * 4):.3f} MiB")
    print(
        "Weights + gradients + two AdamW moment buffers: "
        f"{mib(parameters * 4 * 4):.3f} MiB"
    )
    print("Forward tensors per sample (lower-bound accounting):")
    for name, elements in activations:
        print(f"  {name:16s} {elements:>10,} values  {mib(elements * 4):>8.3f} MiB")
    print(f"  {'total':16s} {'':>10s}         {mib(per_sample_bytes):>8.3f} MiB")
    print(
        f"Batch {args.batch_size} forward-tensor lower bound: "
        f"{mib(per_sample_bytes * args.batch_size):.3f} MiB"
    )

    if not torch.cuda.is_available():
        print("CUDA is unavailable; skipped empirical training-step measurement.")
        return

    device = torch.device("cuda")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    free_before, total_memory = torch.cuda.mem_get_info(device)

    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    loss_function = nn.BCEWithLogitsLoss()
    inputs = torch.randn(args.batch_size, 2, HEIGHT, WIDTH, device=device)
    targets = torch.randint(0, 2, (args.batch_size,), device=device, dtype=torch.float32)

    optimizer.zero_grad(set_to_none=True)
    logits = model(inputs)
    loss = loss_function(logits, targets)
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize(device)

    free_after, _ = torch.cuda.mem_get_info(device)
    print(f"GPU: {torch.cuda.get_device_name(device)}")
    print(f"Total GPU memory: {mib(total_memory):.1f} MiB")
    print(f"Peak PyTorch allocated: {mib(torch.cuda.max_memory_allocated(device)):.1f} MiB")
    print(f"Peak PyTorch reserved: {mib(torch.cuda.max_memory_reserved(device)):.1f} MiB")
    print(f"Change in globally free VRAM: {mib(free_before - free_after):.1f} MiB")


if __name__ == "__main__":
    main()
