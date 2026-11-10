"""Initial visible-light binary CNN specified for the LIBAD project."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class VisibleDefectCNN(nn.Module):
    """Four-stage early-fusion CNN for paired visible-light images."""

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            self._block(2, 16),
            self._block(16, 32),
            self._block(32, 64),
            self._block(64, 128),
        )
        self.global_max_pool = nn.AdaptiveMaxPool2d(output_size=1)
        self.classifier = nn.Linear(128, 1)

    @staticmethod
    def _block(input_channels: int, output_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(
                input_channels,
                output_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def forward(self, images: Tensor) -> Tensor:
        if images.ndim != 4 or images.shape[1] != 2:
            raise ValueError(
                "VisibleDefectCNN expects batch × 2 channels × height × width; "
                f"received {tuple(images.shape)}"
            )
        features = self.features(images)
        pooled = self.global_max_pool(features).flatten(start_dim=1)
        return self.classifier(pooled).squeeze(dim=1)


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def finite_gradient_summary(model: nn.Module) -> dict[str, float]:
    """Return gradient norms and reject missing or non-finite gradients."""
    norms: dict[str, float] = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.grad is None:
            raise RuntimeError(f"Trainable parameter has no gradient: {name}")
        if not torch.isfinite(parameter.grad).all():
            raise RuntimeError(f"Trainable parameter has a non-finite gradient: {name}")
        norms[name] = float(parameter.grad.norm().detach().cpu())
    if not any(value > 0 for value in norms.values()):
        raise RuntimeError("All trainable parameter gradients are zero")
    return norms
