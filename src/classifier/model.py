"""Model architectures: helper layers, an MLP, a CNN, and a generic classifier wrapper."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Reshape(nn.Module):
    """Reshape a flat input to a tensor of the given shape (excluding batch)."""

    def __init__(self, size):
        super().__init__()
        self.size = size

    def forward(self, x):
        assert x.shape[1] == int(np.prod(self.size))
        return x.view(x.shape[0], *self.size)


class Flatten(nn.Module):
    """Flatten everything past the batch dimension."""

    def forward(self, x):
        return x.view(x.shape[0], -1)


def build_mlp(input_dim: int = 64, hidden_dim: int = 256, num_classes: int = 10) -> nn.Sequential:
    """Standard fully-connected classifier: Linear -> ReLU -> Dropout -> Linear -> LogSoftmax."""
    return nn.Sequential(
        Flatten(),
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Linear(hidden_dim, num_classes),
        nn.LogSoftmax(dim=1),
    )


def build_cnn(num_kernels: int = 32, hidden_dim: int = 256, num_classes: int = 10) -> nn.Sequential:
    """Two convolutional blocks + dense head for 8x8 grayscale digit images."""
    return nn.Sequential(
        Reshape(size=(1, 8, 8)),
        nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(in_channels=16, out_channels=num_kernels, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.BatchNorm2d(num_kernels),
        Flatten(),
        nn.Linear(num_kernels * 2 * 2, hidden_dim),
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Linear(hidden_dim, num_classes),
        nn.LogSoftmax(dim=1),
    )


class ClassifierNeuralNet(nn.Module):
    """Generic classifier that wraps any LogSoftmax-producing network and exposes NLL loss."""

    def __init__(self, classnet: nn.Module):
        super().__init__()
        self.classnet = classnet
        self.nll = nn.NLLLoss(reduction="none")

    def classify(self, x: torch.Tensor) -> torch.Tensor:
        y_pred = self.classnet(x)
        return torch.argmax(y_pred, dim=1)

    def forward(self, x: torch.Tensor, y: torch.Tensor, reduction: str = "avg") -> torch.Tensor:
        y_pred = self.classnet(x)
        loss = F.nll_loss(F.log_softmax(y_pred, dim=1), y, reduction="none")
        return loss.sum() if reduction == "sum" else loss.mean()
