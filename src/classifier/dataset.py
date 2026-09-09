"""Scikit-learn 8x8 digits dataset wrapped as a PyTorch Dataset."""

from __future__ import annotations

import numpy as np
from sklearn.datasets import load_digits
from torch.utils.data import Dataset


class Digits(Dataset):
    """Wrapper around the scikit-learn 8x8 digits dataset.

    Splits:
        train: indices  [0:1000]
        val:   indices  [1000:1350]
        test:  indices  [1350:]
    """

    def __init__(self, mode: str = "train", transforms=None):
        digits = load_digits()
        if mode == "train":
            self.data = digits.data[:1000].astype(np.float32)
            self.targets = digits.target[:1000]
        elif mode == "val":
            self.data = digits.data[1000:1350].astype(np.float32)
            self.targets = digits.target[1000:1350]
        else:
            self.data = digits.data[1350:].astype(np.float32)
            self.targets = digits.target[1350:]
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx):
        sample_x = self.data[idx]
        sample_y = self.targets[idx]
        if self.transforms:
            sample_x = self.transforms(sample_x)
        return sample_x, sample_y
