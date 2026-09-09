"""Optimizer implementations for the GD vs DFO comparison."""

from __future__ import annotations

import numpy as np


def f(x: np.ndarray) -> np.ndarray:
    """Non-convex 2D test function.

    f(x) = x1^2 + 2*x2^2 - 0.3*cos(3*pi*x1) - 0.4*cos(4*pi*x2) + 0.7

    Parameters
    ----------
    x : np.ndarray of shape (n, 2)

    Returns
    -------
    np.ndarray of shape (n,)
    """
    return (
        x[:, 0] ** 2
        + 2 * x[:, 1] ** 2
        - 0.3 * np.cos(3.0 * np.pi * x[:, 0])
        - 0.4 * np.cos(4.0 * np.pi * x[:, 1])
        + 0.7
    )


def grad(x: np.ndarray) -> np.ndarray:
    """Analytical gradient of f."""
    grad1 = 2 * x[:, 0] + 0.9 * np.pi * np.sin(3 * np.pi * x[:, 0])
    grad2 = 4 * x[:, 1] + 1.6 * np.pi * np.sin(4 * np.pi * x[:, 1])
    return np.column_stack([grad1, grad2])


class GradientDescent:
    """Vanilla gradient-descent optimizer."""

    def __init__(self, grad_fn, step_size: float = 0.1):
        self.grad = grad_fn
        self.step_size = step_size

    def step(self, x_old: np.ndarray) -> np.ndarray:
        return x_old - self.step_size * self.grad(x_old)


class RandomSearchDFO:
    """Pure random-search DFO. Accepts a candidate only if it improves the objective."""

    def __init__(self, f_fn, search_grid_x1, search_grid_x2, step_size=None):
        self.f = f_fn
        self.x1 = search_grid_x1
        self.x2 = search_grid_x2
        self.step_size = step_size  # kept for API compatibility / labeling only

    def _sample(self) -> np.ndarray:
        cur_x1 = np.random.choice(self.x1)
        cur_x2 = np.random.choice(self.x2)
        return np.array([[cur_x1, cur_x2]])

    def step(self, x_old: np.ndarray) -> np.ndarray:
        x_cand = self._sample()
        if self.f(x_cand) < self.f(x_old):
            return x_cand
        return x_old
