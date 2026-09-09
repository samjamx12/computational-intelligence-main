"""Sampler implementations for Metropolis-Hastings and Simulated Annealing."""

from __future__ import annotations

import numpy as np

EPS = 1.0e-7


class MetropolisHastings:
    """Metropolis-Hastings sampler with a Gaussian random-walk proposal."""

    def __init__(self, x: np.ndarray, prob, std: float = 0.1):
        self.prob = prob
        self.std = std
        self.x_old = x

    def proposal(self, x: np.ndarray) -> np.ndarray:
        return np.random.normal(x, self.std)

    def evaluate(self, x_new: np.ndarray, x_old: np.ndarray) -> np.ndarray:
        p_old = self.prob(x_old)
        p_new = self.prob(x_new)
        A = p_new / (p_old + EPS)
        return np.minimum(1.0, A)

    def select(self, x_new: np.ndarray, A: np.ndarray) -> np.ndarray:
        u = np.random.uniform()
        if u < A:
            self.x_old = x_new
        return self.x_old

    def step(self) -> np.ndarray:
        x_prop = self.proposal(self.x_old)
        A = self.evaluate(x_prop, self.x_old)
        return self.select(x_prop, A)


class SimulatedAnnealing:
    """Simulated annealing with a logarithmic cooling schedule.

    Cooling schedule:  T_t = 1 / (C * log(t + T0))
    """

    def __init__(self, x: np.ndarray, prob, std: float = 0.1, T0: float = 1.0, C: float = 1.0):
        self.prob = prob
        self.std = std
        self.x_old = x
        self.T0 = T0
        self.C = C
        # start at t=1 so the cooling schedule never evaluates log(0) when T0=1
        self.t = 1

    def proposal(self, x: np.ndarray) -> np.ndarray:
        return np.random.normal(x, self.std)

    def evaluate(self, x_new: np.ndarray, x_old: np.ndarray, T: float) -> np.ndarray:
        p_old = self.prob(x_old) ** (1 / T)
        p_new = self.prob(x_new) ** (1 / T)
        A = p_new / (p_old + EPS)
        return np.minimum(1.0, A)

    def select(self, x_new: np.ndarray, A: np.ndarray) -> np.ndarray:
        u = np.random.uniform()
        if u < A:
            self.x_old = x_new
        return self.x_old

    def step(self) -> np.ndarray:
        T = (self.C * np.log(self.t + self.T0)) ** -1
        self.t += 1
        x_prop = self.proposal(self.x_old)
        A = self.evaluate(x_prop, self.x_old, T)
        return self.select(x_prop, A)
