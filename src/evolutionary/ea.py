"""Evolutionary algorithm implementation for parameter estimation."""

from __future__ import annotations

import numpy as np


class EA:
    """Evolutionary algorithm with rank selection, arithmetic crossover, and elitist survival.

    Attributes
    ----------
    repressilator : object
        Anything exposing an `objective(x)` method that returns fitness for one
        or many candidates. The EA is otherwise model-agnostic.
    pop_size : int
        Number of individuals retained each generation.
    bounds_min, bounds_max : list[float] | None
        Optional bounds on each coordinate (used by the caller to initialize
        the population; not enforced inside the EA).
    """

    def __init__(self, repressilator, pop_size: int, bounds_min=None, bounds_max=None):
        self.repressilator = repressilator
        self.pop_size = pop_size
        self.bounds_min = bounds_min
        self.bounds_max = bounds_max

    def rank_selection(self, f_old: np.ndarray, num_parents: int) -> np.ndarray:
        # Lower fitness is better. argsort twice gives the rank of each individual.
        rank = np.argsort(np.argsort(f_old))
        n = len(f_old)
        probabilities = np.array([(2.0 - 2.0 * g / (n - 1)) / n for g in rank])
        return np.random.choice(np.arange(n), p=probabilities, size=num_parents)

    def parent_selection(self, x_old, f_old):
        idx = self.rank_selection(f_old, self.pop_size)
        return x_old[idx], f_old[idx]

    def recombination(self, x_parents, f_parents):
        """Arithmetic-mean crossover applied pairwise."""
        num_parents, num_variables = x_parents.shape
        x_children = np.zeros((num_parents, num_variables))
        for i in range(0, num_parents - 1, 2):
            child = (x_parents[i] + x_parents[i + 1]) / 2
            x_children[i] = child
            x_children[i + 1] = child
        return x_children

    def mutation(self, x_children):
        # Hook for a future Gaussian / uniform mutation operator.
        return x_children

    def survivor_selection(self, x_old, x_children, f_old, f_children):
        """(mu + lambda) elitist selection: keep the top `pop_size` from the merged pool."""
        x = np.concatenate([x_old, x_children])
        f = np.concatenate([f_old, f_children])
        order = np.argsort(f)
        return x[order[: self.pop_size]], f[order[: self.pop_size]]

    def evaluate(self, x):
        return self.repressilator.objective(x)

    def step(self, x_old, f_old):
        x_parents, f_parents = self.parent_selection(x_old, f_old)
        x_children = self.recombination(x_parents, f_parents)
        x_children = self.mutation(x_children)
        f_children = self.evaluate(x_children)
        return self.survivor_selection(x_old, x_children, f_old, f_children)
