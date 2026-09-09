"""Gene repressilator ODE model used as the black-box objective for the EA."""

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp


class Repressilator:
    """Gene repressilator model with a Runge-Kutta 4(5) ODE solver.

    State: (m1, m2, m3, p1, p2, p3) where m_i are mRNA concentrations and
    p_i are protein concentrations. Only the mRNA channels are assumed
    observable; the loss is the mean Euclidean distance between observed
    and simulated mRNA over time.
    """

    def __init__(self, y_real: np.ndarray, params: dict):
        super().__init__()
        self.y_real = y_real.copy()
        self.params = params.copy()

    def repressilator_model(self, t, y):
        m1, m2, m3, p1, p2, p3 = y[0], y[1], y[2], y[3], y[4], y[5]

        alpha0 = self.params["alpha0"]
        n = self.params["n"]
        beta = self.params["beta"]
        alpha = self.params["alpha"]

        dm1_dt = -m1 + alpha / (1.0 + p3 ** n) + alpha0
        dp1_dt = -beta * (p1 - m1)
        dm2_dt = -m2 + alpha / (1.0 + p1 ** n) + alpha0
        dp2_dt = -beta * (p2 - m2)
        dm3_dt = -m3 + alpha / (1.0 + p2 ** n) + alpha0
        dp3_dt = -beta * (p3 - m3)

        return dm1_dt, dm2_dt, dm3_dt, dp1_dt, dp2_dt, dp3_dt

    def solve_repressilator(self):
        solution = solve_ivp(
            lambda t, y: self.repressilator_model(t, y),
            t_span=(self.params["t0"], self.params["t1"]),
            y0=self.params["y0"],
            method="RK45",
            t_eval=self.params["t_points"],
        )
        y_points = np.asarray(solution.y)
        return self.params["t_points"], y_points

    def set_params(self, x):
        self.params["alpha0"] = x[0]
        self.params["n"] = x[1]
        self.params["beta"] = x[2]
        self.params["alpha"] = x[3]

    @staticmethod
    def loss(y_real, y_model):
        # Only m1, m2, m3 are observed
        y_r = y_real[0:3]
        y_m = y_model[0:3]
        if y_r.shape[1] == y_m.shape[1]:
            return np.mean(np.sqrt(np.sum((y_r - y_m) ** 2, 0)))
        return np.inf

    def objective(self, x):
        if len(x.shape) > 1:
            objective_values = []
            for i in range(x.shape[0]):
                self.set_params(x[i])
                _, y_model = self.solve_repressilator()
                objective_values.append(self.loss(self.y_real, y_model))
            return np.asarray(objective_values)

        self.set_params(x)
        _, y_model = self.solve_repressilator()
        return self.loss(self.y_real, y_model)
