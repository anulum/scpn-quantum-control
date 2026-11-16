# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — experimental LLM-QPU logistic readout
"""Multinomial logistic readout with an unpenalised intercept and an L2 weight penalty."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

_FloatArray = np.ndarray[Any, np.dtype[np.float64]]
_IntArray = np.ndarray[Any, np.dtype[np.int64]]


def _log_softmax(logits: _FloatArray) -> _FloatArray:
    """Return row-wise log-softmax, stable for large logits."""
    shifted = logits - logits.max(axis=1, keepdims=True)
    result: _FloatArray = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    return result


@dataclass(frozen=True, slots=True)
class LogisticReadout:
    """Fitted weights (features x classes), intercepts (classes) and the optimiser's outcome."""

    weights: _FloatArray
    intercepts: _FloatArray
    l2: float
    converged: bool
    iterations: int

    @property
    def parameter_count(self) -> int:
        """Return the number of trainable parameters, intercepts included."""
        return int(self.weights.size + self.intercepts.size)

    def log_probabilities(self, features: object) -> _FloatArray:
        """Return the log-probability of every class for every row."""
        matrix = np.asarray(features, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[1] != self.weights.shape[0]:
            raise ValueError("features do not have the fitted width")
        return _log_softmax(matrix @ self.weights + self.intercepts)


def _labels(labels: object, classes: int, rows: int) -> _IntArray:
    """Return integer labels in range for the given row count, or refuse."""
    array = np.asarray(labels)
    if array.shape != (rows,) or array.dtype.kind not in "iu":
        raise ValueError("labels must be one integer per row")
    if array.min() < 0 or array.max() >= classes:
        raise ValueError("labels must name a declared class")
    return array.astype(np.int64)


def fit_logistic(
    features: object, labels: object, *, classes: int, l2: float, max_iter: int = 2000
) -> LogisticReadout:
    """Minimise mean cross-entropy plus l2/2 * ||weights||^2 with L-BFGS and an analytic gradient."""
    from scipy.optimize import minimize

    matrix = np.asarray(features, dtype=np.float64)
    if matrix.ndim != 2 or not matrix.size or not np.isfinite(matrix).all():
        raise ValueError("features must be a nonempty finite matrix")
    if type(classes) is not int or classes < 2:
        raise ValueError("a readout needs at least two classes")
    if not (type(l2) is float and np.isfinite(l2) and l2 >= 0.0):
        raise ValueError("l2 must be a non-negative finite float")
    rows, width = matrix.shape
    target = np.eye(classes)[_labels(labels, classes, rows)]

    def objective(flat: _FloatArray) -> tuple[float, _FloatArray]:
        """Return the penalised mean cross-entropy and its gradient."""
        weights = flat[: width * classes].reshape(width, classes)
        logp = _log_softmax(matrix @ weights + flat[width * classes :])
        residual = (np.exp(logp) - target) / rows
        loss = float(-(target * logp).sum() / rows + 0.5 * l2 * (weights**2).sum())
        grad = np.concatenate([(matrix.T @ residual + l2 * weights).ravel(), residual.sum(axis=0)])
        return loss, grad

    start = np.zeros(width * classes + classes)
    result = minimize(
        objective, start, jac=True, method="L-BFGS-B", options={"maxiter": max_iter, "gtol": 1e-8}
    )
    flat = np.asarray(result.x, dtype=np.float64)
    return LogisticReadout(
        flat[: width * classes].reshape(width, classes),
        flat[width * classes :],
        l2,
        bool(result.success),
        int(result.nit),
    )


def mean_log_loss(readout: LogisticReadout, features: object, labels: object) -> float:
    """Return the mean negative log-probability of the true class."""
    logp = readout.log_probabilities(features)
    index = _labels(labels, readout.intercepts.shape[0], logp.shape[0])
    return float(-logp[np.arange(logp.shape[0]), index].mean())
