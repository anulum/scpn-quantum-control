# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Kernel ridge classifier
"""Custody-checked binary kernel ridge fitting and evaluation."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .schema import FloatArray, IntArray, KernelEvaluation, TopologyKernelMatrix


def _read_only_float(value: NDArray[np.float64]) -> FloatArray:
    result = np.array(value, dtype=np.float64, copy=True)
    result.setflags(write=False)
    return result


def _read_only_int(value: NDArray[np.int64]) -> IntArray:
    result = np.array(value, dtype=np.int64, copy=True)
    result.setflags(write=False)
    return result


def _is_digest(value: str) -> bool:
    return len(value) == 64 and all(char in "0123456789abcdef" for char in value)


@dataclass(frozen=True, slots=True)
class KernelRidgeClassifier:
    """Immutable binary classifier fitted from a precomputed kernel.

    Parameters
    ----------
    train_ids:
        Ordered identifiers for coefficient alignment at prediction time.
    coefficients:
        Solution of ``(K + alpha I) coefficients = labels``.
    alpha:
        Positive diagonal regularisation used during fitting.
    topology_digest:
        Kernel/topology family digest required on every prediction matrix.
    training_kernel_digest:
        Content digest of the exact square training kernel.
    content_digest:
        SHA-256 binding identifiers, coefficients, alpha, and kernel custody.

    """

    train_ids: tuple[str, ...]
    coefficients: FloatArray
    alpha: float
    topology_digest: str
    training_kernel_digest: str
    content_digest: str

    def __post_init__(self) -> None:
        """Validate custody fields and freeze a private coefficient copy."""
        coefficients = np.asarray(self.coefficients, dtype=np.float64)
        if coefficients.ndim != 1 or coefficients.shape != (len(self.train_ids),):
            raise ValueError("coefficients must match train_ids")
        if not self.train_ids or len(set(self.train_ids)) != len(self.train_ids):
            raise ValueError("train_ids must be non-empty and unique")
        if not np.all(np.isfinite(coefficients)):
            raise ValueError("coefficients must be finite")
        if not np.isfinite(self.alpha) or self.alpha <= 0.0:
            raise ValueError("alpha must be finite and positive")
        for name in ("topology_digest", "training_kernel_digest", "content_digest"):
            if not _is_digest(getattr(self, name)):
                raise ValueError(f"{name} must be a lowercase SHA-256 digest")
        object.__setattr__(self, "coefficients", _read_only_float(coefficients))
        object.__setattr__(self, "alpha", float(self.alpha))


def fit_kernel_ridge(
    kernel: TopologyKernelMatrix,
    labels: NDArray[np.int64],
    *,
    alpha: float,
) -> KernelRidgeClassifier:
    """Fit a binary ridge classifier from a square training kernel.

    Parameters
    ----------
    kernel:
        Square, identifier-aligned training kernel.
    labels:
        One ``-1`` or ``+1`` label per training row.
    alpha:
        Positive finite diagonal regularisation.

    Returns
    -------
    KernelRidgeClassifier
        Immutable coefficient vector with exact kernel custody.

    Raises
    ------
    ValueError
        If identifiers are misaligned, labels are not binary, regularisation
        is invalid, or the solve produces non-finite coefficients.

    """
    if not isinstance(kernel, TopologyKernelMatrix):
        raise ValueError("kernel must be a TopologyKernelMatrix")
    if kernel.values.shape[0] != kernel.values.shape[1] or kernel.row_ids != kernel.column_ids:
        raise ValueError("training kernel must be square with identical axis identifiers")
    label_array = np.asarray(labels, dtype=np.int64)
    if label_array.shape != (kernel.values.shape[0],) or not set(label_array.tolist()) <= {-1, 1}:
        raise ValueError("labels must be a binary vector matching the training kernel")
    if not np.isfinite(alpha) or alpha <= 0.0:
        raise ValueError("alpha must be finite and positive")
    regularised = kernel.values + float(alpha) * np.eye(kernel.values.shape[0])
    coefficients = np.asarray(np.linalg.solve(regularised, label_array), dtype=np.float64)
    if not np.all(np.isfinite(coefficients)):
        raise ValueError("kernel ridge solve produced non-finite coefficients")
    digest = hashlib.sha256()
    digest.update("\x00".join(kernel.row_ids).encode())
    digest.update(np.ascontiguousarray(coefficients, dtype="<f8").tobytes())
    digest.update(np.asarray([alpha], dtype="<f8").tobytes())
    digest.update(kernel.topology_digest.encode())
    digest.update(kernel.content_digest.encode())
    return KernelRidgeClassifier(
        train_ids=kernel.row_ids,
        coefficients=coefficients,
        alpha=float(alpha),
        topology_digest=kernel.topology_digest,
        training_kernel_digest=kernel.content_digest,
        content_digest=digest.hexdigest(),
    )


def predict_kernel_ridge(
    model: KernelRidgeClassifier,
    cross_kernel: TopologyKernelMatrix,
) -> IntArray:
    """Predict binary labels from a test-by-train cross-kernel matrix.

    The cross-kernel columns must exactly match the fitted ``train_ids`` and
    its topology/control digest must match the fitted kernel family. Scores at
    exactly zero deterministically map to ``+1``.
    """
    if not isinstance(model, KernelRidgeClassifier):
        raise ValueError("model must be a KernelRidgeClassifier")
    if not isinstance(cross_kernel, TopologyKernelMatrix):
        raise ValueError("cross_kernel must be a TopologyKernelMatrix")
    if cross_kernel.column_ids != model.train_ids:
        raise ValueError("cross-kernel columns must exactly match model train_ids")
    if cross_kernel.topology_digest != model.topology_digest:
        raise ValueError("cross-kernel topology digest does not match the fitted model")
    scores = np.asarray(cross_kernel.values @ model.coefficients, dtype=np.float64)
    predictions = np.where(scores >= 0.0, 1, -1).astype(np.int64)
    return _read_only_int(predictions)


def evaluate_kernel_ridge(
    name: str,
    model: KernelRidgeClassifier,
    cross_kernel: TopologyKernelMatrix,
    labels: NDArray[np.int64],
) -> KernelEvaluation:
    """Predict and return a self-consistent named accuracy record.

    Parameters
    ----------
    name:
        Stable label such as ``ring`` or ``classical_rbf``.
    model:
        Previously fitted kernel ridge classifier.
    cross_kernel:
        Test-by-train kernel with matching custody and training identifiers.
    labels:
        Expected binary test labels, one per cross-kernel row.

    """
    predictions = predict_kernel_ridge(model, cross_kernel)
    label_array = np.asarray(labels, dtype=np.int64)
    if label_array.shape != predictions.shape or not set(label_array.tolist()) <= {-1, 1}:
        raise ValueError("labels must be a binary vector matching cross-kernel rows")
    correct = int(np.sum(predictions == label_array))
    return KernelEvaluation(
        name=name,
        predictions=predictions,
        labels=label_array,
        correct=correct,
        total=predictions.size,
        accuracy=correct / predictions.size,
        kernel_digest=cross_kernel.content_digest,
    )
