# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Validation tests for the QSNN trainer
"""Fail-closed validation tests for the QSNN trainer and diagnostics helper.

Covers the training-diagnostics loss-history guards, the dataset shape and
finiteness checks, the epoch-count guard and the flat parameter-vector setter
guards.
"""

from __future__ import annotations

import numpy as np
import pytest

from scpn_quantum_control.qsnn.qlayer import QuantumDenseLayer
from scpn_quantum_control.qsnn.training import QSNNTrainer, _training_diagnostics


def _trainer() -> QSNNTrainer:
    return QSNNTrainer(QuantumDenseLayer(2, 3, seed=42))


def test_diagnostics_rejects_empty_loss_history() -> None:
    """An empty loss history is rejected."""
    with pytest.raises(ValueError, match="loss_history must contain at least one epoch"):
        _training_diagnostics(())


def test_diagnostics_rejects_non_finite_loss_history() -> None:
    """A non-finite loss is rejected."""
    with pytest.raises(ValueError, match="loss_history must contain only finite losses"):
        _training_diagnostics((0.5, float("nan")))


def test_dataset_rejects_non_two_dimensional_features() -> None:
    """A non-2-D feature matrix is rejected."""
    trainer = _trainer()
    with pytest.raises(ValueError, match="X must be a two-dimensional array"):
        trainer._validate_dataset(
            np.zeros(4, dtype=np.float64), np.zeros((1, 3), dtype=np.float64)
        )


def test_dataset_rejects_non_two_dimensional_targets() -> None:
    """A non-2-D target matrix is rejected."""
    trainer = _trainer()
    with pytest.raises(ValueError, match="y must be a two-dimensional array"):
        trainer._validate_dataset(
            np.zeros((1, 2), dtype=np.float64), np.zeros(3, dtype=np.float64)
        )


def test_dataset_rejects_wrong_feature_columns() -> None:
    """A feature matrix with the wrong column count is rejected."""
    trainer = _trainer()
    with pytest.raises(ValueError, match="X must have 3 input columns"):
        trainer._validate_dataset(
            np.zeros((1, 2), dtype=np.float64), np.zeros((1, 2), dtype=np.float64)
        )


def test_dataset_rejects_wrong_target_columns() -> None:
    """A target matrix with the wrong column count is rejected."""
    trainer = _trainer()
    with pytest.raises(ValueError, match="y must have 2 target columns"):
        trainer._validate_dataset(
            np.zeros((1, 3), dtype=np.float64), np.zeros((1, 3), dtype=np.float64)
        )


def test_dataset_rejects_non_finite_features() -> None:
    """A non-finite feature value is rejected."""
    trainer = _trainer()
    features = np.array([[0.0, 0.0, np.inf]], dtype=np.float64)
    with pytest.raises(ValueError, match="X must contain only finite values"):
        trainer._validate_dataset(features, np.zeros((1, 2), dtype=np.float64))


def test_dataset_rejects_non_finite_targets() -> None:
    """A non-finite target value is rejected."""
    trainer = _trainer()
    targets = np.array([[0.0, np.nan]], dtype=np.float64)
    with pytest.raises(ValueError, match="y must contain only finite values"):
        trainer._validate_dataset(np.zeros((1, 3), dtype=np.float64), targets)


def test_validate_epochs_rejects_non_positive() -> None:
    """A non-positive epoch count is rejected."""
    with pytest.raises(ValueError, match="epochs must be positive"):
        QSNNTrainer._validate_epochs(0)


def test_set_theta_values_rejects_wrong_shape() -> None:
    """A parameter vector of the wrong length is rejected."""
    trainer = _trainer()
    with pytest.raises(ValueError, match=r"values must have shape \(6,\)"):
        trainer._set_theta_values(np.zeros(5, dtype=np.float64))


def test_set_theta_values_rejects_non_finite() -> None:
    """A non-finite parameter angle is rejected."""
    trainer = _trainer()
    angles = np.array([0.0, 0.0, 0.0, 0.0, 0.0, np.inf], dtype=np.float64)
    with pytest.raises(ValueError, match="values must contain only finite angles"):
        trainer._set_theta_values(angles)
