# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Branch tests for the quantum kernel encoder
"""Guard tests for the quantum-kernel coupling, feature and matrix validators."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_quantum_control.applications.quantum_kernel import (
    _validated_coupling_matrix,
    _validated_feature_vector,
    compute_kernel_matrix,
)


def test_coupling_rejects_non_positive_qubits() -> None:
    """A non-positive qubit count is rejected."""
    with pytest.raises(ValueError, match="n_qubits must be positive"):
        _validated_coupling_matrix(np.eye(2, dtype=np.float64), 0)


def test_coupling_rejects_non_finite() -> None:
    """A non-finite coupling matrix is rejected."""
    matrix = np.array([[0.0, np.inf], [np.inf, 0.0]], dtype=np.float64)
    with pytest.raises(ValueError, match="K must contain only finite values"):
        _validated_coupling_matrix(matrix, 2)


def test_feature_vector_rejects_non_one_dimensional() -> None:
    """A non-1-D feature vector is rejected."""
    with pytest.raises(ValueError, match="must be a 1-D feature vector"):
        _validated_feature_vector(np.zeros((2, 2), dtype=np.float64))


def test_feature_vector_rejects_non_finite() -> None:
    """A non-finite feature vector is rejected."""
    with pytest.raises(ValueError, match="must contain only finite values"):
        _validated_feature_vector(np.array([0.1, np.inf], dtype=np.float64))


def test_kernel_matrix_rejects_non_finite_features() -> None:
    """A non-finite feature matrix is rejected before encoding."""
    X = np.array([[0.1, np.inf]], dtype=np.float64)
    with pytest.raises(ValueError, match="X must contain only finite values"):
        compute_kernel_matrix(X, np.eye(2, dtype=np.float64), 2)
