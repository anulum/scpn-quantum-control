# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Branch tests for readout-error mitigation
"""Guard tests for the readout confusion-matrix mitigation helpers.

Covers the qubit-count guard, the observed-vector shape guard and the
zero-vector mitigation guard.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest

from scpn_quantum_control.mitigation.readout_matrix import (
    ReadoutConfusionMatrix,
    computational_basis_labels,
    mitigate_probabilities,
)


def test_basis_labels_rejects_non_positive_qubits() -> None:
    """A non-positive qubit count is rejected."""
    with pytest.raises(ValueError, match="n_qubits must be positive"):
        computational_basis_labels(0)


def test_mitigate_rejects_incompatible_shape() -> None:
    """An observed probability vector of the wrong length is rejected."""
    confusion = cast(
        ReadoutConfusionMatrix,
        SimpleNamespace(labels=("0", "1"), matrix=np.eye(2, dtype=np.float64)),
    )
    with pytest.raises(ValueError, match="incompatible shape"):
        mitigate_probabilities(np.zeros(3, dtype=np.float64), confusion)


def test_mitigate_rejects_zero_vector_result() -> None:
    """A confusion matrix that maps the observed vector below zero is rejected."""
    confusion = cast(
        ReadoutConfusionMatrix,
        SimpleNamespace(labels=("0", "1"), matrix=-np.eye(2, dtype=np.float64)),
    )
    observed = np.array([0.5, 0.5], dtype=np.float64)
    with pytest.raises(ValueError, match="readout mitigation produced a zero vector"):
        mitigate_probabilities(observed, confusion)
