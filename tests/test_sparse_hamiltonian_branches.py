# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Branch tests for the sparse Hamiltonian builder
"""Guard tests for the sparse XY Hamiltonian helpers.

Covers the coupling shape guard, the frequency-vector guard, the Rust import
fallback and the sparsity-statistics dimension guard.
"""

from __future__ import annotations

import sys

import numpy as np
import pytest

from scpn_quantum_control.bridge.sparse_hamiltonian import (
    _canonical_xy_coupling,
    _canonical_xy_inputs,
    _try_rust_sparse,
    sparsity_stats,
)


def test_canonical_coupling_rejects_non_square() -> None:
    """A non-square coupling matrix is rejected."""
    with pytest.raises(ValueError, match="K must be a square coupling matrix"):
        _canonical_xy_coupling(np.zeros((2, 3), dtype=np.float64))


def test_canonical_inputs_rejects_wrong_omega() -> None:
    """A frequency vector of the wrong length is rejected."""
    with pytest.raises(ValueError, match="omega must be a vector of length 2"):
        _canonical_xy_inputs(np.eye(2, dtype=np.float64), np.zeros(3, dtype=np.float64))


def test_try_rust_sparse_returns_none_without_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without the native engine the Rust sparse builder returns None."""
    monkeypatch.setitem(sys.modules, "scpn_quantum_engine", None)
    result = _try_rust_sparse(np.eye(2, dtype=np.float64), np.zeros(2, dtype=np.float64), 2)
    assert result is None


def test_sparsity_stats_rejects_dimension_mismatch() -> None:
    """A coupling matrix that does not match the qubit count is rejected."""
    with pytest.raises(ValueError, match=r"K has shape .*expected \(3, 3\)"):
        sparsity_stats(3, np.zeros((2, 2), dtype=np.float64))
