# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Dense budget guard tests
"""Tests for dense Hilbert-space allocation guards."""

from __future__ import annotations

import numpy as np
import pytest

import scpn_quantum_control.dense_budget as dense_budget_mod
from scpn_quantum_control.bridge.knm_hamiltonian import knm_to_dense_matrix
from scpn_quantum_control.dense_budget import (
    DEFAULT_DENSE_BUDGET_CAP_GIB,
    DEFAULT_DENSE_BUDGET_ENV,
    GIB,
    DenseAllocationError,
    available_memory_bytes,
    dense_budget_bytes,
    dense_object_bytes,
    estimate_dense_allocation,
    hilbert_dimension,
    require_dense_allocation,
    require_dense_eigensolver_workspace,
)


def test_dense_budget_estimates_complex_hamiltonian_bytes() -> None:
    estimate = estimate_dense_allocation(
        4,
        dtype=np.complex128,
        rank=2,
        max_gib=1.0,
        label="test Hamiltonian",
    )

    assert estimate.dimension == 16
    assert estimate.shape == (16, 16)
    assert estimate.bytes_required == 16 * 16 * np.dtype(np.complex128).itemsize
    assert estimate.budget_gib == pytest.approx(1.0)


def test_dense_budget_rejects_invalid_qubit_count() -> None:
    with pytest.raises(ValueError, match="n_qubits"):
        hilbert_dimension(0)


def test_dense_budget_fails_before_large_allocation() -> None:
    with pytest.raises(DenseAllocationError, match="dense XY Hamiltonian"):
        require_dense_allocation(
            16,
            dtype=np.complex128,
            rank=2,
            max_gib=1.0,
            label="dense XY Hamiltonian",
        )


def test_dense_budget_allows_small_allocation() -> None:
    estimate = require_dense_allocation(
        4,
        dtype=np.complex128,
        rank=2,
        max_gib=1.0,
        label="small Hamiltonian",
    )

    assert estimate.bytes_required == dense_object_bytes(4, dtype=np.complex128, rank=2)


def test_knm_dense_matrix_fails_closed_before_rust_or_qiskit_allocation() -> None:
    n = 16
    K = np.zeros((n, n), dtype=float)
    omega = np.ones(n, dtype=float)

    with pytest.raises(DenseAllocationError, match="sparse, sector, tensor-network"):
        knm_to_dense_matrix(K, omega, max_dense_gib=1.0)


def test_dense_eigensolver_workspace_uses_conservative_multiplier() -> None:
    with pytest.raises(DenseAllocationError, match="test dense eigensolver"):
        require_dense_eigensolver_workspace(
            4,
            max_gib=1e-5,
            label="test dense eigensolver",
        )


def test_hilbert_dimension_rejects_non_integer() -> None:
    """A non-integer qubit count is a type error."""
    with pytest.raises(TypeError, match="n_qubits must be an integer"):
        hilbert_dimension(4.0)  # type: ignore[arg-type]


def test_dense_object_bytes_rejects_rank_below_one() -> None:
    """Rank must be at least one to describe a vector or matrix."""
    with pytest.raises(ValueError, match="rank must be >= 1"):
        dense_object_bytes(4, rank=0)


def test_estimate_rejects_object_count_below_one() -> None:
    """At least one object must be allocated for an estimate to be meaningful."""
    with pytest.raises(ValueError, match="object_count must be >= 1"):
        estimate_dense_allocation(4, object_count=0)


def test_dense_budget_bytes_rejects_non_positive_max_gib() -> None:
    """An explicit non-positive budget is rejected."""
    with pytest.raises(ValueError, match="max_gib must be positive"):
        dense_budget_bytes(max_gib=0.0)


def test_dense_budget_bytes_reads_environment_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A valid environment override sets the budget directly."""
    monkeypatch.setenv(DEFAULT_DENSE_BUDGET_ENV, "2")
    assert dense_budget_bytes() == int(2 * GIB)


def test_dense_budget_bytes_rejects_non_numeric_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A non-numeric environment override is rejected."""
    monkeypatch.setenv(DEFAULT_DENSE_BUDGET_ENV, "not-a-number")
    with pytest.raises(ValueError, match="must be a positive number"):
        dense_budget_bytes()


def test_dense_budget_bytes_rejects_non_positive_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A non-positive environment override is rejected."""
    monkeypatch.setenv(DEFAULT_DENSE_BUDGET_ENV, "0")
    with pytest.raises(ValueError, match="must be positive"):
        dense_budget_bytes()


def test_dense_budget_bytes_falls_back_to_cap_without_memory_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When host memory is undiscoverable, the conservative cap applies."""
    monkeypatch.delenv(DEFAULT_DENSE_BUDGET_ENV, raising=False)
    monkeypatch.setattr(dense_budget_mod, "available_memory_bytes", lambda: None)
    assert dense_budget_bytes() == int(DEFAULT_DENSE_BUDGET_CAP_GIB * GIB)


def test_dense_budget_bytes_uses_memory_fraction_when_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A discoverable, tiny host memory caps the budget by its fraction."""
    monkeypatch.delenv(DEFAULT_DENSE_BUDGET_ENV, raising=False)
    monkeypatch.setattr(dense_budget_mod, "available_memory_bytes", lambda: GIB)
    assert dense_budget_bytes() < int(DEFAULT_DENSE_BUDGET_CAP_GIB * GIB)


def test_available_memory_bytes_handles_missing_sysconf(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A platform without SC_AVPHYS_PAGES reports unknown memory."""

    def _raise(_name: str) -> int:
        raise ValueError("unknown sysconf name")

    monkeypatch.setattr("os.sysconf", _raise)
    assert available_memory_bytes() is None


def test_available_memory_bytes_rejects_non_positive_counts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Non-positive page counts are treated as undiscoverable memory."""
    monkeypatch.setattr("os.sysconf", lambda _name: 0)
    assert available_memory_bytes() is None


def test_available_memory_bytes_returns_product_when_discoverable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Discoverable pages and page size multiply into available bytes."""
    monkeypatch.setattr("os.sysconf", lambda _name: 16)
    assert available_memory_bytes() == 16 * 16
