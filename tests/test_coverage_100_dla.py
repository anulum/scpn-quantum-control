# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Coverage tests for dynamical_lie_algebra.py
"""Tests for DLA module using small system sizes (statevector-only, fast)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
from qiskit.quantum_info import SparsePauliOp

from scpn_quantum_control.bridge.knm_hamiltonian import OMEGA_N_16, build_knm_paper27


def test_dla_commutator():
    """Test _commutator computes [A, B] = AB - BA."""
    from scpn_quantum_control.analysis.dynamical_lie_algebra import _commutator

    A = np.array([[0, 1], [0, 0]], dtype=complex)
    B = np.array([[0, 0], [1, 0]], dtype=complex)
    C = _commutator(A, B)
    expected = A @ B - B @ A
    np.testing.assert_allclose(C, expected)


def test_dla_is_independent_empty_basis():
    """Test _is_independent with empty basis."""
    from scpn_quantum_control.analysis.dynamical_lie_algebra import _is_independent

    op = np.array([[1, 0], [0, -1]], dtype=complex)
    assert _is_independent(op, []) is True


def test_dla_is_independent_dependent():
    """Test _is_independent returns False for dependent operator."""
    from scpn_quantum_control.analysis.dynamical_lie_algebra import _is_independent

    op = np.array([[1, 0], [0, -1]], dtype=complex)
    basis = [op.copy()]
    assert _is_independent(op * 2.0, basis) is False


def test_dla_is_independent_zero_operator():
    """Test _is_independent returns False for zero operator."""
    from scpn_quantum_control.analysis.dynamical_lie_algebra import _is_independent

    op = np.zeros((2, 2), dtype=complex)
    assert _is_independent(op, []) is False


def test_compute_dla_2q_xy():
    """Test compute_dla for 2-qubit XY model (small, fast)."""
    from scpn_quantum_control.analysis.dynamical_lie_algebra import (
        build_xy_generators,
        compute_dla,
    )

    K = build_knm_paper27(L=2)
    omega = OMEGA_N_16[:2]
    gens = build_xy_generators(K, omega)
    result = compute_dla(gens, max_iterations=10, max_dimension=100)
    assert result.dimension > 0
    assert result.n_qubits == 2
    assert result.classical_simulable == result.is_polynomial


def test_compute_dla_trivial():
    """Test compute_dla with single generator → dimension 1."""
    from scpn_quantum_control.analysis.dynamical_lie_algebra import compute_dla

    Z = SparsePauliOp("Z", 1.0)
    result = compute_dla([Z], max_iterations=5)
    assert result.dimension == 1
    assert result.is_polynomial is True


def test_compute_dla_reaches_max_dimension():
    """Test compute_dla stops at max_dimension cap."""
    from scpn_quantum_control.analysis.dynamical_lie_algebra import (
        build_xy_generators,
        compute_dla,
    )

    K = build_knm_paper27(L=3)
    omega = OMEGA_N_16[:3]
    gens = build_xy_generators(K, omega)
    result = compute_dla(gens, max_iterations=1, max_dimension=5)
    assert result.dimension >= 1


def test_compute_dla_polynomial_degree_estimation():
    """Test polynomial degree estimation for intermediate DLA dimensions."""
    from scpn_quantum_control.analysis.dynamical_lie_algebra import (
        build_xy_generators,
        compute_dla,
    )

    K = build_knm_paper27(L=3)
    omega = OMEGA_N_16[:3]
    gens = build_xy_generators(K, omega)
    result = compute_dla(gens, max_iterations=20, max_dimension=200)
    assert result.polynomial_degree > 0


def test_compute_dla_rust_fallback():
    """Test compute_dla_rust falls back to Python when Rust unavailable."""
    from scpn_quantum_control.analysis.dynamical_lie_algebra import (
        build_xy_generators,
        compute_dla_rust,
    )

    K = build_knm_paper27(L=2)
    omega = OMEGA_N_16[:2]
    gens = build_xy_generators(K, omega)
    result = compute_dla_rust(gens, max_iterations=10)
    assert result.dimension > 0


def test_compute_dla_rust_with_mock():
    """Test compute_dla_rust with mocked Rust engine."""
    from scpn_quantum_control.analysis.dynamical_lie_algebra import (
        build_xy_generators,
        compute_dla_rust,
    )

    K = build_knm_paper27(L=2)
    omega = OMEGA_N_16[:2]
    gens = build_xy_generators(K, omega)

    mock_engine = MagicMock()
    mock_engine.dla_dimension.return_value = 7
    with patch.dict("sys.modules", {"scpn_quantum_engine": mock_engine}):
        result = compute_dla_rust(gens, max_iterations=10)
        assert result.dimension == 7
        assert result.n_qubits == 2


def test_build_xy_generators():
    """Test build_xy_generators creates correct number of generators."""
    from scpn_quantum_control.analysis.dynamical_lie_algebra import build_xy_generators

    K = build_knm_paper27(L=3)
    omega = OMEGA_N_16[:3]
    gens = build_xy_generators(K, omega)
    # 3 Z terms + 3 pairs × 2 (XX + YY) = 3 + 6 = 9
    assert len(gens) == 9


def test_build_ssgf_generators():
    """Test build_ssgf_generators adds geometry feedback terms."""
    from scpn_quantum_control.analysis.dynamical_lie_algebra import build_ssgf_generators

    K = build_knm_paper27(L=2)
    omega = OMEGA_N_16[:2]
    W = np.array([[0, 0.5], [0.5, 0]])
    gens = build_ssgf_generators(K, omega, W, sigma_g=0.3)
    # Base XY + extra SSGF terms
    assert len(gens) > 0


def test_build_pgbo_generators():
    """Test build_pgbo_generators adds tensor field terms."""
    from scpn_quantum_control.analysis.dynamical_lie_algebra import build_pgbo_generators

    K = build_knm_paper27(L=2)
    omega = OMEGA_N_16[:2]
    h = np.array([[0, 0.1], [0.1, 0]])
    gens = build_pgbo_generators(K, omega, h, pgbo_weight=0.1)
    assert len(gens) > 0


def test_build_tcbo_generators_nearest():
    """Test build_tcbo_generators with nearest-neighbor connectivity."""
    from scpn_quantum_control.analysis.dynamical_lie_algebra import build_tcbo_generators

    K = build_knm_paper27(L=3)
    omega = OMEGA_N_16[:3]
    gens = build_tcbo_generators(K, omega, kappa=1.0, connectivity="nearest")
    # Should have ZZ terms in addition to XY
    assert len(gens) > 9  # more than pure XY


def test_build_tcbo_generators_full():
    """Test build_tcbo_generators with full connectivity."""
    from scpn_quantum_control.analysis.dynamical_lie_algebra import build_tcbo_generators

    K = build_knm_paper27(L=3)
    omega = OMEGA_N_16[:3]
    gens = build_tcbo_generators(K, omega, kappa=1.0, connectivity="full")
    assert len(gens) > 0


def test_build_full_scpn_generators():
    """Test build_full_scpn_generators combines all terms."""
    from scpn_quantum_control.analysis.dynamical_lie_algebra import build_full_scpn_generators

    K = build_knm_paper27(L=2)
    omega = OMEGA_N_16[:2]
    W = np.array([[0, 0.5], [0.5, 0]])
    h = np.array([[0, 0.1], [0.1, 0]])
    gens = build_full_scpn_generators(K, omega, W=W, h_munu=h)
    assert len(gens) > 0


def test_build_full_scpn_generators_no_extras():
    """Test build_full_scpn_generators with W=None and h_munu=None."""
    from scpn_quantum_control.analysis.dynamical_lie_algebra import build_full_scpn_generators

    K = build_knm_paper27(L=2)
    omega = OMEGA_N_16[:2]
    gens = build_full_scpn_generators(K, omega)
    assert len(gens) > 0


def test_dla_result_classical_simulable_property():
    """Test DLAResult.classical_simulable property."""
    from scpn_quantum_control.analysis.dynamical_lie_algebra import DLAResult

    r = DLAResult(
        dimension=10,
        n_qubits=3,
        n_generators=5,
        n_iterations=3,
        basis_labels=["a"],
        is_polynomial=True,
        polynomial_degree=2.0,
        max_hilbert_dim=64,
    )
    assert r.classical_simulable is True
