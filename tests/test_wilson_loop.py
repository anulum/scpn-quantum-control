# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Tests for Wilson Loop
"""Tests for U(1) Wilson loop measurement."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_quantum_control.bridge.knm_hamiltonian import OMEGA_N_16, build_knm_paper27
from scpn_quantum_control.gauge.wilson_loop import (
    WilsonLoopResult,
    _build_wilson_operator,
    _find_loops,
    compute_wilson_loops,
    wilson_loop_expectation,
)
from scpn_quantum_control.hardware.classical import classical_exact_diag


class TestBuildWilsonOperator:
    def test_two_site_hermitian(self):
        W = _build_wilson_operator([0, 1], 4)
        W_mat = W.to_matrix()
        if hasattr(W_mat, "toarray"):
            W_mat = W_mat.toarray()
        np.testing.assert_allclose(W_mat, W_mat.conj().T, atol=1e-12)

    def test_three_site_loop(self):
        W = _build_wilson_operator([0, 1, 2], 4)
        assert W.num_qubits == 4

    def test_raises_for_single_site(self):
        with pytest.raises(ValueError, match="at least 2"):
            _build_wilson_operator([0], 4)


class TestWilsonLoopExpectation:
    def test_product_state(self):
        """Product state |0000> should give well-defined Wilson value."""
        n = 4
        psi = np.zeros(2**n, dtype=complex)
        psi[0] = 1.0
        w = wilson_loop_expectation(psi, [0, 1], n)
        assert isinstance(w, complex)

    def test_magnitude_bounded(self):
        K = build_knm_paper27(L=4)
        omega = OMEGA_N_16[:4]
        exact = classical_exact_diag(4, K=K, omega=omega)
        psi = exact["ground_state"]
        w = wilson_loop_expectation(psi, [0, 1], 4)
        assert abs(w) <= 1.0 + 1e-10


class TestFindLoops:
    def test_complete_graph_has_triangles(self):
        K = build_knm_paper27(L=4)
        loops = _find_loops(K, max_length=3)
        assert len(loops) > 0
        for loop in loops:
            assert len(loop) == 3

    def test_all_to_all_has_squares(self):
        K = build_knm_paper27(L=4)
        loops = _find_loops(K, max_length=4)
        has_square = any(len(lp) == 4 for lp in loops)
        assert has_square

    def test_disconnected_graph_no_loops(self):
        K = np.zeros((4, 4))
        loops = _find_loops(K)
        assert len(loops) == 0


class TestComputeWilsonLoops:
    def test_returns_list(self):
        K = build_knm_paper27(L=4)
        omega = OMEGA_N_16[:4]
        results = compute_wilson_loops(K, omega)
        assert isinstance(results, list)
        assert len(results) > 0

    def test_result_type(self):
        K = build_knm_paper27(L=4)
        omega = OMEGA_N_16[:4]
        results = compute_wilson_loops(K, omega)
        assert isinstance(results[0], WilsonLoopResult)

    def test_magnitude_bounded(self):
        K = build_knm_paper27(L=4)
        omega = OMEGA_N_16[:4]
        results = compute_wilson_loops(K, omega)
        for r in results:
            assert r.magnitude <= 1.0 + 1e-10
            assert r.magnitude >= 0.0

    def test_phase_angle_range(self):
        K = build_knm_paper27(L=4)
        omega = OMEGA_N_16[:4]
        results = compute_wilson_loops(K, omega)
        for r in results:
            assert -np.pi <= r.phase_angle <= np.pi

    def test_max_loops_limit(self):
        K = build_knm_paper27(L=4)
        omega = OMEGA_N_16[:4]
        results = compute_wilson_loops(K, omega, max_loops=3)
        assert len(results) <= 3

    def test_scpn_wilson_loops(self):
        """Record Wilson loop values at SCPN default parameters."""
        K = build_knm_paper27(L=4)
        omega = OMEGA_N_16[:4]
        results = compute_wilson_loops(K, omega, max_loops=5)
        print("\n  Wilson loops (4 osc, default K):")
        for r in results:
            print(f"    Loop {r.loop}: |W|={r.magnitude:.4f}, arg={r.phase_angle:.4f}")
        assert len(results) > 0
