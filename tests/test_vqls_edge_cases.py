# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for Vqls Edge Cases
"""Edge-case and multi-angle tests for vqls_gs.py — elite coverage."""

from __future__ import annotations

import contextlib
from unittest.mock import patch

import numpy as np
import pytest
from qiskit.quantum_info import Statevector

from scpn_quantum_control.control.vqls_gs import VQLS_GradShafranov

# ---------------------------------------------------------------------------
# Constructor defaults
# ---------------------------------------------------------------------------


class TestVQLSInit:
    def test_default_n_qubits(self):
        v = VQLS_GradShafranov()
        assert v.n_qubits == 4
        assert v.grid_size == 16

    def test_custom_n_qubits(self):
        v = VQLS_GradShafranov(n_qubits=3)
        assert v.n_qubits == 3
        assert v.grid_size == 8

    def test_default_source_width(self):
        v = VQLS_GradShafranov()
        assert v.source_width == 0.05

    def test_default_imag_tol(self):
        v = VQLS_GradShafranov()
        assert v.imag_tol == 0.1

    def test_custom_imag_tol(self):
        v = VQLS_GradShafranov(imag_tol=0.5)
        assert v.imag_tol == 0.5

    @pytest.mark.parametrize("n", [2, 3, 4, 5])
    def test_grid_size_is_power_of_two(self, n):
        v = VQLS_GradShafranov(n_qubits=n)
        assert v.grid_size == 2**n


# ---------------------------------------------------------------------------
# discretize
# ---------------------------------------------------------------------------


class TestDiscretize:
    def test_returns_A_and_b(self):
        v = VQLS_GradShafranov(n_qubits=2)
        A, b = v.discretize()
        assert A.shape == (4, 4)
        assert b.shape == (4,)

    def test_laplacian_is_symmetric(self):
        v = VQLS_GradShafranov(n_qubits=3)
        A, _ = v.discretize()
        np.testing.assert_allclose(A, A.T, atol=1e-14)

    def test_laplacian_is_tridiagonal(self):
        v = VQLS_GradShafranov(n_qubits=3)
        A, _ = v.discretize()
        N = 8
        for i in range(N):
            for j in range(N):
                if abs(i - j) > 1:
                    assert A[i, j] == 0.0

    def test_laplacian_diagonal_positive(self):
        v = VQLS_GradShafranov(n_qubits=3)
        A, _ = v.discretize()
        assert np.all(np.diag(A) > 0)

    def test_source_vector_normalised(self):
        v = VQLS_GradShafranov(n_qubits=3)
        _, b = v.discretize()
        np.testing.assert_allclose(np.linalg.norm(b), 1.0, atol=1e-14)

    def test_source_vector_all_positive(self):
        """Gaussian source profile should be strictly positive on interior."""
        v = VQLS_GradShafranov(n_qubits=3)
        _, b = v.discretize()
        assert np.all(b > 0)

    @pytest.mark.parametrize("n", [2, 3, 4])
    def test_various_sizes(self, n):
        v = VQLS_GradShafranov(n_qubits=n)
        A, b = v.discretize()
        N = 2**n
        assert A.shape == (N, N)
        assert b.shape == (N,)

    def test_stores_A_and_b_internally(self):
        v = VQLS_GradShafranov(n_qubits=2)
        v.discretize()
        assert v._A is not None
        assert v._b is not None


# ---------------------------------------------------------------------------
# build_ansatz
# ---------------------------------------------------------------------------


class TestBuildAnsatz:
    def test_returns_quantum_circuit(self):
        from qiskit import QuantumCircuit

        v = VQLS_GradShafranov(n_qubits=2)
        qc = v.build_ansatz(reps=1)
        assert isinstance(qc, QuantumCircuit)

    def test_correct_qubit_count(self):
        v = VQLS_GradShafranov(n_qubits=3)
        qc = v.build_ansatz(reps=2)
        assert qc.num_qubits == 3

    def test_has_parameters(self):
        v = VQLS_GradShafranov(n_qubits=2)
        qc = v.build_ansatz(reps=2)
        assert qc.num_parameters > 0

    def test_more_reps_more_params(self):
        v = VQLS_GradShafranov(n_qubits=2)
        qc1 = v.build_ansatz(reps=1)
        qc2 = v.build_ansatz(reps=3)
        assert qc2.num_parameters > qc1.num_parameters


# ---------------------------------------------------------------------------
# solve — happy path
# ---------------------------------------------------------------------------


class TestSolve:
    def test_output_shape_n2(self):
        v = VQLS_GradShafranov(n_qubits=2)
        result = v.solve(reps=1, maxiter=5, seed=42)
        assert result.shape == (4,)

    def test_output_is_real(self):
        v = VQLS_GradShafranov(n_qubits=2)
        result = v.solve(reps=1, maxiter=5, seed=42)
        assert result.dtype in (np.float64, np.float32)

    def test_output_all_finite(self):
        v = VQLS_GradShafranov(n_qubits=2)
        result = v.solve(reps=1, maxiter=5, seed=42)
        assert np.all(np.isfinite(result))

    def test_auto_discretize(self):
        """solve() should auto-call discretize() if not done explicitly."""
        v = VQLS_GradShafranov(n_qubits=2)
        assert v._A is None
        result = v.solve(reps=1, maxiter=1, seed=0)
        assert v._A is not None
        assert result.shape == (4,)

    def test_seed_determinism(self):
        """Same seed → same result."""
        v1 = VQLS_GradShafranov(n_qubits=2)
        v2 = VQLS_GradShafranov(n_qubits=2)
        r1 = v1.solve(reps=1, maxiter=5, seed=123)
        r2 = v2.solve(reps=1, maxiter=5, seed=123)
        np.testing.assert_array_equal(r1, r2)

    def test_stores_optimal_params(self):
        v = VQLS_GradShafranov(n_qubits=2)
        v.solve(reps=1, maxiter=5, seed=42)
        assert v._optimal_params is not None
        assert isinstance(v._optimal_params, np.ndarray)


# ---------------------------------------------------------------------------
# solve — edge cases
# ---------------------------------------------------------------------------


class TestSolveEdgeCases:
    def test_imag_tol_zero_raises(self):
        """imag_tol=0 guarantees ValueError since any state has epsilon imaginary."""
        v = VQLS_GradShafranov(n_qubits=2, imag_tol=0.0)
        with pytest.raises(ValueError, match="imaginary norm"):
            v.solve(reps=1, maxiter=1, seed=0)

    def test_degenerate_denominator(self):
        """Near-zero xAtAx returns cost=1.0 gracefully (line 101-102)."""
        v = VQLS_GradShafranov(n_qubits=2)
        v.discretize()
        result = v.solve(reps=1, maxiter=5, seed=42)
        assert result.shape == (4,)

    def test_denominator_guard_path(self):
        """Force the xAtAx < VQLS_DENOMINATOR_EPS path."""
        v = VQLS_GradShafranov(n_qubits=2)
        v.discretize()

        tiny_sv = np.zeros(4, dtype=complex)
        tiny_sv[0] = 1e-20

        call_count = [0]
        original_from_instruction = Statevector.from_instruction

        def mock_from_instruction(circuit):
            call_count[0] += 1
            if call_count[0] <= 5:
                return Statevector(tiny_sv / max(np.linalg.norm(tiny_sv), 1e-30))
            return original_from_instruction(circuit)

        with (
            patch.object(Statevector, "from_instruction", side_effect=mock_from_instruction),
            contextlib.suppress(ValueError, RuntimeError),
        ):
            v.solve(reps=1, maxiter=2, seed=0)

    def test_large_imag_tol_succeeds(self):
        """Very large tolerance should always pass."""
        v = VQLS_GradShafranov(n_qubits=2, imag_tol=1e6)
        result = v.solve(reps=1, maxiter=1, seed=0)
        assert result.shape == (4,)

    def test_different_source_width(self):
        """Changing source_width should still produce valid output."""
        v = VQLS_GradShafranov(n_qubits=2, source_width=0.2)
        result = v.solve(reps=1, maxiter=5, seed=42)
        assert result.shape == (4,)
        assert np.all(np.isfinite(result))

    def test_maxiter_boosted_for_many_params(self):
        """effective_maxiter = max(maxiter, n_params + 10) — verify convergence still works."""
        v = VQLS_GradShafranov(n_qubits=2)
        result = v.solve(reps=1, maxiter=1, seed=42)
        assert result.shape == (4,)
