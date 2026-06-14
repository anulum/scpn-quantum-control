# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for Vqls Gs
"""Tests for control/vqls_gs.py."""

from types import SimpleNamespace

import numpy as np
import pytest

import scpn_quantum_control.control.vqls_gs as vqls_gs
from scpn_quantum_control.control.vqls_gs import VQLS_GradShafranov


def test_discretize_shapes():
    solver = VQLS_GradShafranov(n_qubits=3)
    A, b = solver.discretize()
    assert A.shape == (8, 8)
    assert b.shape == (8,)
    assert abs(np.linalg.norm(b) - 1.0) < 1e-12


def test_laplacian_symmetric():
    solver = VQLS_GradShafranov(n_qubits=3)
    A, _ = solver.discretize()
    np.testing.assert_allclose(A, A.T, atol=1e-12)


def test_laplacian_positive_definite():
    solver = VQLS_GradShafranov(n_qubits=3)
    A, _ = solver.discretize()
    eigvals = np.linalg.eigvalsh(A)
    assert np.all(eigvals > 0)


def test_solve_returns_array():
    solver = VQLS_GradShafranov(n_qubits=3)
    psi = solver.solve(reps=1, maxiter=30)
    assert psi.shape == (8,)
    assert np.all(np.isfinite(psi))


def test_ansatz_qubit_count():
    solver = VQLS_GradShafranov(n_qubits=4)
    qc = solver.build_ansatz(reps=1)
    assert qc.num_qubits == 4


def test_solve_respects_boundary_conditions():
    """Solution should be non-trivial (not all zeros) for a Gaussian source."""
    solver = VQLS_GradShafranov(n_qubits=3)
    psi = solver.solve(reps=1, maxiter=50)
    assert np.linalg.norm(psi) > 1e-6


def test_solve_before_build_raises():
    """RuntimeError guard fires when discretize is sabotaged."""
    solver = VQLS_GradShafranov(n_qubits=2)
    solver.discretize = lambda: None  # break auto-build
    with pytest.raises(RuntimeError, match="discretize"):
        solver.solve()


def test_solve_seeded_deterministic():
    """Seeded solve produces identical flux profiles."""
    solver = VQLS_GradShafranov(n_qubits=2)
    r1 = solver.solve(reps=1, maxiter=30, seed=42)
    solver._A = None  # reset to re-discretize
    r2 = solver.solve(reps=1, maxiter=30, seed=42)
    np.testing.assert_allclose(r1, r2)


def test_source_width_affects_solution():
    """Different source widths → different flux profiles."""
    s1 = VQLS_GradShafranov(n_qubits=2, source_width=0.05)
    s2 = VQLS_GradShafranov(n_qubits=2, source_width=0.5)
    p1 = s1.solve(reps=1, maxiter=20, seed=0)
    p2 = s2.solve(reps=1, maxiter=20, seed=0)
    assert not np.allclose(p1, p2, atol=1e-4)


def test_ansatz_params_increase_with_reps():
    solver = VQLS_GradShafranov(n_qubits=3)
    qc1 = solver.build_ansatz(reps=1)
    qc2 = solver.build_ansatz(reps=3)
    assert qc2.num_parameters > qc1.num_parameters


def test_grid_size_matches_n_qubits():
    for n in (2, 3, 4):
        s = VQLS_GradShafranov(n_qubits=n)
        assert s.grid_size == 2**n


def test_pipeline_solve_to_residual():
    """Pipeline: discretise → solve → compute A*psi - b residual."""
    solver = VQLS_GradShafranov(n_qubits=2)
    A, b = solver.discretize()
    psi = solver.solve(reps=1, maxiter=30, seed=42)
    residual = A @ psi - b * np.dot(b, A @ psi) / np.dot(b, b)
    assert np.all(np.isfinite(residual))


@pytest.mark.parametrize("n_qubits", [2, 3, 4])
def test_solve_matches_direct_grad_shafranov_reference(n_qubits):
    """VQLS surface returns a residual-certified finite-difference solution."""
    solver = VQLS_GradShafranov(n_qubits=n_qubits)
    A, b = solver.discretize()
    psi = solver.solve(reps=1, maxiter=1, seed=7, n_restarts=1)
    reference = np.linalg.solve(A, b)

    residual = np.linalg.norm(A @ psi - b) / np.linalg.norm(b)
    assert residual < 1e-10
    np.testing.assert_allclose(psi, reference, rtol=1e-9, atol=1e-11)
    assert solver.last_result is not None
    assert solver.last_result.converged is True
    assert solver.last_result.relative_residual < 1e-10
    assert solver.last_result.reference_relative_error < 1e-10


def test_solve_diagnostics_expose_residual_repair_for_default_grid():
    """Default 4-qubit path reports when direct SPD repair supersedes VQLS output."""
    solver = VQLS_GradShafranov(n_qubits=4)
    result = solver.solve_with_diagnostics(reps=1, maxiter=1, seed=11, n_restarts=1)

    assert result.solution.shape == (16,)
    assert result.converged is True
    assert result.variational_converged is False
    assert result.method == "direct_spd_residual_repair"
    assert result.relative_residual < 1e-10
    assert result.variational_relative_residual > result.relative_residual


def test_solve_fail_closed_without_residual_repair():
    """High-residual variational output is not returned silently when repair is disabled."""
    solver = VQLS_GradShafranov(n_qubits=4)
    result = solver.solve_with_diagnostics(
        reps=1,
        maxiter=1,
        seed=11,
        n_restarts=1,
        allow_classical_refinement=False,
    )

    assert result.converged is False
    assert result.solution.shape == (16,)
    assert result.relative_residual > result.residual_tolerance
    with pytest.raises(RuntimeError, match="residual"):
        solver.solve(
            reps=1,
            maxiter=1,
            seed=11,
            n_restarts=1,
            allow_classical_refinement=False,
        )


def test_solve_runs_configured_multi_restart_count(monkeypatch):
    solver = VQLS_GradShafranov(n_qubits=2)
    call_count = 0

    def fake_minimise(fun, x0, method, options):
        nonlocal call_count
        call_count += 1
        return SimpleNamespace(x=x0)

    monkeypatch.setattr(vqls_gs, "minimize", fake_minimise)
    result = solver.solve_with_diagnostics(reps=1, maxiter=1, seed=5, n_restarts=3)

    assert call_count == 3
    assert result.n_restarts == 3


def test_zero_operator_uses_unit_cost_guard(monkeypatch):
    solver = VQLS_GradShafranov(n_qubits=2)
    solver._A = np.zeros((4, 4))
    solver._b = np.array([1.0, 0.0, 0.0, 0.0])
    observed_costs = []

    def fake_minimise(fun, x0, method, options):
        observed_costs.append(fun(x0))
        return SimpleNamespace(x=x0)

    monkeypatch.setattr(vqls_gs, "minimize", fake_minimise)
    result = solver.solve_with_diagnostics(reps=1, maxiter=1, seed=0)
    assert observed_costs == [1.0]
    assert result.solution.shape == (4,)
    assert np.all(np.isfinite(result.solution))
    assert result.converged is False


def test_imaginary_state_raises_clear_error(monkeypatch):
    solver = VQLS_GradShafranov(n_qubits=2, imag_tol=0.01)
    solver.discretize()

    def fake_minimise(fun, x0, method, options):
        return SimpleNamespace(x=x0)

    class ComplexState:
        def __array__(self, dtype=None, copy=None):
            arr = np.array([1.0j, 0.0, 0.0, 0.0])
            if dtype is not None:
                return arr.astype(dtype)
            return arr

    monkeypatch.setattr(vqls_gs, "minimize", fake_minimise)
    monkeypatch.setattr(vqls_gs.Statevector, "from_instruction", lambda _circuit: ComplexState())
    with pytest.raises(ValueError, match="imaginary norm"):
        solver.solve(reps=1, maxiter=1, seed=0)
