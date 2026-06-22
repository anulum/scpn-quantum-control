# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Branch tests for the VQLS Grad-Shafranov solver
"""Guard and branch tests for the VQLS Grad-Shafranov ground-state solver.

Covers the solve hyper-parameter guards, the no-candidate optimiser guard, the
discretise-before-use guards and the zero-denominator residual/error branches.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

import scpn_quantum_control.control.vqls_gs as vqls_gs
from scpn_quantum_control.control.vqls_gs import VQLS_GradShafranov


def test_solve_rejects_non_positive_restarts() -> None:
    """A restart count below one is rejected."""
    with pytest.raises(ValueError, match="n_restarts must be at least 1"):
        VQLS_GradShafranov(n_qubits=2).solve(n_restarts=0)


def test_solve_rejects_non_positive_maxiter() -> None:
    """A maximum iteration count below one is rejected."""
    with pytest.raises(ValueError, match="maxiter must be at least 1"):
        VQLS_GradShafranov(n_qubits=2).solve(maxiter=0)


def test_solve_rejects_non_positive_residual_tol() -> None:
    """A non-positive residual tolerance is rejected."""
    with pytest.raises(ValueError, match="residual_tol must be positive"):
        VQLS_GradShafranov(n_qubits=2).solve(residual_tol=0.0)


def test_solve_raises_when_optimizer_yields_no_candidate(monkeypatch: pytest.MonkeyPatch) -> None:
    """A non-finite cost on every restart leaves no candidate and is rejected.

    The optimiser returns the valid start point but a degenerate (NaN) state
    vector makes the cost non-finite, so the improvement test never succeeds.
    """
    solver = VQLS_GradShafranov(n_qubits=2)
    dim = solver.grid_size

    class _NanState:
        def __array__(self, *_args: Any, **_kwargs: Any) -> Any:
            return np.full(dim, np.nan, dtype=np.complex128)

    monkeypatch.setattr(
        vqls_gs, "Statevector", SimpleNamespace(from_instruction=lambda _circuit: _NanState())
    )

    def _identity_minimize(_fun: Any, x0: Any, **_kwargs: Any) -> Any:
        return SimpleNamespace(x=x0, success=False, message="degenerate")

    monkeypatch.setattr(vqls_gs, "minimize", _identity_minimize)
    with pytest.raises(RuntimeError, match="VQLS optimizer did not produce a candidate"):
        solver.solve(maxiter=1)


def test_relative_residual_requires_discretize() -> None:
    """Residual evaluation before discretisation is rejected."""
    solver = VQLS_GradShafranov(n_qubits=2)
    with pytest.raises(RuntimeError, match="call discretize\\(\\) before residual evaluation"):
        solver._relative_residual(np.zeros(4, dtype=np.float64))


def test_relative_residual_returns_absolute_for_zero_rhs() -> None:
    """With a near-zero right-hand side the absolute residual is returned."""
    solver = VQLS_GradShafranov(n_qubits=1)
    solver._A = np.eye(2, dtype=np.float64)
    solver._b = np.zeros(2, dtype=np.float64)
    assert solver._relative_residual(np.array([1.0, 0.0], dtype=np.float64)) == 1.0


def test_direct_solution_requires_discretize() -> None:
    """The direct solve before discretisation is rejected."""
    solver = VQLS_GradShafranov(n_qubits=2)
    with pytest.raises(RuntimeError, match="call discretize\\(\\) before direct solve"):
        solver._direct_solution_or_none()


def test_reference_relative_error_returns_absolute_for_zero_reference() -> None:
    """A near-zero reference solution yields the absolute error."""
    error = VQLS_GradShafranov._reference_relative_error(
        np.array([1.0, 0.0], dtype=np.float64), np.zeros(2, dtype=np.float64)
    )
    assert error == 1.0
