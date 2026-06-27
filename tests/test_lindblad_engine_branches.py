# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Lindblad Engine Branch Hardening Tests
"""Branch-level hardening tests for the Lindblad synchronization engine."""

from __future__ import annotations

import ast
import inspect
import textwrap

import numpy as np
import pytest
from numpy.typing import NDArray

from scpn_quantum_control.phase.lindblad_engine import LindbladSyncEngine


def _two_node_coupling() -> NDArray[np.float64]:
    """Return a symmetric two-node coupling matrix."""
    return np.array([[0.0, 0.5], [0.5, 0.0]], dtype=np.float64)


def _two_node_omega() -> NDArray[np.float64]:
    """Return a two-node natural-frequency vector."""
    return np.array([1.0, 1.2], dtype=np.float64)


def test_liouvillian_fails_closed_when_dense_build_leaves_no_hamiltonian(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Liouvillian evaluation should fail explicitly if dense build is incomplete."""
    engine = LindbladSyncEngine(_two_node_coupling(), _two_node_omega())
    rho = np.eye(engine.dim, dtype=np.complex128) / engine.dim

    def leave_hamiltonian_unset(max_dense_gib: float | None = None) -> None:
        """Simulate an invariant break where dense setup returns without H."""
        del max_dense_gib

    monkeypatch.setattr(engine, "_ensure_density_matrix_components", leave_hamiltonian_unset)

    with pytest.raises(RuntimeError, match="dense Hamiltonian"):
        engine.liouvillian(rho.ravel())


def test_liouvillian_contains_no_runtime_asserts() -> None:
    """Optimized bytecode must not remove the liouvillian Hamiltonian guard."""
    source = textwrap.dedent(inspect.getsource(LindbladSyncEngine.liouvillian))
    tree = ast.parse(source)

    assert not any(isinstance(node, ast.Assert) for node in ast.walk(tree))


def test_density_matrix_components_are_reused_after_build() -> None:
    """Repeated dense setup calls should reuse the already-built Hamiltonian."""
    engine = LindbladSyncEngine(_two_node_coupling(), _two_node_omega())
    engine._ensure_density_matrix_components()
    hamiltonian = engine.H_dense
    jump_operators = engine.L_ops_dense

    engine._ensure_density_matrix_components()

    assert engine.H_dense is hamiltonian
    assert engine.L_ops_dense is jump_operators


def test_density_matrix_path_rejects_large_systems_before_dense_allocation() -> None:
    """Density-matrix evolution must fail closed above the documented size limit."""
    n_oscillators = 11
    coupling = np.zeros((n_oscillators, n_oscillators), dtype=np.float64)
    omega = np.ones(n_oscillators, dtype=np.float64)
    engine = LindbladSyncEngine(coupling, omega)

    with pytest.raises(RuntimeError, match="N <= 10"):
        engine.evolve(t_max=0.0, n_steps=1, method="density_matrix")


def test_trajectory_uses_provided_initial_state() -> None:
    """Trajectory evolution should honour an explicit initial state vector."""
    engine = LindbladSyncEngine(_two_node_coupling(), _two_node_omega(), gamma=0.0)
    initial_state = np.zeros(engine.dim, dtype=np.complex128)
    initial_state[0b10] = 1.0

    result = engine.evolve(
        t_max=0.0,
        n_steps=1,
        method="trajectory",
        initial_state=initial_state,
        n_traj=1,
    )

    final_state = result["final_state"]
    assert isinstance(final_state, np.ndarray)
    np.testing.assert_allclose(final_state[0b10, 0b10].real, 1.0, atol=1e-12)


def test_large_trajectory_omits_dense_state_history() -> None:
    """Large trajectory runs should avoid storing density-matrix histories."""
    n_oscillators = 11
    coupling = np.zeros((n_oscillators, n_oscillators), dtype=np.float64)
    omega = np.ones(n_oscillators, dtype=np.float64)
    engine = LindbladSyncEngine(coupling, omega, gamma=0.0)

    result = engine.evolve(t_max=0.0, n_steps=1, method="trajectory", n_traj=1)

    assert result["times"].shape == (2,)
    assert "states" not in result
    assert "final_state" not in result
