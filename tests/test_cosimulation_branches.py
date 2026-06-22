# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Branch tests for the quantum/classical co-simulation
"""Fallback and guard tests for the mean-field quantum/classical co-simulation.

Covers the NumPy Hamiltonian and classical-substep fallbacks, the zero-coupling
skip, the empty order parameter, the provided-state normalisation branch and the
classical phase-vector length guard.
"""

from __future__ import annotations

import numpy as np
import pytest

from scpn_quantum_control.cosimulation import quantum_classical as qc
from scpn_quantum_control.cosimulation.quantum_classical import (
    _classical_substep,
    _initial_quantum_state,
    _internal_half_propagator,
    _order_parameter,
    _xy_hamiltonian_dense_python,
    cosimulate,
)


def test_internal_half_propagator_uses_numpy_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without the native engine the dense XY propagator is built in NumPy."""
    monkeypatch.setattr(qc, "optional_rust_engine", lambda: None)
    k = np.array([[0.0, 0.4], [0.4, 0.0]], dtype=np.float64)
    omega = np.array([0.2, -0.2], dtype=np.float64)
    propagator = _internal_half_propagator(k, omega, 0.1)
    assert propagator.shape == (4, 4)


def test_xy_hamiltonian_skips_zero_couplings() -> None:
    """A near-zero coupling pair is skipped while assembling the Hamiltonian."""
    k = np.array([[0.0, 0.5, 0.0], [0.5, 0.0, 0.5], [0.0, 0.5, 0.0]], dtype=np.float64)
    omega = np.array([0.1, 0.2, 0.3], dtype=np.float64)
    hamiltonian = _xy_hamiltonian_dense_python(k, omega)
    assert hamiltonian.shape == (8, 8)


def test_classical_substep_uses_numpy_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without the native engine the classical substep runs in NumPy."""
    monkeypatch.setattr(qc, "optional_rust_engine", lambda: None)
    theta = np.array([0.1, 0.2], dtype=np.float64)
    omega = np.array([0.3, 0.4], dtype=np.float64)
    k = np.array([[0.0, 0.5], [0.5, 0.0]], dtype=np.float64)
    drive = np.array([0.1, 0.1], dtype=np.float64)
    stepped = _classical_substep(theta, omega, k, drive, drive, 0.05)
    assert stepped.shape == (2,)


def test_order_parameter_zero_for_empty() -> None:
    """An empty phase vector has zero order parameter."""
    assert _order_parameter(np.array([], dtype=np.float64)) == 0.0


def test_initial_quantum_state_normalises_override() -> None:
    """A provided quantum state is reshaped and renormalised."""
    state = _initial_quantum_state(1, np.array([2.0, 0.0], dtype=np.complex128))
    np.testing.assert_allclose(state, np.array([1.0, 0.0], dtype=np.complex128))


def test_cosimulate_rejects_wrong_classical_phase_length() -> None:
    """A classical phase vector of the wrong length is rejected."""
    k = np.array([[0.0, 0.4], [0.4, 0.0]], dtype=np.float64)
    omega = np.array([0.2, -0.2], dtype=np.float64)
    with pytest.raises(ValueError, match="theta0_classical must have length"):
        cosimulate(
            k,
            omega,
            dt=0.1,
            n_steps=1,
            max_quantum_nodes=1,
            theta0_classical=np.zeros(7, dtype=np.float64),
        )
