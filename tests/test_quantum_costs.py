# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Tests for Quantum Costs
"""Tests for quantum cost terms."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_quantum_control.ssgf.quantum_costs import (
    QuantumCosts,
    compute_quantum_costs,
)


def _test_system():
    n = 3
    W = np.array([[0, 0.5, 0.3], [0.5, 0, 0.4], [0.3, 0.4, 0]])
    theta = np.array([0.0, 0.3, 0.6])
    return W, theta, n


class TestComputeQuantumCosts:
    def test_returns_costs(self):
        W, theta, _n = _test_system()
        costs = compute_quantum_costs(W, theta)
        assert isinstance(costs, QuantumCosts)

    def test_c_micro_bounded(self):
        W, theta, _n = _test_system()
        costs = compute_quantum_costs(W, theta)
        assert 0 <= costs.c_micro <= 1.0

    def test_c4_tcbo_bounded(self):
        W, theta, _n = _test_system()
        costs = compute_quantum_costs(W, theta)
        assert 0 <= costs.c4_tcbo <= 1.0

    def test_c_pgbo_bounded(self):
        W, theta, _n = _test_system()
        costs = compute_quantum_costs(W, theta)
        assert 0 <= costs.c_pgbo <= 1.0

    def test_r_global_bounded(self):
        W, theta, _n = _test_system()
        costs = compute_quantum_costs(W, theta)
        assert 0 <= costs.r_global <= 1.0

    def test_c_micro_plus_r_equals_one(self):
        W, theta, _n = _test_system()
        costs = compute_quantum_costs(W, theta)
        assert costs.c_micro + costs.r_global == pytest.approx(1.0, abs=1e-10)

    def test_entropy_non_negative(self):
        W, theta, _n = _test_system()
        costs = compute_quantum_costs(W, theta)
        assert costs.half_chain_entropy >= -1e-10

    def test_variance_non_negative(self):
        W, theta, _n = _test_system()
        costs = compute_quantum_costs(W, theta)
        assert costs.correlator_variance >= -1e-10

    def test_with_omega(self):
        W, theta, _n = _test_system()
        omega = np.array([1.0, 1.5, 0.8])
        costs = compute_quantum_costs(W, theta, omega=omega)
        assert isinstance(costs.c_micro, float)

    def test_2_oscillators(self):
        W = np.array([[0, 0.5], [0.5, 0]])
        theta = np.array([0.0, 0.5])
        costs = compute_quantum_costs(W, theta)
        assert isinstance(costs, QuantumCosts)

    def test_synchronized_low_c_micro(self):
        """All phases aligned → high R → low C_micro."""
        W = np.array([[0, 1.0], [1.0, 0]])
        theta = np.array([0.0, 0.0])
        costs = compute_quantum_costs(W, theta, dt=0.01)
        assert costs.c_micro < 0.5
