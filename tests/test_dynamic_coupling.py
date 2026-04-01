# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li

"""Tests for Dynamic Quantum-Classical Co-Evolution."""

from __future__ import annotations

import numpy as np

from scpn_quantum_control.qsnn.dynamic_coupling import DynamicCouplingEngine


class TestDynamicCoupling:
    def test_dynamic_coupling_engine_step(self):
        """Verify one cycle of the strange loop."""
        n = 3
        # Weak initial coupling
        initial_K = np.array([[0.0, 0.1, 0.1], [0.1, 0.0, 0.1], [0.1, 0.1, 0.0]])
        omega = np.array([10.0, 10.0, 10.0])

        engine = DynamicCouplingEngine(
            n_qubits=n, initial_K=initial_K, omega=omega, learning_rate=0.5, decay_rate=0.0
        )

        # We step it
        res = engine.step(dt=0.5)

        assert "K_updated" in res
        assert "correlation_matrix" in res
        assert res["K_updated"].shape == (3, 3)
        # Verify symmetry
        np.testing.assert_allclose(res["K_updated"], res["K_updated"].T)

    def test_run_coevolution(self):
        n = 2
        # Fully disconnected initially
        initial_K = np.zeros((n, n))
        omega = np.array([5.0, 5.0])

        engine = DynamicCouplingEngine(
            n_qubits=n, initial_K=initial_K, omega=omega, learning_rate=1.0, decay_rate=0.1
        )

        history = engine.run_coevolution(steps=3, dt=0.5)
        assert len(history) == 3
