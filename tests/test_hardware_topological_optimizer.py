# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li

"""Tests for Hardware-in-the-Loop Topological Feedback."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_quantum_control.analysis.quantum_persistent_homology import _RIPSER_AVAILABLE
from scpn_quantum_control.control.hardware_topological_optimizer import (
    HardwareTopologicalOptimizer,
)
from scpn_quantum_control.hardware.runner import HardwareRunner


class TestHardwareTopologicalOptimizer:
    def test_hardware_optimizer_step(self):
        """Verify one cycle of hardware-in-the-loop topological gradient descent."""
        if not _RIPSER_AVAILABLE:
            pytest.skip("ripser not available")

        n = 2
        initial_K = np.array([[0.0, 0.1], [0.1, 0.0]])
        omega = np.array([5.0, 10.0])

        runner = HardwareRunner(use_simulator=True)
        runner.connect()

        opt = HardwareTopologicalOptimizer(
            runner=runner, n_qubits=n, initial_K=initial_K, omega=omega, learning_rate=0.5, dt=0.5
        )

        # We step it with a very small number of samples and shots to keep it fast
        # Note: with n=2, p_h1 is likely 0, so gradient will be 0, but we test the pipeline
        res = opt.step(n_samples=1)

        assert "K_updated" in res
        assert "p_h1_current" in res
        assert "gradient_norm" in res
        assert res["K_updated"].shape == (n, n)
        np.testing.assert_allclose(res["K_updated"], res["K_updated"].T)
