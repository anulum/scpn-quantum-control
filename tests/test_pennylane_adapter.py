# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Tests for PennyLane backend adapter."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_quantum_control.bridge.knm_hamiltonian import OMEGA_N_16, build_knm_paper27

try:
    from scpn_quantum_control.hardware.pennylane_adapter import (
        PennyLaneResult,
        PennyLaneRunner,
        is_pennylane_available,
    )

    _PL_OK = is_pennylane_available()
except (ImportError, AttributeError):
    _PL_OK = False

pytestmark = pytest.mark.skipif(not _PL_OK, reason="PennyLane not available or broken")


class TestPennyLaneRunner:
    def test_trotter_returns_result(self):
        K = build_knm_paper27(L=3)
        omega = OMEGA_N_16[:3]
        runner = PennyLaneRunner(K, omega)
        result = runner.run_trotter(t=0.5, reps=2)
        assert isinstance(result, PennyLaneResult)

    def test_energy_finite(self):
        K = build_knm_paper27(L=3)
        omega = OMEGA_N_16[:3]
        runner = PennyLaneRunner(K, omega)
        result = runner.run_trotter(t=0.5, reps=2)
        assert np.isfinite(result.energy)

    def test_r_global_bounded(self):
        K = build_knm_paper27(L=3)
        omega = OMEGA_N_16[:3]
        runner = PennyLaneRunner(K, omega)
        result = runner.run_trotter(t=0.5, reps=2)
        assert 0 <= result.order_parameter <= 1.0

    def test_n_qubits(self):
        K = build_knm_paper27(L=4)
        omega = OMEGA_N_16[:4]
        runner = PennyLaneRunner(K, omega)
        result = runner.run_trotter(t=0.1, reps=1)
        assert result.n_qubits == 4

    def test_vqe_returns_result(self):
        K = build_knm_paper27(L=2)
        omega = OMEGA_N_16[:2]
        runner = PennyLaneRunner(K, omega)
        result = runner.run_vqe(ansatz_depth=1, maxiter=5, seed=42)
        assert isinstance(result, PennyLaneResult)
        assert np.isfinite(result.energy)

    def test_device_name(self):
        K = build_knm_paper27(L=3)
        omega = OMEGA_N_16[:3]
        runner = PennyLaneRunner(K, omega, device="default.qubit")
        result = runner.run_trotter(t=0.1, reps=1)
        assert result.device_name == "default.qubit"
