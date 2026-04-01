# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for Pennylane Adapter
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

_SKIP_NO_PL = pytest.mark.skipif(not _PL_OK, reason="PennyLane not available or broken")


@_SKIP_NO_PL
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


# ---------------------------------------------------------------------------
# Tests that work WITHOUT PennyLane
# ---------------------------------------------------------------------------


class TestPennyLaneAvailability:
    """These tests always run, regardless of PennyLane installation."""

    def test_is_pennylane_available_returns_bool(self):
        from scpn_quantum_control.hardware.pennylane_adapter import is_pennylane_available

        assert isinstance(is_pennylane_available(), bool)

    def test_module_importable(self):
        from scpn_quantum_control.hardware import pennylane_adapter

        assert hasattr(pennylane_adapter, "is_pennylane_available")
        assert hasattr(pennylane_adapter, "PennyLaneRunner")
        assert hasattr(pennylane_adapter, "PennyLaneResult")

    def test_pennylane_result_dataclass(self):
        from scpn_quantum_control.hardware.pennylane_adapter import PennyLaneResult

        r = PennyLaneResult(
            energy=-1.5,
            order_parameter=0.8,
            n_qubits=4,
            device_name="sim",
            statevector=np.zeros(16),
        )
        assert r.energy == -1.5
        assert r.order_parameter == 0.8

    def test_runner_raises_without_pennylane(self):
        if _PL_OK:
            pytest.skip("PennyLane is installed")
        from scpn_quantum_control.hardware.pennylane_adapter import PennyLaneRunner

        K = build_knm_paper27(L=2)
        omega = OMEGA_N_16[:2]
        with pytest.raises(ImportError):
            PennyLaneRunner(K, omega)

    def test_pipeline_pennylane_availability(self):
        """Pipeline: check availability → import module → verify API.
        Verifies PennyLane adapter is wired into the package.
        """
        import time

        from scpn_quantum_control.hardware.pennylane_adapter import (
            PennyLaneResult,
            PennyLaneRunner,
            is_pennylane_available,
        )

        t0 = time.perf_counter()
        available = is_pennylane_available()
        assert isinstance(available, bool)
        assert PennyLaneRunner is not None
        assert PennyLaneResult is not None
        dt = (time.perf_counter() - t0) * 1000

        print(f"\n  PIPELINE PennyLane adapter: {dt:.2f} ms, available={available}")
