# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Tests for Pea Mitigation
"""Tests for PEA error mitigation and HardwareRunner config — elite coverage."""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit

from scpn_quantum_control.hardware.runner import HardwareRunner, JobResult

# ---------------------------------------------------------------------------
# HardwareRunner constructor
# ---------------------------------------------------------------------------


class TestHardwareRunnerInit:
    def test_default_resilience_level_is_2(self):
        runner = HardwareRunner(use_simulator=True)
        assert runner.resilience_level == 2

    def test_resilience_level_override_zero(self):
        runner = HardwareRunner(use_simulator=True, resilience_level=0)
        assert runner.resilience_level == 0

    def test_resilience_level_override_one(self):
        runner = HardwareRunner(use_simulator=True, resilience_level=1)
        assert runner.resilience_level == 1

    def test_default_optimization_level(self):
        runner = HardwareRunner(use_simulator=True)
        assert runner.optimization_level == 2

    def test_default_use_fractional_gates(self):
        runner = HardwareRunner(use_simulator=True)
        assert runner.use_fractional_gates is True

    def test_use_simulator_flag(self):
        runner = HardwareRunner(use_simulator=True)
        assert runner.use_simulator is True

    def test_backend_name_before_connect(self):
        runner = HardwareRunner(use_simulator=True)
        assert runner.backend_name == "not_connected"

    def test_backend_none_before_connect(self):
        runner = HardwareRunner(use_simulator=True)
        assert runner.backend is None


# ---------------------------------------------------------------------------
# Connect and transpile (simulator path)
# ---------------------------------------------------------------------------


class TestSimulatorConnect:
    def test_connect_simulator(self):
        runner = HardwareRunner(use_simulator=True)
        runner.connect()
        assert runner.backend is not None

    def test_transpile_after_connect(self):
        runner = HardwareRunner(use_simulator=True)
        runner.connect()
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure_all()
        isa = runner.transpile(qc)
        assert isa.num_qubits >= 2

    def test_circuit_stats(self):
        runner = HardwareRunner(use_simulator=True)
        runner.connect()
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure_all()
        isa = runner.transpile(qc)
        stats = runner.circuit_stats(isa)
        assert "depth" in stats
        assert "n_qubits" in stats
        assert "total_gates" in stats
        assert stats["n_qubits"] >= 2
        assert stats["depth"] >= 1


# ---------------------------------------------------------------------------
# JobResult dataclass
# ---------------------------------------------------------------------------


class TestJobResult:
    def test_to_dict_basic(self):
        jr = JobResult(
            job_id="test_1",
            backend_name="aer_simulator",
            experiment_name="exp",
            wall_time_s=1.5,
        )
        d = jr.to_dict()
        assert d["job_id"] == "test_1"
        assert d["backend"] == "aer_simulator"
        assert "counts" not in d

    def test_to_dict_with_counts(self):
        jr = JobResult(
            job_id="test_2",
            backend_name="aer",
            experiment_name="exp",
            counts={"00": 500, "11": 500},
        )
        d = jr.to_dict()
        assert d["counts"]["00"] == 500

    def test_to_dict_with_expectation_values(self):
        evs = np.array([0.5, -0.3])
        jr = JobResult(
            job_id="test_3",
            backend_name="aer",
            experiment_name="exp",
            expectation_values=evs,
        )
        d = jr.to_dict()
        assert d["expectation_values"] == [0.5, -0.3]

    def test_default_metadata(self):
        jr = JobResult(job_id="x", backend_name="y", experiment_name="z")
        assert jr.metadata == {}

    def test_default_wall_time(self):
        jr = JobResult(job_id="x", backend_name="y", experiment_name="z")
        assert jr.wall_time_s == 0.0
