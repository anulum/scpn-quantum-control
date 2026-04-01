# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for Runner Guards
"""Guard tests: HardwareRunner methods raise before connect() — elite coverage."""

from __future__ import annotations

import json

import pytest
from qiskit import QuantumCircuit

from scpn_quantum_control.hardware.runner import HardwareRunner, JobResult

# ---------------------------------------------------------------------------
# Pre-connect guards
# ---------------------------------------------------------------------------


class TestPreConnectGuards:
    def test_transpile_before_connect(self):
        runner = HardwareRunner(use_simulator=True)
        with pytest.raises(RuntimeError, match="connect"):
            runner.transpile(QuantumCircuit(2))

    def test_transpile_with_dd_before_connect(self):
        runner = HardwareRunner(use_simulator=True)
        with pytest.raises(RuntimeError, match="connect"):
            runner.transpile_with_dd(QuantumCircuit(2))

    def test_run_sampler_before_connect(self):
        runner = HardwareRunner(use_simulator=True)
        with pytest.raises(RuntimeError, match="connect"):
            runner.run_sampler(QuantumCircuit(2))

    def test_run_estimator_before_connect(self):
        runner = HardwareRunner(use_simulator=True)
        with pytest.raises(RuntimeError, match="connect"):
            runner.run_estimator(QuantumCircuit(2), [])

    def test_retrieve_job_before_connect(self):
        runner = HardwareRunner(use_simulator=True)
        with pytest.raises(RuntimeError, match="connect"):
            runner.retrieve_job("some_id")


# ---------------------------------------------------------------------------
# Post-connect happy path
# ---------------------------------------------------------------------------


class TestPostConnect:
    def test_transpile_succeeds_after_connect(self):
        runner = HardwareRunner(use_simulator=True)
        runner.connect()
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure_all()
        isa = runner.transpile(qc)
        assert isa.num_qubits >= 2

    def test_backend_name_after_connect(self):
        runner = HardwareRunner(use_simulator=True)
        runner.connect()
        name = runner.backend_name
        assert name != "not_connected"
        assert "simulator" in name.lower() or "aer" in name.lower()

    def test_transpile_with_dd_after_connect(self):
        runner = HardwareRunner(use_simulator=True)
        runner.connect()
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure_all()
        isa = runner.transpile_with_dd(qc)
        assert isa.num_qubits >= 2


# ---------------------------------------------------------------------------
# save_result
# ---------------------------------------------------------------------------


class TestSaveResult:
    def test_save_single_result(self, tmp_path):
        runner = HardwareRunner(use_simulator=True, results_dir=str(tmp_path))
        jr = JobResult(
            job_id="test_1",
            backend_name="aer",
            experiment_name="exp",
            counts={"00": 500, "11": 500},
        )
        path = runner.save_result(jr, filename="test_out.json")
        assert path.exists()
        with open(path) as f:
            data = json.load(f)
        assert data["job_id"] == "test_1"

    def test_save_list_result(self, tmp_path):
        runner = HardwareRunner(use_simulator=True, results_dir=str(tmp_path))
        results = [
            JobResult(job_id=f"t{i}", backend_name="aer", experiment_name=f"e{i}")
            for i in range(3)
        ]
        path = runner.save_result(results, filename="batch.json")
        with open(path) as f:
            data = json.load(f)
        assert len(data) == 3

    def test_auto_filename(self, tmp_path):
        runner = HardwareRunner(use_simulator=True, results_dir=str(tmp_path))
        jr = JobResult(job_id="x", backend_name="y", experiment_name="z")
        path = runner.save_result(jr)
        assert path.exists()
        assert path.suffix == ".json"
