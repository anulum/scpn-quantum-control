"""Guard tests: HardwareRunner methods raise before connect()."""

from __future__ import annotations

import pytest
from qiskit import QuantumCircuit

from scpn_quantum_control.hardware.runner import HardwareRunner


def test_transpile_before_connect():
    runner = HardwareRunner(use_simulator=True)
    with pytest.raises(RuntimeError, match="connect"):
        runner.transpile(QuantumCircuit(2))


def test_transpile_with_dd_before_connect():
    runner = HardwareRunner(use_simulator=True)
    with pytest.raises(RuntimeError, match="connect"):
        runner.transpile_with_dd(QuantumCircuit(2))


def test_run_sampler_before_connect():
    runner = HardwareRunner(use_simulator=True)
    with pytest.raises(RuntimeError, match="connect"):
        runner.run_sampler(QuantumCircuit(2))


def test_run_estimator_before_connect():
    runner = HardwareRunner(use_simulator=True)
    with pytest.raises(RuntimeError, match="connect"):
        runner.run_estimator(QuantumCircuit(2), [])
