# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
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
