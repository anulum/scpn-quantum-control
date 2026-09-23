# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — submit-time provider calibration gate tests
"""Exercise submit-time calibration binding through public HAL cloud routes."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from typing import cast

import pytest
from braket.circuits import Circuit
from qiskit import QuantumCircuit

from scpn_quantum_control.hardware.hal import HardwareAbstractionLayer
from scpn_quantum_control.hardware.hal_braket import (
    BraketAwsHALAdapter,
    braket_circuit_to_workload,
)
from scpn_quantum_control.hardware.hal_qiskit import (
    QiskitRuntimeHALAdapter,
    qiskit_circuit_to_workload,
)
from scpn_quantum_control.hardware.provider_capability_core import ProviderCapabilitySnapshot


def _snapshot(route: str, *, calibrated_at: str | None) -> ProviderCapabilitySnapshot:
    """Return injected metadata for the selected real public adapter route."""
    if route == "qiskit":
        return ProviderCapabilitySnapshot(
            route_id="direct/ibm_quantum",
            aggregator="direct",
            provider="ibm",
            backend_id="ibm_quantum",
            target_name="ibm_offline_target",
            n_qubits=2,
            supported_ir_formats=("qiskit_qpy",),
            online=True,
            max_shots=4,
            calibration_timestamp=calibrated_at,
        )
    return ProviderCapabilitySnapshot(
        route_id="aws_braket/ionq",
        aggregator="aws_braket",
        provider="ionq",
        backend_id="aws_braket_ionq",
        target_name="aws_offline_target",
        n_qubits=2,
        supported_ir_formats=("openqasm3",),
        online=True,
        max_shots=4,
        calibration_timestamp=calibrated_at,
    )


@pytest.mark.parametrize("route", ["qiskit", "braket"])
@pytest.mark.parametrize(
    "mutation,expected",
    [
        ("fresh", None),
        ("stale", "stale"),
        ("missing", "missing"),
        ("future", "future"),
        ("target", "target"),
        ("backend", "backend"),
        ("provider", "provider"),
        ("aggregator", "aggregator"),
        ("ir", "IR"),
        ("qubits", "qubits"),
        ("shots", "shots"),
        ("missing_shot_budget", "shots"),
        ("offline", "offline"),
        ("unknown_online", "online"),
        ("simulator", "simulator"),
        ("bad_probe", "ProviderCapabilitySnapshot"),
        ("bad_age", "max_calibration_age_seconds"),
    ],
)
def test_submit_uses_fresh_matching_capability_probe_before_provider_run(
    route: str, mutation: str, expected: str | None
) -> None:
    """Admit one exact fresh route and refuse mismatches before provider run."""
    now = datetime.now(UTC)
    calibrated_at: str | None = (now - timedelta(minutes=10)).isoformat()
    if mutation == "stale":
        calibrated_at = (now - timedelta(hours=2)).isoformat()
    elif mutation == "missing":
        calibrated_at = None
    elif mutation == "future":
        calibrated_at = (now + timedelta(minutes=10)).isoformat()
    snapshot = _snapshot(route, calibrated_at=calibrated_at)
    if mutation == "target":
        snapshot = replace(snapshot, target_name="different_target")
    elif mutation == "backend":
        snapshot = replace(snapshot, backend_id="different_backend")
    elif mutation == "provider":
        snapshot = replace(snapshot, provider="different_provider")
    elif mutation == "aggregator":
        snapshot = replace(snapshot, aggregator="different_aggregator")
    elif mutation == "ir":
        snapshot = replace(snapshot, supported_ir_formats=("unsupported_ir",))
    elif mutation == "qubits":
        snapshot = replace(snapshot, n_qubits=1)
    elif mutation == "shots":
        snapshot = replace(snapshot, max_shots=3)
    elif mutation == "missing_shot_budget":
        snapshot = replace(snapshot, max_shots=None)
    elif mutation == "offline":
        snapshot = replace(snapshot, online=False)
    elif mutation == "unknown_online":
        snapshot = replace(snapshot, online=None)
    elif mutation == "simulator":
        snapshot = replace(snapshot, simulator=True)
    probe_calls = 0
    provider_runs = 0

    def probe() -> ProviderCapabilitySnapshot:
        nonlocal probe_calls
        probe_calls += 1
        if mutation == "bad_probe":
            return cast(ProviderCapabilitySnapshot, object())
        return snapshot

    max_age = -1 if mutation == "bad_age" else 3600

    if route == "qiskit":

        class RuntimeJob:
            def job_id(self) -> str:
                return "offline-qiskit-submit-freshness"

        class Sampler:
            def __init__(self, mode: object) -> None:
                self.options = type("Options", (), {})()

            def run(self, circuits: Sequence[QuantumCircuit]) -> RuntimeJob:
                nonlocal provider_runs
                provider_runs += 1
                assert len(circuits) == 1
                return RuntimeJob()

        circuit = QuantumCircuit(2, 2)
        circuit.measure([0, 1], [0, 1])
        workload = qiskit_circuit_to_workload(circuit, workload_id="freshness", shots=4)
        hal = HardwareAbstractionLayer.with_builtin_profiles()
        hal.register_backend(
            QiskitRuntimeHALAdapter(
                hal.profile("ibm_quantum"),
                backend=type("Backend", (), {"name": "ibm_offline_target"})(),
                sampler_factory=Sampler,
                capability_probe=probe,
                max_calibration_age_seconds=max_age,
            )
        )
        backend_id = "ibm_quantum"
    else:

        class Task:
            id = "arn:aws:braket:task/offline-submit-freshness"

        class Device:
            name = "aws_offline_target"

            def run(self, circuit: Circuit, shots: int) -> Task:
                nonlocal provider_runs
                provider_runs += 1
                assert shots == 4
                return Task()

        workload = braket_circuit_to_workload(
            Circuit().h(0).cnot(0, 1), workload_id="freshness", shots=4
        )
        hal = HardwareAbstractionLayer.with_builtin_profiles()
        hal.register_backend(
            BraketAwsHALAdapter(
                hal.profile("aws_braket_ionq"),
                device=Device(),
                capability_probe=probe,
                max_calibration_age_seconds=max_age,
            )
        )
        backend_id = "aws_braket_ionq"

    if expected is None:
        job = hal.submit(backend_id, workload, approval_id="offline-fault-injection")
        assert job.metadata["calibration_timestamp"] == calibrated_at
        assert job.metadata["calibration_checked_at"] is not None
        assert job.metadata["calibration_max_age_seconds"] == 3600
        assert provider_runs == 1
    else:
        with pytest.raises(ValueError, match=expected):
            hal.submit(backend_id, workload, approval_id="offline-fault-injection")
        assert provider_runs == 0
    assert probe_calls == 1


@pytest.mark.parametrize("route", ["qiskit", "braket"])
@pytest.mark.parametrize("probe_present,age_present", [(True, False), (False, True)])
def test_submit_calibration_policy_requires_probe_and_age_pair(
    route: str, probe_present: bool, age_present: bool
) -> None:
    """Reject a half-configured submit-time freshness policy at construction."""
    hal = HardwareAbstractionLayer.with_builtin_profiles()
    probe = (
        (lambda: _snapshot(route, calibrated_at=datetime.now(UTC).isoformat()))
        if probe_present
        else None
    )
    age = 3600 if age_present else None
    with pytest.raises(ValueError, match="capability_probe.*max_calibration_age_seconds"):
        if route == "qiskit":
            QiskitRuntimeHALAdapter(
                hal.profile("ibm_quantum"),
                backend=object(),
                capability_probe=probe,
                max_calibration_age_seconds=age,
            )
        else:
            BraketAwsHALAdapter(
                hal.profile("aws_braket_ionq"),
                device=object(),
                capability_probe=probe,
                max_calibration_age_seconds=age,
            )
