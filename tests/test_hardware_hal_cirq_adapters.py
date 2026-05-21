# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Cirq HAL adapter tests
"""Tests for local Cirq execution behind the provider-neutral HAL."""

from __future__ import annotations

from typing import Any

import pytest

from scpn_quantum_control.hardware.hal import HardwareAbstractionLayer, QuantumWorkload
from scpn_quantum_control.hardware.hal_cirq import CirqLocalHALAdapter, cirq_circuit_workload


class _FakeResult:
    def __init__(self, histogram: dict[int, int]) -> None:
        self.id = "cirq-provider-job-1"
        self._histogram = histogram

    def histogram(self, *, key: str) -> dict[int, int]:
        assert key == "m"
        return self._histogram


class _FakeSimulator:
    def __init__(self) -> None:
        self.runs: list[dict[str, object]] = []

    def run(self, circuit: object, *, repetitions: int) -> _FakeResult:
        self.runs.append({"circuit": circuit, "repetitions": repetitions})
        return _FakeResult({0: 3, 3: 5})


def test_cirq_hal_adapter_executes_injected_simulator_without_cloud_approval() -> None:
    simulator = _FakeSimulator()
    hal = HardwareAbstractionLayer.with_builtin_profiles()
    hal.register_backend(
        CirqLocalHALAdapter(
            hal.profile("local_cirq"),
            circuit_factory=lambda source: {"source": source},
            simulator=simulator,
            measurement_key="m",
        )
    )
    workload = cirq_circuit_workload(
        "CIRQ_JSON_OR_TEXT",
        workload_id="cirq_bell",
        n_qubits=2,
        shots=8,
        metadata={"campaign": "hal"},
    )

    job = hal.submit("local_cirq", workload)
    result = hal.result(job)
    cancelled = hal.cancel(job)

    assert job.status == "completed"
    assert job.job_id.startswith("local_cirq:cirq_bell:")
    assert job.metadata["execution_mode"] == "local_cirq_simulator"
    assert job.metadata["provider_job_id"] == "cirq-provider-job-1"
    assert result.counts == {"00": 3, "11": 5}
    assert result.shots == 8
    assert result.metadata["measurement_key"] == "m"
    assert cancelled.status == "cancelled"
    assert simulator.runs == [{"circuit": {"source": "CIRQ_JSON_OR_TEXT"}, "repetitions": 8}]


def test_cirq_hal_adapter_rejects_wrong_profile_and_ir() -> None:
    hal = HardwareAbstractionLayer.with_builtin_profiles()
    with pytest.raises(ValueError, match="local_cirq"):
        CirqLocalHALAdapter(hal.profile("local_qiskit_aer"), simulator=_FakeSimulator())

    adapter = CirqLocalHALAdapter(hal.profile("local_cirq"), simulator=_FakeSimulator())
    with pytest.raises(ValueError, match="cirq workloads"):
        adapter.submit(
            QuantumWorkload(
                workload_id="bad_ir",
                ir_format="openqasm3",
                program="OPENQASM 3.0;",
                n_qubits=2,
                shots=8,
            )
        )


def test_cirq_hal_adapter_validates_histogram_counts() -> None:
    class BadResult:
        def histogram(self, *, key: str) -> dict[int, int]:
            del key
            return {0: -1}

    class BadSimulator:
        def run(self, circuit: object, *, repetitions: int) -> BadResult:
            del circuit, repetitions
            return BadResult()

    adapter = CirqLocalHALAdapter(
        HardwareAbstractionLayer.with_builtin_profiles().profile("local_cirq"),
        circuit_factory=lambda source: source,
        simulator=BadSimulator(),
    )
    with pytest.raises(ValueError, match="non-negative"):
        adapter.submit(
            cirq_circuit_workload("CIRQ", workload_id="bad_counts", n_qubits=1, shots=1)
        )


def test_cirq_hal_adapter_rejects_shot_mismatch() -> None:
    """Cirq adapter must fail closed when decoded counts do not match requested shots."""

    class MismatchResult:
        id = "cirq-provider-job-mismatch"

        def histogram(self, *, key: str) -> dict[int, int]:
            del key
            return {0: 3, 1: 1}

    class MismatchSimulator:
        def run(self, circuit: object, *, repetitions: int) -> MismatchResult:
            del circuit, repetitions
            return MismatchResult()

    adapter = CirqLocalHALAdapter(
        HardwareAbstractionLayer.with_builtin_profiles().profile("local_cirq"),
        circuit_factory=lambda source: source,
        simulator=MismatchSimulator(),
    )
    with pytest.raises(ValueError, match="shot count mismatch"):
        adapter.submit(
            cirq_circuit_workload("CIRQ", workload_id="shot_mismatch", n_qubits=1, shots=5)
        )


def test_cirq_hal_adapter_default_builder_is_sdk_gated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = CirqLocalHALAdapter(
        HardwareAbstractionLayer.with_builtin_profiles().profile("local_cirq")
    )
    workload = cirq_circuit_workload("CIRQ", workload_id="needs_sdk", n_qubits=1, shots=1)

    def fake_import(name: str) -> Any:
        raise ModuleNotFoundError(name)

    monkeypatch.setattr("scpn_quantum_control.hardware.hal_cirq.import_module", fake_import)
    with pytest.raises(RuntimeError, match="cirq-core"):
        adapter.submit(workload)

    def fake_present_import(name: str) -> Any:
        if name == "cirq":
            return object()
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(
        "scpn_quantum_control.hardware.hal_cirq.import_module", fake_present_import
    )
    with pytest.raises(RuntimeError, match="circuit_factory"):
        adapter.submit(workload)


def test_cirq_provider_job_id_extraction_requires_identifier() -> None:
    """Cirq provider job id extraction should fail closed when id is unavailable."""

    from scpn_quantum_control.hardware import hal_cirq as cirq_mod

    assert (
        cirq_mod._provider_job_id(type("Result", (), {"id": "cirq-provider-2"})())
        == "cirq-provider-2"
    )
    with pytest.raises(ValueError, match="provider job id"):
        cirq_mod._provider_job_id(object())
