# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Qiskit HAL adapter tests
"""Tests for concrete Qiskit adapters behind the provider-neutral HAL."""

from __future__ import annotations

import pytest
from qiskit import QuantumCircuit

from scpn_quantum_control.hardware.hal import HardwareAbstractionLayer
from scpn_quantum_control.hardware.hal_qiskit import (
    QiskitAerHALAdapter,
    QiskitRuntimeHALAdapter,
    qiskit_circuit_to_qasm3_workload,
    qiskit_circuit_to_workload,
)


def _bell_circuit() -> QuantumCircuit:
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure([0, 1], [0, 1])
    return qc


def test_qiskit_qpy_workload_round_trips_through_local_aer_hal() -> None:
    """A real Qiskit circuit should execute through HAL and Aer."""

    hal = HardwareAbstractionLayer.with_builtin_profiles()
    hal.register_backend(QiskitAerHALAdapter(hal.profile("local_qiskit_aer")))
    workload = qiskit_circuit_to_workload(
        _bell_circuit(),
        workload_id="bell",
        shots=128,
        metadata={"purpose": "hal_aer_round_trip"},
    )

    job = hal.submit("local_qiskit_aer", workload)
    result = hal.result(job)

    assert job.status == "completed"
    assert result.status == "completed"
    assert result.shots == 128
    assert sum(result.counts.values()) == 128
    assert set(result.counts).issubset({"00", "11"})
    assert result.metadata["execution_mode"] == "qiskit_aer"
    assert result.metadata["ir_format"] == "qiskit_qpy"


def test_qiskit_runtime_adapter_uses_injected_sampler_and_approval_gate() -> None:
    """IBM Runtime adapter should be injectable and still approval-gated."""

    class FakeBackend:
        name = "ibm_fake"
        num_qubits = 127

    class FakeRegister:
        def get_counts(self) -> dict[str, int]:
            return {"0": 3, "1": 5}

    class FakeData:
        c = FakeRegister()

    class FakePubResult:
        data = FakeData()

    class FakeRuntimeResult:
        def __iter__(self):
            return iter((FakePubResult(),))

    class FakeRuntimeJob:
        def job_id(self) -> str:
            return "runtime-job-1"

        def status(self) -> str:
            return "DONE"

        def result(self, timeout: float | None = None) -> FakeRuntimeResult:
            assert timeout == 600.0
            return FakeRuntimeResult()

        def cancel(self) -> None:
            self.cancelled = True

    class FakeSampler:
        def __init__(self, mode):
            assert mode is FakeBackend
            self.options = type("Options", (), {})()

        def run(self, circuits):
            assert len(circuits) == 1
            return FakeRuntimeJob()

    hal = HardwareAbstractionLayer.with_builtin_profiles()
    hal.register_backend(
        QiskitRuntimeHALAdapter(
            hal.profile("ibm_quantum"),
            backend=FakeBackend,
            sampler_factory=FakeSampler,
        )
    )
    workload = qiskit_circuit_to_workload(_bell_circuit(), workload_id="runtime", shots=8)

    job = hal.submit("ibm_quantum", workload, approval_id="approved-runtime")
    result = hal.result(job)

    assert job.job_id == "runtime-job-1"
    assert job.status == "submitted"
    assert hal.status(job) == "completed"
    assert result.status == "completed"
    assert result.counts == {"0": 3, "1": 5}
    assert result.metadata["execution_mode"] == "qiskit_runtime_sampler"
    assert result.metadata["approval_id"] == "approved-runtime"


def test_qiskit_runtime_adapter_rejects_shot_mismatch() -> None:
    """Runtime result decoding must fail closed on count/shot mismatch."""

    class FakeBackend:
        name = "ibm_fake"
        num_qubits = 127

    class FakeRegister:
        def get_counts(self) -> dict[str, int]:
            return {"0": 3, "1": 4}

    class FakeData:
        c = FakeRegister()

    class FakePubResult:
        data = FakeData()

    class FakeRuntimeResult:
        def __iter__(self):
            return iter((FakePubResult(),))

    class FakeRuntimeJob:
        def job_id(self) -> str:
            return "runtime-job-shot-mismatch"

        def status(self) -> str:
            return "DONE"

        def result(self, timeout: float | None = None) -> FakeRuntimeResult:
            assert timeout == 600.0
            return FakeRuntimeResult()

        def cancel(self) -> None:
            self.cancelled = True

    class FakeSampler:
        def __init__(self, mode):
            assert mode is FakeBackend
            self.options = type("Options", (), {})()

        def run(self, circuits):
            assert len(circuits) == 1
            return FakeRuntimeJob()

    hal = HardwareAbstractionLayer.with_builtin_profiles()
    hal.register_backend(
        QiskitRuntimeHALAdapter(
            hal.profile("ibm_quantum"),
            backend=FakeBackend,
            sampler_factory=FakeSampler,
        )
    )
    workload = qiskit_circuit_to_workload(
        _bell_circuit(), workload_id="runtime_bad_shots", shots=8
    )
    job = hal.submit("ibm_quantum", workload, approval_id="approved-runtime")

    with pytest.raises(ValueError, match="shot count mismatch"):
        hal.result(job)


def test_qiskit_runtime_adapter_sums_overlapping_pub_results() -> None:
    """Counts from multiple PUB results should be accumulated, not overwritten."""

    class FakeBackend:
        name = "ibm_fake"
        num_qubits = 127

    class FakeRegisterFirst:
        def get_counts(self) -> dict[str, int]:
            return {"0": 2, "1": 1}

    class FakeRegisterSecond:
        def get_counts(self) -> dict[str, int]:
            return {"0": 5, "11": 3}

    class FakeDataFirst:
        c = FakeRegisterFirst()

    class FakeDataSecond:
        c = FakeRegisterSecond()

    class FakePubResultFirst:
        data = FakeDataFirst()

    class FakePubResultSecond:
        data = FakeDataSecond()

    class FakeRuntimeResult:
        def __iter__(self):
            return iter((FakePubResultFirst(), FakePubResultSecond()))

    class FakeRuntimeJob:
        def job_id(self) -> str:
            return "runtime-job-merge"

        def status(self) -> str:
            return "DONE"

        def result(self, timeout: float | None = None) -> FakeRuntimeResult:
            assert timeout == 600.0
            return FakeRuntimeResult()

        def cancel(self) -> None:
            self.cancelled = True

    class FakeSampler:
        def __init__(self, mode):
            assert mode is FakeBackend
            self.options = type("Options", (), {})()

        def run(self, circuits):
            assert len(circuits) == 1
            return FakeRuntimeJob()

    hal = HardwareAbstractionLayer.with_builtin_profiles()
    hal.register_backend(
        QiskitRuntimeHALAdapter(
            hal.profile("ibm_quantum"),
            backend=FakeBackend,
            sampler_factory=FakeSampler,
        )
    )
    workload = qiskit_circuit_to_workload(_bell_circuit(), workload_id="runtime_merge", shots=11)
    job = hal.submit("ibm_quantum", workload, approval_id="approved-runtime")
    result = hal.result(job)

    assert result.counts == {"0": 7, "1": 1, "11": 3}
    assert result.shots == 11


def test_qiskit_qasm3_workload_round_trips_when_importer_is_installed() -> None:
    """OpenQASM 3 payloads should execute when qiskit-qasm3-import is present."""

    hal = HardwareAbstractionLayer.with_builtin_profiles()
    hal.register_backend(QiskitAerHALAdapter(hal.profile("local_qiskit_aer")))
    workload = qiskit_circuit_to_qasm3_workload(
        _bell_circuit(),
        workload_id="bell_qasm3",
        shots=64,
        metadata={"purpose": "hal_qasm3_round_trip"},
    )

    job = hal.submit("local_qiskit_aer", workload)
    result = hal.result(job)

    assert result.status == "completed"
    assert result.shots == 64
    assert sum(result.counts.values()) == 64
    assert set(result.counts).issubset({"00", "11"})
    assert result.metadata["ir_format"] == "openqasm3"


def test_qiskit_adapter_rejects_non_qiskit_workload_payload() -> None:
    """Concrete Qiskit adapters should not pretend to execute MLIR strings."""

    hal = HardwareAbstractionLayer.with_builtin_profiles()
    hal.register_backend(QiskitAerHALAdapter(hal.profile("local_qiskit_aer")))

    from scpn_quantum_control.hardware.hal import QuantumWorkload

    workload = QuantumWorkload(
        workload_id="bad_payload",
        ir_format="mlir",
        program="module {}",
        n_qubits=1,
        shots=1,
    )

    with pytest.raises(ValueError, match="qiskit_qpy|OpenQASM"):
        hal.submit("local_qiskit_aer", workload)


def test_qiskit_runtime_status_normalisation_maps_provider_tokens() -> None:
    """Qiskit Runtime status values should map to canonical HAL status values."""

    from scpn_quantum_control.hardware import hal_qiskit as qiskit_mod

    assert qiskit_mod._normalise_status("DONE") == "completed"
    assert qiskit_mod._normalise_status("CANCELED") == "cancelled"
    assert qiskit_mod._normalise_status("IN-PROGRESS") == "running"


def test_qiskit_provider_job_id_extraction_requires_identifier() -> None:
    """Qiskit provider job id extraction should fail closed when id is unavailable."""

    from scpn_quantum_control.hardware import hal_qiskit as qiskit_mod

    class MissingJobId:
        def job_id(self) -> str:
            return "   "

    with pytest.raises(ValueError, match="provider job id"):
        qiskit_mod._provider_job_id(MissingJobId(), provider_name="qiskit_runtime")


def test_qiskit_provider_job_id_rejects_control_characters() -> None:
    """Qiskit provider job ids must reject control-character payloads."""

    from scpn_quantum_control.hardware import hal_qiskit as qiskit_mod

    class BadJobId:
        def job_id(self) -> str:
            return "runtime-job-\n1"

    with pytest.raises(ValueError, match="provider job id"):
        qiskit_mod._provider_job_id(BadJobId(), provider_name="qiskit_runtime")


def test_qiskit_provider_job_id_trims_padding() -> None:
    """Qiskit provider job ids should be trimmed to canonical form."""

    from scpn_quantum_control.hardware import hal_qiskit as qiskit_mod

    class PaddedJobId:
        def job_id(self) -> str:
            return "  runtime-job-42  "

    assert (
        qiskit_mod._provider_job_id(PaddedJobId(), provider_name="qiskit_runtime")
        == "runtime-job-42"
    )


def test_qiskit_backend_name_rejects_control_characters() -> None:
    """Qiskit backend names must reject control-character payloads."""

    from scpn_quantum_control.hardware import hal_qiskit as qiskit_mod

    class BadBackend:
        name = "ibm_\nbackend"

    with pytest.raises(ValueError, match="backend name"):
        qiskit_mod._backend_name(BadBackend())


def test_qiskit_backend_name_trims_padding() -> None:
    """Qiskit backend names should be canonicalised by trimming padding."""

    from scpn_quantum_control.hardware import hal_qiskit as qiskit_mod

    class PaddedBackend:
        name = "  ibm_backend  "

    assert qiskit_mod._backend_name(PaddedBackend()) == "ibm_backend"
