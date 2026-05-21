# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- IQM HAL adapter tests
"""Tests for the direct IQM HAL adapter."""

from __future__ import annotations

import types
from typing import Any

import pytest
from qiskit import QuantumCircuit

from scpn_quantum_control.hardware.hal import HardwareAbstractionLayer, QuantumJobRef
from scpn_quantum_control.hardware.hal_iqm import IQMHALAdapter, iqm_qiskit_workload


class _FakeIQMResult:
    def __init__(self, counts: dict[str, int]) -> None:
        self._counts = counts

    def get_counts(self) -> dict[str, int]:
        return dict(self._counts)


class _FakeIQMJob:
    def __init__(
        self,
        counts: dict[str, int],
        *,
        status: str = "DONE",
        expected_timeout: float | None = 9.5,
    ) -> None:
        self._counts = counts
        self._status = status
        self._expected_timeout = expected_timeout
        self.cancelled = False

    def job_id(self) -> str:
        return "iqm-job-123"

    def status(self) -> str:
        return self._status

    def result(self, timeout: float | None = None) -> _FakeIQMResult:
        if self._expected_timeout is not None:
            assert timeout == self._expected_timeout
        return _FakeIQMResult(self._counts)

    def cancel(self) -> None:
        self.cancelled = True


class _FakeIQMBackend:
    name = "fake_garnet"

    def __init__(self) -> None:
        self.jobs: list[_FakeIQMJob] = []
        self.received_shots: list[int] = []

    def run(self, circuits: list[QuantumCircuit], *, shots: int) -> _FakeIQMJob:
        assert len(circuits) == 1
        assert circuits[0].num_qubits == 2
        self.received_shots.append(shots)
        job = _FakeIQMJob({"00": 7, "11": 9})
        self.jobs.append(job)
        return job


class _FakeIQMProvider:
    def __init__(self, url: str, *, quantum_computer: str | None = None) -> None:
        self.url = url
        self.quantum_computer = quantum_computer

    def get_backend(self) -> _FakeIQMBackend:
        return _FakeIQMBackend()


def _bell_circuit() -> QuantumCircuit:
    circuit = QuantumCircuit(2, 2)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.measure([0, 1], [0, 1])
    return circuit


def test_iqm_hal_adapter_executes_injected_backend_with_approval() -> None:
    hal = HardwareAbstractionLayer.with_builtin_profiles()
    backend = _FakeIQMBackend()
    adapter = IQMHALAdapter(hal.profile("iqm_cloud"), backend=backend, timeout_s=9.5)
    hal.register_backend(adapter)
    workload = iqm_qiskit_workload(
        _bell_circuit(),
        workload_id="iqm_bell",
        shots=16,
        metadata={"campaign": "hal"},
    )

    job = hal.submit("iqm_cloud", workload, approval_id="approved-iqm")
    status = hal.status(job)
    result = hal.result(job)
    cancelled = hal.cancel(job)

    assert isinstance(job, QuantumJobRef)
    assert job.job_id.startswith("iqm_cloud:iqm_bell:")
    assert job.metadata["provider_job_id"] == "iqm-job-123"
    assert job.metadata["approval_id"] == "approved-iqm"
    assert job.metadata["execution_mode"] == "iqm_qiskit"
    assert job.metadata["backend_name"] == "fake_garnet"
    assert status == "completed"
    assert result.counts == {"00": 7, "11": 9}
    assert result.shots == 16
    assert result.metadata["backend_name"] == "fake_garnet"
    assert cancelled.status == "cancelled"
    assert backend.jobs[0].cancelled is True


def test_iqm_hal_adapter_requires_cloud_approval() -> None:
    hal = HardwareAbstractionLayer.with_builtin_profiles()
    hal.register_backend(IQMHALAdapter(hal.profile("iqm_cloud"), backend=_FakeIQMBackend()))
    workload = iqm_qiskit_workload(_bell_circuit(), workload_id="needs_approval", shots=8)

    with pytest.raises(PermissionError, match="approval"):
        hal.submit("iqm_cloud", workload)


def test_iqm_hal_adapter_rejects_wrong_profile_and_ir() -> None:
    hal = HardwareAbstractionLayer.with_builtin_profiles()
    with pytest.raises(ValueError, match="iqm_cloud"):
        IQMHALAdapter(hal.profile("ibm_quantum"), backend=_FakeIQMBackend())

    adapter = IQMHALAdapter(hal.profile("iqm_cloud"), backend=_FakeIQMBackend())
    with pytest.raises(ValueError, match="qiskit_qpy"):
        adapter.submit(
            iqm_qiskit_workload(_bell_circuit(), workload_id="bad_ir", shots=4).__class__(
                workload_id="bad_ir",
                ir_format="openqasm3",
                program="OPENQASM 3.0;",
                n_qubits=2,
                shots=4,
            ),
            approval_id="approved",
        )


def test_iqm_hal_adapter_uses_lazy_remote_provider_factory() -> None:
    captured: dict[str, str | None] = {}

    class Provider(_FakeIQMProvider):
        def __init__(self, url: str, *, quantum_computer: str | None = None) -> None:
            captured["url"] = url
            captured["quantum_computer"] = quantum_computer
            super().__init__(url, quantum_computer=quantum_computer)

    def import_module(name: str) -> Any:
        if name == "iqm.qiskit_iqm.iqm_provider":
            return types.SimpleNamespace(IQMProvider=Provider)
        raise ModuleNotFoundError(name)

    hal = HardwareAbstractionLayer.with_builtin_profiles()
    adapter = IQMHALAdapter(
        hal.profile("iqm_cloud"),
        server_url="https://example.iqm.invalid",
        quantum_computer="garnet",
        import_module=import_module,
    )
    job = adapter.submit(
        iqm_qiskit_workload(_bell_circuit(), workload_id="remote_iqm", shots=16),
        approval_id="approved",
    )

    assert job.status == "submitted"
    assert captured == {
        "url": "https://example.iqm.invalid",
        "quantum_computer": "garnet",
    }


def test_iqm_hal_adapter_fails_closed_without_backend_or_server_url() -> None:
    hal = HardwareAbstractionLayer.with_builtin_profiles()
    adapter = IQMHALAdapter(hal.profile("iqm_cloud"))

    with pytest.raises(RuntimeError, match="server_url"):
        adapter.submit(
            iqm_qiskit_workload(_bell_circuit(), workload_id="no_route", shots=4),
            approval_id="approved",
        )


def test_iqm_hal_adapter_reports_unknown_jobs_and_bad_counts() -> None:
    hal = HardwareAbstractionLayer.with_builtin_profiles()
    adapter = IQMHALAdapter(hal.profile("iqm_cloud"), backend=_FakeIQMBackend())

    with pytest.raises(KeyError, match="unknown job_id"):
        adapter.result(
            QuantumJobRef(
                job_id="missing",
                backend_id="iqm_cloud",
                workload_id="missing",
                status="submitted",
            )
        )

    class BadBackend(_FakeIQMBackend):
        def run(self, circuits: list[QuantumCircuit], *, shots: int) -> _FakeIQMJob:
            return _FakeIQMJob({"00": -1}, expected_timeout=None)

    bad = IQMHALAdapter(hal.profile("iqm_cloud"), backend=BadBackend())
    job = bad.submit(
        iqm_qiskit_workload(_bell_circuit(), workload_id="bad_counts", shots=4),
        approval_id="approved",
    )
    with pytest.raises(ValueError, match="non-negative"):
        bad.result(job)


def test_iqm_hal_adapter_rejects_provider_job_without_id() -> None:
    """IQM adapter should fail closed when backend job id is missing."""

    class MissingIdJob:
        def status(self) -> str:
            return "DONE"

        def result(self, timeout: float | None = None) -> _FakeIQMResult:
            del timeout
            return _FakeIQMResult({"0": 1})

        def cancel(self) -> None:
            self.cancelled = True

    class MissingIdBackend(_FakeIQMBackend):
        def run(self, circuits: list[QuantumCircuit], *, shots: int) -> MissingIdJob:
            del circuits, shots
            return MissingIdJob()

    hal = HardwareAbstractionLayer.with_builtin_profiles()
    adapter = IQMHALAdapter(hal.profile("iqm_cloud"), backend=MissingIdBackend())

    with pytest.raises(ValueError, match="provider job id"):
        adapter.submit(
            iqm_qiskit_workload(_bell_circuit(), workload_id="missing_iqm_job_id", shots=4),
            approval_id="approved",
        )


def test_iqm_provider_job_id_rejects_control_characters() -> None:
    """IQM provider identifiers must reject control-character payloads."""

    from scpn_quantum_control.hardware import hal_iqm as iqm_mod

    class BadJob:
        job_id = "iqm-provider-\n2"

    with pytest.raises(ValueError, match="provider job id"):
        iqm_mod._job_id(BadJob())


def test_iqm_provider_job_id_trims_padding() -> None:
    """IQM provider identifiers should be canonicalised by trimming padding."""

    from scpn_quantum_control.hardware import hal_iqm as iqm_mod

    class PaddedJob:
        job_id = "  iqm-provider-2  "

    assert iqm_mod._job_id(PaddedJob()) == "iqm-provider-2"


def test_iqm_backend_name_rejects_control_characters() -> None:
    """IQM backend names must reject control-character payloads."""

    from scpn_quantum_control.hardware import hal_iqm as iqm_mod

    class BadBackend:
        name = "iqm-\nbackend"

    with pytest.raises(ValueError, match="backend name"):
        iqm_mod._backend_name(BadBackend())


def test_iqm_backend_name_trims_padding() -> None:
    """IQM backend names should be canonicalised by trimming padding."""

    from scpn_quantum_control.hardware import hal_iqm as iqm_mod

    class PaddedBackend:
        name = "  iqm-backend  "

    assert iqm_mod._backend_name(PaddedBackend()) == "iqm-backend"


def test_iqm_status_normalisation_maps_provider_tokens() -> None:
    """IQM status values should map to canonical HAL status values."""

    from scpn_quantum_control.hardware import hal_iqm as iqm_mod

    assert iqm_mod._normalise_status("SUCCEEDED") == "completed"
    assert iqm_mod._normalise_status("CANCELED") == "cancelled"
    assert iqm_mod._normalise_status("IN-PROGRESS") == "running"
    assert iqm_mod._normalise_status("INPROGRESS") == "running"
    assert iqm_mod._normalise_status("INITIALIZING") == "submitted"
    assert iqm_mod._normalise_status("STARTING") == "submitted"


def test_iqm_adapter_rejects_shot_mismatch() -> None:
    """IQM adapter must fail closed when decoded counts diverge from requested shots."""

    hal = HardwareAbstractionLayer.with_builtin_profiles()
    backend = _FakeIQMBackend()
    adapter = IQMHALAdapter(hal.profile("iqm_cloud"), backend=backend, timeout_s=9.5)
    hal.register_backend(adapter)
    workload = iqm_qiskit_workload(
        _bell_circuit(),
        workload_id="iqm_shot_mismatch",
        shots=15,
    )

    job = hal.submit("iqm_cloud", workload, approval_id="approved-iqm")
    with pytest.raises(ValueError, match="shot count mismatch"):
        hal.result(job)


def test_iqm_quantum_computer_rejects_control_characters() -> None:
    profile = HardwareAbstractionLayer.with_builtin_profiles().profile("iqm_cloud")
    with pytest.raises(ValueError, match="IQM quantum computer"):
        IQMHALAdapter(profile, backend=_FakeIQMBackend(), quantum_computer="garnet\nbad")


def test_iqm_quantum_computer_trims_padding() -> None:
    profile = HardwareAbstractionLayer.with_builtin_profiles().profile("iqm_cloud")
    adapter = IQMHALAdapter(
        profile,
        backend=_FakeIQMBackend(),
        quantum_computer="  garnet  ",
    )
    assert adapter._quantum_computer == "garnet"
