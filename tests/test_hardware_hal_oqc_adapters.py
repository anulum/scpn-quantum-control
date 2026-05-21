# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — OQC HAL adapter tests
"""Tests for direct OQC QCAAS execution behind the provider-neutral HAL."""

from __future__ import annotations

from typing import Any

import pytest

from scpn_quantum_control.hardware.hal import HardwareAbstractionLayer, QuantumWorkload
from scpn_quantum_control.hardware.hal_oqc import OQCHALAdapter, oqc_openqasm3_workload

_OPENQASM3 = "OPENQASM 3.0;\nqubit[2] q;\nbit[2] c;\nh q[0];\ncx q[0], q[1];"


class _FakeOQCJob:
    def __init__(self) -> None:
        self.id = "oqc-provider-job-1"
        self.status = "COMPLETED"
        self.cancelled = False

    def result(self) -> dict[str, object]:
        return {"counts": {"00": 2, "11": 6}}

    def cancel(self) -> None:
        self.cancelled = True


class _FakeOQCClient:
    def __init__(self) -> None:
        self.submissions: list[dict[str, object]] = []
        self.jobs: list[_FakeOQCJob] = []

    def submit(
        self,
        *,
        program: str,
        shots: int,
        target: str,
        name: str,
    ) -> _FakeOQCJob:
        self.submissions.append(
            {"program": program, "shots": shots, "target": target, "name": name}
        )
        job = _FakeOQCJob()
        self.jobs.append(job)
        return job


def test_oqc_adapter_executes_injected_client_with_approval() -> None:
    hal = HardwareAbstractionLayer.with_builtin_profiles()
    client = _FakeOQCClient()
    hal.register_backend(OQCHALAdapter(hal.profile("oqc_cloud"), client=client, target="Lucy"))
    workload = oqc_openqasm3_workload(
        _OPENQASM3,
        workload_id="oqc_bell",
        n_qubits=2,
        shots=8,
        metadata={"campaign": "hal"},
    )

    with pytest.raises(PermissionError, match="approval"):
        hal.submit("oqc_cloud", workload)

    job = hal.submit("oqc_cloud", workload, approval_id="approved-oqc")
    status = hal.status(job)
    result = hal.result(job)
    cancelled = hal.cancel(job)

    assert job.status == "submitted"
    assert status == "completed"
    assert job.job_id.startswith("oqc_cloud:oqc_bell:")
    assert job.metadata["execution_mode"] == "oqc_qcaas_openqasm3"
    assert job.metadata["target"] == "Lucy"
    assert job.metadata["provider_job_id"] == "oqc-provider-job-1"
    assert result.counts == {"00": 2, "11": 6}
    assert result.shots == 8
    assert cancelled.status == "cancelled"
    assert client.jobs[0].cancelled is True
    assert client.submissions == [
        {"program": _OPENQASM3, "shots": 8, "target": "Lucy", "name": "oqc_bell"}
    ]


def test_oqc_adapter_rejects_wrong_profile_ir_and_program_shape() -> None:
    hal = HardwareAbstractionLayer.with_builtin_profiles()
    with pytest.raises(ValueError, match="oqc_cloud"):
        OQCHALAdapter(hal.profile("ibm_quantum"), client=_FakeOQCClient())

    adapter = OQCHALAdapter(hal.profile("oqc_cloud"), client=_FakeOQCClient())
    with pytest.raises(ValueError, match="openqasm3 workloads"):
        adapter.submit(
            QuantumWorkload(
                workload_id="bad_ir",
                ir_format="qiskit_qpy",
                program="payload",
                n_qubits=2,
                shots=8,
            ),
            approval_id="approved",
        )

    with pytest.raises(ValueError, match="OPENQASM 3.0"):
        oqc_openqasm3_workload(
            "OPENQASM 2.0;\nqreg q[1];",
            workload_id="bad_qasm",
            n_qubits=1,
            shots=8,
        )


def test_oqc_adapter_supports_client_factory_and_validates_counts() -> None:
    captured: dict[str, object] = {}

    def client_factory() -> _FakeOQCClient:
        captured["called"] = True
        return _FakeOQCClient()

    hal = HardwareAbstractionLayer.with_builtin_profiles()
    adapter = OQCHALAdapter(
        hal.profile("oqc_cloud"),
        client_factory=client_factory,
        target="Toshiko",
    )
    job = adapter.submit(
        oqc_openqasm3_workload(_OPENQASM3, workload_id="factory_oqc", n_qubits=2, shots=8),
        approval_id="approved",
    )
    assert adapter.result(job).counts == {"00": 2, "11": 6}
    assert captured == {"called": True}

    class BadJob(_FakeOQCJob):
        def result(self) -> dict[str, object]:
            return {"counts": {"00": -1}}

    class BadClient(_FakeOQCClient):
        def submit(self, *, program: str, shots: int, target: str, name: str) -> BadJob:
            del program, shots, target, name
            return BadJob()

    adapter = OQCHALAdapter(hal.profile("oqc_cloud"), client=BadClient())
    job = adapter.submit(
        oqc_openqasm3_workload(_OPENQASM3, workload_id="bad_counts", n_qubits=2, shots=1),
        approval_id="approved",
    )
    with pytest.raises(ValueError, match="non-negative"):
        adapter.result(job)


def test_oqc_default_builder_is_sdk_gated(monkeypatch: pytest.MonkeyPatch) -> None:
    hal = HardwareAbstractionLayer.with_builtin_profiles()
    adapter = OQCHALAdapter(hal.profile("oqc_cloud"))
    workload = oqc_openqasm3_workload(_OPENQASM3, workload_id="needs_sdk", n_qubits=2, shots=1)

    with pytest.raises(RuntimeError, match="oqc-qcaas-client|calibrated OQC"):
        adapter.submit(workload, approval_id="approved")

    def fake_import(name: str) -> Any:
        if name == "qcaas_client":
            return object()
        raise ModuleNotFoundError(name)

    monkeypatch.setattr("scpn_quantum_control.hardware.hal_oqc.import_module", fake_import)
    with pytest.raises(RuntimeError, match="calibrated OQC"):
        adapter.submit(workload, approval_id="approved")


def test_oqc_status_normalisation_maps_completion_aliases() -> None:
    """OQC status normaliser should map provider completion aliases canonically."""

    from scpn_quantum_control.hardware import hal_oqc as oqc_mod

    assert oqc_mod._normalise_status("SUCCEEDED") == "completed"
    assert oqc_mod._normalise_status("COMPLETE") == "completed"


def test_oqc_provider_job_id_extraction_requires_identifier() -> None:
    """OQC provider job id extraction should fail closed when id is unavailable."""

    from scpn_quantum_control.hardware import hal_oqc as oqc_mod

    assert (
        oqc_mod._provider_job_id(type("Job", (), {"id": "oqc-provider-2"})()) == "oqc-provider-2"
    )
    with pytest.raises(ValueError, match="provider job id"):
        oqc_mod._provider_job_id(object())
