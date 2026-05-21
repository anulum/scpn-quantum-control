# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Pasqal HAL adapter tests
"""Tests for the direct Pasqal/Pulser HAL adapter."""

from __future__ import annotations

import types
from typing import Any

import pytest

from scpn_quantum_control.hardware.hal import HardwareAbstractionLayer, QuantumJobRef
from scpn_quantum_control.hardware.hal_pasqal import (
    PASQAL_PULSER_SCHEMA,
    PasqalPulserHALAdapter,
    pulser_sequence_workload,
)

_PULSER_PLAN = {
    "schema": "pulser_sequence_plan_v1",
    "duration": 1.5,
    "register": {"0": [0.0, 0.0], "1": [4.0, 0.0]},
    "rydberg_channel": "rydberg_global",
    "rabi_envelope": [
        {"time": 0.0, "amplitude": 0.0, "phase": 0.0},
        {"time": 1.5, "amplitude": 1.0, "phase": 0.0},
    ],
    "local_detunings": [
        {"site": 0, "detuning": 0.1},
        {"site": 1, "detuning": -0.1},
    ],
    "interaction_terms": [
        {"source": 0, "target": 1, "coefficient": 0.25},
    ],
    "fim_feedback_terms": [],
}


class _FakePasqalJob:
    def __init__(self) -> None:
        self.id = "pasqal-provider-job-1"
        self.status = "DONE"
        self.cancelled = False

    def result(self) -> dict[str, object]:
        return {"counter": {"00": 5, "11": 7}}

    def cancel(self) -> None:
        self.cancelled = True


class _FakePasqalClient:
    def __init__(self) -> None:
        self.jobs: list[_FakePasqalJob] = []
        self.submissions: list[dict[str, object]] = []

    def submit(self, *, sequence: dict[str, object], shots: int, job_name: str) -> _FakePasqalJob:
        self.submissions.append({"sequence": sequence, "shots": shots, "job_name": job_name})
        job = _FakePasqalJob()
        self.jobs.append(job)
        return job


class _FakePasqalShotMismatchClient(_FakePasqalClient):
    def submit(self, *, sequence: dict[str, object], shots: int, job_name: str) -> _FakePasqalJob:
        del sequence, shots, job_name
        job = _FakePasqalJob()
        job.result = lambda: {"counter": {"00": 1, "11": 1}}  # type: ignore[method-assign]
        self.jobs.append(job)
        return job


def test_pasqal_hal_adapter_executes_injected_client_with_approval() -> None:
    hal = HardwareAbstractionLayer.with_builtin_profiles()
    client = _FakePasqalClient()
    adapter = PasqalPulserHALAdapter(hal.profile("pasqal_cloud"), client=client)
    hal.register_backend(adapter)
    workload = pulser_sequence_workload(
        _PULSER_PLAN,
        workload_id="pasqal_pair",
        n_qubits=2,
        shots=12,
        metadata={"campaign": "hal"},
    )

    job = hal.submit("pasqal_cloud", workload, approval_id="approved-pasqal")
    result = hal.result(job)
    cancelled = hal.cancel(job)

    assert isinstance(job, QuantumJobRef)
    assert job.job_id.startswith("pasqal_cloud:pasqal_pair:")
    assert job.metadata["approval_id"] == "approved-pasqal"
    assert job.metadata["execution_mode"] == "pasqal_pulser"
    assert job.metadata["provider_job_id"] == "pasqal-provider-job-1"
    assert client.submissions == [
        {"sequence": _PULSER_PLAN, "shots": 12, "job_name": "pasqal_pair"}
    ]
    assert result.counts == {"00": 5, "11": 7}
    assert result.shots == 12
    assert cancelled.status == "cancelled"
    assert client.jobs[0].cancelled is True


def test_pasqal_hal_adapter_requires_cloud_approval() -> None:
    hal = HardwareAbstractionLayer.with_builtin_profiles()
    hal.register_backend(
        PasqalPulserHALAdapter(hal.profile("pasqal_cloud"), client=_FakePasqalClient())
    )
    workload = pulser_sequence_workload(
        _PULSER_PLAN, workload_id="needs_approval", n_qubits=2, shots=4
    )

    with pytest.raises(PermissionError, match="approval"):
        hal.submit("pasqal_cloud", workload)


def test_pasqal_hal_adapter_rejects_wrong_profile_ir_and_schema() -> None:
    hal = HardwareAbstractionLayer.with_builtin_profiles()
    with pytest.raises(ValueError, match="pasqal_cloud"):
        PasqalPulserHALAdapter(hal.profile("quera_bloqade"), client=_FakePasqalClient())

    adapter = PasqalPulserHALAdapter(hal.profile("pasqal_cloud"), client=_FakePasqalClient())
    with pytest.raises(ValueError, match="pulser workloads"):
        adapter.submit(
            pulser_sequence_workload(
                _PULSER_PLAN, workload_id="bad_ir", n_qubits=2, shots=4
            ).__class__(
                workload_id="bad_ir",
                ir_format="openqasm3",
                program="OPENQASM 3.0;",
                n_qubits=2,
                shots=4,
            ),
            approval_id="approved",
        )

    bad_schema = dict(_PULSER_PLAN) | {"schema": "other"}
    with pytest.raises(ValueError, match=PASQAL_PULSER_SCHEMA):
        pulser_sequence_workload(bad_schema, workload_id="bad_schema", n_qubits=2, shots=4)


def test_pasqal_hal_adapter_uses_lazy_client_factory() -> None:
    captured: dict[str, object] = {}

    def client_factory(sequence: dict[str, object]) -> _FakePasqalClient:
        captured["schema"] = sequence["schema"]
        return _FakePasqalClient()

    hal = HardwareAbstractionLayer.with_builtin_profiles()
    adapter = PasqalPulserHALAdapter(
        hal.profile("pasqal_cloud"),
        client_factory=client_factory,
        target="FRESNEL",
    )
    job = adapter.submit(
        pulser_sequence_workload(_PULSER_PLAN, workload_id="lazy_pasqal", n_qubits=2, shots=3),
        approval_id="approved",
    )

    assert job.status == "submitted"
    assert job.metadata["target"] == "FRESNEL"
    assert captured == {"schema": PASQAL_PULSER_SCHEMA}


def test_pasqal_hal_adapter_default_builder_is_calibration_gated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hal = HardwareAbstractionLayer.with_builtin_profiles()
    adapter = PasqalPulserHALAdapter(hal.profile("pasqal_cloud"))

    with pytest.raises(RuntimeError, match="client_factory"):
        adapter.submit(
            pulser_sequence_workload(
                _PULSER_PLAN, workload_id="needs_builder", n_qubits=2, shots=1
            ),
            approval_id="approved",
        )

    def fake_import(name: str) -> Any:
        if name == "pulser":
            return types.SimpleNamespace()
        raise ModuleNotFoundError(name)

    monkeypatch.setattr("scpn_quantum_control.hardware.hal_pasqal.import_module", fake_import)
    with pytest.raises(RuntimeError, match="calibrated Pasqal client"):
        adapter.submit(
            pulser_sequence_workload(_PULSER_PLAN, workload_id="has_pulser", n_qubits=2, shots=1),
            approval_id="approved",
        )


def test_pasqal_hal_adapter_validates_payload_shape_and_counts() -> None:
    bad_register = dict(_PULSER_PLAN) | {"register": {"0": [0.0, 0.0]}}
    with pytest.raises(ValueError, match="register"):
        pulser_sequence_workload(bad_register, workload_id="bad_register", n_qubits=2, shots=1)

    bad_envelope = dict(_PULSER_PLAN) | {"rabi_envelope": [{"time": 0.0}]}
    with pytest.raises(ValueError, match="rabi_envelope"):
        pulser_sequence_workload(bad_envelope, workload_id="bad_envelope", n_qubits=2, shots=1)

    class BadClient(_FakePasqalClient):
        def submit(
            self, *, sequence: dict[str, object], shots: int, job_name: str
        ) -> _FakePasqalJob:
            job = _FakePasqalJob()
            job.result = lambda: {"counter": {"00": -1}}  # type: ignore[method-assign]
            return job

    hal = HardwareAbstractionLayer.with_builtin_profiles()
    adapter = PasqalPulserHALAdapter(hal.profile("pasqal_cloud"), client=BadClient())
    job = adapter.submit(
        pulser_sequence_workload(_PULSER_PLAN, workload_id="bad_counts", n_qubits=2, shots=1),
        approval_id="approved",
    )
    with pytest.raises(ValueError, match="non-negative"):
        adapter.result(job)


def test_pasqal_status_normalisation_maps_completion_aliases() -> None:
    """Pasqal status normaliser should map provider completion aliases canonically."""

    from scpn_quantum_control.hardware import hal_pasqal as pasqal_mod

    assert pasqal_mod._normalise_status("SUCCEEDED") == "completed"
    assert pasqal_mod._normalise_status("COMPLETE") == "completed"


def test_pasqal_provider_job_id_extraction_requires_identifier() -> None:
    """Pasqal provider job id extraction should fail closed when id is unavailable."""

    from scpn_quantum_control.hardware import hal_pasqal as pasqal_mod

    assert (
        pasqal_mod._provider_job_id(type("Job", (), {"id": "pasqal-provider-2"})())
        == "pasqal-provider-2"
    )
    with pytest.raises(ValueError, match="provider job id"):
        pasqal_mod._provider_job_id(object())


def test_pasqal_provider_job_id_rejects_control_characters() -> None:
    """Pasqal provider identifiers must reject control-character payloads."""

    from scpn_quantum_control.hardware import hal_pasqal as pasqal_mod

    class BadJob:
        id = "pasqal-provider-\n2"

    with pytest.raises(ValueError, match="provider job id"):
        pasqal_mod._provider_job_id(BadJob())


def test_pasqal_provider_job_id_trims_padding() -> None:
    """Pasqal provider identifiers should be canonicalised by trimming padding."""

    from scpn_quantum_control.hardware import hal_pasqal as pasqal_mod

    class PaddedJob:
        id = "  pasqal-provider-2  "

    assert pasqal_mod._provider_job_id(PaddedJob()) == "pasqal-provider-2"


def test_pasqal_target_rejects_control_characters() -> None:
    """Pasqal targets must reject control-character payloads."""

    profile = HardwareAbstractionLayer.with_builtin_profiles().profile("pasqal_cloud")
    with pytest.raises(ValueError, match="Pasqal target"):
        PasqalPulserHALAdapter(profile, client=_FakePasqalClient(), target="pasqal-\nqpu")


def test_pasqal_target_trims_padding() -> None:
    """Pasqal targets should be canonicalised by trimming padding."""

    profile = HardwareAbstractionLayer.with_builtin_profiles().profile("pasqal_cloud")
    adapter = PasqalPulserHALAdapter(profile, client=_FakePasqalClient(), target="  pasqal-qpu  ")
    assert adapter._target == "pasqal-qpu"


def test_pasqal_hal_adapter_rejects_shot_mismatch() -> None:
    """Pasqal adapter must fail closed when decoded counts diverge from expected shots."""

    hal = HardwareAbstractionLayer.with_builtin_profiles()
    adapter = PasqalPulserHALAdapter(
        hal.profile("pasqal_cloud"),
        client=_FakePasqalShotMismatchClient(),
    )
    hal.register_backend(adapter)
    job = hal.submit(
        "pasqal_cloud",
        pulser_sequence_workload(
            _PULSER_PLAN, workload_id="pasqal_shot_mismatch", n_qubits=2, shots=12
        ),
        approval_id="approved",
    )
    with pytest.raises(ValueError, match="shot count mismatch"):
        hal.result(job)
