# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — hardware HAL pasqal adapters tests
# scpn-quantum-control -- Pasqal HAL adapter tests
"""Tests for the direct Pasqal/Pulser HAL adapter."""

from __future__ import annotations

import json
import types
from dataclasses import replace
from typing import Any

import pytest

from scpn_quantum_control.hardware.hal import (
    HardwareAbstractionLayer,
    QuantumJobRef,
    QuantumWorkload,
)
from scpn_quantum_control.hardware.hal_pasqal import (
    PASQAL_PULSER_SCHEMA,
    PasqalPulserHALAdapter,
    pulser_sequence_workload,
)
from scpn_quantum_control.hardware.provider_modalities import AnalogObservation
from scpn_quantum_control.hardware.provider_semantics import WorkloadSemantics

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
    def __init__(self, counts: dict[str, int] | None = None) -> None:
        self.id = "pasqal-provider-job-1"
        self.status = "DONE"
        self.cancelled = False
        self._counts = {"00": 5, "11": 7} if counts is None else counts

    def result(self) -> dict[str, object]:
        return {"counter": dict(self._counts)}

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
        job = _FakePasqalJob({"00": 1, "11": 1})
        self.jobs.append(job)
        return job


def test_pasqal_hal_adapter_executes_injected_client_with_approval() -> None:
    """An approved client receives original sequence and shots, retaining result and cancellation data."""
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
    """Both HAL and direct adapter entry points refuse before client submission."""
    hal = HardwareAbstractionLayer.with_builtin_profiles()
    client = _FakePasqalClient()
    adapter = PasqalPulserHALAdapter(hal.profile("pasqal_cloud"), client=client)
    hal.register_backend(adapter)
    workload = pulser_sequence_workload(
        _PULSER_PLAN, workload_id="needs_approval", n_qubits=2, shots=4
    )

    with pytest.raises(PermissionError, match="approval"):
        hal.submit("pasqal_cloud", workload)
    with pytest.raises(PermissionError, match="approval"):
        adapter.submit(workload)
    assert client.submissions == []


def test_pasqal_hal_adapter_rejects_wrong_profile_ir_and_schema() -> None:
    """Foreign backend profile, source format or plan schema refuses before execution."""
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
    """Lazy client construction receives the admitted sequence and declared target."""
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
    """Absent SDK and absent calibrated client cannot select a default execution route."""
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
    """Malformed register, schedule and negative counts cannot qualify an analog result."""
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
            job = _FakePasqalJob({"00": -1})
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
    assert pasqal_mod._normalise_status("IN-PROGRESS") == "running"
    assert pasqal_mod._normalise_status("INPROGRESS") == "running"
    assert pasqal_mod._normalise_status("INITIALIZING") == "submitted"
    assert pasqal_mod._normalise_status("STARTING") == "submitted"
    assert pasqal_mod._normalise_status("CREATING") == "submitted"
    assert pasqal_mod._normalise_status("ABORTING") == "cancelled"
    assert pasqal_mod._normalise_status("CANCELLING") == "cancelled"


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


@pytest.mark.parametrize("target", ["pasqal-\nqpu", "", "   "])
def test_pasqal_target_rejects_control_characters(target: str) -> None:
    """Explicit empty or control-character selectors cannot become another route."""
    client = _FakePasqalClient()
    profile = HardwareAbstractionLayer.with_builtin_profiles().profile("pasqal_cloud")
    with pytest.raises(ValueError, match="Pasqal target"):
        PasqalPulserHALAdapter(profile, client=client, target=target)
    assert client.submissions == []


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


def test_pasqal_typed_native_counts_preserve_original_site_order_and_payload() -> None:
    """A caller's JSON plan and ordered sites survive the public transport boundary."""
    from scpn_quantum_control.hardware.provider_modalities import AnalogObservation

    plan = dict(_PULSER_PLAN) | {"register": {"1": [4.0, 0.0], "0": [0.0, 0.0]}}
    original = json.dumps(plan, indent=2)
    client = _FakePasqalClient()
    hal = HardwareAbstractionLayer.with_builtin_profiles()
    hal.register_backend(
        PasqalPulserHALAdapter(
            hal.profile("pasqal_cloud"), client=client, target="declared_FRESNEL"
        )
    )
    workload = pulser_sequence_workload(
        original,
        workload_id="typed_pasqal",
        n_qubits=2,
        shots=12,
        capture_semantics=True,
        requested_target="declared_FRESNEL",
    )
    job = hal.submit("pasqal_cloud", workload, approval_id="transport-contract-only")
    result = hal.result(job)
    assert isinstance(result.provider_observation, AnalogObservation)
    assert result.provider_observation.raw_counts == {"00": 5, "11": 7}
    assert result.provider_observation.request.native_axes == ("1", "0")
    assert job.submission is not None and job.submission.original_program == original
    assert workload.program == original
    submitted = client.submissions[0]["sequence"]
    assert isinstance(submitted, dict)
    assert tuple(submitted["register"]) == ("1", "0")
    with pytest.raises(ValueError, match="workload_id"):
        hal.result(replace(job, workload_id="foreign_plan"))
    assert hal.result(job) is result


def test_pasqal_pinned_target_refuses_before_client_submit() -> None:
    """A different declared target cannot execute or reuse a previous plan."""
    client = _FakePasqalClient()
    hal = HardwareAbstractionLayer.with_builtin_profiles()
    hal.register_backend(
        PasqalPulserHALAdapter(hal.profile("pasqal_cloud"), client=client, target="another_target")
    )
    workload = pulser_sequence_workload(
        _PULSER_PLAN,
        workload_id="target_pin",
        n_qubits=2,
        shots=12,
        capture_semantics=True,
        requested_target="declared_FRESNEL",
    )
    with pytest.raises(ValueError, match="target"):
        hal.submit("pasqal_cloud", workload, approval_id="transport-refusal-only")
    assert client.submissions == []


@pytest.mark.parametrize(
    "mutation,message",
    [
        ({"duration": 0.0}, "positive"),
        ({"register": {"0": "bad", "1": [4.0, 0.0]}}, "coordinates"),
        ({"register": {"0": [0.0], "1": [4.0, 0.0]}}, "coordinates"),
        ({"rydberg_channel": ""}, "non-empty"),
        ({"rabi_envelope": []}, "non-empty"),
        ({"rabi_envelope": [4]}, "mappings"),
        (
            {
                "rabi_envelope": [
                    {"time": 1.0, "amplitude": 0.0, "phase": 0.0},
                    {"time": 0.0, "amplitude": 0.0, "phase": 0.0},
                ]
            },
            "monotonic",
        ),
        ({"local_detunings": "bad"}, "sequence"),
        ({"local_detunings": [4]}, "mappings"),
        ({"interaction_terms": "bad"}, "sequence"),
        ({"interaction_terms": [4]}, "mappings"),
    ],
)
def test_pasqal_malformed_native_sequence_refuses_before_submit_and_keeps_prior_output(
    mutation: dict[str, object],
    message: str,
) -> None:
    """Invalid native register and schedule fields cannot reach a retained client."""
    client = _FakePasqalClient()
    hal = HardwareAbstractionLayer.with_builtin_profiles()
    hal.register_backend(PasqalPulserHALAdapter(hal.profile("pasqal_cloud"), client=client))
    first = hal.submit(
        "pasqal_cloud",
        pulser_sequence_workload(
            _PULSER_PLAN, workload_id="original_analog_anchor", n_qubits=2, shots=12
        ),
        approval_id="native-io-only",
    )
    original = hal.result(first)
    malformed = QuantumWorkload(
        workload_id="malformed_analog_plan",
        ir_format="pulser",
        program=json.dumps(dict(_PULSER_PLAN) | mutation),
        n_qubits=2,
        shots=12,
    )
    with pytest.raises(ValueError, match=message):
        hal.submit("pasqal_cloud", malformed, approval_id="native-io-only")
    assert len(client.submissions) == 1 and hal.result(first) is original


@pytest.mark.parametrize("source", ["{", "[]"])
def test_pasqal_malformed_json_refuses_before_lazy_client(source: str) -> None:
    """Invalid encoded native plans cannot construct a client or choose another target."""
    builds: list[dict[str, object]] = []

    def build(sequence: dict[str, object]) -> _FakePasqalClient:
        builds.append(sequence)
        return _FakePasqalClient()

    hal = HardwareAbstractionLayer.with_builtin_profiles()
    hal.register_backend(PasqalPulserHALAdapter(hal.profile("pasqal_cloud"), client_factory=build))
    malformed = QuantumWorkload(
        workload_id="invalid_json", ir_format="pulser", program=source, n_qubits=2, shots=12
    )
    with pytest.raises(ValueError, match="valid JSON|JSON object"):
        hal.submit("pasqal_cloud", malformed, approval_id="native-io-only")
    assert builds == []


@pytest.mark.parametrize("value", [None, True, "bad", float("nan"), float("inf"), 10**400])
def test_pasqal_invalid_native_coordinates_refuse_before_build(value: object) -> None:
    """Coordinates reject missing, boolean and unbounded numeric values without truncation."""
    plan = dict(_PULSER_PLAN) | {"register": {"0": [value, 0.0], "1": [4.0, 0.0]}}
    with pytest.raises(ValueError, match="numeric|finite"):
        pulser_sequence_workload(plan, workload_id="invalid_coordinate", n_qubits=2, shots=12)


def test_pasqal_retained_lazy_client_receives_each_current_original_plan() -> None:
    """Reusing a client never substitutes an earlier sequence for the current native source."""
    builds: list[dict[str, object]] = []
    client = _FakePasqalClient()

    def build(sequence: dict[str, object]) -> _FakePasqalClient:
        builds.append(sequence)
        return client

    hal = HardwareAbstractionLayer.with_builtin_profiles()
    hal.register_backend(PasqalPulserHALAdapter(hal.profile("pasqal_cloud"), client_factory=build))
    second_plan = dict(_PULSER_PLAN) | {
        "duration": 2.0,
        "register": {"1": [4.0, 0.0], "0": [0.0, 0.0]},
    }
    jobs = []
    for i, plan in enumerate((_PULSER_PLAN, second_plan)):
        jobs.append(
            hal.submit(
                "pasqal_cloud",
                pulser_sequence_workload(
                    plan,
                    workload_id=f"original_plan_{i}",
                    n_qubits=2,
                    shots=12,
                    capture_semantics=True,
                ),
                approval_id="native-io-only",
            )
        )
    assert builds == [_PULSER_PLAN]
    assert [item["sequence"] for item in client.submissions] == [_PULSER_PLAN, second_plan]
    second = hal.result(jobs[1]).provider_observation
    assert isinstance(second, AnalogObservation) and second.request.native_axes == ("1", "0")
    assert hal.result(jobs[0]).job.workload_id == "original_plan_0"


@pytest.mark.parametrize(
    "channel",
    [
        "counts",
        "samples",
        "counts_attribute",
        "counter_attribute",
        "missing",
        "empty",
        "numeric_strings",
    ],
)
def test_pasqal_native_result_channels_preserve_raw_counts_and_recover(channel: str) -> None:
    """Original analog labels survive I/O channels; failures do not poison retained retrieval."""
    counts = {"01": 5, "10": 7}
    replies: dict[str, object] = {
        "counts": {"counts": counts},
        "samples": {"samples": counts},
        "counts_attribute": types.SimpleNamespace(counts=counts),
        "counter_attribute": types.SimpleNamespace(counter=counts),
        "missing": {},
        "empty": {"counts": {}},
        "numeric_strings": {"counts": {"01": "5", "10": "7"}},
    }

    class ChannelJob:
        """Expose an explicit native I/O reply with observable count retrieval."""

        id = "native_analog_channel"

        def __init__(self) -> None:
            """Retain the selected raw channel until a caller repairs it."""
            self.reply = replies[channel]
            self.reads = 0

        def result(self) -> object:
            """Read the original provider reply once for each uncached request."""
            self.reads += 1
            return self.reply

    provider = ChannelJob()

    class ChannelClient:
        """Submit one declared native readout without analog dynamics qualification."""

        def submit(self, *, sequence: dict[str, object], shots: int, job_name: str) -> ChannelJob:
            """Receive the exact current original sequence and native sample count."""
            assert sequence == _PULSER_PLAN and shots == 12 and job_name == "native_result_channel"
            return provider

    hal = HardwareAbstractionLayer.with_builtin_profiles()
    hal.register_backend(
        PasqalPulserHALAdapter(hal.profile("pasqal_cloud"), client=ChannelClient())
    )
    job = hal.submit(
        "pasqal_cloud",
        pulser_sequence_workload(
            _PULSER_PLAN,
            workload_id="native_result_channel",
            n_qubits=2,
            shots=12,
            capture_semantics=True,
        ),
        approval_id="native-io-only",
    )
    if channel in {"missing", "empty", "numeric_strings"}:
        with pytest.raises((RuntimeError, ValueError), match="extract Pasqal|counts|analog count"):
            hal.result(job)
        provider.reply = {"counts": counts}
    result = hal.result(job)
    assert isinstance(result.provider_observation, AnalogObservation)
    assert result.provider_observation.raw_counts == {"01": 5, "10": 7}
    assert (
        result.provider_observation.to_payload()["readout_convention"] == "provider_native_unknown"
    )
    assert hal.result(job) is result and provider.reads == (
        2 if channel in {"missing", "empty", "numeric_strings"} else 1
    )
    counts["01"] = 99
    assert result.provider_observation.raw_counts == {"01": 5, "10": 7}


@pytest.mark.parametrize(
    "fault", ["result", "cancel", "status", "missing_status", "foreign_modality"]
)
def test_pasqal_operation_and_stored_modality_faults_recover_without_losing_original_counts(
    monkeypatch: pytest.MonkeyPatch,
    fault: str,
) -> None:
    """Public HAL operations refuse malformed retention or expose unknown lifecycle honestly."""
    client = _FakePasqalClient()
    hal = HardwareAbstractionLayer.with_builtin_profiles()
    adapter = PasqalPulserHALAdapter(hal.profile("pasqal_cloud"), client=client)
    hal.register_backend(adapter)
    job = hal.submit(
        "pasqal_cloud",
        pulser_sequence_workload(
            _PULSER_PLAN,
            workload_id="retained_analog_request",
            n_qubits=2,
            shots=12,
            capture_semantics=True,
        ),
        approval_id="retention-fault-only",
    )
    provider = client.jobs[0]
    assert job.submission is not None
    with monkeypatch.context() as injected:
        if fault == "result":
            injected.setattr(provider, "result", None)
            with pytest.raises(TypeError, match="result"):
                hal.result(job)
        elif fault == "cancel":
            injected.setattr(provider, "cancel", None)
            with pytest.raises(ValueError, match="cancellation"):
                hal.cancel(job)
        elif fault == "foreign_modality":
            foreign = WorkloadSemantics(
                program_sha256=job.submission.request.program_sha256, n_qubits=2, n_clbits=0
            )
            damaged = replace(job, submission=replace(job.submission, request=foreign))
            injected.setitem(adapter._jobs, job.job_id, damaged)
            with pytest.raises(ValueError, match="analog plan semantics"):
                hal.result(damaged)
        else:
            injected.setattr(provider, "status", (lambda: "DONE") if fault == "status" else None)
            assert hal.status(job) == ("completed" if fault == "status" else "unknown")
    result = hal.result(job)
    assert result.counts == {"00": 5, "11": 7} and hal.result(job) is result


def test_pasqal_legacy_target_pin_requires_native_capture() -> None:
    """A bare legacy annotation cannot claim source-bound target admission."""
    with pytest.raises(ValueError, match="target pin"):
        pulser_sequence_workload(
            _PULSER_PLAN,
            workload_id="uncaptured_target",
            n_qubits=2,
            shots=12,
            requested_target="unbound_target",
        )


def test_pasqal_native_site_reader_refuses_a_nonmapping_internal_register() -> None:
    """Supplement public plan admission with the native site-reader's defensive type contract."""
    from scpn_quantum_control.hardware.hal_pasqal import _site_order

    with pytest.raises(ValueError, match="site mapping"):
        _site_order({"register": ["1", "0"]})


def test_pasqal_callable_native_job_identity_survives_public_retention() -> None:
    """A native callable handle identifies the retained original request and readout."""
    provider = types.SimpleNamespace(
        id=lambda: "native_callable_handle",
        status=lambda: "DONE",
        result=lambda: {"counter": {"01": 5, "10": 7}},
    )

    class NativeClient:
        """Expose an explicit I/O handle without SDK or analog dynamics qualification."""

        def submit(self, *, sequence: dict[str, object], shots: int, job_name: str) -> object:
            """Receive the exact original sequence and return its callable native identity."""
            assert sequence == _PULSER_PLAN and shots == 12 and job_name == "callable_identity"
            return provider

    hal = HardwareAbstractionLayer.with_builtin_profiles()
    hal.register_backend(
        PasqalPulserHALAdapter(hal.profile("pasqal_cloud"), client=NativeClient())
    )
    job = hal.submit(
        "pasqal_cloud",
        pulser_sequence_workload(
            _PULSER_PLAN,
            workload_id="callable_identity",
            n_qubits=2,
            shots=12,
            capture_semantics=True,
        ),
        approval_id="native-io-only",
    )
    assert job.metadata["provider_job_id"] == "native_callable_handle"
    assert hal.status(job) == "completed"
    observation = hal.result(job).provider_observation
    assert isinstance(observation, AnalogObservation)
    assert observation.raw_counts == {"01": 5, "10": 7}
    assert hal.result(job).job is job
