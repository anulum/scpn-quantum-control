# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — hardware HAL quera bloqade adapters tests
# scpn-quantum-control -- QuEra Bloqade HAL adapter tests
"""Tests for QuEra Bloqade execution behind the provider-neutral HAL."""

from __future__ import annotations

import json
import types
from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any

import pytest

from scpn_quantum_control.hardware import hal_quera_bloqade as quera_mod
from scpn_quantum_control.hardware.hal import (
    HardwareAbstractionLayer,
    QuantumJobRef,
    QuantumWorkload,
)
from scpn_quantum_control.hardware.hal_quera_bloqade import (
    QuEraBloqadeHALAdapter,
    bloqade_ahs_workload,
)
from scpn_quantum_control.hardware.provider_modalities import AnalogObservation
from scpn_quantum_control.hardware.provider_semantics import WorkloadSemantics

_ATOM0 = {"index": 0, "position": [0.0, 0.0]}
_ATOM1 = {"index": 1, "position": [5.0, 0.0]}

_BLOQADE_PLAN = {
    "schema": "bloqade_ahs_plan_v1",
    "duration": 2.0,
    "atoms": [_ATOM0, _ATOM1],
    "rabi_amplitude_piecewise_linear": [[0.0, 0.0], [1.0, 1.2], [2.0, 0.0]],
    "rabi_phase_piecewise_linear": [[0.0, 0.0], [2.0, 0.0]],
    "local_detunings": [{"oscillator": 0, "detuning": -0.2}, {"oscillator": 1, "detuning": 0.2}],
    "rydberg_interactions": [{"source": 0, "target": 1, "coefficient": 1.0}],
    "fim_feedback_terms": [],
}


class _FakeBloqadeReport:
    def __init__(self, bitstrings: list[str]) -> None:
        self.bitstrings = bitstrings


class _FakeRawBloqadeReport:
    def __init__(self, bitstrings: list[list[int]]) -> None:
        self.raw_bitstrings = bitstrings


class _FakeBloqadeBatch:
    def __init__(self, bitstrings: list[str] | None = None, report: object | None = None) -> None:
        self.id = "quera-provider-job-1"
        self._bitstrings = bitstrings
        self._report = report
        self.fetched = False
        self.cancelled = False
        self.status: object = _FakeStatusName()

    def fetch(self) -> _FakeBloqadeBatch:
        self.fetched = True
        return self

    def report(self) -> object:
        if self._report is not None:
            return self._report
        return _FakeBloqadeReport(self._bitstrings or [])

    def cancel(self) -> _FakeBloqadeBatch:
        self.cancelled = True
        self.status = "cancelled"
        return self


class _FakeBloqadeRoutine:
    def __init__(self) -> None:
        self.runs: list[dict[str, Any]] = []
        self.batch = _FakeBloqadeBatch(["00", "11", "11"])

    def run(self, *, shots: int, name: str) -> _FakeBloqadeBatch:
        self.runs.append({"shots": shots, "name": name})
        return self.batch


class _FakeStatusName:
    name = "COMPLETED"


class _FakeNoCancelBatch:
    id = "quera-provider-job-no-cancel"
    status = "finished"

    def report(self) -> dict[str, dict[str, int]]:
        return {"counts": {"01": 2, "10": 1}}


class _FakeNoFetchBatch:
    status = "done"
    bitstrings = [[0, 1], [0, 1], [1, 0]]


class _FakeNoCancelRoutine:
    def __init__(self) -> None:
        self.batch = _FakeNoCancelBatch()

    def run(self, *, shots: int, name: str) -> _FakeNoCancelBatch:
        del shots, name
        return self.batch


class _FakeShotMismatchRoutine:
    def run(self, *, shots: int, name: str) -> _FakeBloqadeBatch:
        del shots, name
        return _FakeBloqadeBatch(["00", "11"])


def test_quera_bloqade_adapter_runs_injected_routine_and_approval_gate() -> None:
    """QuEra Bloqade adapter should run approved AHS workloads and normalise reports."""
    routine = _FakeBloqadeRoutine()
    hal = HardwareAbstractionLayer.with_builtin_profiles()
    hal.register_backend(
        QuEraBloqadeHALAdapter(
            hal.profile("quera_bloqade"),
            routine=routine,
            routine_name="aquila-local-emulator",
            routine_factory=lambda workload: workload.program,
        )
    )
    workload = bloqade_ahs_workload(
        _BLOQADE_PLAN,
        workload_id="quera_rydberg_pair",
        n_qubits=2,
        shots=3,
        metadata={"lane": "hal"},
    )

    with pytest.raises(PermissionError, match="approval"):
        hal.submit("quera_bloqade", workload)

    job = hal.submit("quera_bloqade", workload, approval_id="approved-quera")
    status = hal.status(job)
    result = hal.result(job)
    cancelled = hal.cancel(job)

    assert job.status == "submitted"
    assert status == "completed"
    assert result.counts == {"00": 1, "11": 2}
    assert result.shots == 3
    assert result.metadata["execution_mode"] == "quera_bloqade"
    assert result.metadata["routine_name"] == "aquila-local-emulator"
    assert cancelled.status == "cancelled"
    assert routine.runs == [{"shots": 3, "name": "quera_rydberg_pair"}]
    assert routine.batch.fetched is True
    assert routine.batch.cancelled is True
    assert job.job_id.startswith("quera_bloqade:quera_rydberg_pair:")
    assert job.metadata == {
        "approval_id": "approved-quera",
        "provider_job_id": "quera-provider-job-1",
        "execution_mode": "quera_bloqade",
        "routine_name": "aquila-local-emulator",
        "ir_format": "bloqade",
        "n_qubits": 2,
        "shots": 3,
    }


def test_quera_bloqade_adapter_enforces_approval_directly() -> None:
    """The adapter should also enforce approval when bypassing the HAL router."""
    adapter = QuEraBloqadeHALAdapter(
        HardwareAbstractionLayer.with_builtin_profiles().profile("quera_bloqade"),
        routine=_FakeBloqadeRoutine(),
    )

    with pytest.raises(PermissionError, match="approval"):
        adapter.submit(
            bloqade_ahs_workload(
                _BLOQADE_PLAN, workload_id="direct_no_approval", n_qubits=2, shots=1
            )
        )


def test_quera_bloqade_adapter_rejects_non_bloqade_payloads() -> None:
    """Direct Bloqade execution should fail closed until translation is explicit."""
    hal = HardwareAbstractionLayer.with_builtin_profiles()
    hal.register_backend(
        QuEraBloqadeHALAdapter(
            hal.profile("quera_bloqade"),
            routine=_FakeBloqadeRoutine(),
            routine_factory=lambda workload: workload.program,
        )
    )
    workload = QuantumWorkload(
        workload_id="bad_quera",
        ir_format="braket_ahs",
        program="{}",
        n_qubits=2,
        shots=8,
    )

    with pytest.raises(ValueError, match="bloqade workloads"):
        hal.submit("quera_bloqade", workload, approval_id="approved-quera")


def test_quera_bloqade_adapter_rejects_wrong_profile() -> None:
    """The concrete adapter must not attach to a non-QuEra profile."""
    profile = HardwareAbstractionLayer.with_builtin_profiles().profile("local_statevector")

    with pytest.raises(ValueError, match="quera_bloqade"):
        QuEraBloqadeHALAdapter(profile, routine=_FakeBloqadeRoutine())


def test_quera_bloqade_routine_name_rejects_control_characters() -> None:
    """Explicit routine selectors reject control characters before construction."""
    profile = HardwareAbstractionLayer.with_builtin_profiles().profile("quera_bloqade")
    with pytest.raises(ValueError, match="QuEra routine name"):
        QuEraBloqadeHALAdapter(profile, routine=_FakeBloqadeRoutine(), routine_name="aquila\nbad")


def test_quera_bloqade_routine_name_trims_padding() -> None:
    """Surrounding whitespace canonicalises without selecting another routine."""
    profile = HardwareAbstractionLayer.with_builtin_profiles().profile("quera_bloqade")
    adapter = QuEraBloqadeHALAdapter(
        profile,
        routine=_FakeBloqadeRoutine(),
        routine_name="  aquila-local-emulator  ",
    )
    assert adapter._routine_name == "aquila-local-emulator"


def test_quera_bloqade_adapter_uses_lazy_routine_factory_and_caches_results() -> None:
    """Lazy construction should occur once and completed results should be cached."""
    created: list[str] = []
    routine = _FakeBloqadeRoutine()
    routine.batch = _FakeBloqadeBatch(report=_FakeRawBloqadeReport([[0, 1], [1, 0], [1, 0]]))

    def factory(workload: QuantumWorkload) -> _FakeBloqadeRoutine:
        created.append(workload.workload_id)
        return routine

    hal = HardwareAbstractionLayer.with_builtin_profiles()
    hal.register_backend(
        QuEraBloqadeHALAdapter(
            hal.profile("quera_bloqade"),
            routine_name="lazy-bloqade",
            routine_factory=factory,
        )
    )
    job = hal.submit(
        "quera_bloqade",
        bloqade_ahs_workload(_BLOQADE_PLAN, workload_id="lazy_quera", n_qubits=2, shots=3),
        approval_id="approved-quera",
    )

    first = hal.result(job)
    second = hal.result(job)

    assert created == ["lazy_quera"]
    assert first is second
    assert first.counts == {"01": 1, "10": 2}


def test_quera_bloqade_adapter_accepts_mapping_reports_and_missing_cancel() -> None:
    """Reports with count mappings and batches without cancel hooks remain supported."""
    hal = HardwareAbstractionLayer.with_builtin_profiles()
    hal.register_backend(
        QuEraBloqadeHALAdapter(
            hal.profile("quera_bloqade"),
            routine=_FakeNoCancelRoutine(),
            routine_name="count-report",
        )
    )
    job = hal.submit(
        "quera_bloqade",
        bloqade_ahs_workload(_BLOQADE_PLAN, workload_id="count_report", n_qubits=2, shots=3),
        approval_id="approved-quera",
    )

    assert hal.status(job) == "completed"
    assert hal.result(job).counts == {"01": 2, "10": 1}
    assert hal.cancel(job).status == "cancelled"


def test_quera_status_normalisation_maps_completion_aliases() -> None:
    """QuEra status normaliser should map provider completion aliases canonically."""
    assert quera_mod._normalise_status("FINISHED") == "completed"
    assert quera_mod._normalise_status("SUCCEEDED") == "completed"


def test_quera_provider_job_id_extraction_requires_identifier() -> None:
    """QuEra provider job id extraction should fail closed when id is unavailable."""
    assert (
        quera_mod._provider_job_id(type("Batch", (), {"id": "quera-provider-2"})())
        == "quera-provider-2"
    )
    with pytest.raises(ValueError, match="provider job id"):
        quera_mod._provider_job_id(object())


def test_quera_provider_job_id_rejects_control_characters() -> None:
    """QuEra provider identifiers must reject control-character payloads."""
    from scpn_quantum_control.hardware import hal_quera_bloqade as quera_mod

    class BadBatch:
        id = "quera-job-\n1"

    with pytest.raises(ValueError, match="provider job id"):
        quera_mod._provider_job_id(BadBatch())


def test_quera_provider_job_id_trims_padding() -> None:
    """QuEra provider identifiers should be canonicalised by trimming padding."""
    from scpn_quantum_control.hardware import hal_quera_bloqade as quera_mod

    class PaddedBatch:
        id = "  quera-job-42  "

    assert quera_mod._provider_job_id(PaddedBatch()) == "quera-job-42"


def test_quera_bloqade_adapter_rejects_unknown_jobs() -> None:
    """Unknown job handles should not fabricate provider state."""
    adapter = QuEraBloqadeHALAdapter(
        HardwareAbstractionLayer.with_builtin_profiles().profile("quera_bloqade"),
        routine=_FakeBloqadeRoutine(),
    )
    unknown = QuantumJobRef(
        job_id="quera_bloqade:missing",
        backend_id="quera_bloqade",
        workload_id="missing",
        status="submitted",
    )

    with pytest.raises(KeyError, match="unknown job_id"):
        adapter.status(unknown)
    with pytest.raises(KeyError, match="unknown job_id"):
        adapter.result(unknown)
    with pytest.raises(KeyError, match="unknown job_id"):
        adapter.cancel(unknown)


def test_quera_bloqade_default_dependency_errors_are_actionable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing Bloqade should produce route-specific dependency errors."""

    def fail_import(name: str) -> object:
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(quera_mod, "import_module", fail_import)

    with pytest.raises(RuntimeError, match="bloqade"):
        quera_mod._default_routine_factory(
            bloqade_ahs_workload(_BLOQADE_PLAN, workload_id="missing_bloqade", n_qubits=2, shots=1)
        )


def test_quera_bloqade_default_builder_requires_calibration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Automatic Bloqade construction should not pretend calibration is known."""
    monkeypatch.setattr(quera_mod, "import_module", lambda name: object())

    with pytest.raises(RuntimeError, match="calibrated provider builder"):
        quera_mod._default_routine_factory(
            bloqade_ahs_workload(_BLOQADE_PLAN, workload_id="needs_builder", n_qubits=2, shots=1)
        )


@pytest.mark.parametrize(
    ("source", "message"),
    [
        ([], "shots"),
        ({}, "shots"),
        ({"00": -1}, "non-negative"),
        ({"": 1}, "empty"),
        ({"02": 1}, "binary"),
        ({object(): 1}, "strings"),
    ],
)
def test_quera_bloqade_count_validation_rejects_malformed_results(
    source: Sequence[Any] | Mapping[Any, object], message: str
) -> None:
    """Malformed Bloqade result data should fail before HAL results are reported."""
    with pytest.raises(ValueError, match=message):
        quera_mod._normalise_counts(source)


def test_quera_bloqade_extracts_mapping_and_attribute_variants() -> None:
    """Bloqade reports can expose counts or bitstrings through common shapes."""
    assert quera_mod._normalise_counts(quera_mod._extract_bitstrings({"bitstrings": ["00"]})) == {
        "00": 1
    }
    assert quera_mod._normalise_counts(quera_mod._extract_bitstrings({"counts": {"11": 2}})) == {
        "11": 2
    }
    assert quera_mod._normalise_counts(quera_mod._extract_bitstrings(_FakeNoFetchBatch())) == {
        "01": 2,
        "10": 1,
    }
    assert quera_mod._normalise_counts(
        quera_mod._extract_bitstrings({"report": "ignored", "bitstrings": ["10"]})
    ) == {"10": 1}


def test_quera_bloqade_extracts_report_mapping_bitstrings() -> None:
    """Report mappings with bitstrings should normalise like attribute reports."""

    class _ReportMappingBatch:
        def report(self) -> dict[str, list[str]]:
            return {"bitstrings": ["00", "01", "01"]}

    assert quera_mod._normalise_counts(quera_mod._extract_bitstrings(_ReportMappingBatch())) == {
        "00": 1,
        "01": 2,
    }


def test_quera_bloqade_rejects_reports_without_counts() -> None:
    """A provider report with no observable shots should fail closed."""
    with pytest.raises(ValueError, match="bitstrings or counts"):
        quera_mod._extract_bitstrings(object())


def test_bloqade_workload_accepts_json_string() -> None:
    """The workload helper should accept canonical JSON payloads."""
    workload = bloqade_ahs_workload(
        json.dumps(_BLOQADE_PLAN),
        workload_id="json_quera",
        n_qubits=2,
        shots=2,
    )

    assert workload.ir_format == "bloqade"


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ({"schema": "bad"}, "schema"),
        ({"atoms": "bad"}, "atoms"),
        ({"atoms": [_ATOM0]}, "atom count"),
        ({"atoms": ["bad", _ATOM1]}, "atom entries"),
        (
            {"atoms": [{"index": 0, "position": [0.0]}, _ATOM1]},
            "position",
        ),
        ({"duration": 0.0}, "duration"),
        ({"rabi_amplitude_piecewise_linear": []}, "non-empty"),
        ({"rabi_phase_piecewise_linear": [[0.0]]}, "time and value"),
    ],
)
def test_bloqade_workload_rejects_malformed_ahs_plans(
    mutation: dict[str, object], message: str
) -> None:
    """AHS plans should be structurally checked before provider submission."""
    payload = dict(_BLOQADE_PLAN)
    payload.update(mutation)

    with pytest.raises(ValueError, match=message):
        bloqade_ahs_workload(payload, workload_id="bad_plan", n_qubits=2, shots=1)


def test_bloqade_json_payload_rejects_invalid_json() -> None:
    """Invalid JSON payloads should fail before becoming HAL workloads."""
    with pytest.raises(ValueError, match="valid JSON"):
        bloqade_ahs_workload("{", workload_id="bad_json", n_qubits=2, shots=1)


def test_bloqade_json_payload_requires_object() -> None:
    """JSON arrays should not be accepted as provider plans."""
    with pytest.raises(ValueError, match="JSON object"):
        bloqade_ahs_workload("[]", workload_id="bad_json_object", n_qubits=2, shots=1)


def test_bloqade_numeric_validation_errors_are_explicit() -> None:
    """Non-numeric atom and schedule fields should produce targeted errors."""
    bad_index = dict(_BLOQADE_PLAN)
    bad_index["atoms"] = [{"index": "bad", "position": [0.0, 0.0]}, _ATOM1]
    with pytest.raises(ValueError, match="integer"):
        bloqade_ahs_workload(bad_index, workload_id="bad_index", n_qubits=2, shots=1)

    bad_coordinate = dict(_BLOQADE_PLAN)
    bad_coordinate["atoms"] = [{"index": 0, "position": ["bad", 0.0]}, _ATOM1]
    with pytest.raises(ValueError, match="numeric"):
        bloqade_ahs_workload(bad_coordinate, workload_id="bad_coordinate", n_qubits=2, shots=1)

    bad_schedule = dict(_BLOQADE_PLAN)
    bad_schedule["rabi_amplitude_piecewise_linear"] = [["bad", 0.0]]
    with pytest.raises(ValueError, match="numeric"):
        bloqade_ahs_workload(bad_schedule, workload_id="bad_schedule", n_qubits=2, shots=1)


def test_quera_bloqade_adapter_requires_execution_route() -> None:
    """Construction should fail closed without an injected routine or routine factory."""
    profile = HardwareAbstractionLayer.with_builtin_profiles().profile("quera_bloqade")

    with pytest.raises(ValueError, match="routine"):
        QuEraBloqadeHALAdapter(profile)


def test_quera_bloqade_adapter_rejects_shot_mismatch() -> None:
    """QuEra adapter must fail closed when decoded counts diverge from expected shots."""
    hal = HardwareAbstractionLayer.with_builtin_profiles()
    hal.register_backend(
        QuEraBloqadeHALAdapter(
            hal.profile("quera_bloqade"),
            routine=_FakeShotMismatchRoutine(),
            routine_name="shot-mismatch",
        )
    )
    job = hal.submit(
        "quera_bloqade",
        bloqade_ahs_workload(
            _BLOQADE_PLAN, workload_id="quera_shot_mismatch", n_qubits=2, shots=3
        ),
        approval_id="approved-quera",
    )

    with pytest.raises(ValueError, match="shot count mismatch"):
        hal.result(job)


def test_quera_factory_receives_each_changed_native_plan() -> None:
    """A changed AHS plan reaches its builder rather than a cached earlier routine."""
    programs: list[str] = []

    def build(workload: QuantumWorkload) -> _FakeBloqadeRoutine:
        """Record actual build inputs without claiming analog dynamics."""
        programs.append(workload.program)
        return _FakeBloqadeRoutine()

    hal = HardwareAbstractionLayer.with_builtin_profiles()
    hal.register_backend(
        QuEraBloqadeHALAdapter(hal.profile("quera_bloqade"), routine_factory=build)
    )
    first = bloqade_ahs_workload(_BLOQADE_PLAN, workload_id="first_plan", n_qubits=2, shots=3)
    second = bloqade_ahs_workload(
        dict(_BLOQADE_PLAN) | {"duration": 3.0}, workload_id="second_plan", n_qubits=2, shots=3
    )
    a = hal.submit("quera_bloqade", first, approval_id="transport-contract-only")
    b = hal.submit("quera_bloqade", second, approval_id="transport-contract-only")
    assert programs == [first.program, second.program]
    assert hal.result(a).job.workload_id == first.workload_id
    assert hal.result(b).job.workload_id == second.workload_id


def test_quera_typed_samples_preserve_site_order_and_reject_foreign_cache_handles() -> None:
    """Native I/O samples retain correlations and cannot leak across workload IDs."""
    from scpn_quantum_control.hardware.provider_modalities import AnalogObservation

    hal = HardwareAbstractionLayer.with_builtin_profiles()
    hal.register_backend(
        QuEraBloqadeHALAdapter(
            hal.profile("quera_bloqade"),
            routine_factory=lambda workload: _FakeBloqadeRoutine(),
            routine_name="declared_local_route",
        )
    )
    workload = bloqade_ahs_workload(
        _BLOQADE_PLAN,
        workload_id="typed_analog",
        n_qubits=2,
        shots=3,
        capture_semantics=True,
        requested_target="declared_local_route",
    )
    job = hal.submit("quera_bloqade", workload, approval_id="transport-contract-only")
    result = hal.result(job)
    assert isinstance(result.provider_observation, AnalogObservation)
    assert result.provider_observation.raw_samples == ("00", "11", "11")
    assert result.provider_observation.request.native_axes == (0, 1)
    assert job.submission is not None and job.submission.original_program == workload.program
    with pytest.raises(ValueError, match="workload_id"):
        hal.result(replace(job, workload_id="foreign_request"))
    assert hal.result(job) is result


def test_quera_fractional_native_readout_is_never_truncated_to_a_binary_sample() -> None:
    """A fractional result fails through the public HAL result boundary."""

    class FractionalBatch:
        """Transport fault fixture with an invalid native sample."""

        id = "fractional_native_readout"

        def report(self) -> dict[str, list[list[float]]]:
            """Return an inexact occupation, not a quantum dynamics model."""
            return {"bitstrings": [[0.5, 1.0]]}

    class FractionalRoutine:
        """Expose the malformed provider data at the routine boundary."""

        def run(self, *, shots: int, name: str) -> FractionalBatch:
            """Return one explicitly invalid native readout."""
            return FractionalBatch()

    hal = HardwareAbstractionLayer.with_builtin_profiles()
    hal.register_backend(
        QuEraBloqadeHALAdapter(hal.profile("quera_bloqade"), routine=FractionalRoutine())
    )
    workload = bloqade_ahs_workload(_BLOQADE_PLAN, workload_id="fractional", n_qubits=2, shots=1)
    job = hal.submit("quera_bloqade", workload, approval_id="transport-refusal-only")
    with pytest.raises(ValueError, match="integer"):
        hal.result(job)


def test_quera_explicit_falsey_factory_is_not_replaced_by_default_builder() -> None:
    """A callable with false truth value remains the explicitly selected plan builder."""
    calls: list[str] = []
    routine = _FakeBloqadeRoutine()

    class FalseyFactory:
        def __bool__(self) -> bool:
            return False

        def __call__(self, workload: QuantumWorkload) -> _FakeBloqadeRoutine:
            calls.append(workload.program)
            return routine

    hal = HardwareAbstractionLayer.with_builtin_profiles()
    hal.register_backend(
        QuEraBloqadeHALAdapter(hal.profile("quera_bloqade"), routine_factory=FalseyFactory())
    )
    workload = bloqade_ahs_workload(
        _BLOQADE_PLAN, workload_id="explicit_falsey_factory", n_qubits=2, shots=3
    )
    job = hal.submit("quera_bloqade", workload, approval_id="io-contract-only")
    assert hal.result(job).counts == {"00": 1, "11": 2}
    assert calls == [workload.program]


@pytest.mark.parametrize("name", ["", "   "])
def test_quera_present_empty_selector_never_becomes_an_injected_default(name: str) -> None:
    """An explicitly empty routine selector refuses rather than naming another route."""
    routine = _FakeBloqadeRoutine()
    profile = HardwareAbstractionLayer.with_builtin_profiles().profile("quera_bloqade")
    with pytest.raises(ValueError, match="QuEra routine name"):
        QuEraBloqadeHALAdapter(profile, routine=routine, routine_name=name)
    assert routine.runs == []


@pytest.mark.parametrize(
    "channel",
    [
        "direct_counts",
        "direct_bits",
        "report_counts",
        "report_bits",
        "attribute_counts",
        "raw_bits",
        "tuple_key",
        "unsupported",
        "empty_mapping",
        "empty_report",
    ],
)
def test_quera_native_readout_channels_preserve_permuted_site_order_and_recover(
    channel: str,
) -> None:
    """Native I/O channels retain original axes and refuse malformed data without caching it."""
    raw_counts = {"01": 2, "10": 1}
    raw_bits = [[0, 1], [0, 1], [1, 0]]

    class MappingReadout(dict[str, object]):
        """Expose a provider identifier on a direct native readout mapping."""

        id = "native_readout_mapping"

    values: dict[str, object] = {
        "direct_counts": MappingReadout(counts=raw_counts),
        "direct_bits": MappingReadout(bitstrings=raw_bits),
        "report_counts": types.SimpleNamespace(
            id="native_readout", report=lambda: {"counts": raw_counts}
        ),
        "report_bits": types.SimpleNamespace(
            id="native_readout", report=lambda: {"bitstrings": raw_bits}
        ),
        "attribute_counts": types.SimpleNamespace(id="native_readout", counts=raw_counts),
        "raw_bits": types.SimpleNamespace(id="native_readout", raw_bitstrings=raw_bits),
        "tuple_key": types.SimpleNamespace(id="native_readout", counts={(0, 1): 3}),
        "unsupported": types.SimpleNamespace(id="native_readout"),
        "empty_mapping": MappingReadout(),
        "empty_report": types.SimpleNamespace(id="native_readout", report=lambda: {}),
    }

    class ReadoutRoutine:
        """Return explicit native I/O fixtures without modelling analog dynamics."""

        def __init__(self) -> None:
            """Select the original result channel for this contract case."""
            self.reply = values[channel]

        def run(self, *, shots: int, name: str) -> object:
            """Return the channel after validating exact requested sampling settings."""
            assert shots == 3 and name == "permuted_native_sites"
            return self.reply

    plan = dict(_BLOQADE_PLAN) | {"atoms": [_ATOM1, _ATOM0]}
    source = "\n" + json.dumps(plan, indent=2) + "\n"
    workload = bloqade_ahs_workload(
        source,
        workload_id="permuted_native_sites",
        n_qubits=2,
        shots=3,
        capture_semantics=True,
        requested_target="readout_route",
    )
    routine = ReadoutRoutine()
    hal = HardwareAbstractionLayer.with_builtin_profiles()
    adapter = QuEraBloqadeHALAdapter(
        hal.profile("quera_bloqade"),
        routine_factory=lambda request: routine,
        routine_name="readout_route",
    )
    hal.register_backend(adapter)
    job = hal.submit("quera_bloqade", workload, approval_id="native-io-only")
    assert job.submission is not None and job.submission.original_program == source
    assert job.submission.target_origin == "adapter_selector"
    if channel in {"tuple_key", "unsupported", "empty_mapping", "empty_report"}:
        with pytest.raises(ValueError, match="unchanged string keys|bitstrings or counts"):
            hal.result(job)
        batch = values[channel]
        if isinstance(batch, MappingReadout):
            batch["counts"] = raw_counts
        else:
            assert isinstance(batch, types.SimpleNamespace)
            batch.counts = raw_counts
            if channel == "empty_report":
                batch.report = lambda: {"counts": raw_counts}
    result = hal.result(job)
    observation = result.provider_observation
    assert isinstance(observation, AnalogObservation)
    assert observation.request.native_axes == (1, 0)
    assert result.counts == {"01": 2, "10": 1} and result.shots == 3
    if channel in {"direct_bits", "report_bits", "raw_bits"}:
        assert observation.raw_samples == ((0, 1), (0, 1), (1, 0))
        raw_bits[0][0] = 1
        assert observation.raw_samples == ((0, 1), (0, 1), (1, 0))
    else:
        assert observation.raw_counts == {"01": 2, "10": 1}
        raw_counts["01"] = 99
        assert observation.raw_counts == {"01": 2, "10": 1}
    assert observation.to_payload()["readout_convention"] == "provider_native_unknown"
    assert hal.result(job) is result


@pytest.mark.parametrize("value", [True, None, float("nan"), float("inf"), 10**400])
def test_bloqade_nonfinite_or_non_numeric_coordinates_refuse_before_build(value: object) -> None:
    """Analog coordinates cannot silently coerce booleans, missing or unbounded values."""
    plan = dict(_BLOQADE_PLAN) | {"atoms": [{"index": 0, "position": [value, 0.0]}, _ATOM1]}
    with pytest.raises(ValueError, match="numeric|finite"):
        bloqade_ahs_workload(plan, workload_id="invalid_coordinate", n_qubits=2, shots=3)


def test_bloqade_target_pin_requires_native_capture() -> None:
    """A legacy plan cannot imply exact route admission from a bare target annotation."""
    with pytest.raises(ValueError, match="target pin"):
        bloqade_ahs_workload(
            _BLOQADE_PLAN,
            workload_id="uncaptured_pin",
            n_qubits=2,
            shots=3,
            requested_target="unbound_route",
        )


def test_quera_stored_foreign_modality_refuses_without_losing_original_readout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Controlled retention damage refuses through HAL and restoration recovers the native result."""
    hal = HardwareAbstractionLayer.with_builtin_profiles()
    adapter = QuEraBloqadeHALAdapter(
        hal.profile("quera_bloqade"), routine_factory=lambda request: _FakeBloqadeRoutine()
    )
    hal.register_backend(adapter)
    workload = bloqade_ahs_workload(
        _BLOQADE_PLAN, workload_id="damaged_modality", n_qubits=2, shots=3, capture_semantics=True
    )
    job = hal.submit("quera_bloqade", workload, approval_id="retention-fault-only")
    assert job.submission is not None
    foreign = WorkloadSemantics(
        program_sha256=job.submission.request.program_sha256, n_qubits=2, n_clbits=0
    )
    damaged = replace(job, submission=replace(job.submission, request=foreign))
    with monkeypatch.context() as fault:
        fault.setitem(adapter._jobs, job.job_id, damaged)
        with pytest.raises(ValueError, match="analog plan semantics"):
            hal.result(damaged)
    original = hal.result(job)
    assert isinstance(original.provider_observation, AnalogObservation)
    assert original.provider_observation.raw_samples == ("00", "11", "11")
    assert hal.result(job) is original


def test_quera_callable_provider_identifier_retains_exact_job_identity() -> None:
    """A native-compatible callable job identifier remains bound to the stored source."""

    class CallableBatch:
        """Supply a callable identity and one valid I/O histogram."""

        counts = {"01": 3}

        def job_id(self) -> str:
            """Expose one canonical provider identity."""
            return "callable-native-id"

    class CallableRoutine:
        """Submit a declared I/O readout without physical qualification."""

        def run(self, *, shots: int, name: str) -> CallableBatch:
            """Accept the original sampling settings exactly once."""
            assert shots == 3 and name == "callable_identity"
            return CallableBatch()

    hal = HardwareAbstractionLayer.with_builtin_profiles()
    hal.register_backend(
        QuEraBloqadeHALAdapter(hal.profile("quera_bloqade"), routine=CallableRoutine())
    )
    job = hal.submit(
        "quera_bloqade",
        bloqade_ahs_workload(_BLOQADE_PLAN, workload_id="callable_identity", n_qubits=2, shots=3),
        approval_id="native-io-only",
    )
    assert job.metadata["provider_job_id"] == "callable-native-id"
    assert hal.result(job).counts == {"01": 3}


@pytest.mark.parametrize("digest_form", ["missing", "other", "exact"])
def test_quera_prepared_routine_requires_original_plan_digest_and_keeps_prior_result(
    digest_form: str,
) -> None:
    """Opaque prepared routines cannot attest a different captured plan or discard prior output."""
    workload = bloqade_ahs_workload(
        _BLOQADE_PLAN,
        workload_id="captured_prepared_plan",
        n_qubits=2,
        shots=3,
        capture_semantics=True,
    )
    assert workload.semantics is not None
    digest = {"missing": None, "other": "0" * 64, "exact": workload.semantics.program_sha256}[
        digest_form
    ]
    routine = _FakeBloqadeRoutine()
    hal = HardwareAbstractionLayer.with_builtin_profiles()
    adapter = QuEraBloqadeHALAdapter(
        hal.profile("quera_bloqade"), routine=routine, prepared_program_sha256=digest
    )
    hal.register_backend(adapter)
    legacy = hal.submit(
        "quera_bloqade",
        bloqade_ahs_workload(
            _BLOQADE_PLAN, workload_id="original_legacy_anchor", n_qubits=2, shots=3
        ),
        approval_id="native-io-only",
    )
    original = hal.result(legacy)
    if digest_form == "exact":
        job = hal.submit("quera_bloqade", workload, approval_id="native-io-only")
        assert job.submission is not None and job.submission.compilation == "caller_precompiled"
        assert isinstance(hal.result(job).provider_observation, AnalogObservation)
        assert len(routine.runs) == 2
    else:
        with pytest.raises(ValueError, match="exact original program digest"):
            hal.submit("quera_bloqade", workload, approval_id="native-io-only")
        assert len(routine.runs) == 1
    assert hal.result(legacy) is original and original.counts == {"00": 1, "11": 2}
