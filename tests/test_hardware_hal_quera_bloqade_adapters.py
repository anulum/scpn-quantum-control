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

from typing import Any

import pytest

from scpn_quantum_control.hardware import hal_quera_bloqade as quera_mod
from scpn_quantum_control.hardware.hal import HardwareAbstractionLayer, QuantumWorkload
from scpn_quantum_control.hardware.hal_quera_bloqade import (
    QuEraBloqadeHALAdapter,
    bloqade_ahs_workload,
)

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

    def report(self) -> _FakeBloqadeReport:
        if self._report is not None:
            return self._report  # type: ignore[return-value]
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
    profile = HardwareAbstractionLayer.with_builtin_profiles().profile("quera_bloqade")
    with pytest.raises(ValueError, match="QuEra routine name"):
        QuEraBloqadeHALAdapter(profile, routine=_FakeBloqadeRoutine(), routine_name="aquila\nbad")


def test_quera_bloqade_routine_name_trims_padding() -> None:
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
    unknown = quera_mod.QuantumJobRef(
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
    source: object, message: str
) -> None:
    """Malformed Bloqade result data should fail before HAL results are reported."""

    with pytest.raises(ValueError, match=message):
        quera_mod._normalise_counts(source)  # type: ignore[arg-type]


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
        quera_mod.json.dumps(_BLOQADE_PLAN),
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
