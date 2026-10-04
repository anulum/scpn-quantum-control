# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — D-Wave Leap HAL adapter tests
"""Tests for direct D-Wave Leap annealing behind the provider-neutral HAL."""

from __future__ import annotations

import json
import types
from dataclasses import replace
from typing import Any

import numpy as np
import pytest

from scpn_quantum_control.hardware.hal import HardwareAbstractionLayer, QuantumWorkload
from scpn_quantum_control.hardware.hal_dwave import (
    DWAVE_BQM_SCHEMA,
    DWaveLeapHALAdapter,
    dwave_bqm_workload,
)
from scpn_quantum_control.hardware.provider_modalities import (
    AnnealingObservation,
    ModalitySemantics,
)


class _FakeSampleRow:
    def __init__(self, sample: dict[str, int], num_occurrences: int) -> None:
        self.sample = sample
        self.num_occurrences = num_occurrences


class _FakeSampleSet:
    info = {"problem_id": "dwave-problem-1"}

    def data(self, fields: list[str]) -> list[_FakeSampleRow]:
        assert fields == ["sample", "num_occurrences"]
        return [
            _FakeSampleRow({"0": 0, "1": 1}, 3),
            _FakeSampleRow({"0": 1, "1": 0}, 5),
        ]


class _FakeSampler:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def sample(self, bqm: dict[str, Any], *, num_reads: int, label: str) -> _FakeSampleSet:
        self.calls.append({"bqm": bqm, "num_reads": num_reads, "label": label})
        return _FakeSampleSet()


class _FakeShotMismatchSampleSet:
    info = {"problem_id": "dwave-problem-shot-mismatch"}

    def data(self, fields: list[str]) -> list[_FakeSampleRow]:
        assert fields == ["sample", "num_occurrences"]
        return [_FakeSampleRow({"0": 0, "1": 1}, 3)]


class _FakeShotMismatchSampler:
    def sample(
        self, bqm: dict[str, Any], *, num_reads: int, label: str
    ) -> _FakeShotMismatchSampleSet:
        del bqm, num_reads, label
        return _FakeShotMismatchSampleSet()


def _bqm_factory(payload: dict[str, object]) -> dict[str, object]:
    return {"factory_payload": payload}


def test_dwave_leap_adapter_samples_injected_sampler_with_approval() -> None:
    """Both approval entry points refuse before sampling; approved data retains native settings."""
    hal = HardwareAbstractionLayer.with_builtin_profiles()
    sampler = _FakeSampler()
    adapter = DWaveLeapHALAdapter(
        hal.profile("dwave_leap"),
        sampler=sampler,
        bqm_factory=_bqm_factory,
        solver="Advantage_system_test",
    )
    hal.register_backend(adapter)
    workload = dwave_bqm_workload(
        linear={"0": -1.0, "1": 0.5},
        quadratic={("0", "1"): -0.25},
        workload_id="dwave_pair",
        n_variables=2,
        reads=8,
        offset=0.125,
        vartype="BINARY",
        metadata={"campaign": "hal"},
    )

    with pytest.raises(PermissionError, match="approval"):
        hal.submit("dwave_leap", workload)
    with pytest.raises(PermissionError, match="approval"):
        adapter.submit(workload)
    assert sampler.calls == []

    job = hal.submit("dwave_leap", workload, approval_id="approved-dwave")
    result = hal.result(job)
    cancelled = hal.cancel(job)

    assert job.status == "completed"
    assert job.job_id.startswith("dwave_leap:dwave_pair:")
    assert job.metadata["provider_job_id"] == "dwave-problem-1"
    assert job.metadata["execution_mode"] == "dwave_leap_bqm"
    assert job.metadata["solver"] == "Advantage_system_test"
    assert result.counts == {"01": 3, "10": 5}
    assert result.shots == 8
    assert result.metadata["vartype"] == "BINARY"
    assert cancelled.status == "cancelled"
    assert sampler.calls == [
        {
            "bqm": {
                "factory_payload": {
                    "schema": DWAVE_BQM_SCHEMA,
                    "vartype": "BINARY",
                    "variables": ["0", "1"],
                    "linear": {"0": -1.0, "1": 0.5},
                    "quadratic": [{"u": "0", "v": "1", "bias": -0.25}],
                    "offset": 0.125,
                }
            },
            "num_reads": 8,
            "label": "dwave_pair",
        }
    ]


def test_dwave_leap_adapter_rejects_wrong_profile_ir_and_schema() -> None:
    """Foreign profiles, unsupported IR and wrong model schema cannot execute."""
    hal = HardwareAbstractionLayer.with_builtin_profiles()
    with pytest.raises(ValueError, match="dwave_leap"):
        DWaveLeapHALAdapter(hal.profile("pasqal_cloud"), sampler=_FakeSampler())

    adapter = DWaveLeapHALAdapter(
        hal.profile("dwave_leap"), sampler=_FakeSampler(), bqm_factory=_bqm_factory
    )
    with pytest.raises(ValueError, match="bqm workloads"):
        adapter.submit(
            QuantumWorkload(
                workload_id="bad_ir",
                ir_format="openqasm3",
                program="OPENQASM 3.0;",
                n_qubits=2,
                shots=4,
            ),
            approval_id="approved",
        )

    with pytest.raises(ValueError, match=DWAVE_BQM_SCHEMA):
        dwave_bqm_workload(
            linear={"0": 1.0},
            quadratic={},
            workload_id="bad_schema",
            n_variables=1,
            reads=4,
            schema="other",
        )


def test_dwave_leap_adapter_validates_bqm_payload_and_counts() -> None:
    """Wrong variable/edge declarations and negative multiplicities refuse instead of changing the model."""
    with pytest.raises(ValueError, match="variables"):
        dwave_bqm_workload(
            linear={"0": 1.0},
            quadratic={},
            workload_id="bad_variables",
            n_variables=2,
            reads=4,
        )

    with pytest.raises(ValueError, match="quadratic"):
        dwave_bqm_workload(
            linear={"0": 1.0},
            quadratic={("0", "2"): 1.0},
            workload_id="bad_edge",
            n_variables=1,
            reads=4,
        )

    class BadSampleSet:
        info = {"problem_id": "dwave-bad-counts"}

        def data(self, fields: list[str]) -> list[_FakeSampleRow]:
            del fields
            return [_FakeSampleRow({"0": 0}, -1)]

    class BadSampler:
        def sample(self, bqm: dict[str, Any], *, num_reads: int, label: str) -> BadSampleSet:
            del bqm, num_reads, label
            return BadSampleSet()

    hal = HardwareAbstractionLayer.with_builtin_profiles()
    adapter = DWaveLeapHALAdapter(
        hal.profile("dwave_leap"), sampler=BadSampler(), bqm_factory=_bqm_factory
    )
    with pytest.raises(ValueError, match="non-negative"):
        adapter.submit(
            dwave_bqm_workload(
                linear={"0": 1.0},
                quadratic={},
                workload_id="bad_counts",
                n_variables=1,
                reads=1,
            ),
            approval_id="approved",
        )


@pytest.mark.parametrize("vartype", ["BINARY", "SPIN"])
def test_dwave_leap_adapter_default_builder_is_sdk_gated(
    monkeypatch: pytest.MonkeyPatch,
    vartype: str,
) -> None:
    """Missing SDK paths refuse both native domains; controlled import conformance is not SDK qualification."""
    hal = HardwareAbstractionLayer.with_builtin_profiles()
    adapter = DWaveLeapHALAdapter(hal.profile("dwave_leap"))
    workload = dwave_bqm_workload(
        linear={"0": 1.0},
        quadratic={},
        workload_id="needs_sdk",
        n_variables=1,
        reads=1,
        vartype=vartype,
    )

    with pytest.raises(RuntimeError, match="dimod|dwave-system|DWaveSampler"):
        adapter.submit(workload, approval_id="approved")

    def fake_import(name: str) -> Any:
        if name == "dimod":

            class _BQM:
                @classmethod
                def from_qubo(
                    cls, qubo: dict[tuple[str, str], float], offset: float = 0.0
                ) -> dict[str, Any]:
                    return {"qubo": qubo, "offset": offset}

                @classmethod
                def from_ising(
                    cls,
                    linear: dict[str, float],
                    quadratic: dict[tuple[str, str], float],
                    offset: float = 0.0,
                ) -> dict[str, Any]:
                    """Expose only constructor I/O while the sampler import remains refused."""
                    return {"linear": linear, "quadratic": quadratic, "offset": offset}

            return type("Dimod", (), {"BinaryQuadraticModel": _BQM})
        raise ModuleNotFoundError(name)

    monkeypatch.setattr("scpn_quantum_control.hardware.hal_dwave.import_module", fake_import)
    with pytest.raises(RuntimeError, match="DWaveSampler"):
        adapter.submit(workload, approval_id="approved")


def test_dwave_provider_job_id_extraction_requires_identifier() -> None:
    """D-Wave provider job id extraction should fail closed when id is unavailable."""
    from scpn_quantum_control.hardware import hal_dwave as dwave_mod

    assert (
        dwave_mod._provider_job_id(type("SampleSet", (), {"info": {"problem_id": "dw-2"}})())
        == "dw-2"
    )
    with pytest.raises(ValueError, match="provider job id"):
        dwave_mod._provider_job_id(type("SampleSet", (), {"info": {}})())


def test_dwave_provider_job_id_rejects_control_characters() -> None:
    """D-Wave provider identifiers must reject control-character payloads."""
    from scpn_quantum_control.hardware import hal_dwave as dwave_mod

    SampleSet = type("SampleSet", (), {"info": {"problem_id": "dw-\n2"}})
    with pytest.raises(ValueError, match="provider job id"):
        dwave_mod._provider_job_id(SampleSet())


def test_dwave_provider_job_id_trims_padding() -> None:
    """D-Wave provider identifiers should be canonicalised by trimming padding."""
    from scpn_quantum_control.hardware import hal_dwave as dwave_mod

    SampleSet = type("SampleSet", (), {"info": {"problem_id": "  dw-2  "}})
    assert dwave_mod._provider_job_id(SampleSet()) == "dw-2"


def test_dwave_leap_adapter_rejects_shot_mismatch() -> None:
    """A native occurrence total must equal the original requested reads."""
    hal = HardwareAbstractionLayer.with_builtin_profiles()
    adapter = DWaveLeapHALAdapter(
        hal.profile("dwave_leap"),
        sampler=_FakeShotMismatchSampler(),
        bqm_factory=_bqm_factory,
    )
    with pytest.raises(ValueError, match="shot count mismatch"):
        adapter.submit(
            dwave_bqm_workload(
                linear={"0": -1.0, "1": 0.5},
                quadratic={("0", "1"): -0.25},
                workload_id="dwave_shot_mismatch",
                n_variables=2,
                reads=8,
            ),
            approval_id="approved-dwave",
        )


@pytest.mark.parametrize("solver", ["Advantage\nbad", "", "   "])
def test_dwave_solver_rejects_control_characters(solver: str) -> None:
    """Present malformed solver identifiers cannot select a default route."""
    profile = HardwareAbstractionLayer.with_builtin_profiles().profile("dwave_leap")
    with pytest.raises(ValueError, match="D-Wave solver"):
        DWaveLeapHALAdapter(profile, sampler=_FakeSampler(), solver=solver)


def test_dwave_solver_trims_padding() -> None:
    """Surrounding permitted solver padding canonicalizes to the declared identifier."""
    profile = HardwareAbstractionLayer.with_builtin_profiles().profile("dwave_leap")
    adapter = DWaveLeapHALAdapter(
        profile,
        sampler=_FakeSampler(),
        solver="  Advantage_system6.4  ",
    )
    assert adapter._solver == "Advantage_system6.4"


def test_dwave_native_spin_record_retains_energy_order_and_raw_bytes() -> None:
    """Structured native I/O data survives HAL without a dimod/device qualification."""
    from scpn_quantum_control.hardware.provider_modalities import AnnealingObservation

    class RecordSampleSet:
        """Match the documented SampleSet record format using a real NumPy record."""

        info = {"problem_id": "native_spin_format_contract"}
        variables = ("b", "a")
        vartype = "SPIN"
        record = np.array(
            [([1, -1], -0.25, 2), ([-1, 1], 0.5, 1)],
            dtype=[("sample", "i1", (2,)), ("energy", "f8"), ("num_occurrences", "i8")],
        )

    class RecordSampler:
        """Return one fixed native-format record, without simulating an annealer."""

        def sample(self, bqm: object, *, num_reads: int, label: str) -> RecordSampleSet:
            """Expose original record order and all requested occurrences."""
            assert num_reads == 3
            return RecordSampleSet()

    hal = HardwareAbstractionLayer.with_builtin_profiles()
    hal.register_backend(
        DWaveLeapHALAdapter(
            hal.profile("dwave_leap"),
            sampler=RecordSampler(),
            bqm_factory=_bqm_factory,
            solver="declared_solver",
        )
    )
    workload = dwave_bqm_workload(
        linear={"a": -0.5, "b": 0.25},
        quadratic={},
        workload_id="native_spin",
        n_variables=2,
        reads=3,
        vartype="SPIN",
        capture_semantics=True,
        requested_target="declared_solver",
    )
    job = hal.submit("dwave_leap", workload, approval_id="native-format-contract-only")
    result = hal.result(job)
    assert isinstance(result.provider_observation, AnnealingObservation)
    observation = result.provider_observation
    assert observation.returned_axes == ("b", "a")
    assert [sample.values for sample in observation.samples] == [(1, -1), (-1, 1)]
    assert [sample.energy for sample in observation.samples] == [-0.25, 0.5]
    assert result.counts == {"01": 2, "10": 1}
    assert observation.raw_record is not None
    assert observation.raw_record.data == RecordSampleSet.record.tobytes(order="C")
    assert job.submission is not None and job.submission.original_program == workload.program
    with pytest.raises(ValueError, match="workload_id"):
        hal.result(replace(job, workload_id="foreign_model"))
    assert hal.result(job) is result


def test_dwave_explicit_spin_domain_rejects_zero_native_values() -> None:
    """A provider zero cannot be silently accepted as a SPIN -1 sample."""

    class ZeroSpinSamples:
        """Malformed native-domain transport fixture."""

        info = {"problem_id": "invalid_spin_zero"}

        def data(self, fields: list[str]) -> list[_FakeSampleRow]:
            """Return a zero in a source-declared SPIN domain."""
            return [_FakeSampleRow({"a": 0}, 1)]

    class ZeroSpinSampler:
        """Expose a malformed sample through the existing native API boundary."""

        def sample(self, bqm: object, *, num_reads: int, label: str) -> ZeroSpinSamples:
            """Return the invalid spin data without a numerical engine."""
            return ZeroSpinSamples()

    hal = HardwareAbstractionLayer.with_builtin_profiles()
    hal.register_backend(
        DWaveLeapHALAdapter(
            hal.profile("dwave_leap"), sampler=ZeroSpinSampler(), bqm_factory=_bqm_factory
        )
    )
    workload = dwave_bqm_workload(
        linear={"a": 0.0},
        quadratic={},
        workload_id="bad_spin",
        n_variables=1,
        reads=1,
        vartype="SPIN",
    )
    with pytest.raises(ValueError, match="domain"):
        hal.submit("dwave_leap", workload, approval_id="transport-refusal-only")


@pytest.mark.parametrize("factory_kind", ["bqm", "sampler"])
def test_dwave_explicit_falsey_factories_remain_the_native_execution_owners(
    factory_kind: str,
) -> None:
    """An explicit callable's truth value cannot replace its native BQM or sampler route."""
    sampler = _FakeSampler()
    builds: list[dict[str, object]] = []
    selections: list[str] = []

    class BQMFactory:
        """Retain a falsey but explicitly supplied current-plan constructor."""

        def __bool__(self) -> bool:
            """Expose a legal false truth value independent of constructor admission."""
            return False

        def __call__(self, payload: dict[str, object]) -> dict[str, object]:
            """Record the original current payload before returning its native I/O model."""
            builds.append(payload)
            return _bqm_factory(payload)

    class SamplerFactory:
        """Retain a falsey but explicitly supplied native-compatible sampler builder."""

        def __bool__(self) -> bool:
            """Expose false truth without indicating factory absence."""
            return False

        def __call__(self) -> _FakeSampler:
            """Return the explicitly selected I/O sampler without a default SDK route."""
            selections.append("explicit")
            return sampler

    hal = HardwareAbstractionLayer.with_builtin_profiles()
    adapter = DWaveLeapHALAdapter(
        hal.profile("dwave_leap"),
        sampler=None if factory_kind == "sampler" else sampler,
        sampler_factory=SamplerFactory() if factory_kind == "sampler" else None,
        bqm_factory=BQMFactory() if factory_kind == "bqm" else _bqm_factory,
    )
    hal.register_backend(adapter)
    workload = dwave_bqm_workload(
        linear={"0": -1.0, "1": 0.5},
        quadratic={},
        workload_id="explicit_factory_owner",
        n_variables=2,
        reads=8,
        capture_semantics=True,
    )
    job = hal.submit("dwave_leap", workload, approval_id="native-io-only")
    assert len(sampler.calls) == 1
    assert len(builds) == (1 if factory_kind == "bqm" else 0)
    assert selections == (["explicit"] if factory_kind == "sampler" else [])
    assert job.submission is not None and job.submission.original_program == workload.program
    assert hal.result(job).counts == {"01": 3, "10": 5}


@pytest.mark.parametrize(
    "mutation,message",
    [
        ({"variables": "bad"}, "sequence"),
        ({"variables": ["0", "0"]}, "exactly once"),
        ({"vartype": "unknown"}, "vartype"),
        ({"linear": []}, "mapping"),
        ({"linear": {"0": 1.0, "foreign": 0.5}}, "cover every variable"),
        ({"quadratic": "bad"}, "sequence"),
        ({"quadratic": [4]}, "mappings"),
        ({"quadratic": [{"u": "0", "v": "0", "bias": 0.25}]}, "invalid variables"),
        ({"offset": float("nan")}, "finite"),
        ({"linear": {"0": True, "1": 0.5}}, "numeric"),
    ],
)
def test_dwave_invalid_original_plan_refuses_before_model_or_sampler_and_retains_prior_result(
    mutation: dict[str, object],
    message: str,
) -> None:
    """Schema, variable, native-domain and finite-bias failures cannot replace an earlier sample set."""
    sampler = _FakeSampler()
    builds: list[dict[str, object]] = []

    def build(payload: dict[str, object]) -> dict[str, object]:
        """Track the exact admitted current model at the public adapter boundary."""
        builds.append(payload)
        return _bqm_factory(payload)

    hal = HardwareAbstractionLayer.with_builtin_profiles()
    hal.register_backend(
        DWaveLeapHALAdapter(hal.profile("dwave_leap"), sampler=sampler, bqm_factory=build)
    )
    workload = dwave_bqm_workload(
        linear={"0": -1.0, "1": 0.5},
        quadratic={},
        workload_id="native_plan_anchor",
        n_variables=2,
        reads=8,
    )
    original = hal.submit("dwave_leap", workload, approval_id="native-io-only")
    retained = hal.result(original)
    payload = json.loads(workload.program) | mutation
    malformed = replace(workload, workload_id="invalid_plan", program=json.dumps(payload))
    with pytest.raises(ValueError, match=message):
        hal.submit("dwave_leap", malformed, approval_id="native-io-only")
    assert len(builds) == len(sampler.calls) == 1
    assert hal.result(original) is retained


@pytest.mark.parametrize("source", ["{", "[]"])
def test_dwave_original_json_refuses_before_native_factories(source: str) -> None:
    """Malformed or non-object JSON cannot construct a model or sampler."""
    builds: list[dict[str, object]] = []

    def build(payload: dict[str, object]) -> dict[str, object]:
        """Record an unexpected factory invocation on an invalid source."""
        builds.append(payload)
        return payload

    adapter = DWaveLeapHALAdapter(
        HardwareAbstractionLayer.with_builtin_profiles().profile("dwave_leap"),
        bqm_factory=build,
        sampler=_FakeSampler(),
    )
    with pytest.raises(ValueError, match="JSON"):
        adapter.submit(
            QuantumWorkload(
                workload_id="invalid_json", ir_format="bqm", program=source, n_qubits=2, shots=8
            ),
            approval_id="refusal-only",
        )
    assert builds == []


@pytest.mark.parametrize("value", [True, None, "bad", float("nan"), float("inf"), 10**1000])
def test_dwave_original_bias_builder_refuses_nonfinite_or_non_numeric_values(value: Any) -> None:
    """A native linear bias must be finite without boolean or overflow coercion."""
    with pytest.raises(ValueError, match="numeric|finite"):
        dwave_bqm_workload(
            linear={"0": value}, quadratic={}, workload_id="invalid_bias", n_variables=1, reads=1
        )


@pytest.mark.parametrize("fault", ["target", "digest", "axes", "vartype", "modality"])
def test_dwave_native_request_disagreement_refuses_before_bqm_or_sampling(fault: str) -> None:
    """The original model, requested domain and exact target bind before factory admission."""
    sampler = _FakeSampler()
    builds: list[dict[str, object]] = []

    def build(payload: dict[str, object]) -> dict[str, object]:
        """Track unexpected model construction for a disagreeing companion."""
        builds.append(payload)
        return payload

    hal = HardwareAbstractionLayer.with_builtin_profiles()
    hal.register_backend(
        DWaveLeapHALAdapter(
            hal.profile("dwave_leap"), sampler=sampler, bqm_factory=build, solver="actual_selector"
        )
    )
    workload = dwave_bqm_workload(
        linear={"0": -1.0, "1": 0.5},
        quadratic={},
        workload_id="source_disagreement",
        n_variables=2,
        reads=8,
        capture_semantics=True,
        requested_target="other_selector" if fault == "target" else "actual_selector",
    )
    semantics = workload.semantics
    assert isinstance(semantics, ModalitySemantics)
    if fault == "digest":
        with pytest.raises(ValueError, match="digest"):
            replace(workload, semantics=replace(semantics, program_sha256="0" * 64))
        assert builds == [] and sampler.calls == []
        return
    elif fault == "axes":
        workload = replace(workload, semantics=replace(semantics, native_axes=("1", "0")))
    elif fault == "vartype":
        workload = replace(workload, semantics=replace(semantics, vartype="SPIN"))
    elif fault == "modality":
        workload = replace(
            workload,
            semantics=ModalitySemantics(
                program_sha256=semantics.program_sha256,
                modality="analog",
                native_axes=("0", "1"),
            ),
        )
    with pytest.raises(ValueError, match="target|digest|order|domain|modality"):
        hal.submit("dwave_leap", workload, approval_id="refusal-only")
    assert builds == [] and sampler.calls == []


@pytest.mark.parametrize("embedded", [False, True])
def test_dwave_default_sampler_import_contract_uses_returned_native_owner(
    monkeypatch: pytest.MonkeyPatch,
    embedded: bool,
) -> None:
    """Controlled optional-loader I/O retains the declared sampler, without installed SDK qualification."""
    sampler = _FakeSampler()
    imports: list[str] = []
    selections: list[object] = []

    def embed(native: object) -> _FakeSampler:
        """Retain the same supplied sampler through the native embedding constructor contract."""
        selections.append(native)
        return sampler

    def load(name: str) -> object:
        """Supply a controlled loader boundary, not an actual dimod or dwave-system runtime."""
        imports.append(name)
        assert name == "dwave.system"
        return types.SimpleNamespace(
            DWaveSampler=lambda: sampler, EmbeddingComposite=embed if embedded else None
        )

    monkeypatch.setattr("scpn_quantum_control.hardware.hal_dwave.import_module", load)
    hal = HardwareAbstractionLayer.with_builtin_profiles()
    hal.register_backend(DWaveLeapHALAdapter(hal.profile("dwave_leap"), bqm_factory=_bqm_factory))
    workload = dwave_bqm_workload(
        linear={"0": -1.0, "1": 0.5},
        quadratic={},
        workload_id="optional_loader_contract",
        n_variables=2,
        reads=8,
        capture_semantics=True,
    )
    job = hal.submit("dwave_leap", workload, approval_id="loader-contract-only")
    assert imports == ["dwave.system"]
    assert selections == ([sampler] if embedded else [])
    assert hal.result(job).counts == {"01": 3, "10": 5}


@pytest.mark.parametrize("fault", ["missing_result", "foreign_identity", "unknown_job"])
def test_dwave_public_retention_failures_keep_the_original_completed_sample_set(
    monkeypatch: pytest.MonkeyPatch,
    fault: str,
) -> None:
    """Unknown or damaged retained identity refuses without resampling or erasing the original result."""
    sampler = _FakeSampler()
    hal = HardwareAbstractionLayer.with_builtin_profiles()
    adapter = DWaveLeapHALAdapter(
        hal.profile("dwave_leap"), sampler=sampler, bqm_factory=_bqm_factory
    )
    hal.register_backend(adapter)
    job = hal.submit(
        "dwave_leap",
        dwave_bqm_workload(
            linear={"0": -1.0, "1": 0.5},
            quadratic={},
            workload_id="retained_sample_set",
            n_variables=2,
            reads=8,
            capture_semantics=True,
        ),
        approval_id="native-io-only",
    )
    original = hal.result(job)
    with monkeypatch.context() as injected:
        if fault == "missing_result":
            injected.delitem(adapter._results, job.job_id)
            with pytest.raises(KeyError, match="unknown job"):
                hal.result(job)
        elif fault == "foreign_identity":
            with pytest.raises(ValueError, match="workload_id"):
                hal.status(replace(job, workload_id="another_source"))
        else:
            with pytest.raises(KeyError, match="unknown job"):
                hal.cancel(replace(job, job_id="not_retained"))
    assert hal.result(job) is original and len(sampler.calls) == 1
    cancelled = hal.cancel(job)
    assert hal.status(cancelled) == "cancelled" and hal.result(cancelled) is original
    assert cancelled.submission == job.submission


@pytest.mark.parametrize(
    "fault",
    [
        "not_array",
        "rank",
        "unstructured",
        "object",
        "missing_fields",
        "text_samples",
        "text_occurrences",
        "text_energy",
    ],
)
def test_dwave_invalid_native_record_layout_refuses_without_replacing_prior_result(
    fault: str,
) -> None:
    """Native records require numeric fields and a one-dimensional original storage layout."""
    record = np.array(
        [([0, 1], -0.25, 8)],
        dtype=[("sample", "i1", (2,)), ("energy", "f8"), ("num_occurrences", "i8")],
    )
    malformed: dict[str, object] = {
        "not_array": [],
        "rank": record.reshape(1, 1),
        "unstructured": np.array([1]),
        "object": np.array(
            [([0, 1], -0.25, 8)],
            dtype=[("sample", "O"), ("energy", "f8"), ("num_occurrences", "i8")],
        ),
        "missing_fields": np.array(
            [([0, 1], 8)], dtype=[("sample", "i1", (2,)), ("num_occurrences", "i8")]
        ),
        "text_samples": np.array(
            [(["0", "1"], -0.25, 8)],
            dtype=[("sample", "U1", (2,)), ("energy", "f8"), ("num_occurrences", "i8")],
        ),
        "text_occurrences": np.array(
            [([0, 1], -0.25, "8")],
            dtype=[("sample", "i1", (2,)), ("energy", "f8"), ("num_occurrences", "U1")],
        ),
        "text_energy": np.array(
            [([0, 1], "-0.25", 8)],
            dtype=[("sample", "i1", (2,)), ("energy", "U5"), ("num_occurrences", "i8")],
        ),
    }

    class RecordSampler:
        """Expose actual NumPy bytes through native I/O, without simulating annealing."""

        def __init__(self) -> None:
            """Retain a valid native record until the fault boundary changes its layout."""
            self.record: object = record
            self.calls = 0

        def sample(self, bqm: object, *, num_reads: int, label: str) -> object:
            """Return the current original record with exact declared variable order."""
            self.calls += 1
            assert num_reads == 8
            return types.SimpleNamespace(
                info={"problem_id": "record_layout"},
                variables=("0", "1"),
                vartype=types.SimpleNamespace(name="BINARY"),
                record=self.record,
            )

    sampler = RecordSampler()
    hal = HardwareAbstractionLayer.with_builtin_profiles()
    hal.register_backend(
        DWaveLeapHALAdapter(hal.profile("dwave_leap"), sampler=sampler, bqm_factory=_bqm_factory)
    )
    workload = dwave_bqm_workload(
        linear={"0": -1.0, "1": 0.5},
        quadratic={},
        workload_id="original_record_anchor",
        n_variables=2,
        reads=8,
        capture_semantics=True,
    )
    prior = hal.submit("dwave_leap", workload, approval_id="native-io-only")
    original = hal.result(prior)
    sampler.record = malformed[fault]
    with pytest.raises(ValueError, match="numeric structured"):
        hal.submit(
            "dwave_leap",
            replace(workload, workload_id="malformed_record"),
            approval_id="native-io-only",
        )
    assert hal.result(prior) is original and sampler.calls == 2
    assert isinstance(original.provider_observation, AnnealingObservation)
    assert original.provider_observation.raw_record is not None
    assert original.provider_observation.raw_record.data == record.tobytes(order="C")


@pytest.mark.parametrize("capture", [False, True])
@pytest.mark.parametrize(
    "channel",
    [
        "mapping",
        "data",
        "spin",
        "empty",
        "data_bad_mapping",
        "missing_variable",
        "mapping_bad_mapping",
        "mapping_wrong_lengths",
        "missing_samples",
        "missing_info",
        "missing_occurrence",
        "unsupported_payload",
        "no_sample_method",
    ],
)
def test_dwave_native_sample_channels_and_legacy_projection_preserve_prior_model(
    monkeypatch: pytest.MonkeyPatch,
    capture: bool,
    channel: str,
) -> None:
    """Native row order/domain survives supported channels; malformed replies keep earlier results intact."""
    spin = channel == "spin"
    sample = {"1": 1, "0": -1 if spin else 0}
    rows = [_FakeSampleRow(sample, 3), _FakeSampleRow(dict(sample), 5)]

    class MappingReply(dict[str, object]):
        """Expose original mapping samples with native identity and returned variable order."""

        info = {"task_id": "mapping_sample_identity"}
        variables = ("1", "0")
        vartype = "SPIN" if spin else "BINARY"

    def read(fields: list[str]) -> list[_FakeSampleRow]:
        """Require the exact documented row fields without inventing missing native energy."""
        assert fields == ["sample", "num_occurrences"]
        return rows

    valid_reply = types.SimpleNamespace(
        info={"id": "data_sample_identity"},
        variables=("1", "0"),
        vartype="SPIN" if spin else "BINARY",
        data=read,
    )
    replies: dict[str, object] = {
        "mapping": MappingReply(samples=[sample, dict(sample)], counts=[3, 5]),
        "data": valid_reply,
        "spin": valid_reply,
        "empty": MappingReply(samples=[], num_occurrences=[]),
        "data_bad_mapping": types.SimpleNamespace(
            data=lambda fields: [types.SimpleNamespace(sample=[], num_occurrences=8)]
        ),
        "missing_variable": types.SimpleNamespace(
            data=lambda fields: [_FakeSampleRow({"0": 0}, 8)], variables=("1", "0")
        ),
        "mapping_bad_mapping": MappingReply(samples=[4], num_occurrences=[8]),
        "mapping_wrong_lengths": MappingReply(samples=[sample], num_occurrences=[3, 5]),
        "missing_samples": MappingReply(samples=None, counts=[8]),
        "missing_info": types.SimpleNamespace(data=read),
        "missing_occurrence": types.SimpleNamespace(
            data=lambda fields: [types.SimpleNamespace(sample=sample)]
        ),
        "unsupported_payload": object(),
        "no_sample_method": valid_reply,
    }

    class NativeSampler:
        """Return explicit sample-set I/O without dimod or physical annealer qualification."""

        def __init__(self) -> None:
            """Start with a valid original reply and count real public sampling calls."""
            self.reply: object = valid_reply
            self.calls = 0

        def sample(self, bqm: object, *, num_reads: int, label: str) -> object:
            """Expose only the current admitted model's reply and exact requested reads."""
            assert num_reads == 8
            self.calls += 1
            return self.reply

    sampler = NativeSampler()
    hal = HardwareAbstractionLayer.with_builtin_profiles()
    hal.register_backend(
        DWaveLeapHALAdapter(hal.profile("dwave_leap"), sampler=sampler, bqm_factory=_bqm_factory)
    )
    workload = dwave_bqm_workload(
        linear={"0": -1.0, "1": 0.5},
        quadratic={},
        workload_id="original_sample_anchor",
        n_variables=2,
        reads=8,
        vartype="SPIN" if spin else "BINARY",
        capture_semantics=capture,
    )
    prior = hal.submit("dwave_leap", workload, approval_id="native-io-only")
    original = hal.result(prior)
    sampler.reply = replies[channel]
    current = replace(workload, workload_id="current_sample_channel")
    if channel not in {"mapping", "data", "spin"}:
        with monkeypatch.context() as injected:
            if channel == "no_sample_method":
                injected.setattr(sampler, "sample", None)
            with pytest.raises(TypeError if channel == "no_sample_method" else ValueError):
                hal.submit("dwave_leap", current, approval_id="native-io-only")
        assert sampler.calls == (1 if channel == "no_sample_method" else 2)
    else:
        job = hal.submit("dwave_leap", current, approval_id="native-io-only")
        result = hal.result(job)
        assert result.counts == {"01": 8} and result.shots == 8
        if capture:
            observation = result.provider_observation
            assert isinstance(observation, AnnealingObservation)
            assert observation.returned_axes == ("1", "0")
            assert [row.values for row in observation.samples] == [(1, -1 if spin else 0)] * 2
            assert [row.energy for row in observation.samples] == [None, None]
            assert observation.raw_record is None
            assert (
                job.submission is not None and job.submission.original_program == workload.program
            )
        else:
            assert result.provider_observation is None
    assert hal.result(prior) is original


def test_dwave_legacy_target_pin_requires_native_source_capture() -> None:
    """A bare legacy selector annotation cannot claim binding to an original model."""
    with pytest.raises(ValueError, match="target pin"):
        dwave_bqm_workload(
            linear={"0": 1.0},
            quadratic={},
            workload_id="unbound_target",
            n_variables=1,
            reads=1,
            requested_target="declared_only",
        )
