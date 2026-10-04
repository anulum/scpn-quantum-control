# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Quandela HAL adapter tests
"""Tests for direct Quandela/Perceval execution behind the HAL."""

from __future__ import annotations

import json
import types
from dataclasses import replace
from typing import Any

import pytest

from scpn_quantum_control.hardware.hal import HardwareAbstractionLayer, QuantumWorkload
from scpn_quantum_control.hardware.hal_quandela import (
    QUANDELA_PERCEVAL_SCHEMA,
    QuandelaPercevalHALAdapter,
    quandela_perceval_workload,
)
from scpn_quantum_control.hardware.provider_modalities import (
    ModalitySemantics,
    PhotonicObservation,
)

_PHOTONIC_PLAN = {
    "schema": "scpn.quandela.perceval.v1",
    "modes": 2,
    "input_state": [1, 0],
    "components": [
        {"type": "beam_splitter", "modes": [0, 1], "theta": 0.7853981633974483},
        {"type": "phase_shifter", "mode": 1, "phi": 0.25},
    ],
    "postselection": {"min_detected_photons": 1},
}


class _FakeSampler:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def samples(self, *, count: int) -> dict[str, object]:
        self.calls.append({"count": count})
        return {"id": "quandela-provider-job-1", "results": {"10": 6, "01": 4}}


class _FakeProcessor:
    def __init__(self) -> None:
        self.samples_calls: list[int] = []

    def samples(self, count: int) -> dict[str, object]:
        self.samples_calls.append(count)
        return {"job_id": "quandela-provider-job-2", "counts": {"10": 3, "01": 1}}


def test_quandela_adapter_executes_injected_sampler_with_approval() -> None:
    """An approved I/O sampler receives the original plan and exact requested count."""
    sampler = _FakeSampler()
    captured: dict[str, object] = {}

    def processor_factory(plan: dict[str, object]) -> dict[str, object]:
        captured["plan"] = plan
        return {"processor_plan": plan}

    def sampler_factory(processor: object) -> _FakeSampler:
        captured["processor"] = processor
        return sampler

    hal = HardwareAbstractionLayer.with_builtin_profiles()
    hal.register_backend(
        QuandelaPercevalHALAdapter(
            hal.profile("quandela_cloud"),
            processor_factory=processor_factory,
            sampler_factory=sampler_factory,
            target="ascella",
        )
    )
    workload = quandela_perceval_workload(
        _PHOTONIC_PLAN,
        workload_id="quandela_pair",
        n_modes=2,
        shots=10,
        metadata={"campaign": "hal"},
    )

    with pytest.raises(PermissionError, match="approval"):
        hal.submit("quandela_cloud", workload)
    with pytest.raises(PermissionError, match="approval"):
        QuandelaPercevalHALAdapter(hal.profile("quandela_cloud"), processor=object()).submit(
            workload
        )

    job = hal.submit("quandela_cloud", workload, approval_id="approved-quandela")
    result = hal.result(job)
    cancelled = hal.cancel(job)

    assert job.status == "completed"
    assert job.job_id.startswith("quandela_cloud:quandela_pair:")
    assert job.metadata["provider_job_id"] == "quandela-provider-job-1"
    assert job.metadata["execution_mode"] == "quandela_perceval"
    assert job.metadata["target"] == "ascella"
    assert result.counts == {"10": 6, "01": 4}
    assert result.shots == 10
    assert result.metadata["target"] == "ascella"
    assert cancelled.status == "cancelled"
    assert captured["plan"] == _PHOTONIC_PLAN
    assert captured["processor"] == {"processor_plan": _PHOTONIC_PLAN}
    assert sampler.calls == [{"count": 10}]


def test_quandela_adapter_supports_direct_processor_samples() -> None:
    """The direct processor sampling route preserves declared observations."""
    processor = _FakeProcessor()
    hal = HardwareAbstractionLayer.with_builtin_profiles()
    adapter = QuandelaPercevalHALAdapter(
        hal.profile("quandela_cloud"),
        processor=processor,
        target="local-processor",
    )
    workload = quandela_perceval_workload(
        _PHOTONIC_PLAN,
        workload_id="direct_processor",
        n_modes=2,
        shots=4,
    )

    job = adapter.submit(workload, approval_id="approved")
    result = adapter.result(job)

    assert result.counts == {"10": 3, "01": 1}
    assert processor.samples_calls == [4]


def test_quandela_adapter_rejects_wrong_profile_ir_and_schema() -> None:
    """Wrong provider profile, source format or schema refuses before sampling."""
    hal = HardwareAbstractionLayer.with_builtin_profiles()
    with pytest.raises(ValueError, match="quandela_cloud"):
        QuandelaPercevalHALAdapter(hal.profile("pasqal_cloud"), processor=_FakeProcessor())

    adapter = QuandelaPercevalHALAdapter(hal.profile("quandela_cloud"), processor=_FakeProcessor())
    with pytest.raises(ValueError, match="perceval workloads"):
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

    bad_schema = dict(_PHOTONIC_PLAN) | {"schema": "other"}
    with pytest.raises(ValueError, match=QUANDELA_PERCEVAL_SCHEMA):
        quandela_perceval_workload(
            bad_schema,
            workload_id="bad_schema",
            n_modes=2,
            shots=4,
        )


def test_quandela_adapter_validates_payload_shape_and_counts() -> None:
    """Malformed native plan fields and negative occurrences cannot qualify a result."""
    bad_modes = dict(_PHOTONIC_PLAN) | {"input_state": [1]}
    with pytest.raises(ValueError, match="input_state"):
        quandela_perceval_workload(
            bad_modes,
            workload_id="bad_modes",
            n_modes=2,
            shots=4,
        )

    bad_component = dict(_PHOTONIC_PLAN) | {"components": [{"type": "phase_shifter"}]}
    with pytest.raises(ValueError, match="mode"):
        quandela_perceval_workload(
            bad_component,
            workload_id="bad_component",
            n_modes=2,
            shots=4,
        )

    class BadProcessor:
        def samples(self, count: int) -> dict[str, object]:
            del count
            return {"job_id": "quandela-provider-bad", "counts": {"10": -1}}

    adapter = QuandelaPercevalHALAdapter(
        HardwareAbstractionLayer.with_builtin_profiles().profile("quandela_cloud"),
        processor=BadProcessor(),
    )
    with pytest.raises(ValueError, match="non-negative"):
        adapter.submit(
            quandela_perceval_workload(
                _PHOTONIC_PLAN,
                workload_id="bad_counts",
                n_modes=2,
                shots=1,
            ),
            approval_id="approved",
        )


def test_quandela_provider_job_id_extraction_requires_identifier() -> None:
    """An observed histogram without provider identity cannot become a stored job."""

    class MissingProviderIdProcessor:
        def samples(self, count: int) -> dict[str, object]:
            del count
            return {"counts": {"10": 1}}

    adapter = QuandelaPercevalHALAdapter(
        HardwareAbstractionLayer.with_builtin_profiles().profile("quandela_cloud"),
        processor=MissingProviderIdProcessor(),
    )
    with pytest.raises(ValueError, match="provider job id"):
        adapter.submit(
            quandela_perceval_workload(
                _PHOTONIC_PLAN,
                workload_id="missing_job_id",
                n_modes=2,
                shots=1,
            ),
            approval_id="approved",
        )


def test_quandela_provider_job_id_rejects_control_characters() -> None:
    """Quandela provider identifiers must reject control-character payloads."""

    class BadProviderIdProcessor:
        def samples(self, count: int) -> dict[str, object]:
            del count
            return {"job_id": "quandela-provider-\n2", "counts": {"10": 1}}

    adapter = QuandelaPercevalHALAdapter(
        HardwareAbstractionLayer.with_builtin_profiles().profile("quandela_cloud"),
        processor=BadProviderIdProcessor(),
    )
    with pytest.raises(ValueError, match="provider job id"):
        adapter.submit(
            quandela_perceval_workload(
                _PHOTONIC_PLAN,
                workload_id="bad_provider_id",
                n_modes=2,
                shots=1,
            ),
            approval_id="approved",
        )


def test_quandela_provider_job_id_trims_padding() -> None:
    """Quandela provider identifiers should be canonicalised by trimming padding."""

    class PaddedProviderIdProcessor:
        def samples(self, count: int) -> dict[str, object]:
            del count
            return {"job_id": "  quandela-provider-2  ", "counts": {"10": 1}}

    adapter = QuandelaPercevalHALAdapter(
        HardwareAbstractionLayer.with_builtin_profiles().profile("quandela_cloud"),
        processor=PaddedProviderIdProcessor(),
    )
    job = adapter.submit(
        quandela_perceval_workload(
            _PHOTONIC_PLAN,
            workload_id="padded_provider_id",
            n_modes=2,
            shots=1,
        ),
        approval_id="approved",
    )
    assert job.metadata["provider_job_id"] == "quandela-provider-2"


@pytest.mark.parametrize("target", ["quandela-\nqpu", "", "   "])
def test_quandela_target_rejects_control_characters(target: str) -> None:
    """Explicit control-character or empty selectors cannot become another route."""
    processor = _FakeProcessor()
    profile = HardwareAbstractionLayer.with_builtin_profiles().profile("quandela_cloud")
    with pytest.raises(ValueError, match="Quandela target"):
        QuandelaPercevalHALAdapter(profile, processor=processor, target=target)
    assert processor.samples_calls == []


def test_quandela_target_trims_padding() -> None:
    """Quandela targets should be canonicalised by trimming padding."""
    profile = HardwareAbstractionLayer.with_builtin_profiles().profile("quandela_cloud")
    adapter = QuandelaPercevalHALAdapter(profile, processor=object(), target="  quandela-qpu  ")
    assert adapter._target == "quandela-qpu"


def test_quandela_default_builder_is_calibration_gated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Absent SDK and absent calibrated builder refuse without inventing a processor."""
    hal = HardwareAbstractionLayer.with_builtin_profiles()
    adapter = QuandelaPercevalHALAdapter(hal.profile("quandela_cloud"))
    workload = quandela_perceval_workload(
        _PHOTONIC_PLAN,
        workload_id="needs_builder",
        n_modes=2,
        shots=1,
    )

    with pytest.raises(RuntimeError, match="perceval|calibrated Quandela"):
        adapter.submit(workload, approval_id="approved")

    def fake_import(name: str) -> Any:
        if name == "perceval":
            return object()
        raise ModuleNotFoundError(name)

    monkeypatch.setattr("scpn_quantum_control.hardware.hal_quandela.import_module", fake_import)
    with pytest.raises(RuntimeError, match="calibrated Quandela"):
        adapter.submit(workload, approval_id="approved")


def test_quandela_adapter_rejects_shot_mismatch() -> None:
    """A provider total different from requested observations refuses before retention."""

    class ShotMismatchProcessor:
        def samples(self, count: int) -> dict[str, object]:
            del count
            return {"job_id": "quandela-provider-shot-mismatch", "counts": {"10": 3, "01": 1}}

    adapter = QuandelaPercevalHALAdapter(
        HardwareAbstractionLayer.with_builtin_profiles().profile("quandela_cloud"),
        processor=ShotMismatchProcessor(),
    )
    with pytest.raises(ValueError, match="shot count mismatch"):
        adapter.submit(
            quandela_perceval_workload(
                _PHOTONIC_PLAN,
                workload_id="quandela_shot_mismatch",
                n_modes=2,
                shots=10,
            ),
            approval_id="approved",
        )


def test_quandela_factory_rebuilds_each_distinct_native_plan() -> None:
    """A second plan must reach its builder instead of sampling the first processor."""
    plans: list[dict[str, object]] = []

    def build(plan: dict[str, object]) -> _FakeProcessor:
        """Record the actual public build boundary without numerical qualification."""
        plans.append(plan)
        return _FakeProcessor()

    hal = HardwareAbstractionLayer.with_builtin_profiles()
    adapter = QuandelaPercevalHALAdapter(hal.profile("quandela_cloud"), processor_factory=build)
    hal.register_backend(adapter)
    first = quandela_perceval_workload(_PHOTONIC_PLAN, workload_id="first", n_modes=2, shots=4)
    second_plan = dict(_PHOTONIC_PLAN) | {
        "components": [{"type": "phase_shifter", "mode": 0, "phi": 0.5}]
    }
    second = quandela_perceval_workload(second_plan, workload_id="second", n_modes=2, shots=4)
    a = hal.submit("quandela_cloud", first, approval_id="transport-contract-only")
    b = hal.submit("quandela_cloud", second, approval_id="transport-contract-only")
    assert plans == [_PHOTONIC_PLAN, second_plan]
    assert hal.result(a).job.workload_id == "first"
    assert hal.result(b).job.workload_id == "second"


def test_quandela_typed_occupation_output_and_stored_identity() -> None:
    """Public HAL preserves nonbinary photon occupations from a contract fixture."""
    from scpn_quantum_control.hardware.provider_modalities import PhotonicObservation

    class OccupationProcessor:
        """Return fixed native-domain data, without a simulated-device claim."""

        def samples(self, count: int) -> dict[str, object]:
            """Expose original occupation keys and exact requested occurrences."""
            assert count == 3
            return {"job_id": "photonic-contract", "counts": {(2, 0): 2, (0, 2): 1}}

    hal = HardwareAbstractionLayer.with_builtin_profiles()
    adapter = QuandelaPercevalHALAdapter(
        hal.profile("quandela_cloud"),
        processor_factory=lambda plan: OccupationProcessor(),
        target="declared_ascella",
    )
    hal.register_backend(adapter)
    workload = quandela_perceval_workload(
        _PHOTONIC_PLAN,
        workload_id="occupation",
        n_modes=2,
        shots=3,
        capture_semantics=True,
        requested_target="declared_ascella",
    )
    job = hal.submit("quandela_cloud", workload, approval_id="transport-contract-only")
    result = hal.result(job)
    assert isinstance(result.provider_observation, PhotonicObservation)
    assert result.provider_observation.samples[0].occupations == (2, 0)
    assert job.submission is not None
    assert job.submission.original_program == workload.program
    assert job.submission.target_origin == "adapter_selector"
    with pytest.raises(ValueError, match="workload_id"):
        hal.result(replace(job, workload_id="another_source"))
    assert hal.result(job) is result


@pytest.mark.parametrize(
    "mutation,message",
    [
        ({"modes": 3}, "mode count"),
        ({"input_state": "10"}, "sequence"),
        ({"input_state": [-1, 0]}, "non-negative"),
        ({"components": "beam_splitter"}, "sequence"),
        ({"components": [4]}, "mappings"),
        ({"components": [{"type": ""}]}, "non-empty"),
        ({"components": [{"type": "unsupported"}]}, "unsupported"),
        ({"components": [{"type": "beam_splitter", "modes": [0]}]}, "two mode indices"),
        ({"components": [{"type": "beam_splitter", "modes": [0, 0], "theta": 0.5}]}, "distinct"),
        ({"components": [{"type": "phase_shifter", "mode": -1, "phi": 0.25}]}, "out of range"),
        ({"components": [{"type": "phase_shifter", "mode": 2, "phi": 0.25}]}, "out of range"),
        ({"postselection": []}, "mapping"),
        ({"postselection": {"min_detected_photons": -1}}, "non-negative"),
    ],
)
def test_quandela_original_plan_refusals_precede_sampling_and_keep_prior_output(
    mutation: dict[str, object],
    message: str,
) -> None:
    """Malformed native plans refuse through HAL without replacing a prior result."""
    processor = _FakeProcessor()
    hal = HardwareAbstractionLayer.with_builtin_profiles()
    hal.register_backend(
        QuandelaPercevalHALAdapter(hal.profile("quandela_cloud"), processor=processor)
    )
    first = hal.submit(
        "quandela_cloud",
        quandela_perceval_workload(
            _PHOTONIC_PLAN, workload_id="prior_photonic_result", n_modes=2, shots=4
        ),
        approval_id="io-contract-only",
    )
    original = hal.result(first)
    malformed = QuantumWorkload(
        workload_id="invalid_native_plan",
        ir_format="perceval",
        program=json.dumps(dict(_PHOTONIC_PLAN) | mutation),
        n_qubits=2,
        shots=4,
    )
    with pytest.raises(ValueError, match=message):
        hal.submit("quandela_cloud", malformed, approval_id="io-contract-only")
    assert processor.samples_calls == [4] and hal.result(first) is original


@pytest.mark.parametrize("source", ["{", "[]"])
def test_quandela_invalid_json_refuses_before_processor_factory(source: str) -> None:
    """Malformed native source cannot invoke a builder or select a default processor."""
    plans: list[dict[str, object]] = []

    def build(plan: dict[str, object]) -> _FakeProcessor:
        plans.append(plan)
        return _FakeProcessor()

    hal = HardwareAbstractionLayer.with_builtin_profiles()
    hal.register_backend(
        QuandelaPercevalHALAdapter(hal.profile("quandela_cloud"), processor_factory=build)
    )
    raw = QuantumWorkload(
        workload_id="malformed_json", ir_format="perceval", program=source, n_qubits=2, shots=4
    )
    with pytest.raises(ValueError, match="valid JSON|JSON object"):
        hal.submit("quandela_cloud", raw, approval_id="io-contract-only")
    assert plans == []


@pytest.mark.parametrize("value", [None, True, "bad", float("nan"), float("inf"), 10**400])
def test_quandela_invalid_native_angles_refuse_before_builder(value: object) -> None:
    """Photonic angles cannot silently coerce missing, boolean or unbounded values."""
    plan = dict(_PHOTONIC_PLAN) | {
        "components": [{"type": "phase_shifter", "mode": 0, "phi": value}]
    }
    with pytest.raises(ValueError, match="numeric|finite"):
        quandela_perceval_workload(plan, workload_id="invalid_native_angle", n_modes=2, shots=4)


@pytest.mark.parametrize("digest_form", ["missing", "other", "exact"])
def test_quandela_prepared_processor_requires_exact_captured_source_and_keeps_prior_output(
    digest_form: str,
) -> None:
    """Opaque prepared processors cannot execute a different native plan under its companion."""
    source = "\n" + json.dumps(_PHOTONIC_PLAN, indent=2) + "\n"
    workload = quandela_perceval_workload(
        source, workload_id="captured_prepared_source", n_modes=2, shots=4, capture_semantics=True
    )
    assert workload.semantics is not None
    digest = {"missing": None, "other": "0" * 64, "exact": workload.semantics.program_sha256}[
        digest_form
    ]

    class NativeProcessor:
        """Return ordered native occupation fixtures without photonic dynamics claims."""

        def __init__(self) -> None:
            """Retain an observable submission count."""
            self.calls: list[int] = []

        def samples(self, count: int) -> dict[str, object]:
            """Return nonbinary occupations with exactly the requested total."""
            self.calls.append(count)
            return {"id": "native_prepared_photonic", "counts": {(2, 0): 3, (0, 2): 1}}

    processor = NativeProcessor()
    hal = HardwareAbstractionLayer.with_builtin_profiles()
    hal.register_backend(
        QuandelaPercevalHALAdapter(
            hal.profile("quandela_cloud"), processor=processor, prepared_program_sha256=digest
        )
    )
    prior = hal.submit(
        "quandela_cloud",
        quandela_perceval_workload(
            _PHOTONIC_PLAN, workload_id="prior_prepared_result", n_modes=2, shots=4
        ),
        approval_id="native-io-only",
    )
    original = hal.result(prior)
    if digest_form == "exact":
        job = hal.submit("quandela_cloud", workload, approval_id="native-io-only")
        assert job.submission is not None and job.submission.original_program == source
        assert job.submission.compilation == "caller_precompiled"
        result = hal.result(job)
        assert isinstance(result.provider_observation, PhotonicObservation)
        assert result.provider_observation.samples[0].occupations == (2, 0)
        assert hal.status(job) == "completed"
        cancelled = hal.cancel(job)
        assert cancelled.submission == job.submission and hal.status(cancelled) == "cancelled"
        assert hal.result(cancelled) is result and processor.calls == [4, 4]
    else:
        with pytest.raises(ValueError, match="exact original program digest"):
            hal.submit("quandela_cloud", workload, approval_id="native-io-only")
        assert processor.calls == [4]
    assert hal.result(prior) is original


@pytest.mark.parametrize(
    "channel",
    [
        "samples",
        "counts_attribute",
        "results_attribute",
        "missing",
        "state_string",
        "state_scalar",
    ],
)
def test_quandela_native_channels_preserve_occupations_or_refuse_without_prior_loss(
    channel: str,
) -> None:
    """Native occupations and labels survive supported I/O channels; incompatible domains refuse."""
    counts: dict[object, object] = {(2, 0): 3, (0, 2): 1}
    values: dict[str, object] = {
        "samples": {"task_id": "native_photonic_channel", "samples": counts},
        "counts_attribute": types.SimpleNamespace(
            task_id="native_photonic_channel", counts=counts
        ),
        "results_attribute": types.SimpleNamespace(
            id=lambda: "native_photonic_channel", results=counts
        ),
        "missing": types.SimpleNamespace(id="native_photonic_channel"),
        "state_string": {"id": "native_photonic_channel", "counts": {"20": 4}},
        "state_scalar": {"id": "native_photonic_channel", "counts": {2: 4}},
    }

    class ChannelProcessor:
        """Expose independent I/O channels and explicit recovery without SDK attestation."""

        def __init__(self) -> None:
            """Start with valid native data before injecting the selected result fault."""
            self.reply: object = {"id": "original_photonic_channel", "counts": counts}

        def sample_count(self, count: int) -> object:
            """Support the legacy sample-count operation with exact requested observations."""
            assert count == 4
            return self.reply

    processor = ChannelProcessor()
    hal = HardwareAbstractionLayer.with_builtin_profiles()
    hal.register_backend(
        QuandelaPercevalHALAdapter(
            hal.profile("quandela_cloud"), processor_factory=lambda plan: processor
        )
    )
    first = hal.submit(
        "quandela_cloud",
        quandela_perceval_workload(
            _PHOTONIC_PLAN,
            workload_id="prior_native_channel",
            n_modes=2,
            shots=4,
            capture_semantics=True,
        ),
        approval_id="native-io-only",
    )
    original = hal.result(first)
    processor.reply = values[channel]
    workload = quandela_perceval_workload(
        _PHOTONIC_PLAN,
        workload_id="native_result_channel",
        n_modes=2,
        shots=4,
        capture_semantics=True,
    )
    if channel in {"missing", "state_string", "state_scalar"}:
        with pytest.raises(
            (RuntimeError, ValueError),
            match="extract Quandela|iterable occupations|ordered occupations",
        ):
            hal.submit("quandela_cloud", workload, approval_id="native-io-only")
        assert hal.result(first) is original
        processor.reply = {"id": "recovered_photonic_channel", "counts": counts}
    job = hal.submit("quandela_cloud", workload, approval_id="native-io-only")
    result = hal.result(job)
    assert isinstance(result.provider_observation, PhotonicObservation)
    assert [
        (sample.occupations, sample.occurrences, sample.native_label)
        for sample in result.provider_observation.samples
    ] == [((2, 0), 3, "(2, 0)"), ((0, 2), 1, "(0, 2)")]
    counts[(2, 0)] = 99
    assert result.provider_observation.samples[0].occurrences == 3
    assert hal.result(first) is original


def test_quandela_legacy_target_pin_requires_native_capture() -> None:
    """A legacy plan cannot infer exact source-bound target admission from a bare name."""
    with pytest.raises(ValueError, match="target pin"):
        quandela_perceval_workload(
            _PHOTONIC_PLAN,
            workload_id="uncaptured_target",
            n_modes=2,
            shots=4,
            requested_target="unbound_target",
        )


@pytest.mark.parametrize(
    "fault",
    [
        "sampler",
        "processor",
        "mapping_without_counts",
        "nonmapping_without_id",
        "empty_label",
        "empty_counts",
    ],
)
def test_quandela_invalid_sampling_contracts_refuse_without_retaining_a_job(fault: str) -> None:
    """Unsupported operations, result channels and legacy labels cannot become completed jobs."""
    values: dict[str, object] = {
        "mapping_without_counts": {"id": "invalid_channel"},
        "nonmapping_without_id": types.SimpleNamespace(counts={"10": 4}),
        "empty_label": {"id": "invalid_channel", "counts": {"": 4}},
        "empty_counts": {"id": "invalid_channel", "counts": {}},
    }
    calls: list[int] = []

    class ResultProcessor:
        """Expose each explicit malformed provider I/O reply."""

        def samples(self, count: int) -> object:
            """Retain the observable native sampling count before returning the fault."""
            calls.append(count)
            return values[fault]

    profile = HardwareAbstractionLayer.with_builtin_profiles().profile("quandela_cloud")
    if fault == "sampler":
        adapter = QuandelaPercevalHALAdapter(
            profile, processor=object(), sampler_factory=lambda processor: object()
        )
    elif fault == "processor":
        adapter = QuandelaPercevalHALAdapter(profile, processor=object())
    else:
        adapter = QuandelaPercevalHALAdapter(profile, processor=ResultProcessor())
    workload = quandela_perceval_workload(
        _PHOTONIC_PLAN, workload_id="invalid_sampling_contract", n_modes=2, shots=4
    )
    with pytest.raises(
        (ValueError, TypeError, RuntimeError),
        match="samples|counts|provider job id|photonic states",
    ):
        adapter.submit(workload, approval_id="io-refusal-only")
    assert calls == ([] if fault in {"sampler", "processor"} else [4])


def test_quandela_lost_retained_result_refuses_and_restoration_preserves_native_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Controlled local retention loss cannot fabricate results or resample a completed job."""
    processor = _FakeProcessor()
    hal = HardwareAbstractionLayer.with_builtin_profiles()
    adapter = QuandelaPercevalHALAdapter(hal.profile("quandela_cloud"), processor=processor)
    hal.register_backend(adapter)
    job = hal.submit(
        "quandela_cloud",
        quandela_perceval_workload(
            _PHOTONIC_PLAN, workload_id="retained_photonic_result", n_modes=2, shots=4
        ),
        approval_id="retention-fault-only",
    )
    result = hal.result(job)
    with monkeypatch.context() as loss:
        loss.delitem(adapter._results, job.job_id)
        with pytest.raises(KeyError, match="unknown job_id"):
            hal.result(job)
    assert hal.result(job) is result and processor.samples_calls == [4]


def test_quandela_plan_without_components_or_postselection_keeps_original_native_source() -> None:
    """Optional postselection and an empty component sequence remain valid native plans."""
    plan = dict(_PHOTONIC_PLAN) | {"components": []}
    del plan["postselection"]
    source = json.dumps(plan, indent=2)
    workload = quandela_perceval_workload(
        source, workload_id="untransformed_plan", n_modes=2, shots=4, capture_semantics=True
    )
    assert workload.program == source and isinstance(workload.semantics, ModalitySemantics)
    assert workload.semantics.native_axes == (0, 1)
