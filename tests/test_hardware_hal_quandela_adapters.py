# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Quandela HAL adapter tests
"""Tests for direct Quandela/Perceval execution behind the HAL."""

from __future__ import annotations

from typing import Any

import pytest

from scpn_quantum_control.hardware.hal import HardwareAbstractionLayer, QuantumWorkload
from scpn_quantum_control.hardware.hal_quandela import (
    QUANDELA_PERCEVAL_SCHEMA,
    QuandelaPercevalHALAdapter,
    quandela_perceval_workload,
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


def test_quandela_default_builder_is_calibration_gated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
