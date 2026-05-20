# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — D-Wave Leap HAL adapter tests
"""Tests for direct D-Wave Leap annealing behind the provider-neutral HAL."""

from __future__ import annotations

from typing import Any

import pytest

from scpn_quantum_control.hardware.hal import HardwareAbstractionLayer, QuantumWorkload
from scpn_quantum_control.hardware.hal_dwave import (
    DWAVE_BQM_SCHEMA,
    DWaveLeapHALAdapter,
    dwave_bqm_workload,
)


class _FakeSampleRow:
    def __init__(self, sample: dict[str, int], num_occurrences: int) -> None:
        self.sample = sample
        self.num_occurrences = num_occurrences


class _FakeSampleSet:
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


def _bqm_factory(payload: dict[str, object]) -> dict[str, object]:
    return {"factory_payload": payload}


def test_dwave_leap_adapter_samples_injected_sampler_with_approval() -> None:
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

    job = hal.submit("dwave_leap", workload, approval_id="approved-dwave")
    result = hal.result(job)
    cancelled = hal.cancel(job)

    assert job.status == "completed"
    assert job.job_id == "dwave_leap:dwave_pair"
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


def test_dwave_leap_adapter_default_builder_is_sdk_gated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hal = HardwareAbstractionLayer.with_builtin_profiles()
    adapter = DWaveLeapHALAdapter(hal.profile("dwave_leap"))
    workload = dwave_bqm_workload(
        linear={"0": 1.0},
        quadratic={},
        workload_id="needs_sdk",
        n_variables=1,
        reads=1,
    )

    with pytest.raises(RuntimeError, match="dimod"):
        adapter.submit(workload, approval_id="approved")

    def fake_import(name: str) -> Any:
        if name == "dimod":

            class _BQM:
                @classmethod
                def from_qubo(
                    cls, qubo: dict[tuple[str, str], float], offset: float = 0.0
                ) -> dict[str, Any]:
                    return {"qubo": qubo, "offset": offset}

            return type("Dimod", (), {"BinaryQuadraticModel": _BQM})
        raise ModuleNotFoundError(name)

    monkeypatch.setattr("scpn_quantum_control.hardware.hal_dwave.import_module", fake_import)
    with pytest.raises(RuntimeError, match="DWaveSampler"):
        adapter.submit(workload, approval_id="approved")
