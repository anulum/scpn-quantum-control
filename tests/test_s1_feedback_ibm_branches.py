# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — S1 IBM feedback refusal and offline branch tests
"""Exercise public S1 IBM refusal and fake-provider branches."""

from __future__ import annotations

import sys
from collections.abc import Mapping, Sequence
from dataclasses import replace
from types import ModuleType, SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
from qiskit import QuantumCircuit

from scpn_quantum_control.control.realtime_feedback import RealtimeSyncFeedbackController
from scpn_quantum_control.hardware import s1_feedback_ibm as s1
from scpn_quantum_control.hardware.feedback_loop import FeedbackResult


def _controller() -> RealtimeSyncFeedbackController:
    """Return a small deterministic controller for offline circuit building."""
    return RealtimeSyncFeedbackController(
        np.array([[0.0, 0.3], [0.3, 0.0]], dtype=np.float64),
        np.array([0.1, 0.4], dtype=np.float64),
    )


def _arm(*, repetitions: int = 1) -> s1.S1FeedbackArmCircuit:
    """Return a valid one-qubit arm for command and sampler tests."""
    circuit = QuantumCircuit(1, 1)
    circuit.measure(0, 0)
    return s1.S1FeedbackArmCircuit(
        s1.S1_FEEDBACK_ARM,
        circuit,
        shots=8,
        repetitions=repetitions,
        estimated_qpu_seconds=0.5,
    )


def _feedback_result(
    arm: str,
    *,
    job_id: str | None = "job",
    observable: Any = "XX",
    records: Any = None,
) -> FeedbackResult:
    """Build a result with caller-controlled public metadata."""
    if records is None:
        records = [{"counts": {"00": 4}, "source_index": 0}]
    return FeedbackResult(
        job_id=job_id,
        metadata={"arm": arm, "observable": observable, "records": records},
    )


def _binary_package(
    *,
    experiment_id: str = "s1",
    target_r: Any = 0.5,
    feedback_result: FeedbackResult | None = None,
    control_result: FeedbackResult | None = None,
) -> dict[str, Any]:
    """Call the public binary-synchrony packager with typed defaults."""
    return s1.raw_count_package_from_feedback_results(
        experiment_id=experiment_id,
        target_r=target_r,
        n_qubits=2,
        feedback_result=feedback_result or _feedback_result(s1.S1_FEEDBACK_ARM),
        control_result=control_result or _feedback_result(s1.S1_CONTROL_ARM),
    )


def test_arm_and_circuit_builders_reject_invalid_public_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Arm metadata and circuit-builder boundaries fail closed."""
    arm = _arm()
    for kwargs, match in (
        ({"label": "other"}, "label"),
        ({"shots": 0}, "shots"),
        ({"repetitions": 0}, "repetitions"),
        ({"estimated_qpu_seconds": -0.1}, "estimated_qpu_seconds"),
        ({"observable": ""}, "observable"),
    ):
        with pytest.raises(ValueError, match=match):
            replace(arm, **kwargs)

    with pytest.raises(TypeError, match="controller"):
        s1.build_s1_feedback_arm_circuits(cast(Any, object()), n_rounds=1, shots=8, repetitions=1)
    with pytest.raises(ValueError, match="n_rounds"):
        s1.build_s1_feedback_arm_circuits(_controller(), n_rounds=0, shots=8, repetitions=1)
    with pytest.raises(ValueError, match="non-empty"):
        s1.build_s1_xy_observable_arm_circuits(
            _controller(), observables=(), n_rounds=1, shots=8, repetitions=1
        )
    with pytest.raises(ValueError, match="only I, X, Y, or Z"):
        s1.build_s1_xy_observable_arm_circuits(
            _controller(), observables=("AX",), n_rounds=1, shots=8, repetitions=1
        )
    with pytest.raises(ValueError, match="length"):
        s1.build_s1_xy_observable_arm_circuits(
            _controller(), observables=("X",), n_rounds=1, shots=8, repetitions=1
        )

    unmeasured = s1.S1FeedbackArmCircuit(
        s1.S1_FEEDBACK_ARM,
        QuantumCircuit(2, 2),
        shots=8,
        repetitions=1,
        estimated_qpu_seconds=0.5,
    )

    def fake_pair(
        controller: RealtimeSyncFeedbackController,
        *,
        n_rounds: int,
        shots: int,
        repetitions: int,
        estimated_seconds_per_circuit: float = 1.0,
    ) -> tuple[s1.S1FeedbackArmCircuit, s1.S1FeedbackArmCircuit]:
        del controller, n_rounds, shots, repetitions, estimated_seconds_per_circuit
        return unmeasured, replace(unmeasured, label=s1.S1_CONTROL_ARM)

    monkeypatch.setattr(s1, "build_s1_feedback_arm_circuits", fake_pair)
    with pytest.raises(ValueError, match="end with system measurements"):
        s1.build_s1_xy_observable_arm_circuits(
            _controller(), observables=("ZZ",), n_rounds=1, shots=8, repetitions=1
        )


def test_count_estimators_reject_malformed_inputs() -> None:
    """Synchrony and Pauli estimators reject malformed count dictionaries."""
    with pytest.raises(ValueError, match="n_qubits"):
        s1.binary_phase_synchrony_from_counts({"0": 1}, n_qubits=0)
    with pytest.raises(ValueError, match="bitstrings"):
        s1.binary_phase_synchrony_from_counts(cast(Mapping[str, int], {1: 1}), n_qubits=1)
    with pytest.raises(ValueError, match="non-negative"):
        s1.binary_phase_synchrony_from_counts({"0": -1}, n_qubits=1)
    with pytest.raises(ValueError, match="at least one shot"):
        s1.binary_phase_synchrony_from_counts({}, n_qubits=1)

    with pytest.raises(ValueError, match="non-identity"):
        s1.pauli_expectation_from_counts({"00": 1}, observable="II", n_qubits=2)
    with pytest.raises(ValueError, match="bitstrings"):
        s1.pauli_expectation_from_counts(
            cast(Mapping[str, int], {1: 1}), observable="X", n_qubits=1
        )
    with pytest.raises(ValueError, match="non-negative"):
        s1.pauli_expectation_from_counts({"0": -1}, observable="X", n_qubits=1)
    with pytest.raises(ValueError, match="at least one shot"):
        s1.pauli_expectation_from_counts({}, observable="X", n_qubits=1)
    with pytest.raises(ValueError, match="observable"):
        s1.pauli_expectation_from_counts({"0": 1}, observable="", n_qubits=1)


def test_feedback_package_rejects_invalid_public_results() -> None:
    """Binary-synchrony package validation rejects corrupt result custody."""
    with pytest.raises(ValueError, match="experiment_id"):
        _binary_package(experiment_id="")
    for target in (float("nan"), cast(Any, "bad")):
        with pytest.raises(ValueError, match="target_r"):
            _binary_package(target_r=target)
    with pytest.raises(ValueError, match="expected arm"):
        _binary_package(
            feedback_result=_feedback_result(s1.S1_CONTROL_ARM),
        )
    for records, match in (
        ("bad", "non-empty records"),
        ([1], "mappings"),
        ([{}], "preserve counts"),
    ):
        with pytest.raises(ValueError, match=match):
            _binary_package(
                feedback_result=_feedback_result(s1.S1_FEEDBACK_ARM, records=records),
            )

    payload = _binary_package(
        feedback_result=_feedback_result(s1.S1_FEEDBACK_ARM, records=[{"counts": {0: "4"}}]),
    )
    assert payload["arms"][0]["records"][0]["source_index"] == 0


def test_xy_package_rejects_incomplete_or_corrupt_result_groups() -> None:
    """Observable result grouping fails closed on missing or corrupt metadata."""
    with pytest.raises(ValueError, match="experiment_id"):
        s1.raw_count_package_from_xy_observable_results(
            experiment_id="", n_qubits=2, results=[_feedback_result(s1.S1_FEEDBACK_ARM)]
        )
    with pytest.raises(ValueError, match="non-empty"):
        s1.raw_count_package_from_xy_observable_results(
            experiment_id="s1b", n_qubits=2, results=[]
        )
    with pytest.raises(ValueError, match="job_id"):
        s1.raw_count_package_from_xy_observable_results(
            experiment_id="s1b",
            n_qubits=2,
            results=[_feedback_result(s1.S1_FEEDBACK_ARM, job_id=None)],
        )
    with pytest.raises(ValueError, match="supported arm"):
        s1.raw_count_package_from_xy_observable_results(
            experiment_id="s1b", n_qubits=2, results=[_feedback_result("other")]
        )
    with pytest.raises(ValueError, match="observable label"):
        s1.raw_count_package_from_xy_observable_results(
            experiment_id="s1b",
            n_qubits=2,
            results=[_feedback_result(s1.S1_FEEDBACK_ARM, observable=1)],
        )
    with pytest.raises(ValueError, match="both S1 arms"):
        s1.raw_count_package_from_xy_observable_results(
            experiment_id="s1b",
            n_qubits=2,
            results=[_feedback_result(s1.S1_FEEDBACK_ARM)],
        )

    control = _feedback_result(s1.S1_CONTROL_ARM)
    for records, match in (
        ("bad", "non-empty records"),
        ([1], "mappings"),
        ([{}], "preserve counts"),
    ):
        with pytest.raises(ValueError, match=match):
            s1.raw_count_package_from_xy_observable_results(
                experiment_id="s1b",
                n_qubits=2,
                results=[
                    _feedback_result(s1.S1_FEEDBACK_ARM, records=records),
                    control,
                ],
            )

    payload = s1.raw_count_package_from_xy_observable_results(
        experiment_id="s1b",
        n_qubits=2,
        results=[
            _feedback_result(s1.S1_FEEDBACK_ARM, records=[{"counts": {0: "4"}}]),
            control,
        ],
    )
    assert payload["observables"][0]["arms"][0]["records"][0]["source_index"] == 0


class _ChangingMetadata(dict[str, Any]):
    """Return one grouping value and a different summary value for one key."""

    def __init__(self, key: str, first: Any, second: Any) -> None:
        super().__init__(
            arm=s1.S1_FEEDBACK_ARM,
            observable="XX",
            records=[{"counts": {"00": 1}}],
        )
        self._key = key
        self._values = iter((first, second))

    def get(self, key: str, default: Any = None) -> Any:
        """Return stateful values only for the selected metadata key."""
        if key == self._key:
            return next(self._values)
        return super().get(key, default)


def test_xy_package_detects_metadata_mutation_between_grouping_and_summary() -> None:
    """Mutable provider metadata cannot evade arm or observable custody checks."""
    control = _feedback_result(s1.S1_CONTROL_ARM)
    for metadata, match in (
        (_ChangingMetadata("arm", s1.S1_FEEDBACK_ARM, s1.S1_CONTROL_ARM), "expected arm"),
        (_ChangingMetadata("observable", "XX", "YY"), "expected observable"),
    ):
        with pytest.raises(ValueError, match=match):
            s1.raw_count_package_from_xy_observable_results(
                experiment_id="s1b",
                n_qubits=2,
                results=[FeedbackResult(job_id="job", metadata=metadata), control],
            )


def test_command_and_fake_sampler_paths_fail_closed_without_provider_contact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Command and sampler validation stays entirely on fake local objects."""
    arm = _arm(repetitions=2)
    with pytest.raises(ValueError, match="length"):
        s1.build_s1_arm_command(arm, isa_circuits=[arm.circuit], timeout_s=1.0)
    with pytest.raises(ValueError, match="timeout_s"):
        s1.build_s1_arm_command(arm, isa_circuits=[arm.circuit, arm.circuit], timeout_s=0.0)
    with pytest.raises(ValueError, match="length"):
        s1.run_ibm_sampler_arm(
            backend=object(),
            arm=arm,
            isa_circuits=[arm.circuit],
            timeout_s=1.0,
            sampler_cls=object,
        )

    class Register:
        def __init__(self, counts: Mapping[Any, Any]) -> None:
            self._counts = counts

        def get_counts(self) -> Mapping[Any, Any]:
            return self._counts

    class PubResult:
        def __init__(self, counts: Mapping[Any, Any]) -> None:
            self.data = SimpleNamespace(readout=Register(counts))

    class Job:
        job_id = "attribute-job"

        def __init__(self, rows: Sequence[Mapping[Any, Any]]) -> None:
            self._rows = rows

        def result(self, timeout: float) -> list[PubResult]:
            assert timeout == 2.0
            return [PubResult(row) for row in self._rows]

    class Sampler:
        rows: Sequence[Mapping[Any, Any]] = ({0: 2}, {0: 3})

        def __init__(self, mode: object) -> None:
            self.mode = mode
            self.options = SimpleNamespace()

        def run(self, circuits: Sequence[QuantumCircuit]) -> Job:
            assert len(circuits) == 2
            return Job(self.rows)

    fake_runtime = ModuleType("qiskit_ibm_runtime")
    fake_runtime.SamplerV2 = Sampler  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "qiskit_ibm_runtime", fake_runtime)
    result = s1.run_ibm_sampler_arm(
        backend=object(),
        arm=arm,
        isa_circuits=[arm.circuit, arm.circuit],
        timeout_s=2.0,
    )
    assert result.job_id == "attribute-job"
    assert result.counts == {"0": 5}

    class MissingJobId:
        def result(self, timeout: float) -> list[PubResult]:
            del timeout
            return []

    class MissingJobSampler:
        def __init__(self, mode: object) -> None:
            self.mode = mode
            self.options = SimpleNamespace()

        def run(self, circuits: Sequence[QuantumCircuit]) -> MissingJobId:
            del circuits
            return MissingJobId()

    with pytest.raises(RuntimeError, match="job_id"):
        s1.run_ibm_sampler_arm(
            backend=object(),
            arm=arm,
            isa_circuits=[arm.circuit, arm.circuit],
            timeout_s=2.0,
            sampler_cls=MissingJobSampler,
        )

    for rows, match in ((({"0": -1},), "non-negative"), (({},), "at least one shot")):
        Sampler.rows = rows
        with pytest.raises(ValueError, match=match):
            s1.run_ibm_sampler_arm(
                backend=object(),
                arm=arm,
                isa_circuits=[arm.circuit, arm.circuit],
                timeout_s=2.0,
                sampler_cls=Sampler,
            )
