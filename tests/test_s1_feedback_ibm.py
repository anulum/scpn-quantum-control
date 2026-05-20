# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for S1 IBM feedback execution contracts
"""Tests for S1 IBM paired-arm submission and raw-count conversion."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_quantum_control.control.realtime_feedback import RealtimeSyncFeedbackController
from scpn_quantum_control.hardware.feedback_loop import FeedbackResult
from scpn_quantum_control.hardware.feedback_submission import summarise_feedback_circuit
from scpn_quantum_control.hardware.s1_feedback_ibm import (
    S1_CONTROL_ARM,
    S1_FEEDBACK_ARM,
    binary_phase_synchrony_from_counts,
    build_s1_arm_command,
    build_s1_feedback_arm_circuits,
    build_s1_xy_observable_arm_circuits,
    pauli_expectation_from_counts,
    raw_count_package_from_feedback_results,
    raw_count_package_from_xy_observable_results,
    run_ibm_sampler_arm,
)


def _controller() -> RealtimeSyncFeedbackController:
    return RealtimeSyncFeedbackController(
        np.array([[0.0, 0.35, 0.20], [0.35, 0.0, 0.25], [0.20, 0.25, 0.0]], dtype=float),
        np.array([0.1, 0.4, 0.7], dtype=float),
    )


def test_build_s1_feedback_arm_circuits_returns_preregistered_pair() -> None:
    arms = build_s1_feedback_arm_circuits(_controller(), n_rounds=3, shots=1024, repetitions=12)

    assert [arm.label for arm in arms] == [S1_FEEDBACK_ARM, S1_CONTROL_ARM]
    assert all(arm.shots == 1024 for arm in arms)
    assert all(arm.repetitions == 12 for arm in arms)
    assert arms[0].circuit.count_ops()["if_else"] == 3
    assert arms[0].circuit.count_ops()["reset"] == 3
    assert "if_else" not in arms[1].circuit.count_ops()


def test_corrected_feedback_summary_does_not_require_conditional_reset() -> None:
    arm = build_s1_feedback_arm_circuits(_controller(), n_rounds=3, shots=1024, repetitions=12)[0]

    summary = summarise_feedback_circuit(arm.circuit, n_rounds=3)

    assert summary.has_conditional_control is True
    assert summary.has_conditional_reset is False


def test_binary_phase_synchrony_from_counts_uses_raw_bitstrings() -> None:
    assert binary_phase_synchrony_from_counts({"000": 10}, n_qubits=3) == pytest.approx(1.0)
    assert binary_phase_synchrony_from_counts({"001": 10}, n_qubits=3) == pytest.approx(1.0 / 3.0)
    assert binary_phase_synchrony_from_counts({"000": 5, "001": 5}, n_qubits=3) == pytest.approx(
        2.0 / 3.0
    )


def test_raw_count_package_from_feedback_results_preserves_jobs_and_records() -> None:
    feedback = FeedbackResult(
        job_id="job-feedback",
        qpu_seconds=4.0,
        metadata={
            "arm": S1_FEEDBACK_ARM,
            "records": [
                {"counts": {"000": 10}, "source_index": 0},
                {"counts": {"001": 10}, "source_index": 1},
            ],
        },
    )
    control = FeedbackResult(
        job_id="job-control",
        qpu_seconds=4.0,
        metadata={"arm": S1_CONTROL_ARM, "records": [{"counts": {"011": 10}, "source_index": 0}]},
    )

    package = raw_count_package_from_feedback_results(
        experiment_id="s1",
        target_r=0.72,
        n_qubits=3,
        feedback_result=feedback,
        control_result=control,
    )

    assert package["job_ids"] == ["job-feedback", "job-control"]
    assert package["arms"][0]["label"] == S1_FEEDBACK_ARM
    assert package["arms"][0]["records"][0]["r_live"] == pytest.approx(1.0)
    assert package["arms"][0]["records"][1]["r_live"] == pytest.approx(1.0 / 3.0)
    assert package["arms"][1]["records"][0]["counts"] == {"011": 10}


def test_raw_count_package_rejects_missing_job_or_records() -> None:
    feedback = FeedbackResult(job_id=None, metadata={"arm": S1_FEEDBACK_ARM, "records": []})
    control = FeedbackResult(job_id="job-control", metadata={"arm": S1_CONTROL_ARM, "records": []})

    with pytest.raises(ValueError, match="job_id"):
        raw_count_package_from_feedback_results(
            experiment_id="s1",
            target_r=0.72,
            n_qubits=3,
            feedback_result=feedback,
            control_result=control,
        )


def test_run_ibm_sampler_arm_preserves_per_repetition_counts() -> None:
    arm = build_s1_feedback_arm_circuits(_controller(), n_rounds=1, shots=128, repetitions=2)[0]

    class _Register:
        def __init__(self, counts: dict[str, int]) -> None:
            self._counts = counts

        def get_counts(self) -> dict[str, int]:
            return self._counts

    class _PubResult:
        def __init__(self, counts: dict[str, int]) -> None:
            self.data = type("Data", (), {"readout": _Register(counts)})()

    class _Job:
        def job_id(self) -> str:
            return "job-feedback"

        def result(self, timeout: float):
            assert timeout == 30.0
            return [_PubResult({"000": 128}), _PubResult({"001": 128})]

    class _Sampler:
        def __init__(self, mode) -> None:
            self.mode = mode
            self.options = type("Options", (), {})()

        def run(self, circuits):
            assert len(circuits) == 2
            assert self.options.default_shots == 128
            return _Job()

    result = run_ibm_sampler_arm(
        backend=object(),
        arm=arm,
        isa_circuits=[arm.circuit, arm.circuit],
        timeout_s=30.0,
        sampler_cls=_Sampler,
    )

    assert result.job_id == "job-feedback"
    assert result.metadata["arm"] == S1_FEEDBACK_ARM
    assert result.metadata["records"] == [
        {"source_index": 0, "counts": {"000": 128}},
        {"source_index": 1, "counts": {"001": 128}},
    ]
    assert result.qpu_seconds > 0.0


def test_build_s1_arm_command_carries_approval_budget_payload() -> None:
    arm = build_s1_feedback_arm_circuits(_controller(), n_rounds=1, shots=128, repetitions=2)[1]

    command = build_s1_arm_command(arm, isa_circuits=[arm.circuit, arm.circuit], timeout_s=45.0)

    assert command.label == S1_CONTROL_ARM
    assert command.estimated_qpu_seconds == pytest.approx(arm.estimated_qpu_seconds)
    assert command.payload["shots"] == 128
    assert command.payload["timeout_s"] == 45.0


def test_build_s1_xy_observable_arm_circuits_preserves_dynamic_body() -> None:
    arms = build_s1_xy_observable_arm_circuits(
        _controller(),
        observables=("XXI", "YYI", "IXX", "IYY"),
        n_rounds=3,
        shots=1024,
        repetitions=2,
    )

    assert len(arms) == 8
    assert {(arm.label, arm.observable) for arm in arms} == {
        (S1_FEEDBACK_ARM, "XXI"),
        (S1_FEEDBACK_ARM, "YYI"),
        (S1_FEEDBACK_ARM, "IXX"),
        (S1_FEEDBACK_ARM, "IYY"),
        (S1_CONTROL_ARM, "XXI"),
        (S1_CONTROL_ARM, "YYI"),
        (S1_CONTROL_ARM, "IXX"),
        (S1_CONTROL_ARM, "IYY"),
    }
    feedback_xx = next(
        arm for arm in arms if arm.label == S1_FEEDBACK_ARM and arm.observable == "XXI"
    )
    control_yy = next(
        arm for arm in arms if arm.label == S1_CONTROL_ARM and arm.observable == "YYI"
    )
    assert feedback_xx.circuit.count_ops()["if_else"] == 3
    assert "if_else" not in control_yy.circuit.count_ops()
    assert feedback_xx.circuit.count_ops()["h"] >= 2
    assert control_yy.circuit.count_ops()["sdg"] >= 2


def test_pauli_expectation_from_counts_reduces_selected_non_identity_bits() -> None:
    assert pauli_expectation_from_counts(
        {"000": 10}, observable="XXI", n_qubits=3
    ) == pytest.approx(1.0)
    assert pauli_expectation_from_counts(
        {"010": 10}, observable="XXI", n_qubits=3
    ) == pytest.approx(-1.0)
    assert pauli_expectation_from_counts(
        {"010": 5, "000": 5}, observable="XXI", n_qubits=3
    ) == pytest.approx(0.0)


def test_raw_count_package_from_xy_observable_results_groups_by_observable() -> None:
    feedback_xx = FeedbackResult(
        job_id="job-feedback-xx",
        metadata={
            "arm": S1_FEEDBACK_ARM,
            "observable": "XXI",
            "records": [{"counts": {"000": 10}, "source_index": 0}],
        },
    )
    control_xx = FeedbackResult(
        job_id="job-control-xx",
        metadata={
            "arm": S1_CONTROL_ARM,
            "observable": "XXI",
            "records": [{"counts": {"010": 10}, "source_index": 0}],
        },
    )

    package = raw_count_package_from_xy_observable_results(
        experiment_id="s1b",
        n_qubits=3,
        results=[feedback_xx, control_xx],
    )

    assert package["experiment_id"] == "s1b"
    assert package["observable_family"] == "direct_xy_pauli_correlators"
    assert package["job_ids"] == ["job-feedback-xx", "job-control-xx"]
    assert package["observables"][0]["basis"] == "XXI"
    assert package["observables"][0]["arms"][0]["mean_expectation"] == pytest.approx(1.0)
    assert package["observables"][0]["arms"][1]["mean_expectation"] == pytest.approx(-1.0)
    assert package["observables"][0]["feedback_minus_control"] == pytest.approx(2.0)
