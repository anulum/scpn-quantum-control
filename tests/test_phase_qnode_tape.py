# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for Phase QNode Tape
"""Tests for phase/qnode_tape.py QNode-style differentiable tape evidence."""

from __future__ import annotations

from dataclasses import replace
from typing import cast

import numpy as np
import pytest
from numpy.typing import NDArray

from scpn_quantum_control.differentiable import Parameter, multi_frequency_parameter_shift_rule
from scpn_quantum_control.differentiable_result_contracts import ParameterShiftSampleRecord
from scpn_quantum_control.phase import (
    PhaseQNodeTape,
    PhaseQNodeTapeRecord,
    phase_qnode_tape,
    run_phase_qnode_tape_readiness_suite,
)

FloatArray = NDArray[np.float64]
SAMPLE_PROVENANCE = {
    "sample_seed": "phase-qnode-tape-test-seed",
    "shot_batch_id": "phase-qnode-tape-test-batch",
    "source_class": "caller_supplied",
}


def test_phase_qnode_tape_records_deterministic_parameter_shift() -> None:
    """Deterministic QNode tape records should preserve parameter-shift metadata."""

    def energy(params: FloatArray) -> float:
        return float(np.cos(params[0]) + 0.25 * np.sin(params[1]))

    params = np.array([0.2, -0.4], dtype=float)

    with phase_qnode_tape(
        qnode_name="kuramoto_xy_vqe",
        observable="energy",
        backend="statevector",
    ) as tape:
        record = tape.record_parameter_shift("energy_expectation", energy, params)

    expected = np.array([-np.sin(params[0]), 0.25 * np.cos(params[1])], dtype=float)
    assert isinstance(record, PhaseQNodeTapeRecord)
    np.testing.assert_allclose(record.gradient, expected, atol=1e-12)
    assert record.qnode_name == "kuramoto_xy_vqe"
    assert record.objective_name == "energy_expectation"
    assert record.kind == "deterministic"
    assert record.supported
    assert not record.fail_closed
    assert record.hardware_execution is False
    assert record.parameter_shift_evaluations == 4
    assert record.total_shots is None
    assert "not hardware" in record.claim_boundary
    payload = record.to_dict()
    assert payload["qnode_name"] == "kuramoto_xy_vqe"
    assert payload["backend"] == "statevector_simulator"
    assert payload["gradient"] == pytest.approx(expected.tolist())
    assert tape.to_dict()["record_count"] == 1


def test_phase_qnode_tape_records_multi_term_finite_shot_replay() -> None:
    """Finite-shot QNode tape records should expose shifted-sample provenance."""

    rule = multi_frequency_parameter_shift_rule([1.0, 2.0])
    plus_values = np.array([[1.2, -0.3], [0.9, 0.4]], dtype=float)
    minus_values = np.array([[0.8, -0.7], [0.5, -0.2]], dtype=float)
    plus_variances = np.array([[0.04, 0.09], [0.08, 0.05]], dtype=float)
    minus_variances = np.array([[0.05, 0.07], [0.06, 0.03]], dtype=float)

    with phase_qnode_tape(
        qnode_name="bounded_phase_qnn",
        observable="mean_squared_error",
        backend="finite_shot_simulator",
        shots=1024,
        seed=31,
    ) as tape:
        record = tape.record_finite_shot_parameter_shift(
            "finite_shot_loss",
            plus_values=plus_values,
            minus_values=minus_values,
            plus_variances=plus_variances,
            minus_variances=minus_variances,
            sample_provenance=SAMPLE_PROVENANCE,
            parameters=[Parameter("theta"), Parameter("frozen", trainable=False)],
            rule=rule,
            value=0.375,
        )

    expected = np.zeros(plus_values.shape[1], dtype=float)
    for term_index, (_, coefficient) in enumerate(rule.terms):
        expected[0] += coefficient * (plus_values[term_index, 0] - minus_values[term_index, 0])

    np.testing.assert_allclose(record.gradient, expected, atol=1e-12)
    assert record.kind == "finite_shot"
    assert record.supported
    assert record.shot_count == 1024
    assert record.sample_records[0].sample_seed == SAMPLE_PROVENANCE["sample_seed"]
    assert record.sample_records[0].shot_batch_id == SAMPLE_PROVENANCE["shot_batch_id"]
    assert record.sample_records[0].source_class == SAMPLE_PROVENANCE["source_class"]
    assert record.seed == 31
    assert record.parameter_shift_evaluations == 8
    assert record.total_shots == 8192
    assert record.standard_error is not None
    assert record.standard_error[0] > 0.0
    assert record.standard_error[1] == pytest.approx(0.0)
    assert record.confidence_radius is not None
    assert record.confidence_radius[0] > 0.0
    assert record.confidence_radius[1] == pytest.approx(0.0)
    assert record.sample_record_count == len(rule.terms) * plus_values.shape[1]
    assert {sample.parameter_name for sample in record.sample_records} == {"theta", "frozen"}
    assert all(
        sample.trainable == (sample.parameter_name == "theta") for sample in record.sample_records
    )
    assert all(
        sample.gradient_contribution == 0.0
        for sample in record.sample_records
        if sample.parameter_index == 1
    )
    payload = record.to_dict()
    sample_records = payload["sample_records"]
    assert payload["sample_record_count"] == len(rule.terms) * plus_values.shape[1]
    assert payload["total_shots"] == 8192
    assert isinstance(sample_records, list)
    assert sample_records[0]["plus_shots"] == 1024
    assert sample_records[0]["minus_shots"] == 1024
    assert sample_records[0]["term_index"] == 0
    assert sample_records[-1]["parameter_index"] == 1


def test_phase_qnode_tape_records_fail_closed_provider_boundary() -> None:
    """Provider-boundary QNode tape records should fail closed before submission."""

    with phase_qnode_tape(
        qnode_name="hardware_vqe_candidate",
        observable="energy",
        backend="hardware",
        shots=4096,
        seed=7,
    ) as tape:
        record = tape.record_provider_boundary(
            "hardware_gradient",
            provider="ibm_quantum",
            requested_job_id="blocked-before-submit",
        )

    assert record.kind == "provider_boundary"
    assert not record.supported
    assert record.fail_closed
    assert record.hardware_execution is False
    assert record.provider == "ibm_quantum"
    assert record.requested_job_id == "blocked-before-submit"
    assert record.gradient.size == 0
    assert record.parameter_shift_evaluations == 0
    assert record.total_shots == 0
    assert record.plan.requires_hardware_approval
    assert "hardware gradient execution requires" in record.failure_reason
    payload = tape.to_dict()
    record_payload = record.to_dict()
    assert payload["fail_closed_count"] == 1
    assert record_payload["requested_job_id"] == "blocked-before-submit"


def test_phase_qnode_tape_readiness_suite_reports_supported_and_blocked_routes() -> None:
    """The readiness suite should aggregate supported local and blocked provider routes."""

    suite = run_phase_qnode_tape_readiness_suite()

    assert suite.passed
    assert suite.record_count == 3
    assert suite.supported_count == 2
    assert suite.fail_closed_count == 1
    assert suite.total_parameter_shift_evaluations > 0
    assert suite.total_shots > 0
    assert not suite.hardware_execution
    assert {record.kind for record in suite.records} == {
        "deterministic",
        "finite_shot",
        "provider_boundary",
    }
    assert suite.to_dict()["passed"] is True


def test_phase_qnode_tape_fails_closed_on_invalid_inputs() -> None:
    """QNode tape construction and finite-shot records should reject invalid inputs."""

    with pytest.raises(ValueError, match="qnode_name"):
        PhaseQNodeTape(qnode_name="", observable="energy")

    tape = PhaseQNodeTape(qnode_name="outside", observable="energy")
    with pytest.raises(RuntimeError, match="active context"):
        tape.record_parameter_shift(
            "outside",
            lambda params: float(np.cos(params[0])),
            np.array([0.1], dtype=float),
        )

    with (
        phase_qnode_tape(qnode_name="bad_finite", observable="energy", shots=128) as active,
        pytest.raises(ValueError, match="plus_values"),
    ):
        active.record_finite_shot_parameter_shift(
            "bad",
            plus_values=np.array([1.0, 2.0], dtype=float),
            minus_values=np.array([0.5], dtype=float),
            plus_variances=np.array([0.1, 0.1], dtype=float),
            minus_variances=np.array([0.1], dtype=float),
        )


def test_phase_qnode_tape_record_rejects_inconsistent_metadata() -> None:
    """Record validation should reject inconsistent uncertainty and sample metadata."""
    suite = run_phase_qnode_tape_readiness_suite()
    deterministic = suite.records[0]
    finite_shot = suite.records[1]

    with pytest.raises(ValueError, match="standard_error must match gradient shape"):
        replace(deterministic, standard_error=np.zeros(1, dtype=float))
    with pytest.raises(ValueError, match="confidence_radius must match gradient shape"):
        replace(deterministic, confidence_radius=np.zeros(1, dtype=float))
    with pytest.raises(ValueError, match="shot_count must be positive"):
        replace(deterministic, shot_count=0)
    with pytest.raises(ValueError, match="seed must be non-negative"):
        replace(deterministic, seed=-1)

    invalid_samples = cast(tuple[ParameterShiftSampleRecord, ...], ("not-a-sample",))
    with pytest.raises(ValueError, match="must contain ParameterShiftSampleRecord"):
        replace(deterministic, sample_records=invalid_samples)

    out_of_range = replace(
        finite_shot.sample_records[0],
        parameter_index=deterministic.gradient.size,
    )
    with pytest.raises(ValueError, match="parameter_index must fit"):
        replace(deterministic, sample_records=(out_of_range,))
    with pytest.raises(ValueError, match="must include sample_records"):
        replace(finite_shot, sample_records=())
    with pytest.raises(ValueError, match="only valid for finite-shot"):
        replace(deterministic, sample_records=(finite_shot.sample_records[0],))


def test_phase_qnode_tape_context_reentry_clear_and_supported_provider_boundary() -> None:
    """Tape lifecycle guards should preserve a fail-closed local provider boundary."""
    tape = PhaseQNodeTape(
        qnode_name="local_provider_candidate",
        observable="energy",
        backend="statevector",
    )

    with tape:
        with pytest.raises(RuntimeError, match="already active"):
            tape.__enter__()
        record = tape.record_provider_boundary(
            "provider_boundary",
            provider="local_simulator",
        )
        assert record.plan.supported
        assert not record.supported
        assert record.failure_reason == (
            "provider execution not submitted by QNode tape readiness record"
        )
        assert tape.records == (record,)
        tape.clear()
        assert len(tape.records) == 0


def test_phase_qnode_tape_rejects_invalid_constructor_controls() -> None:
    """Tape construction should reject invalid shot, seed, and confidence controls."""
    with pytest.raises(ValueError, match="shots must be a positive integer"):
        PhaseQNodeTape(qnode_name="invalid", observable="energy", shots=True)
    with pytest.raises(ValueError, match="seed must be a non-negative integer"):
        PhaseQNodeTape(qnode_name="invalid", observable="energy", seed=-1)
    with pytest.raises(ValueError, match="confidence_level must be between"):
        PhaseQNodeTape(qnode_name="invalid", observable="energy", confidence_level=np.nan)


def test_phase_qnode_tape_record_rejects_nonfinite_and_nonscalar_vectors() -> None:
    """Record construction should reject non-finite scalars and malformed vectors."""
    record = run_phase_qnode_tape_readiness_suite().records[0]

    with pytest.raises(ValueError, match="value must be finite"):
        replace(record, value=np.inf)
    with pytest.raises(ValueError, match="gradient must be a one-dimensional array"):
        replace(record, gradient=np.zeros((1, 2), dtype=float))
    with pytest.raises(ValueError, match="gradient must contain only finite values"):
        replace(record, gradient=np.array([np.nan, 0.0], dtype=float))
