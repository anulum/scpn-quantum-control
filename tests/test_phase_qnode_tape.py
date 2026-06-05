# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for Phase QNode Tape
"""Tests for phase/qnode_tape.py QNode-style differentiable tape evidence."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_quantum_control.differentiable import multi_frequency_parameter_shift_rule
from scpn_quantum_control.phase import (
    PhaseQNodeTape,
    PhaseQNodeTapeRecord,
    phase_qnode_tape,
    run_phase_qnode_tape_readiness_suite,
)


def test_phase_qnode_tape_records_deterministic_parameter_shift() -> None:
    def energy(params: np.ndarray) -> float:
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
            rule=rule,
            value=0.375,
        )

    expected = np.zeros(plus_values.shape[1], dtype=float)
    for term_index, (_, coefficient) in enumerate(rule.terms):
        expected += coefficient * (plus_values[term_index] - minus_values[term_index])

    np.testing.assert_allclose(record.gradient, expected, atol=1e-12)
    assert record.kind == "finite_shot"
    assert record.supported
    assert record.shot_count == 1024
    assert record.seed == 31
    assert record.parameter_shift_evaluations == 8
    assert record.total_shots == 8192
    assert record.standard_error is not None
    assert np.all(record.standard_error > 0.0)
    assert record.confidence_radius is not None
    assert np.all(record.confidence_radius > 0.0)
    assert record.to_dict()["total_shots"] == 8192


def test_phase_qnode_tape_records_fail_closed_provider_boundary() -> None:
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
