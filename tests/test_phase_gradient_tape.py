# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for Phase Gradient Tape
"""Tests for phase/gradient_tape.py quantum-gradient tape semantics."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from scpn_quantum_control.differentiable import multi_frequency_parameter_shift_rule
from scpn_quantum_control.phase import QuantumGradientTape, gradient_tape

FloatArray = NDArray[np.float64]
SAMPLE_PROVENANCE = {
    "sample_seed": "phase-gradient-tape-test-seed",
    "shot_batch_id": "phase-gradient-tape-test-batch",
    "source_class": "caller_supplied",
}


def test_gradient_tape_records_deterministic_parameter_shift() -> None:
    def objective(params: FloatArray) -> float:
        return float(np.cos(params[0]) + 0.25 * np.sin(params[1]))

    params = np.array([0.2, -0.4], dtype=float)

    with gradient_tape(backend="statevector") as tape:
        record = tape.record_parameter_shift("xy_expectation", objective, params)

    expected = np.array([-np.sin(params[0]), 0.25 * np.cos(params[1])], dtype=float)
    np.testing.assert_allclose(record.gradient, expected, atol=1e-12)
    assert record.name == "xy_expectation"
    assert record.kind == "deterministic"
    assert record.plan.method == "parameter_shift"
    assert record.evaluations == 4
    assert len(tape.records) == 1
    assert tape.records[0] is record


def test_gradient_tape_records_multi_frequency_parameter_shift() -> None:
    rule = multi_frequency_parameter_shift_rule([1.0, 2.0])

    def objective(params: FloatArray) -> float:
        return float(np.sin(params[0]) + 0.1 * np.cos(2.0 * params[0]) + 0.25 * np.sin(params[1]))

    params = np.array([0.31, -0.17], dtype=float)

    with gradient_tape(backend="statevector") as tape:
        record = tape.record_parameter_shift(
            "multi_frequency_xy",
            objective,
            params,
            rule=rule,
        )

    expected = np.array(
        [
            np.cos(params[0]) - 0.2 * np.sin(2.0 * params[0]),
            0.25 * np.cos(params[1]),
        ],
        dtype=float,
    )
    np.testing.assert_allclose(record.gradient, expected, atol=1e-12)
    assert record.method == "multi_frequency_parameter_shift"
    assert record.shift_terms == len(rule.terms)
    assert record.evaluations == 2 * len(rule.terms) * params.size
    assert record.to_dict()["shift_terms"] == len(rule.terms)


def test_gradient_tape_records_finite_shot_uncertainty() -> None:
    with gradient_tape(backend="finite_shot_simulator", shots=512) as tape:
        record = tape.record_finite_shot_parameter_shift(
            "finite_shot_xy",
            plus_values=np.array([1.2, -0.3], dtype=float),
            minus_values=np.array([0.8, -0.7], dtype=float),
            plus_variances=np.array([0.04, 0.09], dtype=float),
            minus_variances=np.array([0.04, 0.09], dtype=float),
            sample_provenance=SAMPLE_PROVENANCE,
            value=0.5,
        )

    np.testing.assert_allclose(record.gradient, np.array([0.2, 0.2], dtype=float))
    assert record.kind == "stochastic"
    assert record.plan.method == "stochastic_parameter_shift"
    assert record.plan.shots == 512
    assert record.standard_error is not None
    assert np.all(record.standard_error > 0.0)


def test_gradient_tape_records_multi_term_finite_shot_uncertainty() -> None:
    rule = multi_frequency_parameter_shift_rule([1.0, 2.0])
    plus_values = np.array([[1.2, -0.3], [0.9, 0.4]], dtype=float)
    minus_values = np.array([[0.8, -0.7], [0.5, -0.2]], dtype=float)
    plus_variances = np.array([[0.04, 0.09], [0.08, 0.05]], dtype=float)
    minus_variances = np.array([[0.05, 0.07], [0.06, 0.03]], dtype=float)

    with gradient_tape(backend="finite_shot_simulator", shots=512) as tape:
        record = tape.record_finite_shot_parameter_shift(
            "multi_term_finite_shot_xy",
            plus_values=plus_values,
            minus_values=minus_values,
            plus_variances=plus_variances,
            minus_variances=minus_variances,
            sample_provenance=SAMPLE_PROVENANCE,
            rule=rule,
            value=0.5,
        )

    expected = np.zeros(plus_values.shape[1], dtype=float)
    for term_index, (_, coefficient) in enumerate(rule.terms):
        expected += coefficient * (plus_values[term_index] - minus_values[term_index])

    np.testing.assert_allclose(record.gradient, expected, atol=1e-12)
    assert record.kind == "stochastic"
    assert record.method == "multi_frequency_parameter_shift_shot_noise"
    assert record.shift_terms == len(rule.terms)
    assert record.evaluations == 2 * len(rule.terms) * plus_values.shape[1]
    assert record.standard_error is not None
    assert np.all(record.standard_error > 0.0)


def test_gradient_tape_rejects_flat_multi_term_finite_shot_records() -> None:
    rule = multi_frequency_parameter_shift_rule([1.0, 2.0])

    with (
        gradient_tape(backend="finite_shot_simulator", shots=512) as tape,
        pytest.raises(ValueError, match="two-dimensional plus_values"),
    ):
        tape.record_finite_shot_parameter_shift(
            "invalid_multi_term",
            plus_values=np.array([1.0, 0.5], dtype=float),
            minus_values=np.array([0.8, 0.2], dtype=float),
            plus_variances=np.array([0.04, 0.04], dtype=float),
            minus_variances=np.array([0.04, 0.04], dtype=float),
            rule=rule,
        )


def test_gradient_tape_rejects_finite_shot_records_without_sample_provenance() -> None:
    with (
        gradient_tape(backend="finite_shot_simulator", shots=512) as tape,
        pytest.raises(ValueError, match="sample provenance"),
    ):
        tape.record_finite_shot_parameter_shift(
            "missing_provenance",
            plus_values=np.array([1.0], dtype=float),
            minus_values=np.array([0.5], dtype=float),
            plus_variances=np.array([0.1], dtype=float),
            minus_variances=np.array([0.1], dtype=float),
        )


def test_gradient_tape_fails_closed_for_hardware_without_policy() -> None:
    with (
        gradient_tape(backend="hardware", shots=256) as tape,
        pytest.raises(ValueError, match="hardware gradient execution requires"),
    ):
        tape.record_finite_shot_parameter_shift(
            "hardware_blocked",
            plus_values=np.array([1.0], dtype=float),
            minus_values=np.array([0.5], dtype=float),
            plus_variances=np.array([0.1], dtype=float),
            minus_variances=np.array([0.1], dtype=float),
        )


def test_gradient_tape_rejects_recording_outside_context() -> None:
    tape = QuantumGradientTape(backend="statevector")

    with pytest.raises(RuntimeError, match="active context"):
        tape.record_parameter_shift(
            "outside",
            lambda params: float(np.cos(params[0])),
            np.array([0.2], dtype=float),
        )


def test_gradient_tape_rejects_empty_record_names() -> None:
    with (
        gradient_tape(backend="statevector") as tape,
        pytest.raises(ValueError, match="record name"),
    ):
        tape.record_parameter_shift(
            "",
            lambda params: float(np.cos(params[0])),
            np.array([0.2], dtype=float),
        )
