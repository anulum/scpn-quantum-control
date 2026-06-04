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

from scpn_quantum_control.phase import QuantumGradientTape, gradient_tape


def test_gradient_tape_records_deterministic_parameter_shift() -> None:
    def objective(params: np.ndarray) -> float:
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


def test_gradient_tape_records_finite_shot_uncertainty() -> None:
    with gradient_tape(backend="finite_shot_simulator", shots=512) as tape:
        record = tape.record_finite_shot_parameter_shift(
            "finite_shot_xy",
            plus_values=np.array([1.2, -0.3], dtype=float),
            minus_values=np.array([0.8, -0.7], dtype=float),
            plus_variances=np.array([0.04, 0.09], dtype=float),
            minus_variances=np.array([0.04, 0.09], dtype=float),
            value=0.5,
        )

    np.testing.assert_allclose(record.gradient, np.array([0.2, 0.2], dtype=float))
    assert record.kind == "stochastic"
    assert record.plan.method == "stochastic_parameter_shift"
    assert record.plan.shots == 512
    assert record.standard_error is not None
    assert np.all(record.standard_error > 0.0)


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
