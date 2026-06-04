# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for Phase PennyLane Bridge
"""Tests for optional PennyLane gradient-agreement checks."""

from __future__ import annotations

import numpy as np
import pytest

import scpn_quantum_control.phase.pennylane_bridge as pennylane_bridge
from scpn_quantum_control.phase import (
    PennyLaneGradientAgreementResult,
    PennyLaneRoundTripResult,
    check_pennylane_parameter_shift_agreement,
    check_pennylane_qnode_round_trip,
    is_phase_pennylane_available,
    multi_frequency_parameter_shift_rule,
)


def _objective(values: np.ndarray) -> float:
    return float(np.cos(values[0]) + 0.25 * np.sin(values[1]))


def _closed_form_gradient(values: np.ndarray) -> np.ndarray:
    return np.array([-np.sin(values[0]), 0.25 * np.cos(values[1])], dtype=float)


def test_pennylane_bridge_reports_gradient_agreement(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pennylane_bridge, "_load_pennylane", lambda: object())

    result = check_pennylane_parameter_shift_agreement(
        _objective,
        _closed_form_gradient,
        np.array([0.2, -0.4], dtype=float),
        tolerance=1e-12,
    )

    assert isinstance(result, PennyLaneGradientAgreementResult)
    assert is_phase_pennylane_available()
    assert result.passed
    assert result.max_abs_error <= 1e-12
    assert result.evaluations == 5
    np.testing.assert_allclose(result.scpn_gradient, result.pennylane_gradient, atol=1e-12)


def test_pennylane_bridge_reports_gradient_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pennylane_bridge, "_load_pennylane", lambda: object())

    def shifted_gradient(values: np.ndarray) -> np.ndarray:
        return _closed_form_gradient(values) + np.array([0.01, 0.0], dtype=float)

    result = check_pennylane_parameter_shift_agreement(
        _objective,
        shifted_gradient,
        np.array([0.2, -0.4], dtype=float),
        tolerance=1e-4,
    )

    assert not result.passed
    assert result.max_abs_error > result.tolerance
    assert result.l2_error > 0.0


def test_pennylane_bridge_reports_multi_frequency_gradient_agreement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(pennylane_bridge, "_load_pennylane", lambda: object())
    rule = multi_frequency_parameter_shift_rule([1.0, 2.0])

    def objective(values: np.ndarray) -> float:
        return float(np.sin(values[0]) + 0.1 * np.cos(2.0 * values[0]))

    def external_gradient(values: np.ndarray) -> np.ndarray:
        return np.array([np.cos(values[0]) - 0.2 * np.sin(2.0 * values[0])], dtype=float)

    result = check_pennylane_parameter_shift_agreement(
        objective,
        external_gradient,
        np.array([0.4], dtype=float),
        tolerance=1e-12,
        rule=rule,
    )
    payload = result.to_dict()

    assert result.passed
    assert result.method == "multi_frequency_parameter_shift"
    assert result.shift_terms == len(rule.terms)
    assert result.evaluations == 1 + 2 * len(rule.terms)
    assert payload["shift_terms"] == len(rule.terms)
    np.testing.assert_allclose(result.scpn_gradient, result.pennylane_gradient, atol=1e-12)


def test_pennylane_bridge_reports_qnode_round_trip(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pennylane_bridge, "_load_pennylane", lambda: object())

    result = check_pennylane_qnode_round_trip(
        _objective,
        _objective,
        _closed_form_gradient,
        np.array([0.2, -0.4], dtype=float),
        value_tolerance=1e-12,
        gradient_tolerance=1e-12,
    )
    payload = result.to_dict()

    assert isinstance(result, PennyLaneRoundTripResult)
    assert result.passed
    assert result.value_abs_error <= 1e-12
    assert result.gradient_max_abs_error <= 1e-12
    assert result.evaluations == 5
    assert payload["passed"] is True
    assert payload["evaluations"] == 5
    np.testing.assert_allclose(result.scpn_gradient, result.pennylane_gradient, atol=1e-12)


def test_pennylane_bridge_reports_multi_frequency_qnode_round_trip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(pennylane_bridge, "_load_pennylane", lambda: object())
    rule = multi_frequency_parameter_shift_rule([1.0, 2.0])

    def objective(values: np.ndarray) -> float:
        return float(np.sin(values[0]) + 0.1 * np.cos(2.0 * values[0]))

    def external_gradient(values: np.ndarray) -> np.ndarray:
        return np.array([np.cos(values[0]) - 0.2 * np.sin(2.0 * values[0])], dtype=float)

    result = check_pennylane_qnode_round_trip(
        objective,
        objective,
        external_gradient,
        np.array([0.4], dtype=float),
        value_tolerance=1e-12,
        gradient_tolerance=1e-12,
        rule=rule,
    )

    assert result.passed
    assert result.method == "multi_frequency_parameter_shift"
    assert result.shift_terms == len(rule.terms)
    assert result.evaluations == 1 + 2 * len(rule.terms)
    np.testing.assert_allclose(result.scpn_gradient, result.pennylane_gradient, atol=1e-12)


def test_pennylane_bridge_reports_qnode_round_trip_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(pennylane_bridge, "_load_pennylane", lambda: object())

    def shifted_objective(values: np.ndarray) -> float:
        return _objective(values) + 0.01

    result = check_pennylane_qnode_round_trip(
        _objective,
        shifted_objective,
        _closed_form_gradient,
        np.array([0.2, -0.4], dtype=float),
        value_tolerance=1e-4,
        gradient_tolerance=1e-12,
    )

    assert not result.passed
    assert result.value_abs_error > result.value_tolerance
    assert result.gradient_max_abs_error <= result.gradient_tolerance


def test_pennylane_bridge_round_trip_rejects_bad_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(pennylane_bridge, "_load_pennylane", lambda: object())

    def non_finite_objective(values: np.ndarray) -> float:
        return float("nan")

    with pytest.raises(ValueError, match="PennyLane objective"):
        check_pennylane_qnode_round_trip(
            _objective,
            non_finite_objective,
            _closed_form_gradient,
            np.array([0.2, -0.4], dtype=float),
        )


def test_pennylane_bridge_fails_closed_when_pennylane_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unavailable():
        raise ImportError("blocked")

    monkeypatch.setattr(pennylane_bridge, "_load_pennylane", unavailable)

    assert not is_phase_pennylane_available()
    with pytest.raises(ImportError, match="blocked"):
        check_pennylane_parameter_shift_agreement(
            _objective,
            _closed_form_gradient,
            np.array([0.2, -0.4], dtype=float),
        )
    with pytest.raises(ImportError, match="blocked"):
        check_pennylane_qnode_round_trip(
            _objective,
            _objective,
            _closed_form_gradient,
            np.array([0.2, -0.4], dtype=float),
        )
