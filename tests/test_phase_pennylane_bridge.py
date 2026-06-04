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
    check_pennylane_parameter_shift_agreement,
    is_phase_pennylane_available,
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
