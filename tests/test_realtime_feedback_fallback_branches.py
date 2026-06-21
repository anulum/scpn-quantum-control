# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Fallback-branch tests for the real-time feedback loop
"""Validation and native-engine fallback tests for the real-time feedback loop.

Covers the correction-sign config guard, the batch feedback-policy engine
fallback, the negligible-correction short-circuit, and the qiskit XY-expectation
fallback taken when the native expectation kernel is unavailable.
"""

from __future__ import annotations

import sys
import types
from typing import Any

import numpy as np
import pytest
from qiskit.quantum_info import Statevector

from scpn_quantum_control.control import realtime_feedback as rf
from scpn_quantum_control.control.realtime_feedback import (
    RealtimeFeedbackConfig,
    RealtimeSyncFeedbackController,
)
from scpn_quantum_control.phase.xy_kuramoto import QuantumKuramotoSolver

_K = np.array([[0.0, 0.3], [0.3, 0.0]], dtype=np.float64)
_OMEGA = np.array([0.2, 0.5], dtype=np.float64)


def test_config_rejects_invalid_correction_sign() -> None:
    """The feedback correction sign must be exactly -1.0 or 1.0."""
    with pytest.raises(ValueError, match="feedback_correction_sign must be either -1.0 or 1.0"):
        RealtimeFeedbackConfig(feedback_correction_sign=2.0)


def test_feedback_policy_falls_back_on_engine_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """A raising native batch policy falls back to the NumPy policy."""

    def _boom(*_args: Any, **_kwargs: Any) -> None:
        raise ValueError("engine refused the batch policy")

    stub = types.ModuleType("scpn_quantum_engine")
    stub.feedback_policy_batch = _boom  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "scpn_quantum_engine", stub)

    r_values = np.array([0.2, 0.75, 0.95], dtype=np.float64)
    actions, gains, errors = rf._feedback_policy(
        r_values, target_r=0.75, deadband=0.03, base_gain=0.8, max_gain=1.5
    )
    expected = rf.feedback_policy_numpy(
        r_values, target_r=0.75, deadband=0.03, base_gain=0.8, max_gain=1.5
    )
    np.testing.assert_array_equal(actions, expected[0])
    np.testing.assert_allclose(gains, expected[1])
    np.testing.assert_allclose(errors, expected[2])


def test_feedback_policy_uses_native_batch_result(monkeypatch: pytest.MonkeyPatch) -> None:
    """A successful native batch policy is adopted verbatim."""

    def _batch(
        _r: Any, _target: float, _deadband: float, _base: float, _max: float
    ) -> tuple[list[int], list[float], list[float]]:
        return [1], [1.5], [0.55]

    stub = types.ModuleType("scpn_quantum_engine")
    stub.feedback_policy_batch = _batch  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "scpn_quantum_engine", stub)

    actions, gains, errors = rf._feedback_policy(
        np.array([0.2], dtype=np.float64),
        target_r=0.75,
        deadband=0.03,
        base_gain=0.8,
        max_gain=1.5,
    )
    assert actions.tolist() == [1]
    assert gains.tolist() == [1.5]
    assert errors.tolist() == [0.55]


def test_apply_correction_short_circuits_on_negligible_error() -> None:
    """A negligible error produces no correction even for an active action."""
    controller = RealtimeSyncFeedbackController(_K, _OMEGA)
    assert controller._apply_feedback_correction(0.0, "synchronise") == 0.0


def test_xy_expectations_falls_back_to_qiskit(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without the native expectation kernel the qiskit operator path is used."""
    stub = types.ModuleType("scpn_quantum_engine")
    # No ``all_xy_expectations`` attribute -> AttributeError -> qiskit fallback.
    monkeypatch.setitem(sys.modules, "scpn_quantum_engine", stub)

    solver = QuantumKuramotoSolver(2, _K, _OMEGA)
    state = Statevector.from_label("01")
    exp_x, exp_y = rf._xy_expectations(state, 2, solver)

    assert exp_x.shape == (2,)
    assert exp_y.shape == (2,)
    # |01> is a computational-basis state, so all transverse expectations vanish.
    np.testing.assert_allclose(exp_x, np.zeros(2), atol=1e-12)
    np.testing.assert_allclose(exp_y, np.zeros(2), atol=1e-12)
