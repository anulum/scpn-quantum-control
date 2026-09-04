# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Differentiable Gradient Audit Edge Tests
"""Exercise public differentiable-audit validation and framework-result edges."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import pytest

import scpn_quantum_control.phase.differentiable_audit as audit_module
from scpn_quantum_control.differentiable import (
    ShotAllocationResult,
    multi_frequency_parameter_shift_rule,
)
from scpn_quantum_control.phase import (
    run_finite_shot_gradient_uncertainty_audit,
    run_ml_framework_gradient_audit,
    run_parameter_shift_audit_suite,
    verify_parameter_shift_analytic_gradient,
)


@dataclass(frozen=True)
class _AdapterResult:
    value: float
    gradient: np.ndarray[Any, np.dtype[np.float64]]


def _objective(values: np.ndarray[Any, np.dtype[np.float64]]) -> float:
    return float(np.mean(1.0 - np.cos(values)))


def _analytic_gradient(
    values: np.ndarray[Any, np.dtype[np.float64]],
) -> np.ndarray[Any, np.dtype[np.float64]]:
    return cast(
        np.ndarray[Any, np.dtype[np.float64]],
        (np.sin(values) / values.size).astype(np.float64, copy=False),
    )


def _adapter(
    objective: audit_module.ScalarObjective,
    values: np.ndarray[Any, np.dtype[np.float64]],
    **_kwargs: object,
) -> _AdapterResult:
    params = np.asarray(values, dtype=np.float64)
    return _AdapterResult(
        value=float(objective(params)),
        gradient=_analytic_gradient(params),
    )


def _set_framework_availability(
    monkeypatch: pytest.MonkeyPatch,
    *,
    jax: bool = False,
    torch: bool = False,
    tensorflow: bool = False,
    pennylane: bool = False,
) -> None:
    monkeypatch.setattr(audit_module, "is_phase_jax_available", lambda: jax)
    monkeypatch.setattr(audit_module, "is_phase_torch_available", lambda: torch)
    monkeypatch.setattr(audit_module, "is_phase_tensorflow_available", lambda: tensorflow)
    monkeypatch.setattr(audit_module, "is_phase_pennylane_available", lambda: pennylane)


def test_public_audit_entry_points_reject_remaining_threshold_edges() -> None:
    """Reject negative tolerances and non-positive uncertainty targets."""
    values = np.array([0.2], dtype=np.float64)

    with pytest.raises(ValueError, match="target_value_tolerance"):
        run_parameter_shift_audit_suite(
            _objective,
            _analytic_gradient,
            values,
            target_value_tolerance=-1.0,
        )
    with pytest.raises(ValueError, match="tolerance"):
        verify_parameter_shift_analytic_gradient(
            _objective,
            _analytic_gradient,
            values,
            tolerance=-1.0,
        )
    with pytest.raises(ValueError, match="tolerance"):
        run_ml_framework_gradient_audit(tolerance=-1.0)
    with pytest.raises(ValueError, match="target_standard_error"):
        run_finite_shot_gradient_uncertainty_audit(
            _objective,
            values,
            target_standard_error=0.0,
        )


def test_parameter_shift_audit_accepts_an_unconstrained_target() -> None:
    """Accept a caller that intentionally omits an optimisation target."""
    report = run_parameter_shift_audit_suite(
        _objective,
        _analytic_gradient,
        np.array([0.3, -0.2], dtype=np.float64),
        target_value=None,
        max_steps=4,
    )

    assert report.training_certificate.within_target_value_tolerance is None
    assert report.training.best_value <= report.training.initial_value


def test_finite_shot_audit_accepts_scalar_and_multiterm_variances() -> None:
    """Accept scalar variance broadcasting and multi-term matrix allocation."""
    values = np.array([0.3, -0.2], dtype=np.float64)
    scalar = run_finite_shot_gradient_uncertainty_audit(
        _objective,
        values,
        plus_variances=0.04,
        minus_variances=0.03,
    )
    rule = multi_frequency_parameter_shift_rule([1.0, 2.0])
    matrix = run_finite_shot_gradient_uncertainty_audit(
        _objective,
        values,
        rule=rule,
        plus_variances=np.array([[0.04, 0.03], [0.02, 0.01]], dtype=np.float64),
        minus_variances=np.array([[0.03, 0.02], [0.01, 0.04]], dtype=np.float64),
    )

    assert scalar.passed
    assert scalar.shot_allocation.shots.shape == (2, 2)
    assert matrix.passed
    assert matrix.shot_allocation.shots.shape == (2, 2, 2)


@pytest.mark.parametrize(
    ("variances", "message"),
    (
        (np.array([0.04], dtype=np.float64), "width must match"),
        (np.array([[0.04], [0.03]], dtype=np.float64), "shape must be"),
        (
            np.array([[0.04, np.nan], [0.03, 0.02]], dtype=np.float64),
            "finite real numeric values",
        ),
        (np.ones((2, 2, 1), dtype=np.float64), "scalar, vector, or shift-term matrix"),
    ),
)
def test_finite_shot_audit_rejects_remaining_variance_shapes(
    variances: np.ndarray[Any, np.dtype[np.float64]],
    message: str,
) -> None:
    """Reject wrong-width, malformed, non-finite and rank-three variances."""
    rule = multi_frequency_parameter_shift_rule([1.0, 2.0])

    with pytest.raises(ValueError, match=message):
        run_finite_shot_gradient_uncertainty_audit(
            _objective,
            np.array([0.3, -0.2], dtype=np.float64),
            rule=rule,
            plus_variances=variances,
        )


def test_finite_shot_audit_rejects_a_malformed_allocation_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fail closed when an injected allocation violates the upstream contract."""

    @dataclass(frozen=True)
    class _MalformedAllocation:
        shots: np.ndarray[Any, np.dtype[np.float64]]

    def malformed_plan(*_args: object, **_kwargs: object) -> ShotAllocationResult:
        malformed = _MalformedAllocation(np.array([64.0], dtype=np.float64))
        return cast(ShotAllocationResult, cast(object, malformed))

    monkeypatch.setattr(audit_module, "plan_parameter_shift_shots", malformed_plan)

    with pytest.raises(ValueError, match="shot allocation shape"):
        run_finite_shot_gradient_uncertainty_audit(
            _objective,
            np.array([0.2], dtype=np.float64),
        )


def test_ml_audit_executes_torch_and_tensorflow_adapters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Execute both available framework adapters through the public audit."""
    _set_framework_availability(monkeypatch, torch=True, tensorflow=True)
    monkeypatch.setattr(audit_module, "torch_parameter_shift_value_and_grad", _adapter)
    monkeypatch.setattr(audit_module, "tensorflow_parameter_shift_value_and_grad", _adapter)

    suite = run_ml_framework_gradient_audit(initial_values=np.array([0.3, -0.2]))

    assert suite.audit_passed
    assert suite.executed_frameworks == ("torch", "tensorflow")
    assert suite.unavailable_frameworks == ("jax", "pennylane")


def test_ml_audit_records_blocked_and_executed_pennylane_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Distinguish absent caller QNode evidence from executed PennyLane parity."""
    _set_framework_availability(monkeypatch, pennylane=True)
    blocked = run_ml_framework_gradient_audit()
    executed = run_ml_framework_gradient_audit(pennylane_gradient=_analytic_gradient)

    assert blocked.blocked_frameworks == ("pennylane",)
    assert blocked.audit_passed
    assert executed.executed_frameworks == ("pennylane",)
    assert executed.audit_passed


def test_ml_audit_rejects_shape_drift_and_records_numeric_disagreement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject adapter shape drift and retain a same-shape failed parity record."""
    _set_framework_availability(monkeypatch, jax=True)

    def wrong_shape_adapter(
        objective: audit_module.ScalarObjective,
        values: np.ndarray[Any, np.dtype[np.float64]],
        **_kwargs: object,
    ) -> _AdapterResult:
        return _AdapterResult(float(objective(values)), np.array([0.0], dtype=np.float64))

    monkeypatch.setattr(audit_module, "jax_parameter_shift_value_and_grad", wrong_shape_adapter)
    with pytest.raises(ValueError, match="gradient shape must match reference"):
        run_ml_framework_gradient_audit(initial_values=np.array([0.3, -0.2]))

    def disagreeing_adapter(
        objective: audit_module.ScalarObjective,
        values: np.ndarray[Any, np.dtype[np.float64]],
        **_kwargs: object,
    ) -> _AdapterResult:
        return _AdapterResult(float(objective(values)), np.ones_like(values, dtype=np.float64))

    monkeypatch.setattr(audit_module, "jax_parameter_shift_value_and_grad", disagreeing_adapter)
    suite = run_ml_framework_gradient_audit(initial_values=np.array([0.3, -0.2]))

    assert not suite.audit_passed
    assert suite.failed_frameworks == ("jax",)
