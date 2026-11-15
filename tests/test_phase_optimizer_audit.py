# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for Phase Optimizer Audit
"""Tests for phase/optimizer_audit.py convergence comparison evidence."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_quantum_control.phase import (
    OptimizerComparisonSuiteResult,
    run_parameter_shift_optimizer_comparison,
)


def test_optimizer_comparison_default_audit_passes_and_prefers_metric_route() -> None:
    """The default three-start audit passes and natural gradient is never worse."""
    suite = run_parameter_shift_optimizer_comparison(max_steps=8)

    assert isinstance(suite, OptimizerComparisonSuiteResult)
    assert suite.passed
    assert suite.start_count == 3
    assert suite.optimizers == (
        "parameter_shift_gradient_descent",
        "parameter_shift_natural_gradient_descent",
    )
    assert len(suite.records) == 6
    assert suite.natural_gradient_not_worse_count == suite.start_count
    assert suite.natural_gradient_best_value <= suite.gradient_descent_best_value
    assert suite.best_optimizer == "parameter_shift_natural_gradient_descent"
    assert suite.certificate_failures == ()
    assert "non-isolated functional audit" in suite.claim_boundary


def test_optimizer_comparison_serializes_per_start_records() -> None:
    """Per-start records serialise, and the default metric is conditioned above one."""
    suite = run_parameter_shift_optimizer_comparison(
        starts=np.array([[0.5, -0.4]], dtype=float),
        max_steps=4,
    )
    payload = suite.to_dict()
    records = suite.records_for_start(0)

    assert suite.passed
    assert len(records) == 2
    assert payload["start_count"] == 1
    audit_records = payload["records"]
    assert isinstance(audit_records, list)
    first_record = audit_records[0]
    assert isinstance(first_record, dict)
    assert first_record["start_index"] == 0
    assert records[1].metric_source == "array"
    assert records[1].max_metric_condition_number is not None
    assert records[1].max_metric_condition_number > 1.0


def test_optimizer_comparison_custom_identity_metric_is_bounded_baseline() -> None:
    """A caller's identity metric is a bounded baseline with a caller-supplied claim."""

    def objective(params: np.ndarray) -> float:
        return float(np.sum(1.0 - np.cos(params)))

    suite = run_parameter_shift_optimizer_comparison(
        objective,
        starts=np.array([[0.6, -0.3]], dtype=float),
        metric_tensor=np.eye(2, dtype=float),
        max_steps=5,
        require_natural_not_worse=False,
    )
    natural = suite.records_for_start(0)[1]

    assert suite.passed
    assert natural.metric_source == "array"
    assert natural.best_value <= suite.records_for_start(0)[0].best_value + 1e-8
    assert "caller-supplied metric" in natural.claim_boundary


def test_optimizer_comparison_rejects_invalid_inputs() -> None:
    """Empty starts, a zero learning rate and a negative comparison tolerance are refused."""
    with pytest.raises(ValueError, match="starts"):
        run_parameter_shift_optimizer_comparison(starts=np.empty((0, 2), dtype=float))

    with pytest.raises(ValueError, match="learning_rate"):
        run_parameter_shift_optimizer_comparison(learning_rate=0.0)

    with pytest.raises(ValueError, match="comparison_tolerance"):
        run_parameter_shift_optimizer_comparison(comparison_tolerance=-1.0)


def test_optimizer_comparison_refuses_bad_starts_steps_and_an_orphan_target_tolerance() -> None:
    """Starts must be finite, steps a positive integer, and a target tolerance needs a target."""
    with pytest.raises(ValueError, match="^starts must contain only finite values$"):
        run_parameter_shift_optimizer_comparison(starts=np.array([[0.8, np.nan]]))
    for steps in (0, True):
        with pytest.raises(ValueError, match="^max_steps must be a positive integer$"):
            run_parameter_shift_optimizer_comparison(max_steps=steps)
    with pytest.raises(ValueError, match="^target_value_tolerance requires target_value$"):
        run_parameter_shift_optimizer_comparison(target_value_tolerance=1e-3)


def test_optimizer_comparison_runs_one_start_of_one_phase_against_a_target() -> None:
    """A single one-dimensional start uses unit weights and is certified against its target."""
    suite = run_parameter_shift_optimizer_comparison(
        starts=np.array([0.8]),
        max_steps=40,
        certificate_gradient_tolerance=1e-2,
        certificate_natural_gradient_tolerance=1e-2,
        target_value=0.0,
        target_value_tolerance=1e-3,
    )

    assert suite.start_count == 1
    gradient, natural = suite.records_for_start(0)
    np.testing.assert_allclose(gradient.initial_params, [0.8])
    assert gradient.initial_value == pytest.approx(1.0 - np.cos(0.8))
    assert gradient.certificate_passed and natural.certificate_passed
    assert suite.passed
    assert suite.gradient_descent_best_value <= 1e-3


def test_optimizer_comparison_fails_when_a_poor_metric_slows_natural_gradient() -> None:
    """A metric that shrinks every natural step leaves that route worse and fails the comparison."""

    def objective(params: np.ndarray) -> float:
        """Return a unit-weight phase cost."""
        return float(np.sum(1.0 - np.cos(params)))

    suite = run_parameter_shift_optimizer_comparison(
        objective,
        starts=np.array([[0.6, -0.3]], dtype=float),
        metric_tensor=100.0 * np.eye(2, dtype=float),
        max_steps=5,
    )

    gradient, natural = suite.records_for_start(0)
    assert natural.best_value > gradient.best_value
    assert suite.natural_gradient_not_worse_count == 0
    assert suite.best_optimizer == "parameter_shift_gradient_descent"
    assert not suite.passed


def test_optimizer_comparison_applies_strict_certificate_tolerances() -> None:
    """Certificate tolerances tighter than five steps can reach fail both optimizers."""
    loose = run_parameter_shift_optimizer_comparison(max_steps=5)
    strict_gradient = run_parameter_shift_optimizer_comparison(
        max_steps=5, certificate_gradient_tolerance=1e-12
    )
    strict_natural = run_parameter_shift_optimizer_comparison(
        max_steps=5, certificate_natural_gradient_tolerance=1e-12
    )

    assert all(record.certificate_passed for record in loose.records)
    assert not any(record.certificate_passed for record in strict_gradient.records)
    natural_records = [
        record
        for record in strict_natural.records
        if record.optimizer == "parameter_shift_natural_gradient_descent"
    ]
    assert not any(record.certificate_passed for record in natural_records)


def test_optimizer_comparison_certifies_against_the_target_within_its_tolerance() -> None:
    """One step leaves 0.129 above a zero target: certified within 0.4, refused with no tolerance."""
    within = run_parameter_shift_optimizer_comparison(
        starts=np.array([0.8]), max_steps=1, target_value=0.0, target_value_tolerance=0.4
    )
    exact = run_parameter_shift_optimizer_comparison(
        starts=np.array([0.8]), max_steps=1, target_value=0.0
    )

    assert within.gradient_descent_best_value == pytest.approx(0.128752, abs=1e-6)
    assert all(record.certificate_passed for record in within.records)
    assert not any(record.certificate_passed for record in exact.records)
