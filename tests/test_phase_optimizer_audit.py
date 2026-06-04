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
    suite = run_parameter_shift_optimizer_comparison(
        starts=np.array([[0.5, -0.4]], dtype=float),
        max_steps=4,
    )
    payload = suite.to_dict()
    records = suite.records_for_start(0)

    assert suite.passed
    assert len(records) == 2
    assert payload["start_count"] == 1
    assert payload["records"][0]["start_index"] == 0
    assert records[1].metric_source == "array"
    assert records[1].max_metric_condition_number is not None
    assert records[1].max_metric_condition_number > 1.0


def test_optimizer_comparison_custom_identity_metric_is_bounded_baseline() -> None:
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
    with pytest.raises(ValueError, match="starts"):
        run_parameter_shift_optimizer_comparison(starts=np.empty((0, 2), dtype=float))

    with pytest.raises(ValueError, match="learning_rate"):
        run_parameter_shift_optimizer_comparison(learning_rate=0.0)

    with pytest.raises(ValueError, match="comparison_tolerance"):
        run_parameter_shift_optimizer_comparison(comparison_tolerance=-1.0)
