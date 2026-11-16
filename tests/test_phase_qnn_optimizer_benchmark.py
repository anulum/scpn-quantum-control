# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for Phase QNN Optimizer Benchmark
"""Tests for phase/qnn_optimizer_benchmark.py bounded QNN optimizer evidence."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from scpn_quantum_control.phase import (
    ParameterShiftQNNOptimizerBenchmarkSuiteResult,
    QNNOptimizerBaselineResult,
    run_parameter_shift_qnn_optimizer_benchmark_suite,
)


def test_qnn_optimizer_benchmark_records_non_isolated_functional_evidence() -> None:
    """The default suite records functional, non-isolated evidence for every baseline."""
    suite = run_parameter_shift_qnn_optimizer_benchmark_suite()

    assert isinstance(suite, ParameterShiftQNNOptimizerBenchmarkSuiteResult)
    assert suite.passed
    assert suite.case_count == 2
    assert suite.parameter_shift_not_worse_count == 2
    assert suite.derivative_free_win_count == 0
    assert suite.optimizer_names == (
        "parameter_shift",
        "finite_difference",
        "sgd",
        "adam",
        "lbfgs",
        "natural_gradient",
        "spsa",
        "derivative_free_grid",
    )
    assert suite.optimizer_count == 8
    assert suite.evidence_class == "functional_non_isolated"
    assert not suite.production_benchmark
    assert "not a throughput benchmark" in suite.claim_boundary

    for case in suite.cases:
        assert case.passed
        assert case.optimizer_names == suite.optimizer_names
        assert case.parameter_shift_best_loss <= case.finite_difference_best_loss + 1e-6
        assert case.parameter_shift_best_loss <= case.derivative_free_best_loss
        assert case.parameter_shift_evaluations > 0
        assert case.finite_difference_evaluations > 0
        assert case.derivative_free_evaluations > 0
        assert len(case.optimizer_results) == suite.optimizer_count
        for optimizer in case.optimizer_results:
            assert isinstance(optimizer, QNNOptimizerBaselineResult)
            assert np.isfinite(optimizer.best_loss)
            assert optimizer.evaluations > 0
            assert optimizer.steps >= 0
            assert optimizer.wall_time_seconds >= 0.0
            assert optimizer.evidence_class == "functional_non_isolated"
            assert "not a throughput benchmark" in optimizer.claim_boundary
        assert case.evidence_class == "functional_non_isolated"
        assert case.to_dict()["passed"] is True
        assert case.optimizer_by_name("lbfgs").method == (
            "scipy_l_bfgs_b_with_parameter_shift_jacobian"
        )
        assert case.optimizer_by_name("natural_gradient").method == (
            "parameter_shift_diagonal_fisher_natural_gradient"
        )

    payload = suite.to_dict()
    assert payload["passed"] is True
    assert payload["case_count"] == 2
    assert payload["optimizer_count"] == 8
    assert payload["optimizer_names"] == list(suite.optimizer_names)
    assert payload["production_benchmark"] is False


def test_qnn_optimizer_benchmark_supports_external_case_selection() -> None:
    """A caller can select one case, which then passes and is found by name."""
    suite = run_parameter_shift_qnn_optimizer_benchmark_suite(
        case_names=("phase_separable_single_feature",)
    )

    assert suite.passed
    assert suite.case_count == 1
    assert suite.cases[0].name == "phase_separable_single_feature"
    assert suite.cases[0].parameter_shift_accuracy == 1.0
    assert suite.case_by_name("phase_separable_single_feature") is suite.cases[0]


def test_qnn_optimizer_benchmark_rejects_unknown_case_names() -> None:
    """An unknown case name is refused."""
    with pytest.raises(ValueError, match="unknown QNN optimizer benchmark case"):
        run_parameter_shift_qnn_optimizer_benchmark_suite(case_names=("missing",))


def test_qnn_optimizer_benchmark_rejects_invalid_controls() -> None:
    """Non-positive rates, steps and perturbations, a negative tolerance and seed are refused."""
    with pytest.raises(ValueError, match="learning_rate"):
        run_parameter_shift_qnn_optimizer_benchmark_suite(learning_rate=0.0)

    with pytest.raises(ValueError, match="max_steps"):
        run_parameter_shift_qnn_optimizer_benchmark_suite(max_steps=0)

    with pytest.raises(ValueError, match="tolerance"):
        run_parameter_shift_qnn_optimizer_benchmark_suite(tolerance=-1.0)

    with pytest.raises(ValueError, match="spsa_perturbation"):
        run_parameter_shift_qnn_optimizer_benchmark_suite(spsa_perturbation=0.0)

    with pytest.raises(ValueError, match="spsa_seed"):
        run_parameter_shift_qnn_optimizer_benchmark_suite(spsa_seed=-1)


def test_qnn_optimizer_benchmark_derivative_free_baseline_uses_declared_candidates() -> None:
    """Declared derivative-free candidates are scanned, and parameter shift beats them here."""
    candidates = (np.array([0.8], dtype=float), np.array([0.9], dtype=float))
    suite = run_parameter_shift_qnn_optimizer_benchmark_suite(
        case_names=("phase_separable_single_feature",),
        derivative_free_candidates={"phase_separable_single_feature": candidates},
    )

    case = suite.cases[0]
    assert case.derivative_free_evaluations == len(candidates)
    assert case.parameter_shift_best_loss < case.derivative_free_best_loss


def test_qnn_optimizer_benchmark_stops_a_diverging_optimizer_instead_of_failing() -> None:
    """An Adam step that overflows stops that optimiser; the suite still reports every baseline.

    Regression: the divergence guard tested the loss, but the loss refuses non-finite
    parameters first, so an overflow aborted the whole suite with a misleading error.
    """
    with np.errstate(over="ignore"):
        suite = run_parameter_shift_qnn_optimizer_benchmark_suite(
            learning_rate=1e308, max_steps=5, case_names=("two_feature_mixed_phase",)
        )

    case = suite.case_by_name("two_feature_mixed_phase")
    adam = case.optimizer_by_name("adam")
    assert adam.steps < 5
    assert np.isfinite(adam.best_loss)
    assert all(np.isfinite(result.best_loss) for result in case.optimizer_results)


def test_qnn_optimizer_benchmark_lookups_refuse_unknown_names_and_empty_suites_list_none() -> None:
    """Unknown case and optimizer names raise KeyError; a suite without cases lists no optimizers."""
    suite = run_parameter_shift_qnn_optimizer_benchmark_suite(
        case_names=("phase_separable_single_feature",), max_steps=2
    )
    case = suite.cases[0]

    assert suite.case_by_name("phase_separable_single_feature") is case
    assert case.optimizer_by_name("sgd").name == "sgd"
    with pytest.raises(KeyError, match="unknown QNN optimizer benchmark case: missing"):
        suite.case_by_name("missing")
    with pytest.raises(KeyError, match="unknown QNN optimizer baseline: missing"):
        case.optimizer_by_name("missing")
    empty = run_parameter_shift_qnn_optimizer_benchmark_suite(case_names=())
    assert empty.optimizer_names == ()
    assert empty.optimizer_count == 0


@pytest.mark.parametrize(
    ("controls", "message"),
    [
        ({"max_steps": True}, "^max_steps must be a positive integer$"),
        ({"spsa_seed": True}, "^spsa_seed must be a non-negative integer$"),
        ({"spsa_seed": -1}, "^spsa_seed must be a non-negative integer$"),
    ],
)
def test_qnn_optimizer_benchmark_refuses_boolean_or_negative_integer_controls(
    controls: dict[str, Any], message: str
) -> None:
    """Integer controls refuse booleans, and the SPSA seed refuses negatives."""
    with pytest.raises(ValueError, match=message):
        run_parameter_shift_qnn_optimizer_benchmark_suite(**controls)


@pytest.mark.parametrize(
    ("candidates", "message"),
    [
        ([np.array([0.1, 0.2])], r"must have shape \(1,\)$"),
        ([np.array([np.nan])], "must contain only finite values$"),
        ([], "must be non-empty$"),
    ],
)
def test_qnn_optimizer_benchmark_refuses_malformed_derivative_free_candidates(
    candidates: list[np.ndarray], message: str
) -> None:
    """Declared derivative-free candidates must match the case width, be finite and exist."""
    with pytest.raises(ValueError, match=message):
        run_parameter_shift_qnn_optimizer_benchmark_suite(
            case_names=("phase_separable_single_feature",),
            max_steps=1,
            derivative_free_candidates={"phase_separable_single_feature": candidates},
        )
