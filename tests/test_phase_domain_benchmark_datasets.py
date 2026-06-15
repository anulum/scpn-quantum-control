# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for Differentiable Domain Benchmark Datasets
"""Tests for exact-answer differentiable phase-domain benchmark datasets."""

from __future__ import annotations

import math

import numpy as np
import pytest

from scpn_quantum_control.phase import (
    DifferentiableDomainBenchmarkDatasetSuite,
    DifferentiableDomainBenchmarkValidationSuite,
    load_differentiable_domain_benchmark_datasets,
    parameter_shift_qnn_classifier_gradient,
    parameter_shift_qnn_classifier_loss,
    predict_parameter_shift_qnn_classifier,
    run_differentiable_domain_benchmark_dataset_validation,
)


def test_differentiable_domain_benchmark_datasets_carry_exact_answers() -> None:
    suite = load_differentiable_domain_benchmark_datasets()

    assert isinstance(suite, DifferentiableDomainBenchmarkDatasetSuite)
    assert suite.evidence_class == "synthetic_exact_answer"
    assert "not hardware" in suite.claim_boundary
    assert suite.case_count == 3
    assert suite.dataset_ids == (
        "bounded_qnn_phase_separable_exact",
        "bounded_qnn_two_feature_mixed_exact",
        "kuramoto_xy_two_oscillator_pi_over_3",
    )

    qnn_case = suite.qnn_cases[0]
    prediction = predict_parameter_shift_qnn_classifier(qnn_case.features, qnn_case.params)
    assert np.allclose(prediction.probabilities, qnn_case.expected_probabilities)
    assert parameter_shift_qnn_classifier_loss(
        qnn_case.features, qnn_case.labels, qnn_case.params
    ) == pytest.approx(qnn_case.expected_loss)
    assert np.allclose(
        parameter_shift_qnn_classifier_gradient(
            qnn_case.features,
            qnn_case.labels,
            qnn_case.params,
        ),
        qnn_case.expected_gradient,
    )
    assert qnn_case.to_dict()["n_samples"] == 2

    kuramoto_case = suite.kuramoto_cases[0]
    assert kuramoto_case.expected_order_parameter == pytest.approx(math.sqrt(3.0) / 2.0)
    assert kuramoto_case.expected_mean_phase == pytest.approx(math.pi / 6.0)
    assert kuramoto_case.expected_xy_energy == pytest.approx(-0.25)
    assert np.allclose(
        kuramoto_case.expected_energy_gradient,
        np.array([-math.sqrt(3.0) / 4.0, math.sqrt(3.0) / 4.0], dtype=float),
    )


def test_differentiable_domain_benchmark_validation_passes_all_cases() -> None:
    validation = run_differentiable_domain_benchmark_dataset_validation()

    assert isinstance(validation, DifferentiableDomainBenchmarkValidationSuite)
    assert validation.passed
    assert validation.case_count == 3
    assert validation.max_abs_error <= 1e-12
    assert tuple(result.dataset_id for result in validation.results) == (
        "bounded_qnn_phase_separable_exact",
        "bounded_qnn_two_feature_mixed_exact",
        "kuramoto_xy_two_oscillator_pi_over_3",
    )
    assert validation.to_dict()["evidence_class"] == "synthetic_exact_answer"


def test_differentiable_domain_benchmark_dataset_selection_is_fail_closed() -> None:
    suite = load_differentiable_domain_benchmark_datasets(
        dataset_ids=("kuramoto_xy_two_oscillator_pi_over_3",)
    )
    assert not suite.qnn_cases
    assert len(suite.kuramoto_cases) == 1

    validation = run_differentiable_domain_benchmark_dataset_validation(
        dataset_ids=("bounded_qnn_phase_separable_exact",)
    )
    assert validation.passed
    assert validation.case_count == 1
    assert validation.results[0].domain == "bounded_phase_qnn"

    with pytest.raises(ValueError, match="unknown differentiable benchmark dataset"):
        load_differentiable_domain_benchmark_datasets(dataset_ids=("missing",))

    with pytest.raises(ValueError, match="dataset_ids"):
        load_differentiable_domain_benchmark_datasets(dataset_ids=("",))

    with pytest.raises(ValueError, match="tolerance"):
        run_differentiable_domain_benchmark_dataset_validation(tolerance=-1.0)
