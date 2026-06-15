# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for Phase QNN Framework Agreement
"""Tests for phase/qnn_framework_agreement.py bounded QNN adapter evidence."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_quantum_control.phase import (
    ParameterShiftQNNFrameworkAgreementSuiteResult,
    parameter_shift_qnn_classifier_gradient,
    run_parameter_shift_qnn_framework_agreement_suite,
    verify_parameter_shift_qnn_framework_agreement,
)


def test_qnn_framework_agreement_records_named_adapter_gradients() -> None:
    features = np.array(
        [[0.2, -0.4], [1.1, 0.7], [-0.8, 0.3]],
        dtype=float,
    )
    labels = np.array([0.0, 1.0, 0.25], dtype=float)
    params = np.array([0.4, -0.2], dtype=float)
    expected = parameter_shift_qnn_classifier_gradient(features, labels, params)

    result = verify_parameter_shift_qnn_framework_agreement(
        features,
        labels,
        params,
        framework_gradients={
            "jax": lambda _values: expected.copy(),
            "pennylane": lambda _values: expected + np.array([2e-8, -2e-8], dtype=float),
        },
        tolerance=1e-6,
    )

    assert result.passed
    assert result.framework_count == 2
    assert result.passed_count == 2
    assert result.failed_count == 0
    assert result.evidence_class == "caller_supplied_qnn_framework_agreement"
    assert "not native framework autodiff" in result.claim_boundary
    assert tuple(agreement.framework for agreement in result.agreements) == (
        "jax",
        "pennylane",
    )
    assert {agreement.source_class for agreement in result.agreements} == {
        "caller_supplied_gradient"
    }
    assert not any(agreement.native_framework_autodiff for agreement in result.agreements)
    assert all(
        "not native framework autodiff" in agreement.claim_boundary
        for agreement in result.agreements
    )
    assert result.to_dict()["passed"] is True
    assert result.to_dict()["agreements"][0]["source_class"] == "caller_supplied_gradient"


def test_qnn_framework_agreement_suite_covers_default_cases() -> None:
    suite = run_parameter_shift_qnn_framework_agreement_suite()

    assert isinstance(suite, ParameterShiftQNNFrameworkAgreementSuiteResult)
    assert suite.passed
    assert suite.case_count == 2
    assert suite.framework_count == 4
    assert suite.failed_count == 0
    assert suite.evidence_class == "caller_supplied_qnn_framework_agreement"
    assert not suite.native_framework_autodiff
    assert all(
        agreement.source_class == "deterministic_manual_reference"
        for case in suite.cases
        for agreement in case.agreements
    )
    assert tuple(case.name for case in suite.cases) == (
        "phase_separable_single_feature",
        "two_feature_mixed_phase",
    )


def test_qnn_framework_agreement_suite_accepts_external_frameworks() -> None:
    features = np.array([[0.0], [np.pi]], dtype=float)
    labels = np.array([0.0, 1.0], dtype=float)
    params = np.array([0.45], dtype=float)
    expected = parameter_shift_qnn_classifier_gradient(features, labels, params)

    suite = run_parameter_shift_qnn_framework_agreement_suite(
        case_names=("phase_separable_single_feature",),
        framework_gradients={
            "phase_separable_single_feature": {"custom_autograd": lambda _values: expected.copy()}
        },
    )

    assert suite.passed
    assert suite.case_count == 1
    assert suite.framework_count == 1
    assert suite.case_by_name("phase_separable_single_feature").framework_count == 1


def test_qnn_framework_agreement_fails_closed_on_invalid_gradient() -> None:
    features = np.array([[0.0], [np.pi]], dtype=float)
    labels = np.array([0.0, 1.0], dtype=float)
    params = np.array([0.45], dtype=float)

    with pytest.raises(ValueError, match="framework gradient"):
        verify_parameter_shift_qnn_framework_agreement(
            features,
            labels,
            params,
            framework_gradients={"jax": lambda _values: np.array([1.0, 2.0], dtype=float)},
        )

    with pytest.raises(ValueError, match="framework_gradients"):
        verify_parameter_shift_qnn_framework_agreement(
            features,
            labels,
            params,
            framework_gradients={},
        )

    with pytest.raises(ValueError, match="source_class"):
        verify_parameter_shift_qnn_framework_agreement(
            features,
            labels,
            params,
            framework_gradients={"jax": lambda _values: np.array([0.0], dtype=float)},
            source_class="native autodiff",
        )


def test_qnn_framework_agreement_suite_rejects_unknown_case() -> None:
    with pytest.raises(ValueError, match="unknown QNN framework agreement case"):
        run_parameter_shift_qnn_framework_agreement_suite(case_names=("missing",))
