# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Surrogate fidelity certificate tests
"""Tests for disjoint surrogate value and gradient fidelity gates."""

from __future__ import annotations

from typing import cast

import numpy as np
import pytest
from numpy.typing import NDArray

from scpn_quantum_control.surrogates import (
    GaussianRBFSurrogate,
    SurrogateFidelityThresholds,
    SurrogateGradientCertificate,
    certify_surrogate_fidelity,
    certify_surrogate_gradient,
    fit_gaussian_rbf_surrogate,
)


def _fitted_model() -> GaussianRBFSurrogate:
    """Fit the production surrogate to a smooth one-parameter objective."""
    inputs = np.linspace(-1.0, 1.0, 9, dtype=np.float64)[:, None]
    targets = np.sin(inputs[:, 0])
    return fit_gaussian_rbf_surrogate(inputs, targets)


def _exact_objective(parameters: NDArray[np.float64]) -> float:
    """Return the analytic reference objective used to test gradients."""
    return float(np.sin(parameters[0]))


def test_value_fidelity_certifies_disjoint_validation_data() -> None:
    """Held-out values pass a frozen threshold and retain digest evidence."""
    model = _fitted_model()
    inputs = np.array([[-0.875], [-0.125], [0.375], [0.875]], dtype=np.float64)
    targets = np.sin(inputs[:, 0])
    certificate = certify_surrogate_fidelity(
        model,
        inputs,
        targets,
        thresholds=SurrogateFidelityThresholds(0.01, 0.02, 0.99),
    )

    assert certificate.passed
    assert certificate.training_overlap_count == 0
    assert certificate.n_training == 9
    assert certificate.n_validation == 4
    assert certificate.r_squared >= 0.99
    assert len(certificate.prediction_digest) == 64
    threshold_payload = cast(dict[str, object], certificate.to_dict()["thresholds"])
    assert threshold_payload["max_rmse"] == 0.01


def test_value_fidelity_reports_a_failed_frozen_gate() -> None:
    """A certificate reports failure instead of weakening its thresholds."""
    model = _fitted_model()
    inputs = np.array([[-0.875], [0.375]], dtype=np.float64)
    targets = np.sin(inputs[:, 0]) + 0.2
    certificate = certify_surrogate_fidelity(
        model,
        inputs,
        targets,
        thresholds=SurrogateFidelityThresholds(0.01, 0.02, 0.99),
    )

    assert not certificate.passed
    assert certificate.max_absolute_error > certificate.thresholds.max_absolute_error


def test_gradient_fidelity_compares_analytic_and_reference_derivatives() -> None:
    """Analytic RBF gradients pass against central differences."""
    model = _fitted_model()
    inputs = np.array([[-0.875], [-0.125], [0.375], [0.875]], dtype=np.float64)
    certificate = certify_surrogate_gradient(
        model,
        inputs,
        _exact_objective,
        finite_difference_step=1.0e-5,
        max_absolute_error=0.05,
    )

    assert isinstance(certificate, SurrogateGradientCertificate)
    assert certificate.passed
    assert certificate.n_points == 4
    assert certificate.n_parameters == 1
    assert certificate.training_overlap_count == 0
    assert len(certificate.exact_gradient_digest) == 64
    assert certificate.to_dict()["hardware_execution"] is False


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"max_rmse": 0.0, "max_absolute_error": 1.0, "min_r_squared": 0.0}, "max_rmse"),
        (
            {"max_rmse": 1.0, "max_absolute_error": np.nan, "min_r_squared": 0.0},
            "max_absolute_error",
        ),
        ({"max_rmse": 1.0, "max_absolute_error": 1.0, "min_r_squared": 1.1}, "r_squared"),
    ],
)
def test_fidelity_thresholds_reject_invalid_values(
    kwargs: dict[str, float],
    message: str,
) -> None:
    """Value-fidelity thresholds must remain finite and meaningful."""
    with pytest.raises(ValueError, match=message):
        SurrogateFidelityThresholds(**kwargs)


def test_value_fidelity_rejects_leakage_constant_targets_and_bad_shapes() -> None:
    """Value certification rejects leakage and ill-posed validation data."""
    model = _fitted_model()
    thresholds = SurrogateFidelityThresholds(1.0, 1.0, -1.0)
    with pytest.raises(ValueError, match="overlap"):
        certify_surrogate_fidelity(
            model,
            np.array([[-1.0], [0.2]]),
            np.array([0.0, 0.1]),
            thresholds=thresholds,
        )
    with pytest.raises(ValueError, match="non-zero variation"):
        certify_surrogate_fidelity(
            model,
            np.array([[-0.875], [0.875]]),
            np.ones(2),
            thresholds=thresholds,
        )
    with pytest.raises(ValueError, match="2-D"):
        certify_surrogate_fidelity(
            model,
            np.array([0.1, 0.2]),
            np.array([0.1, 0.2]),
            thresholds=thresholds,
        )
    with pytest.raises(ValueError, match="exact targets"):
        certify_surrogate_fidelity(
            model,
            np.array([[-0.875], [0.875]]),
            np.array([0.1]),
            thresholds=thresholds,
        )
    with pytest.raises(ValueError, match="parameter dimension"):
        certify_surrogate_fidelity(
            model,
            np.ones((2, 2)),
            np.array([0.1, 0.2]),
            thresholds=thresholds,
        )
    with pytest.raises(ValueError, match="inputs must contain only finite"):
        certify_surrogate_fidelity(
            model,
            np.array([[-0.875], [np.nan]]),
            np.array([0.1, 0.2]),
            thresholds=thresholds,
        )
    with pytest.raises(ValueError, match="targets must contain only finite"):
        certify_surrogate_fidelity(
            model,
            np.array([[-0.875], [0.875]]),
            np.array([0.1, np.inf]),
            thresholds=thresholds,
        )


def test_gradient_fidelity_rejects_leakage_bad_controls_and_nonfinite_objective() -> None:
    """Gradient certification fails closed before emitting misleading metrics."""
    model = _fitted_model()
    with pytest.raises(ValueError, match="overlap"):
        certify_surrogate_gradient(
            model,
            np.array([[-1.0]]),
            _exact_objective,
            finite_difference_step=1.0e-5,
            max_absolute_error=0.1,
        )
    with pytest.raises(ValueError, match="dimension-matched"):
        certify_surrogate_gradient(
            model,
            np.ones((2, 2)),
            _exact_objective,
            finite_difference_step=1.0e-5,
            max_absolute_error=0.1,
        )
    with pytest.raises(ValueError, match="contain only finite"):
        certify_surrogate_gradient(
            model,
            np.array([[np.nan]]),
            _exact_objective,
            finite_difference_step=1.0e-5,
            max_absolute_error=0.1,
        )
    with pytest.raises(ValueError, match="finite_difference_step"):
        certify_surrogate_gradient(
            model,
            np.array([[-0.875]]),
            _exact_objective,
            finite_difference_step=0.0,
            max_absolute_error=0.1,
        )
    with pytest.raises(ValueError, match="max_absolute_error"):
        certify_surrogate_gradient(
            model,
            np.array([[-0.875]]),
            _exact_objective,
            finite_difference_step=1.0e-5,
            max_absolute_error=np.inf,
        )

    def nonfinite_objective(parameters: NDArray[np.float64]) -> float:
        """Return a non-finite result for refusal testing."""
        assert parameters.shape == (1,)
        return np.nan

    with pytest.raises(ValueError, match="exact_objective"):
        certify_surrogate_gradient(
            model,
            np.array([[-0.875]]),
            nonfinite_objective,
            finite_difference_step=1.0e-5,
            max_absolute_error=0.1,
        )
