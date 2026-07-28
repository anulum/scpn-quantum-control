# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Exact-validated surrogate proposal tests
"""Tests for the unapplied BL-33 surrogate proposal boundary."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from scpn_quantum_control.applications import ReservoirLinearObjective
from scpn_quantum_control.surrogates import (
    ExactValidatedSurrogateProposal,
    GaussianRBFSurrogate,
    SurrogateFidelityCertificate,
    SurrogateFidelityThresholds,
    certify_surrogate_fidelity,
    fit_gaussian_rbf_surrogate,
    propose_and_validate_surrogate_step,
)


def _product_surfaces() -> tuple[
    GaussianRBFSurrogate,
    ReservoirLinearObjective,
    SurrogateFidelityCertificate,
]:
    """Build real exact-objective, surrogate, and held-out certificate surfaces."""
    coupling = np.array([[0.0, 0.6], [0.6, 0.0]], dtype=np.float64)
    objective = ReservoirLinearObjective(
        K=coupling,
        omega=np.array([0.1, -0.1]),
        feature_labels=("IZ", "ZI"),
        feature_weights=(0.6, -0.4),
        t=0.5,
        max_weight=1,
    )
    training = np.array(
        [[0.1, 0.1], [0.1, 0.9], [0.5, 0.5], [0.9, 0.1], [0.9, 0.9]],
        dtype=np.float64,
    )
    targets = np.array([objective(point) for point in training])
    model = fit_gaussian_rbf_surrogate(training, targets)
    validation = np.array([[0.25, 0.35], [0.7, 0.8]], dtype=np.float64)
    exact = np.array([objective(point) for point in validation])
    fidelity = certify_surrogate_fidelity(
        model,
        validation,
        exact,
        thresholds=SurrogateFidelityThresholds(0.5, 0.8, -1.0),
    )
    return model, objective, fidelity


def test_surrogate_proposal_is_bounded_exact_validated_and_unapplied() -> None:
    """The hybrid path returns a bounded proposal without applying it."""
    model, objective, fidelity = _product_surfaces()
    result = propose_and_validate_surrogate_step(
        model,
        np.array([0.3, 0.7]),
        objective,
        fidelity,
        learning_rate=0.1,
        max_step_norm=0.03,
    )

    assert isinstance(result, ExactValidatedSurrogateProposal)
    assert np.linalg.norm(result.proposal.update) <= 0.03 + 1.0e-12
    assert result.applied is False
    assert result.hardware_execution is False
    assert result.reason in {"exact_local_improvement", "exact_local_non_improvement"}
    assert result.accepted_by_exact_objective == (result.exact_observed_improvement > 0.0)
    assert result.to_dict()["proposal"] == result.proposal.to_dict()


def test_surrogate_proposal_can_record_exact_non_improvement() -> None:
    """Exact validation can refuse a surrogate direction without applying it."""
    model, _, fidelity = _product_surfaces()

    def opposing_objective(parameters: NDArray[np.float64]) -> float:
        """Return the negative surrogate value to force an opposing objective."""
        return -model.value(parameters)

    result = propose_and_validate_surrogate_step(
        model,
        np.array([0.3, 0.7]),
        opposing_objective,
        fidelity,
        learning_rate=0.1,
        max_step_norm=0.03,
    )
    assert not result.accepted_by_exact_objective
    assert result.reason == "exact_local_non_improvement"


def test_surrogate_proposal_rejects_failed_fidelity_and_invalid_controls() -> None:
    """Hybrid composition requires a passing certificate and bounded controls."""
    model, objective, fidelity = _product_surfaces()
    failed = certify_surrogate_fidelity(
        model,
        np.array([[0.25, 0.35], [0.7, 0.8]]),
        np.array([10.0, -10.0]),
        thresholds=SurrogateFidelityThresholds(0.01, 0.01, 0.99),
    )
    with pytest.raises(ValueError, match="passing disjoint"):
        propose_and_validate_surrogate_step(
            model,
            np.array([0.3, 0.7]),
            objective,
            failed,
            learning_rate=0.1,
            max_step_norm=0.03,
        )
    with pytest.raises(ValueError, match="parameter dimension"):
        propose_and_validate_surrogate_step(
            model,
            np.array([0.3]),
            objective,
            fidelity,
            learning_rate=0.1,
            max_step_norm=0.03,
        )
    with pytest.raises(ValueError, match="contain only finite"):
        propose_and_validate_surrogate_step(
            model,
            np.array([np.nan, 0.7]),
            objective,
            fidelity,
            learning_rate=0.1,
            max_step_norm=0.03,
        )
    with pytest.raises(ValueError, match="learning_rate"):
        propose_and_validate_surrogate_step(
            model,
            np.array([0.3, 0.7]),
            objective,
            fidelity,
            learning_rate=0.0,
            max_step_norm=0.03,
        )
    with pytest.raises(ValueError, match="max_step_norm"):
        propose_and_validate_surrogate_step(
            model,
            np.array([0.3, 0.7]),
            objective,
            fidelity,
            learning_rate=0.1,
            max_step_norm=np.inf,
        )


def test_surrogate_proposal_rejects_nonfinite_exact_objective() -> None:
    """A non-finite exact result cannot create an acceptance observation."""
    model, _, fidelity = _product_surfaces()

    def nonfinite_objective(parameters: NDArray[np.float64]) -> float:
        """Return a non-finite exact objective for refusal testing."""
        assert parameters.shape == (2,)
        return np.nan

    with pytest.raises(ValueError, match="exact_objective"):
        propose_and_validate_surrogate_step(
            model,
            np.array([0.3, 0.7]),
            nonfinite_objective,
            fidelity,
            learning_rate=0.1,
            max_step_norm=0.03,
        )
