# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Partial-observation forecast objective tests
"""Production-surface tests for multimodal-forecasting observed/physics objective evidence."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

import numpy as np
import pytest

from scpn_quantum_control.forecasting.multimodal_forecaster import (
    fit_multimodal_ridge_forecaster,
)
from scpn_quantum_control.forecasting.partial_observation import (
    PartialObservationWeights,
    evaluate_partial_observation_batch,
    evaluate_partial_observation_objective,
)
from scpn_quantum_control.forecasting.synthetic_multimodal import (
    SyntheticMultimodalConfig,
    SyntheticMultimodalDataset,
    generate_synthetic_multimodal_dataset,
)


def _dataset() -> SyntheticMultimodalDataset:
    return generate_synthetic_multimodal_dataset(
        SyntheticMultimodalConfig(
            train_samples=24,
            calibration_samples=8,
            test_samples=8,
            history_steps=8,
            horizon_steps=4,
            missing_fraction=0.2,
            seed=3703,
        )
    )


def test_exact_synthetic_target_has_zero_observation_error_and_small_residual() -> None:
    """Exact synthetic targets retain zero error and a small physics residual."""
    dataset = _dataset()
    batch = dataset.test
    truth = batch.targets[0]
    score = evaluate_partial_observation_objective(
        truth,
        truth,
        np.ones_like(truth, dtype=np.bool_),
        batch.frequencies[0],
        batch.graphs[0],
        dt=batch.dt,
    )

    assert score.observed_wrapped_rmse == 0.0
    assert score.normalised_observation_loss == 0.0
    assert score.kuramoto_residual_rmse < 0.01
    assert score.physics_loss >= 0.0
    assert score.total_objective >= 0.0
    assert score.to_dict()["observed_values"] == truth.size


def test_batch_certificate_uses_explicit_partial_mask_and_complete_couplings() -> None:
    """Batch evidence binds the explicit mask and complete simulator graph."""
    dataset = _dataset()
    model = fit_multimodal_ridge_forecaster(dataset.train, ridge=10.0)
    forecast = model.predict(dataset.test)
    mask = np.zeros_like(dataset.test.target_mask)
    mask[:, :, ::2] = True
    certificate = evaluate_partial_observation_batch(forecast, dataset.test, mask)

    assert certificate.samples == dataset.test.n_samples
    assert certificate.observed_fraction == 0.5
    assert certificate.mean_observed_wrapped_rmse >= 0.0
    assert certificate.mean_kuramoto_residual_rmse >= 0.0
    assert len(certificate.scores) == dataset.test.n_samples
    assert certificate.forecast_model_digest == model.model_digest
    assert len(certificate.observation_mask_digest) == 64
    assert "not arbitrary" in certificate.claim_boundary
    payload = certificate.to_dict()
    assert isinstance(payload["scores"], list)
    assert len(payload["scores"]) == dataset.test.n_samples


@pytest.mark.parametrize(
    "weights",
    [
        PartialObservationWeights(observation=0.0, physics=1.0),
        PartialObservationWeights(observation=1.0, physics=0.0),
    ],
)
def test_single_term_objectives_remain_supported(weights: PartialObservationWeights) -> None:
    """Either observation or physics scoring may operate independently."""
    predicted = np.array([[0.0, 0.2], [0.1, 0.25]])
    observed = predicted + 0.01
    score = evaluate_partial_observation_objective(
        predicted,
        observed,
        np.ones_like(predicted, dtype=np.bool_),
        np.array([0.1, -0.1]),
        np.array([[0.0, 0.2], [0.2, 0.0]]),
        dt=0.1,
        weights=weights,
    )
    assert np.isfinite(score.total_objective)


@pytest.mark.parametrize(
    ("builder", "message"),
    [
        (lambda: PartialObservationWeights(observation=-1.0), "observation weight"),
        (lambda: PartialObservationWeights(physics=-1.0), "physics weight"),
        (
            lambda: PartialObservationWeights(observation=0.0, physics=0.0),
            "at least one",
        ),
        (lambda: PartialObservationWeights(observation_noise_std=0.0), "noise_std"),
    ],
)
def test_weights_reject_invalid_values(
    builder: Callable[[], PartialObservationWeights], message: str
) -> None:
    """Invalid objective weights and noise scales fail closed."""
    with pytest.raises(ValueError, match=message):
        builder()


def test_objective_rejects_shape_nonfinite_empty_mask_and_time_errors() -> None:
    """Malformed, non-finite, unobserved, and invalid-time inputs are rejected."""
    predicted = np.array([[0.0, 0.2], [0.1, 0.25]])
    observed = predicted.copy()
    mask = np.ones_like(predicted, dtype=np.bool_)
    omega = np.array([0.1, -0.1])
    coupling = np.array([[0.0, 0.2], [0.2, 0.0]])

    with pytest.raises(ValueError, match="share rank-two"):
        evaluate_partial_observation_objective(
            predicted[0], observed[0], mask[0], omega, coupling, dt=0.1
        )
    with pytest.raises(ValueError, match="dimensions"):
        evaluate_partial_observation_objective(
            predicted, observed, mask, omega[:1], coupling, dt=0.1
        )
    with pytest.raises(ValueError, match="must be finite"):
        evaluate_partial_observation_objective(
            np.full_like(predicted, np.inf), observed, mask, omega, coupling, dt=0.1
        )
    with pytest.raises(ValueError, match="must be finite"):
        evaluate_partial_observation_objective(
            predicted,
            np.full_like(observed, np.inf),
            mask,
            omega,
            coupling,
            dt=0.1,
        )
    with pytest.raises(ValueError, match="coupling must be finite"):
        evaluate_partial_observation_objective(
            predicted,
            observed,
            mask,
            np.array([np.inf, -0.1]),
            coupling,
            dt=0.1,
        )
    with pytest.raises(ValueError, match="coupling must be finite"):
        evaluate_partial_observation_objective(
            predicted,
            observed,
            mask,
            omega,
            np.array([[0.0, np.inf], [0.2, 0.0]]),
            dt=0.1,
        )
    with pytest.raises(ValueError, match="dt must"):
        evaluate_partial_observation_objective(predicted, observed, mask, omega, coupling, dt=0.0)
    with pytest.raises(ValueError, match="select at least one"):
        evaluate_partial_observation_objective(
            predicted,
            observed,
            np.zeros_like(mask),
            omega,
            coupling,
            dt=0.1,
        )
    with pytest.raises(ValueError, match="physics horizon"):
        evaluate_partial_observation_objective(
            predicted[:1], observed[:1], mask[:1], omega, coupling, dt=0.1
        )


def test_batch_certificate_rejects_custody_shape_and_incomplete_graph() -> None:
    """Batch certification rejects custody drift and incomplete couplings."""
    dataset = _dataset()
    model = fit_multimodal_ridge_forecaster(dataset.train, ridge=10.0)
    forecast = model.predict(dataset.test)
    with pytest.raises(ValueError, match="custody/shape"):
        evaluate_partial_observation_batch(
            replace(forecast, sample_ids=tuple(reversed(forecast.sample_ids))),
            dataset.test,
            dataset.test.target_mask,
        )
    with pytest.raises(ValueError, match="mask must match"):
        evaluate_partial_observation_batch(
            forecast,
            dataset.test,
            dataset.test.target_mask[:, :-1],
        )
    graph_mask = dataset.test.graph_mask.copy()
    graph_mask[0, 0, 1] = False
    with pytest.raises(ValueError, match="complete known"):
        evaluate_partial_observation_batch(
            forecast,
            replace(dataset.test, graph_mask=graph_mask),
            dataset.test.target_mask,
        )
