# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Partial-observation forecast objective
"""Observed-phase and exact Kuramoto-residual scoring for multimodal-forecasting forecasts."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass

import numpy as np
from numpy.typing import NDArray

from .multimodal_forecaster import MultimodalPointForecast
from .multimodal_schema import MultimodalObservationBatch

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


@dataclass(frozen=True, slots=True)
class PartialObservationWeights:
    """Weights for observed-phase and Kuramoto-residual terms.

    Parameters
    ----------
    observation
        Non-negative weight for wrapped observed-phase squared error.
    physics
        Non-negative weight for the exact Kuramoto derivative residual.
    observation_noise_std
        Positive scale used to normalise the observed-phase error.

    """

    observation: float = 1.0
    physics: float = 0.1
    observation_noise_std: float = 0.05

    def __post_init__(self) -> None:
        """Require finite non-negative weights and a positive noise scale."""
        if not np.isfinite(self.observation) or self.observation < 0.0:
            raise ValueError("observation weight must be finite and non-negative")
        if not np.isfinite(self.physics) or self.physics < 0.0:
            raise ValueError("physics weight must be finite and non-negative")
        if self.observation == 0.0 and self.physics == 0.0:
            raise ValueError("at least one objective weight must be positive")
        if not np.isfinite(self.observation_noise_std) or self.observation_noise_std <= 0.0:
            raise ValueError("observation_noise_std must be finite and positive")


@dataclass(frozen=True, slots=True)
class PartialObservationScore:
    """Objective decomposition for one forecast trajectory."""

    observed_values: int
    possible_values: int
    observed_fraction: float
    observed_wrapped_rmse: float
    normalised_observation_loss: float
    kuramoto_residual_rmse: float
    physics_loss: float
    total_objective: float

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready objective decomposition."""
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PartialObservationBatchCertificate:
    """Aggregate partial-observation objective evidence over one batch."""

    samples: int
    observed_fraction: float
    mean_observed_wrapped_rmse: float
    mean_kuramoto_residual_rmse: float
    mean_total_objective: float
    scores: tuple[PartialObservationScore, ...]
    forecast_model_digest: str
    batch_digest: str
    observation_mask_digest: str
    claim_boundary: str = (
        "synthetic held-out phase scoring with complete known simulator couplings; "
        "not arbitrary state/parameter inference, data assimilation, or operational control"
    )

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready batch certificate."""
        payload = asdict(self)
        payload["scores"] = [score.to_dict() for score in self.scores]
        return payload


def _validated_inputs(
    predicted_phases: object,
    observed_phases: object,
    observation_mask: object,
    frequencies: object,
    coupling: object,
    *,
    dt: float,
    physics_required: bool,
) -> tuple[FloatArray, FloatArray, BoolArray, FloatArray, FloatArray]:
    predicted = np.asarray(predicted_phases, dtype=np.float64)
    observed = np.asarray(observed_phases, dtype=np.float64)
    mask = np.asarray(observation_mask, dtype=np.bool_)
    omega = np.asarray(frequencies, dtype=np.float64)
    matrix = np.asarray(coupling, dtype=np.float64)
    if predicted.ndim != 2 or predicted.shape != observed.shape or predicted.shape != mask.shape:
        raise ValueError("predicted_phases, observed_phases, and mask must share rank-two shape")
    horizon, nodes = predicted.shape
    if horizon < 1 or nodes < 2 or (physics_required and horizon < 2):
        raise ValueError("objective requires nodes >= 2 and a physics horizon >= 2")
    if omega.shape != (nodes,) or matrix.shape != (nodes, nodes):
        raise ValueError("frequencies and coupling dimensions must match forecast nodes")
    if not np.all(np.isfinite(predicted)) or not np.all(np.isfinite(observed[mask])):
        raise ValueError("predicted and observed entries selected by the mask must be finite")
    if not np.all(np.isfinite(omega)) or not np.all(np.isfinite(matrix)):
        raise ValueError("frequencies and coupling must be finite")
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be finite and positive")
    if not np.any(mask):
        raise ValueError("observation_mask must select at least one phase value")
    return predicted, observed, mask, omega, matrix


def evaluate_partial_observation_objective(
    predicted_phases: object,
    observed_phases: object,
    observation_mask: object,
    frequencies: object,
    coupling: object,
    *,
    dt: float,
    weights: PartialObservationWeights | None = None,
) -> PartialObservationScore:
    r"""Score observed phase error plus an exact Kuramoto derivative residual.

    The physics term uses
    ``omega_i + sum_j K_ij sin(theta_j - theta_i)`` and a wrapped forward
    phase difference. Complete simulator couplings are required; this is not a
    hidden-state or coupling-inference algorithm.
    """
    resolved = PartialObservationWeights() if weights is None else weights
    predicted, observed, mask, omega, matrix = _validated_inputs(
        predicted_phases,
        observed_phases,
        observation_mask,
        frequencies,
        coupling,
        dt=dt,
        physics_required=resolved.physics > 0.0,
    )
    wrapped_error = (predicted - observed + np.pi) % (2.0 * np.pi) - np.pi
    observed_mse = float(np.mean(np.square(wrapped_error[mask])))
    normalised_observation = observed_mse / resolved.observation_noise_std**2

    physics_loss = 0.0
    physics_rmse = 0.0
    if resolved.physics > 0.0:
        wrapped_increment = (predicted[1:] - predicted[:-1] + np.pi) % (2.0 * np.pi) - np.pi
        derivative = wrapped_increment / dt
        phase_delta = predicted[:-1, None, :] - predicted[:-1, :, None]
        right_hand_side = omega[None, :] + np.sum(matrix[None, :, :] * np.sin(phase_delta), axis=2)
        residual = derivative - right_hand_side
        physics_loss = float(np.mean(np.square(residual)))
        physics_rmse = float(np.sqrt(physics_loss))
    return PartialObservationScore(
        observed_values=int(np.count_nonzero(mask)),
        possible_values=mask.size,
        observed_fraction=float(np.mean(mask)),
        observed_wrapped_rmse=float(np.sqrt(observed_mse)),
        normalised_observation_loss=normalised_observation,
        kuramoto_residual_rmse=physics_rmse,
        physics_loss=physics_loss,
        total_objective=(
            resolved.observation * normalised_observation + resolved.physics * physics_loss
        ),
    )


def evaluate_partial_observation_batch(
    forecast: MultimodalPointForecast,
    batch: MultimodalObservationBatch,
    observation_mask: object,
    *,
    weights: PartialObservationWeights | None = None,
) -> PartialObservationBatchCertificate:
    """Evaluate a custody-matched batch under an explicit partial target mask."""
    mask = np.asarray(observation_mask, dtype=np.bool_)
    if forecast.sample_ids != batch.sample_ids or forecast.values.shape != batch.targets.shape:
        raise ValueError("forecast and batch custody/shape must match exactly")
    if mask.shape != batch.targets.shape:
        raise ValueError("observation_mask must match the batch target shape")
    mask = np.logical_and(mask, batch.target_mask)
    if not np.all(batch.graph_mask):
        raise ValueError("physics certificate requires complete known simulator couplings")
    resolved = PartialObservationWeights() if weights is None else weights
    scores = tuple(
        evaluate_partial_observation_objective(
            forecast.values[index],
            batch.targets[index],
            mask[index],
            batch.frequencies[index],
            batch.graphs[index],
            dt=batch.dt,
            weights=resolved,
        )
        for index in range(batch.n_samples)
    )
    digest = hashlib.sha256()
    digest.update(b"scpn.partial_observation_mask.v1\0")
    digest.update(mask.tobytes(order="C"))
    return PartialObservationBatchCertificate(
        samples=batch.n_samples,
        observed_fraction=float(np.mean(mask)),
        mean_observed_wrapped_rmse=float(
            np.mean([score.observed_wrapped_rmse for score in scores])
        ),
        mean_kuramoto_residual_rmse=float(
            np.mean([score.kuramoto_residual_rmse for score in scores])
        ),
        mean_total_objective=float(np.mean([score.total_objective for score in scores])),
        scores=scores,
        forecast_model_digest=forecast.model_digest,
        batch_digest=batch.content_digest(),
        observation_mask_digest=digest.hexdigest(),
    )


__all__ = [
    "PartialObservationBatchCertificate",
    "PartialObservationScore",
    "PartialObservationWeights",
    "evaluate_partial_observation_batch",
    "evaluate_partial_observation_objective",
]
