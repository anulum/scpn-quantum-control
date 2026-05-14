# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 Layer 5 Active Inference math validation fixtures
"""Simulator-only Layer 5 Active Inference mathematical fixtures."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from .spec_loader import load_l5_active_inference_math_validation_spec

CLAIM_BOUNDARY = (
    "source-bounded Layer 5 Active Inference mathematical fixture; not empirical evidence"
)
SOURCE_LEDGER_SPAN = ("P0R06450", "P0R06484")


@dataclass(frozen=True, slots=True)
class L5ActiveInferenceMathConfig:
    """Finite simulator settings for Layer 5 Active Inference math fixtures."""

    kappa: float = 0.25
    spec_bundle_path: Path | None = None

    def __post_init__(self) -> None:
        if not np.isfinite(self.kappa) or self.kappa <= 0.0:
            raise ValueError("kappa must be finite and positive")


@dataclass(frozen=True, slots=True)
class LayerFreeEnergyTerms:
    """Finite layer-free-energy decomposition."""

    complexity_kl: float
    accuracy_loss: float
    total_free_energy: float


@dataclass(frozen=True, slots=True)
class MessagePassingUpdate:
    """Finite upward/downward message-passing terms."""

    upward_error: NDArray[np.float64]
    downward_error: NDArray[np.float64]
    delta_mu: NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class ExpectedFreeEnergyScores:
    """Policy-wise expected free-energy scores."""

    ambiguity: NDArray[np.float64]
    divergence_from_prior: NDArray[np.float64]
    expected_free_energy: NDArray[np.float64]
    selected_policy_index: int


@dataclass(frozen=True, slots=True)
class PrecisionWeightedUpdate:
    """Source-bounded inverse-precision update terms."""

    delta_mu: NDArray[np.float64]
    source_formula_consistency_warning: bool


@dataclass(frozen=True, slots=True)
class L5ActiveInferenceMathFixtureResult:
    """Combined Layer 5 Active Inference mathematical fixture result."""

    spec_keys: tuple[str, str, str, str]
    hardware_status: str
    source_ledger_span: tuple[str, str]
    free_energy_total: float
    free_energy_residual: float
    message_update_norm: float
    selected_policy_index: int
    precision_update_norm: float
    null_controls: MappingProxyType[str, float]
    claim_boundary: str
    problem_metadata: MappingProxyType[str, Any]


def layer_free_energy_terms(
    *,
    q_psi: NDArray[np.float64],
    p_psi: NDArray[np.float64],
    p_o_given_psi: NDArray[np.float64],
) -> LayerFreeEnergyTerms:
    """Compute finite KL complexity and negative expected log likelihood."""
    q_raw = _finite_vector("q_psi", q_psi)
    p_raw = _finite_vector("p_psi", p_psi)
    likelihood = _finite_vector("p_o_given_psi", p_o_given_psi)
    if q_raw.shape != p_raw.shape or q_raw.shape != likelihood.shape:
        raise ValueError("probability vectors must have the same shape")
    q = _normalise_positive_vector("q_psi", q_raw)
    p = _normalise_positive_vector("p_psi", p_raw)
    likelihood = _require_strictly_positive("p_o_given_psi", likelihood)
    complexity = float(np.sum(q * np.log(q / p)))
    accuracy_loss = float(np.sum(q * (-np.log(likelihood))))
    total = complexity + accuracy_loss
    return LayerFreeEnergyTerms(
        complexity_kl=complexity,
        accuracy_loss=accuracy_loss,
        total_free_energy=total,
    )


def message_passing_update(
    *,
    observation: NDArray[np.float64],
    generated_prediction: NDArray[np.float64],
    local_mu: NDArray[np.float64],
    parent_prediction: NDArray[np.float64],
    kappa: float,
) -> MessagePassingUpdate:
    """Compute source-bounded upward/downward errors and belief update."""
    if not np.isfinite(kappa) or kappa <= 0.0:
        raise ValueError("kappa must be finite and positive")
    observation_arr = _finite_vector("observation", observation)
    generated_arr = _finite_vector("generated_prediction", generated_prediction)
    local_arr = _finite_vector("local_mu", local_mu)
    parent_arr = _finite_vector("parent_prediction", parent_prediction)
    if not (observation_arr.shape == generated_arr.shape == local_arr.shape == parent_arr.shape):
        raise ValueError("message-passing vectors must have the same shape")
    upward = observation_arr - generated_arr
    downward = local_arr - parent_arr
    return MessagePassingUpdate(
        upward_error=upward,
        downward_error=downward,
        delta_mu=-float(kappa) * (upward + downward),
    )


def expected_free_energy(
    *,
    ambiguity: NDArray[np.float64],
    divergence_from_prior: NDArray[np.float64],
) -> ExpectedFreeEnergyScores:
    """Compute policy-wise expected free energy and source argmin action."""
    ambiguity_arr = _finite_vector("ambiguity", ambiguity)
    divergence_arr = _finite_vector("divergence_from_prior", divergence_from_prior)
    if ambiguity_arr.shape != divergence_arr.shape:
        raise ValueError("expected-free-energy vectors must have the same shape")
    if np.any(ambiguity_arr < 0.0) or np.any(divergence_arr < 0.0):
        raise ValueError("expected-free-energy terms must be non-negative")
    scores = ambiguity_arr + divergence_arr
    return ExpectedFreeEnergyScores(
        ambiguity=ambiguity_arr,
        divergence_from_prior=divergence_arr,
        expected_free_energy=scores,
        selected_policy_index=int(np.argmin(scores)),
    )


def precision_weighted_update(
    *,
    precision_matrix: NDArray[np.float64],
    prediction_error: NDArray[np.float64],
) -> PrecisionWeightedUpdate:
    """Apply the manuscript's Delta mu = Pi^(-1) x epsilon formula."""
    precision = np.array(precision_matrix, dtype=np.float64, copy=True)
    error = _finite_vector("prediction_error", prediction_error)
    if precision.ndim != 2 or precision.shape[0] != precision.shape[1]:
        raise ValueError("precision_matrix must be square")
    if precision.shape[0] != error.size:
        raise ValueError("precision_matrix and prediction_error dimensions must match")
    delta = np.linalg.solve(precision, error)
    high_precision = np.eye(error.size, dtype=np.float64) * 4.0
    low_precision = np.eye(error.size, dtype=np.float64) * 2.0
    warning = bool(
        np.linalg.norm(np.linalg.solve(high_precision, error))
        < np.linalg.norm(np.linalg.solve(low_precision, error))
    )
    return PrecisionWeightedUpdate(
        delta_mu=cast(NDArray[np.float64], delta),
        source_formula_consistency_warning=warning,
    )


def validate_l5_active_inference_math_fixture(
    config: L5ActiveInferenceMathConfig | None = None,
) -> L5ActiveInferenceMathFixtureResult:
    """Run the combined Layer 5 Active Inference mathematical fixture."""
    cfg = config or L5ActiveInferenceMathConfig()
    keys = (
        "l5_active_inference_math.generative_hierarchy",
        "l5_active_inference_math.layer_free_energy",
        "l5_active_inference_math.message_passing_update",
        "l5_active_inference_math.action_and_precision_control",
    )
    specs = tuple(
        load_l5_active_inference_math_validation_spec(
            key,
            spec_bundle_path=cfg.spec_bundle_path,
        )
        for key in keys
    )
    free_energy = layer_free_energy_terms(
        q_psi=np.array([0.2, 0.5, 0.3], dtype=np.float64),
        p_psi=np.array([0.25, 0.45, 0.3], dtype=np.float64),
        p_o_given_psi=np.array([0.7, 0.4, 0.9], dtype=np.float64),
    )
    message = message_passing_update(
        observation=np.array([1.2, 0.8], dtype=np.float64),
        generated_prediction=np.array([1.0, 0.9], dtype=np.float64),
        local_mu=np.array([0.4, 0.6], dtype=np.float64),
        parent_prediction=np.array([0.3, 0.7], dtype=np.float64),
        kappa=cfg.kappa,
    )
    action = expected_free_energy(
        ambiguity=np.array([0.4, 0.3, 0.2], dtype=np.float64),
        divergence_from_prior=np.array([0.2, 0.1, 0.5], dtype=np.float64),
    )
    precision = precision_weighted_update(
        precision_matrix=np.diag(np.array([2.0, 4.0], dtype=np.float64)),
        prediction_error=np.array([0.5, 1.0], dtype=np.float64),
    )
    controls = {
        "shape_mismatch_rejection_label": _shape_mismatch_rejection_label(),
        "non_positive_likelihood_rejection_label": _non_positive_likelihood_rejection_label(),
        "singular_precision_rejection_label": _singular_precision_rejection_label(),
        "source_precision_wording_warning_label": float(
            precision.source_formula_consistency_warning
        ),
    }
    return L5ActiveInferenceMathFixtureResult(
        spec_keys=keys,
        hardware_status=str(specs[0]["hardware_status"]),
        source_ledger_span=SOURCE_LEDGER_SPAN,
        free_energy_total=free_energy.total_free_energy,
        free_energy_residual=abs(
            free_energy.total_free_energy - (free_energy.complexity_kl + free_energy.accuracy_loss)
        ),
        message_update_norm=float(np.linalg.norm(message.delta_mu)),
        selected_policy_index=action.selected_policy_index,
        precision_update_norm=float(np.linalg.norm(precision.delta_mu)),
        null_controls=MappingProxyType(controls),
        claim_boundary=CLAIM_BOUNDARY,
        problem_metadata=MappingProxyType(
            {
                "source_ledger_ids": tuple(str(item) for item in specs[0]["source_ledger_ids"]),
                "source_ledger_span": SOURCE_LEDGER_SPAN,
                "claim_boundary": CLAIM_BOUNDARY,
                "precision_formula_warning": (
                    "source formula uses inverse precision while adjacent wording says "
                    "higher precision gives larger updates"
                ),
            }
        ),
    )


def _shape_mismatch_rejection_label() -> float:
    try:
        layer_free_energy_terms(
            q_psi=np.array([0.5, 0.5], dtype=np.float64),
            p_psi=np.array([0.4, 0.4, 0.2], dtype=np.float64),
            p_o_given_psi=np.array([0.8, 0.7], dtype=np.float64),
        )
    except ValueError as exc:
        return float("same shape" in str(exc))
    return 0.0


def _non_positive_likelihood_rejection_label() -> float:
    try:
        layer_free_energy_terms(
            q_psi=np.array([0.5, 0.5], dtype=np.float64),
            p_psi=np.array([0.5, 0.5], dtype=np.float64),
            p_o_given_psi=np.array([0.8, 0.0], dtype=np.float64),
        )
    except ValueError as exc:
        return float("strictly positive" in str(exc))
    return 0.0


def _singular_precision_rejection_label() -> float:
    try:
        precision_weighted_update(
            precision_matrix=np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.float64),
            prediction_error=np.array([0.5, 1.0], dtype=np.float64),
        )
    except np.linalg.LinAlgError:
        return 1.0
    return 0.0


def _normalise_positive_vector(name: str, values: NDArray[np.float64]) -> NDArray[np.float64]:
    array = _require_strictly_positive(name, values)
    total = float(np.sum(array))
    if total <= np.finfo(np.float64).eps:
        raise ValueError(f"{name} must have positive mass")
    return cast(NDArray[np.float64], array / total)


def _require_strictly_positive(
    name: str,
    values: NDArray[np.float64],
) -> NDArray[np.float64]:
    if np.any(values <= 0.0):
        raise ValueError(f"{name} must be strictly positive")
    return values


def _finite_vector(name: str, values: NDArray[np.float64]) -> NDArray[np.float64]:
    array = np.array(values, dtype=np.float64, copy=True)
    if array.ndim != 1 or array.size < 1:
        raise ValueError(f"{name} must be a one-dimensional vector")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return cast(NDArray[np.float64], array)


__all__ = [
    "CLAIM_BOUNDARY",
    "ExpectedFreeEnergyScores",
    "L5ActiveInferenceMathConfig",
    "L5ActiveInferenceMathFixtureResult",
    "LayerFreeEnergyTerms",
    "MessagePassingUpdate",
    "PrecisionWeightedUpdate",
    "expected_free_energy",
    "layer_free_energy_terms",
    "message_passing_update",
    "precision_weighted_update",
    "validate_l5_active_inference_math_fixture",
]
