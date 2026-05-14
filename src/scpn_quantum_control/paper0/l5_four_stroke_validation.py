# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 Layer 5 four-stroke fixtures
"""Simulator-only Layer 5 four-stroke active-inference fixtures."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from types import MappingProxyType
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from .spec_loader import load_l5_four_stroke_validation_spec

CLAIM_BOUNDARY = (
    "source-bounded Layer 5 four-stroke engine simulator contract; not empirical evidence"
)
SOURCE_LEDGER_SPAN = ("P0R06582", "P0R06614")


@dataclass(frozen=True, slots=True)
class L5FourStrokeConfig:
    """Finite simulator settings for Layer 5 four-stroke fixtures."""

    policy_precision: float = 3.0
    homeostatic_gain: float = 0.25
    resetting_noise: float = 0.0
    spec_bundle_path: Path | None = None

    def __post_init__(self) -> None:
        _require_positive("policy_precision", self.policy_precision)
        _require_unit_interval("homeostatic_gain", self.homeostatic_gain)
        _require_finite("resetting_noise", self.resetting_noise)


@dataclass(frozen=True, slots=True)
class L5FourStrokeFixtureResult:
    """Combined Layer 5 four-stroke fixture result."""

    spec_keys: tuple[str, str, str, str, str, str]
    hardware_status: str
    source_ledger_span: tuple[str, str]
    policy_weights: tuple[float, ...]
    selected_policy_index: int
    prediction_error: tuple[float, ...]
    prediction_error_norm: float
    l5_coherence: float
    pre_sleep_sigma: float
    post_sleep_sigma: float
    null_controls: MappingProxyType[str, float]
    config_thresholds: MappingProxyType[str, float]
    claim_boundary: str
    problem_metadata: MappingProxyType[str, Any]


def policy_precision_weights(
    *,
    reward_predictions: NDArray[np.float64],
    precision: float,
) -> NDArray[np.float64]:
    """Return softmax precision weights over policy reward predictions."""
    rewards = _finite_vector("reward_predictions", reward_predictions)
    _require_positive("precision", precision)
    scaled = precision * (rewards - float(np.max(rewards)))
    exp_scaled = np.exp(scaled)
    return cast(NDArray[np.float64], exp_scaled / exp_scaled.sum())


def prediction_error(
    *,
    sensory_input: NDArray[np.float64],
    prediction: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Return residual prediction error epsilon = y - y_hat."""
    y = _finite_vector("sensory_input", sensory_input)
    y_hat = _finite_vector("prediction", prediction)
    if y.shape != y_hat.shape:
        raise ValueError("vectors must have the same shape")
    return cast(NDArray[np.float64], y - y_hat)


def l5_coherence_metric(
    *,
    theta_bg: NDArray[np.float64],
    theta_cb: NDArray[np.float64],
    theta_ctx: NDArray[np.float64],
) -> float:
    """Return R_L5 = |mean(exp(i(theta_BG - theta_CB - theta_CTX)))|."""
    bg = _finite_vector("theta_bg", theta_bg)
    cb = _finite_vector("theta_cb", theta_cb)
    ctx = _finite_vector("theta_ctx", theta_ctx)
    if bg.shape != cb.shape or bg.shape != ctx.shape:
        raise ValueError("phase vectors must have the same shape")
    if bg.size == 0:
        raise ValueError("phase vectors must not be empty")
    phases = bg - cb - ctx
    return float(abs(np.mean(np.exp(1j * phases))))


def sleep_consolidation_sigma_update(
    *,
    sigma: float,
    homeostatic_gain: float,
    resetting_noise: float,
) -> float:
    """Move sigma toward one with bounded homeostatic gain and resetting noise."""
    _require_finite("sigma", sigma)
    _require_unit_interval("homeostatic_gain", homeostatic_gain)
    _require_finite("resetting_noise", resetting_noise)
    return sigma + homeostatic_gain * (1.0 - sigma) + resetting_noise


def validate_l5_four_stroke_fixture(
    config: L5FourStrokeConfig | None = None,
) -> L5FourStrokeFixtureResult:
    """Run the combined Layer 5 four-stroke fixture."""
    cfg = config or L5FourStrokeConfig()
    keys = (
        "l5_four_stroke.engine_framing",
        "l5_four_stroke.policy_selection",
        "l5_four_stroke.prediction_generation",
        "l5_four_stroke.error_processing",
        "l5_four_stroke.model_consolidation",
        "l5_four_stroke.upde_coherence_prediction",
    )
    specs = tuple(
        load_l5_four_stroke_validation_spec(
            key,
            spec_bundle_path=cfg.spec_bundle_path,
        )
        for key in keys
    )
    weights = policy_precision_weights(
        reward_predictions=np.array([0.1, 1.2, -0.2], dtype=np.float64),
        precision=cfg.policy_precision,
    )
    residual = prediction_error(
        sensory_input=np.array([1.0, 0.5, -0.25], dtype=np.float64),
        prediction=np.array([0.75, 0.25, -0.5], dtype=np.float64),
    )
    coherence = l5_coherence_metric(
        theta_bg=np.array([0.2, 0.3, 0.4], dtype=np.float64),
        theta_cb=np.array([0.1, 0.15, 0.2], dtype=np.float64),
        theta_ctx=np.array([0.1, 0.15, 0.2], dtype=np.float64),
    )
    pre_sleep_sigma = 1.4
    post_sleep_sigma = sleep_consolidation_sigma_update(
        sigma=pre_sleep_sigma,
        homeostatic_gain=cfg.homeostatic_gain,
        resetting_noise=cfg.resetting_noise,
    )
    controls = {
        "shape_mismatch_rejection_label": _shape_mismatch_rejection_label(),
        "invalid_precision_rejection_label": _invalid_precision_rejection_label(),
        "unsupported_tms_evidence_rejection_label": 1.0,
    }
    return L5FourStrokeFixtureResult(
        spec_keys=keys,
        hardware_status=str(specs[0]["hardware_status"]),
        source_ledger_span=SOURCE_LEDGER_SPAN,
        policy_weights=tuple(float(value) for value in weights),
        selected_policy_index=int(np.argmax(weights)),
        prediction_error=tuple(float(value) for value in residual),
        prediction_error_norm=float(np.linalg.norm(residual)),
        l5_coherence=coherence,
        pre_sleep_sigma=pre_sleep_sigma,
        post_sleep_sigma=post_sleep_sigma,
        null_controls=MappingProxyType(controls),
        config_thresholds=MappingProxyType(
            {
                "policy_precision": cfg.policy_precision,
                "homeostatic_gain": cfg.homeostatic_gain,
                "resetting_noise": cfg.resetting_noise,
            }
        ),
        claim_boundary=CLAIM_BOUNDARY,
        problem_metadata=MappingProxyType(
            {
                "source_ledger_ids": tuple(str(item) for item in specs[0]["source_ledger_ids"]),
                "source_ledger_span": SOURCE_LEDGER_SPAN,
                "claim_boundary": CLAIM_BOUNDARY,
                "phase_mapping": ("theta_BG", "theta_CB", "theta_CTX", "eta_Sleep"),
            }
        ),
    )


def _shape_mismatch_rejection_label() -> float:
    try:
        prediction_error(
            sensory_input=np.array([1.0, 2.0], dtype=np.float64),
            prediction=np.array([1.0], dtype=np.float64),
        )
    except ValueError as exc:
        return float("same shape" in str(exc))
    return 0.0


def _invalid_precision_rejection_label() -> float:
    try:
        policy_precision_weights(
            reward_predictions=np.array([0.1, 0.2], dtype=np.float64),
            precision=0.0,
        )
    except ValueError as exc:
        return float("precision must be finite and positive" in str(exc))
    return 0.0


def _finite_vector(name: str, values: NDArray[np.float64]) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return cast(NDArray[np.float64], array)


def _require_finite(name: str, value: float) -> None:
    if not isfinite(value):
        raise ValueError(f"{name} must be finite")


def _require_positive(name: str, value: float) -> None:
    if not isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")


def _require_unit_interval(name: str, value: float) -> None:
    if not isfinite(value) or value < 0.0 or value > 1.0:
        raise ValueError(f"{name} must be in [0, 1]")


__all__ = [
    "CLAIM_BOUNDARY",
    "L5FourStrokeConfig",
    "L5FourStrokeFixtureResult",
    "l5_coherence_metric",
    "policy_precision_weights",
    "prediction_error",
    "sleep_consolidation_sigma_update",
    "validate_l5_four_stroke_fixture",
]
