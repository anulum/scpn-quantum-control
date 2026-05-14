# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 two-timescale quasicritical fixtures
"""Simulator-only two-timescale quasicritical controller fixtures."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, tanh
from pathlib import Path
from types import MappingProxyType
from typing import Any

from .spec_loader import load_two_timescale_quasicritical_validation_spec

CLAIM_BOUNDARY = (
    "source-bounded two-timescale quasicritical controller simulator contract; "
    "not empirical evidence"
)
SOURCE_LEDGER_SPAN = ("P0R06646", "P0R06676")


@dataclass(frozen=True, slots=True)
class TwoTimescaleQuasicriticalConfig:
    """Finite simulator settings for the two-timescale controller."""

    tau_f: float = 1.0
    tau_s: float = 25.0
    max_timescale_ratio: float = 0.1
    delta: float = 0.08
    g_f_min: float = 0.2
    k_f: float = 0.7
    k_f_prime: float = 0.4
    g_s_max: float = 1.0
    c: float = 1.1
    beta: float = 0.75
    alpha_f: float = 1.2
    alpha_s: float = 0.4
    spec_bundle_path: Path | None = None

    def __post_init__(self) -> None:
        _require_positive("tau_f", self.tau_f)
        _require_positive("tau_s", self.tau_s)
        if self.tau_s <= self.tau_f:
            raise ValueError("tau_s must be greater than tau_f")
        _require_positive("max_timescale_ratio", self.max_timescale_ratio)
        _require_positive("delta", self.delta)
        _require_non_negative("g_f_min", self.g_f_min)
        _require_non_negative("k_f", self.k_f)
        _require_non_negative("k_f_prime", self.k_f_prime)
        _require_non_negative("g_s_max", self.g_s_max)
        _require_positive("c", self.c)
        _require_positive("beta", self.beta)
        _require_positive("alpha_f", self.alpha_f)
        _require_positive("alpha_s", self.alpha_s)


@dataclass(frozen=True, slots=True)
class TwoTimescaleQuasicriticalFixtureResult:
    """Combined two-timescale quasicritical fixture result."""

    spec_keys: tuple[str, str, str, str, str]
    hardware_status: str
    source_ledger_span: tuple[str, str]
    timescale_ratio: float
    high_surprise_fast_gain: float
    low_surprise_fast_gain: float
    high_surprise_slow_gain: float
    low_surprise_slow_gain: float
    outside_band_slow_gain: float
    lyapunov_value: float
    lyapunov_drift_upper_bound: float
    null_controls: MappingProxyType[str, float]
    config_thresholds: MappingProxyType[str, float]
    claim_boundary: str
    problem_metadata: MappingProxyType[str, Any]


def fast_stabilizer_gain(
    *,
    sigma: float,
    affective_gradient: float,
    config: TwoTimescaleQuasicriticalConfig,
) -> float:
    """Return G_f = G_f,min + k_f |dA/dsigma| + k_f_prime |sigma - 1|."""
    _require_finite("sigma", sigma)
    _require_finite("affective_gradient", affective_gradient)
    return (
        config.g_f_min + config.k_f * abs(affective_gradient) + config.k_f_prime * abs(sigma - 1.0)
    )


def slow_explorer_gain(
    *,
    sigma: float,
    affective_gradient: float,
    config: TwoTimescaleQuasicriticalConfig,
) -> float:
    """Return G_s inside the quasicritical window and zero outside it."""
    _require_finite("sigma", sigma)
    _require_finite("affective_gradient", affective_gradient)
    if abs(sigma - 1.0) > config.delta:
        return 0.0
    return config.g_s_max * (1.0 - tanh(config.c * abs(affective_gradient)))


def lyapunov_total(
    *,
    sigma: float,
    coherence: float,
    target_coherence: float,
    beta: float,
) -> float:
    """Return V_total = (sigma - 1)^2 + beta (R - R_star)^2."""
    _require_finite("sigma", sigma)
    _require_finite("coherence", coherence)
    _require_finite("target_coherence", target_coherence)
    _require_positive("beta", beta)
    return (sigma - 1.0) ** 2 + beta * (coherence - target_coherence) ** 2


def lyapunov_drift_bound(
    *,
    sigma: float,
    coherence: float,
    target_coherence: float,
    bounded_noise: float,
    config: TwoTimescaleQuasicriticalConfig,
) -> float:
    """Return dV/dt upper bound under timescale separation and bounded noise."""
    _require_non_negative("bounded_noise", bounded_noise)
    v_fast = (sigma - 1.0) ** 2
    v_slow = config.beta * (coherence - target_coherence) ** 2
    return -config.alpha_f * v_fast - config.alpha_s * v_slow + bounded_noise


def validate_two_timescale_quasicritical_fixture(
    config: TwoTimescaleQuasicriticalConfig | None = None,
) -> TwoTimescaleQuasicriticalFixtureResult:
    """Run the combined two-timescale quasicritical fixture."""
    cfg = config or TwoTimescaleQuasicriticalConfig()
    keys = (
        "two_timescale_quasicritical.block_framing",
        "two_timescale_quasicritical.dual_channel_architecture",
        "two_timescale_quasicritical.affective_gain_scheduling",
        "two_timescale_quasicritical.bibo_stability_certificate",
        "two_timescale_quasicritical.operational_consequence",
    )
    specs = tuple(
        load_two_timescale_quasicritical_validation_spec(
            key,
            spec_bundle_path=cfg.spec_bundle_path,
        )
        for key in keys
    )
    sigma = 1.02
    high_gradient = 2.0
    low_gradient = 0.05
    high_fast = fast_stabilizer_gain(
        sigma=sigma,
        affective_gradient=high_gradient,
        config=cfg,
    )
    low_fast = fast_stabilizer_gain(
        sigma=sigma,
        affective_gradient=low_gradient,
        config=cfg,
    )
    high_slow = slow_explorer_gain(
        sigma=sigma,
        affective_gradient=high_gradient,
        config=cfg,
    )
    low_slow = slow_explorer_gain(
        sigma=sigma,
        affective_gradient=low_gradient,
        config=cfg,
    )
    outside_slow = slow_explorer_gain(sigma=1.4, affective_gradient=low_gradient, config=cfg)
    lyapunov = lyapunov_total(
        sigma=1.2,
        coherence=0.7,
        target_coherence=0.9,
        beta=cfg.beta,
    )
    drift_bound = lyapunov_drift_bound(
        sigma=1.2,
        coherence=0.7,
        target_coherence=0.9,
        bounded_noise=0.001,
        config=cfg,
    )
    controls = {
        "invalid_timescale_rejection_label": _invalid_timescale_rejection_label(),
        "invalid_delta_rejection_label": _invalid_delta_rejection_label(),
        "unsupported_bibo_empirical_claim_rejection_label": 1.0,
    }
    return TwoTimescaleQuasicriticalFixtureResult(
        spec_keys=keys,
        hardware_status=str(specs[0]["hardware_status"]),
        source_ledger_span=SOURCE_LEDGER_SPAN,
        timescale_ratio=cfg.tau_f / cfg.tau_s,
        high_surprise_fast_gain=high_fast,
        low_surprise_fast_gain=low_fast,
        high_surprise_slow_gain=high_slow,
        low_surprise_slow_gain=low_slow,
        outside_band_slow_gain=outside_slow,
        lyapunov_value=lyapunov,
        lyapunov_drift_upper_bound=drift_bound,
        null_controls=MappingProxyType(controls),
        config_thresholds=MappingProxyType(
            {
                "max_timescale_ratio": cfg.max_timescale_ratio,
                "delta": cfg.delta,
                "beta": cfg.beta,
                "alpha_f": cfg.alpha_f,
                "alpha_s": cfg.alpha_s,
            }
        ),
        claim_boundary=CLAIM_BOUNDARY,
        problem_metadata=MappingProxyType(
            {
                "source_ledger_ids": tuple(str(item) for item in specs[0]["source_ledger_ids"]),
                "source_ledger_span": SOURCE_LEDGER_SPAN,
                "claim_boundary": CLAIM_BOUNDARY,
                "controller_channels": ("fast_stabilizer", "slow_explorer"),
            }
        ),
    )


def _invalid_timescale_rejection_label() -> float:
    try:
        TwoTimescaleQuasicriticalConfig(tau_f=1.0, tau_s=1.0)
    except ValueError as exc:
        return float("tau_s must be greater than tau_f" in str(exc))
    return 0.0


def _invalid_delta_rejection_label() -> float:
    try:
        TwoTimescaleQuasicriticalConfig(delta=0.0)
    except ValueError as exc:
        return float("delta must be finite and positive" in str(exc))
    return 0.0


def _require_finite(name: str, value: float) -> None:
    if not isfinite(value):
        raise ValueError(f"{name} must be finite")


def _require_positive(name: str, value: float) -> None:
    if not isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")


def _require_non_negative(name: str, value: float) -> None:
    if not isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")


__all__ = [
    "CLAIM_BOUNDARY",
    "TwoTimescaleQuasicriticalConfig",
    "TwoTimescaleQuasicriticalFixtureResult",
    "fast_stabilizer_gain",
    "lyapunov_drift_bound",
    "lyapunov_total",
    "slow_explorer_gain",
    "validate_two_timescale_quasicritical_fixture",
]
