# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Paper 0 UPDE adaptive-coupling validation
"""Executable simulator fixture for the Paper 0 adaptive-coupling law."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import cast

import numpy as np

from .spec_loader import load_upde_validation_spec


@dataclass(frozen=True, slots=True)
class AdaptiveCouplingConfig:
    """Configuration for the Paper 0 adaptive-coupling fixture."""

    gamma_L: float = 0.8
    lambda_L: float = 0.15
    alpha_L: float = 0.45
    max_abs_gain: float = 100.0
    max_abs_update: float = 10.0

    def __post_init__(self) -> None:
        _require_bounded_gain(self.gamma_L, "gamma_L", self.max_abs_gain)
        _require_bounded_gain(self.alpha_L, "alpha_L", self.max_abs_gain)
        if not np.isfinite(self.lambda_L) or self.lambda_L < 0.0:
            raise ValueError("lambda_L must be finite and non-negative")
        if not np.isfinite(self.max_abs_gain) or self.max_abs_gain <= 0.0:
            raise ValueError("max_abs_gain must be finite and positive")
        if not np.isfinite(self.max_abs_update) or self.max_abs_update <= 0.0:
            raise ValueError("max_abs_update must be finite and positive")


@dataclass(frozen=True, slots=True)
class AdaptiveCouplingRates:
    """Instantaneous rates for Paper 0 adaptive coupling and eta dynamics."""

    K_dot: np.ndarray
    eta_dot: float


@dataclass(frozen=True, slots=True)
class AdaptiveCouplingStep:
    """One explicit-Euler adaptive-coupling step."""

    K_next: np.ndarray
    eta_next: float
    rates: AdaptiveCouplingRates


@dataclass(frozen=True, slots=True)
class AdaptiveCouplingValidationResult:
    """Result of the Paper 0 adaptive-coupling fixture."""

    spec_key: str
    validation_protocol: str
    hardware_status: str
    source_equation_ids: tuple[str, ...]
    source_ledger_ids: tuple[str, ...]
    K_dot_linf: float
    eta_dot: float
    K_next_linf: float
    eta_next: float
    bounded_update_linf: float
    null_controls: MappingProxyType[str, float]


def adaptive_coupling_rates(
    K_ij_L: np.ndarray,
    R_L: float,
    R_L_star: float,
    sigma_L: float,
    noise_xi: np.ndarray | None,
    eta_L: float,
    *,
    config: AdaptiveCouplingConfig | None = None,
) -> AdaptiveCouplingRates:
    """Return Paper 0 ``dot K_ij^L`` and ``dot eta_L`` source rates."""
    cfg = config or AdaptiveCouplingConfig()
    K = _validated_symmetric_matrix(K_ij_L, "K_ij_L")
    noise = (
        np.zeros_like(K) if noise_xi is None else _validated_symmetric_matrix(noise_xi, "noise_xi")
    )
    _require_finite_scalar(R_L, "R_L")
    _require_finite_scalar(R_L_star, "R_L_star")
    _require_finite_scalar(sigma_L, "sigma_L")
    _require_finite_scalar(eta_L, "eta_L")
    K_dot = cfg.gamma_L * (float(R_L) - float(R_L_star)) - cfg.lambda_L * K + noise
    K_dot = np.asarray(K_dot, dtype=np.float64)
    np.fill_diagonal(K_dot, 0.0)
    eta_dot = -cfg.alpha_L * (float(sigma_L) - 1.0)
    return AdaptiveCouplingRates(K_dot=K_dot, eta_dot=float(eta_dot))


def apply_adaptive_coupling_step(
    K_ij_L: np.ndarray,
    R_L: float,
    R_L_star: float,
    sigma_L: float,
    noise_xi: np.ndarray | None,
    eta_L: float,
    *,
    dt: float,
    config: AdaptiveCouplingConfig | None = None,
) -> AdaptiveCouplingStep:
    """Apply one bounded explicit-Euler step of the Paper 0 adaptive law."""
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be finite and positive")
    cfg = config or AdaptiveCouplingConfig()
    K = _validated_symmetric_matrix(K_ij_L, "K_ij_L")
    rates = adaptive_coupling_rates(
        K,
        R_L,
        R_L_star,
        sigma_L,
        noise_xi,
        eta_L,
        config=cfg,
    )
    update_norm = float(max(np.max(np.abs(dt * rates.K_dot)), abs(dt * rates.eta_dot)))
    if update_norm > cfg.max_abs_update:
        raise ValueError(
            f"adaptive update exceeds max_abs_update: {update_norm:.6g} > {cfg.max_abs_update:.6g}"
        )
    K_next = K + dt * rates.K_dot
    K_next = 0.5 * (K_next + K_next.T)
    np.fill_diagonal(K_next, 0.0)
    eta_next = float(eta_L) + dt * rates.eta_dot
    return AdaptiveCouplingStep(K_next=K_next, eta_next=eta_next, rates=rates)


def validate_upde_adaptive_coupling_fixture(
    K_ij_L: np.ndarray,
    R_L: float,
    R_L_star: float,
    sigma_L: float,
    noise_xi: np.ndarray | None,
    eta_L: float,
    *,
    dt: float = 0.05,
    config: AdaptiveCouplingConfig | None = None,
) -> AdaptiveCouplingValidationResult:
    """Run the source-anchored adaptive-coupling UPDE simulator fixture."""
    cfg = config or AdaptiveCouplingConfig()
    spec = load_upde_validation_spec("upde.adaptive_coupling")
    step = apply_adaptive_coupling_step(
        K_ij_L,
        R_L,
        R_L_star,
        sigma_L,
        noise_xi,
        eta_L,
        dt=dt,
        config=cfg,
    )
    zero = adaptive_coupling_rates(
        K_ij_L,
        R_L,
        R_L_star,
        sigma_L,
        np.zeros_like(np.asarray(K_ij_L, dtype=np.float64)),
        eta_L,
        config=AdaptiveCouplingConfig(gamma_L=0.0, lambda_L=0.0, alpha_L=0.0),
    )
    wrong = adaptive_coupling_rates(
        K_ij_L,
        R_L,
        R_L_star,
        sigma_L,
        noise_xi,
        eta_L,
        config=AdaptiveCouplingConfig(
            gamma_L=-cfg.gamma_L,
            lambda_L=cfg.lambda_L,
            alpha_L=-cfg.alpha_L,
            max_abs_gain=cfg.max_abs_gain,
            max_abs_update=cfg.max_abs_update,
        ),
    )
    update_linf = float(max(np.max(np.abs(dt * step.rates.K_dot)), abs(dt * step.rates.eta_dot)))
    null_controls = {
        "zero_gain_K_dot_linf": float(np.max(np.abs(zero.K_dot))),
        "zero_gain_eta_dot_abs": abs(float(zero.eta_dot)),
        "wrong_sign_K_dot_l2": float(np.linalg.norm(wrong.K_dot - step.rates.K_dot)),
        "wrong_sign_eta_dot_abs": abs(float(wrong.eta_dot - step.rates.eta_dot)),
    }
    return AdaptiveCouplingValidationResult(
        spec_key="upde.adaptive_coupling",
        validation_protocol=str(spec["validation_protocol"]),
        hardware_status=str(spec["hardware_status"]),
        source_equation_ids=tuple(str(item) for item in spec["source_equation_ids"]),
        source_ledger_ids=tuple(str(item) for item in spec["source_ledger_ids"]),
        K_dot_linf=float(np.max(np.abs(step.rates.K_dot))),
        eta_dot=float(step.rates.eta_dot),
        K_next_linf=float(np.max(np.abs(step.K_next))),
        eta_next=float(step.eta_next),
        bounded_update_linf=update_linf,
        null_controls=MappingProxyType(null_controls),
    )


def _validated_symmetric_matrix(values: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(f"{name} must be a square matrix")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    if not np.allclose(arr, arr.T, atol=1e-12, rtol=1e-12):
        raise ValueError(f"{name} must be symmetric")
    out = arr.copy()
    np.fill_diagonal(out, 0.0)
    return cast(np.ndarray, out)


def _require_finite_scalar(value: float, name: str) -> None:
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite")


def _require_bounded_gain(value: float, name: str, max_abs_gain: float) -> None:
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if abs(value) > max_abs_gain:
        raise ValueError(f"{name} exceeds max_abs_gain")


__all__ = [
    "AdaptiveCouplingConfig",
    "AdaptiveCouplingRates",
    "AdaptiveCouplingStep",
    "AdaptiveCouplingValidationResult",
    "adaptive_coupling_rates",
    "apply_adaptive_coupling_step",
    "validate_upde_adaptive_coupling_fixture",
]
