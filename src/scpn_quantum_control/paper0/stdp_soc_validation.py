# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 STDP/SOC validation fixtures
"""Simulator-only STDP/SOC fixtures."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np

from .spec_loader import load_stdp_soc_validation_spec

CLAIM_BOUNDARY = "source-bounded STDP/SOC simulator contract; not empirical evidence"
SOURCE_LEDGER_SPAN = ("P0R06402", "P0R06413")


@dataclass(frozen=True, slots=True)
class STDPSOCConfig:
    """Finite simulator settings for STDP/SOC fixtures."""

    tau_ltp: float = 0.02
    tau_ltd: float = 0.02
    criticality_threshold: float = 2.0
    spec_bundle_path: Path | None = None

    def __post_init__(self) -> None:
        _require_positive("tau_ltp", self.tau_ltp)
        _require_positive("tau_ltd", self.tau_ltd)
        _require_positive("criticality_threshold", self.criticality_threshold)


@dataclass(frozen=True, slots=True)
class STDPSOCFixtureResult:
    """Combined STDP/SOC fixture result."""

    spec_keys: tuple[str, str, str, str]
    hardware_status: str
    ltp_update: float
    ltd_update: float
    avalanche_density_small: float
    avalanche_density_large: float
    power_law_ratio: float
    relaxation_above_critical: float
    relaxation_below_critical: float
    null_controls: MappingProxyType[str, float]
    config_thresholds: MappingProxyType[str, float]
    source_ledger_span: tuple[str, str]
    claim_boundary: str
    problem_metadata: MappingProxyType[str, Any]


def stdp_weight_update(
    *,
    delta_t: float,
    amplitude_ltp: float,
    amplitude_ltd: float,
    tau_ltp: float = 0.02,
    tau_ltd: float = 0.02,
) -> float:
    """Return an asymmetric STDP update with LTP for positive timing and LTD for negative timing."""
    _require_positive("amplitude_ltp", amplitude_ltp)
    _require_positive("amplitude_ltd", amplitude_ltd)
    _require_positive("tau_ltp", tau_ltp)
    _require_positive("tau_ltd", tau_ltd)
    if not np.isfinite(delta_t):
        raise ValueError("delta_t must be finite")
    if delta_t > 0.0:
        return float(amplitude_ltp * np.exp(-delta_t / tau_ltp))
    if delta_t < 0.0:
        return float(-amplitude_ltd * np.exp(delta_t / tau_ltd))
    return 0.0


def avalanche_power_law_density(*, size: float, tau: float, normalisation: float = 1.0) -> float:
    """Return source-bounded avalanche density P(S) proportional_to S^-tau."""
    _require_positive("size", size)
    _require_positive("tau", tau)
    _require_positive("normalisation", normalisation)
    return float(normalisation * size ** (-tau))


def branching_parameter_relaxation_derivative(
    *,
    sigma_l: float,
    kappa_l: float,
    eta_l: float,
) -> float:
    """Return d sigma_L / dt = -kappa_L * (sigma_L - 1) + eta_L(t)."""
    if not np.isfinite(sigma_l):
        raise ValueError("sigma_l must be finite")
    _require_positive("kappa_l", kappa_l)
    if not np.isfinite(eta_l):
        raise ValueError("eta_l must be finite")
    return float(-kappa_l * (sigma_l - 1.0) + eta_l)


def validate_stdp_soc_fixture(config: STDPSOCConfig | None = None) -> STDPSOCFixtureResult:
    """Run the combined STDP/SOC fixture."""
    cfg = config or STDPSOCConfig()
    keys = (
        "stdp_soc.asymmetric_learning_window",
        "stdp_soc.avalanche_power_law_signature",
        "stdp_soc.quasicritical_relaxation_mapping",
        "stdp_soc.l4_microscopic_engine_boundary",
    )
    specs = tuple(
        load_stdp_soc_validation_spec(key, spec_bundle_path=cfg.spec_bundle_path) for key in keys
    )
    ltp = stdp_weight_update(
        delta_t=0.012,
        amplitude_ltp=0.08,
        amplitude_ltd=0.07,
        tau_ltp=cfg.tau_ltp,
        tau_ltd=cfg.tau_ltd,
    )
    ltd = stdp_weight_update(
        delta_t=-0.012,
        amplitude_ltp=0.08,
        amplitude_ltd=0.07,
        tau_ltp=cfg.tau_ltp,
        tau_ltd=cfg.tau_ltd,
    )
    density_small = avalanche_power_law_density(size=4.0, tau=1.5)
    density_large = avalanche_power_law_density(size=16.0, tau=1.5)
    relaxation_above = branching_parameter_relaxation_derivative(
        sigma_l=1.2,
        kappa_l=0.4,
        eta_l=0.0,
    )
    relaxation_below = branching_parameter_relaxation_derivative(
        sigma_l=0.8,
        kappa_l=0.4,
        eta_l=0.0,
    )
    controls = {
        "wrong_stdp_sign_rejection_label": float(ltp > 0.0 and ltd < 0.0),
        "missing_relaxation_rejection_label": float(
            relaxation_above < 0.0 and relaxation_below > 0.0
        ),
        "unsupported_empirical_evidence_rejection_label": 1.0,
    }
    return STDPSOCFixtureResult(
        spec_keys=keys,
        hardware_status=str(specs[0]["hardware_status"]),
        ltp_update=ltp,
        ltd_update=ltd,
        avalanche_density_small=density_small,
        avalanche_density_large=density_large,
        power_law_ratio=float(density_small / density_large),
        relaxation_above_critical=relaxation_above,
        relaxation_below_critical=relaxation_below,
        null_controls=MappingProxyType(controls),
        config_thresholds=MappingProxyType(
            {
                "criticality_threshold": cfg.criticality_threshold,
                "tau_ltp": cfg.tau_ltp,
                "tau_ltd": cfg.tau_ltd,
            }
        ),
        source_ledger_span=SOURCE_LEDGER_SPAN,
        claim_boundary=CLAIM_BOUNDARY,
        problem_metadata=MappingProxyType(
            {
                "source_ledger_ids": tuple(str(item) for item in specs[0]["source_ledger_ids"]),
                "source_ledger_span": SOURCE_LEDGER_SPAN,
                "claim_boundary": CLAIM_BOUNDARY,
            }
        ),
    )


def _require_positive(name: str, value: float) -> None:
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")


__all__ = [
    "CLAIM_BOUNDARY",
    "STDPSOCConfig",
    "STDPSOCFixtureResult",
    "avalanche_power_law_density",
    "branching_parameter_relaxation_derivative",
    "stdp_weight_update",
    "validate_stdp_soc_fixture",
]
