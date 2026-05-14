# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 three-channel coupling fixtures
"""Source-bounded fixtures for Paper 0 unified coupling parameter scan."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from .spec_loader import load_three_channel_coupling_validation_spec

CLAIM_BOUNDARY = "source-bounded parameter scan; not empirical support"
HARDWARE_STATUS = "parameter_scan_protocol_no_execution"
SOURCE_LEDGER_SPAN = ("P0R07081", "P0R07129")


@dataclass(frozen=True, slots=True)
class ThreeChannelCouplingConfig:
    """Finite settings for the three-channel coupling fixture."""

    expected_channel_count: int = 3
    lambda0: float = 1.0e-5
    spec_bundle_path: Path | None = None

    def __post_init__(self) -> None:
        if self.expected_channel_count < 1:
            raise ValueError("expected_channel_count must be at least 1")
        if self.lambda0 <= 0.0:
            raise ValueError("lambda0 must be positive")


@dataclass(frozen=True, slots=True)
class ChannelConstraint:
    """Source-stated current-limit constraint for one channel."""

    channel: str
    scaling_factor: float
    current_limit: str
    lambda0_at_limit: float


@dataclass(frozen=True, slots=True)
class ThreeChannelCouplingFixtureResult:
    """Combined three-channel coupling fixture result."""

    spec_keys: tuple[str, str, str, str, str, str]
    hardware_status: str
    source_ledger_span: tuple[str, str]
    channel_count: int
    expected_channel_count: int
    spec_count: int
    coupling_factors: MappingProxyType[str, float]
    coupling_ratios: MappingProxyType[str, float]
    sweet_spot_window: tuple[float, float]
    sweet_spot_predictions: MappingProxyType[str, float]
    propagated_bounds: MappingProxyType[str, float]
    null_controls: MappingProxyType[str, float]
    claim_boundary: str
    problem_metadata: MappingProxyType[str, Any]


def coupling_factors() -> dict[str, float]:
    """Return the source-stated canonical geometry factors."""
    return {"G": 1.1e-122, "EM": 9.3e-2, "Q": 8.0e-2, "S": 1.3e-6}


def coupling_ratios(factors: dict[str, float] | None = None) -> dict[str, float]:
    """Return fixed coupling ratios derived from geometry factors."""
    values = factors or coupling_factors()
    return {
        "EM_over_G": values["EM"] / values["G"],
        "Q_over_EM": values["Q"] / values["EM"],
        "S_over_EM": values["S"] / values["EM"],
    }


def constraint_catalogue() -> tuple[ChannelConstraint, ChannelConstraint, ChannelConstraint]:
    """Return the source-stated three independent channel constraints."""
    return (
        ChannelConstraint("G", 6.4e-5, "acceleration < 1e-8 m/s^2", 1.6e-4),
        ChannelConstraint("EM", 5.7e-13, "delta alpha / alpha < 1e-17/year", 1.8e-5),
        ChannelConstraint("Q", 5.0e-1, "decoherence < 1e-5 for 1e4 amu", 2.0e-5),
    )


def sweet_spot_predictions(lambda0: float = 1.0e-5) -> dict[str, float]:
    """Return source-normalised sweet-spot observable magnitudes."""
    if lambda0 <= 0.0:
        raise ValueError("lambda0 must be positive")
    scale = lambda0 / 1.0e-5
    return {
        "extra_acceleration_m_s2": 1.0e-9 * scale,
        "alpha_drift_per_year": 5.0e-18 * scale,
        "decoherence_fraction": 5.0e-6 * scale,
    }


def propagate_gravitational_bound(lambda_psi_g_bound: float) -> dict[str, float]:
    """Propagate a gravitational Psi-sector bound to EM and quantum sectors."""
    if lambda_psi_g_bound <= 0.0:
        raise ValueError("lambda_psi_g_bound must be positive")
    return {
        "lambda_psi_EM_bound": lambda_psi_g_bound * 1.2e120,
        "lambda_psi_Q_bound": lambda_psi_g_bound * 9.4e119,
    }


def classify_three_channel_outcome(outcomes: dict[str, bool]) -> str:
    """Classify source-stated all-null, isolated-signal, and three-signal outcomes."""
    channels = ("G", "EM", "Q")
    observed = tuple(outcomes.get(channel) is True for channel in channels)
    count = sum(observed)
    if count == 3:
        return "single-lambda0-correlation-supported"
    if count == 1:
        return "single-channel-signal-falsifies-unified-coupling"
    if count == 0:
        return "all-null-window-tightened"
    return "partial-pattern-requires-ratio-and-sensitivity-audit"


def validate_three_channel_coupling_fixture(
    config: ThreeChannelCouplingConfig | None = None,
) -> ThreeChannelCouplingFixtureResult:
    """Run the three-channel coupling parameter scan fixture."""
    cfg = config or ThreeChannelCouplingConfig()
    keys = (
        "three_channel_coupling.section_boundary",
        "three_channel_coupling.geometry_factors",
        "three_channel_coupling.fixed_ratios",
        "three_channel_coupling.experimental_constraints",
        "three_channel_coupling.cross_channel_propagation",
        "three_channel_coupling.falsification_fingerprint",
    )
    specs = tuple(
        load_three_channel_coupling_validation_spec(key, spec_bundle_path=cfg.spec_bundle_path)
        for key in keys
    )
    factors = coupling_factors()
    ratios = coupling_ratios(factors)
    controls = {
        "single_channel_overclaim_rejection_label": float(
            classify_three_channel_outcome({"G": True, "EM": False, "Q": False})
            == "single-channel-signal-falsifies-unified-coupling"
        ),
        "ratio_mismatch_rejection_label": _ratio_audit_label(ratios),
        "missing_constraint_propagation_rejection_label": float(
            set(propagate_gravitational_bound(1.0e-126))
            == {"lambda_psi_EM_bound", "lambda_psi_Q_bound"}
        ),
    }
    return ThreeChannelCouplingFixtureResult(
        spec_keys=keys,
        hardware_status=str(specs[0]["hardware_status"]),
        source_ledger_span=SOURCE_LEDGER_SPAN,
        channel_count=3,
        expected_channel_count=cfg.expected_channel_count,
        spec_count=len(keys),
        coupling_factors=MappingProxyType(factors),
        coupling_ratios=MappingProxyType(ratios),
        sweet_spot_window=(1.0e-6, 1.0e-5),
        sweet_spot_predictions=MappingProxyType(sweet_spot_predictions(cfg.lambda0)),
        propagated_bounds=MappingProxyType(propagate_gravitational_bound(1.0e-126)),
        null_controls=MappingProxyType(controls),
        claim_boundary=CLAIM_BOUNDARY,
        problem_metadata=MappingProxyType(
            {
                "source_ledger_ids": tuple(str(item) for item in specs[0]["source_ledger_ids"]),
                "source_ledger_span": SOURCE_LEDGER_SPAN,
                "claim_boundary": CLAIM_BOUNDARY,
                "protocol_state": "parameter_scan_protocol_only_no_execution",
            }
        ),
    )


def _ratio_audit_label(ratios: dict[str, float]) -> float:
    return float(
        abs(ratios["EM_over_G"] / 8.5e120 - 1.0) < 0.02
        and abs(ratios["Q_over_EM"] / 0.86 - 1.0) < 0.02
        and abs(ratios["S_over_EM"] / 1.4e-5 - 1.0) < 0.02
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "ChannelConstraint",
    "ThreeChannelCouplingConfig",
    "ThreeChannelCouplingFixtureResult",
    "classify_three_channel_outcome",
    "constraint_catalogue",
    "coupling_factors",
    "coupling_ratios",
    "propagate_gravitational_bound",
    "sweet_spot_predictions",
    "validate_three_channel_coupling_fixture",
]
