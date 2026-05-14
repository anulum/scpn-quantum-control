# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 dark-sector validation fixtures
"""Simulator-only dark-energy and psi-DM fixtures."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np

from .spec_loader import load_dark_sector_validation_spec

CLAIM_BOUNDARY = "source-bounded dark-sector simulator contract; not empirical evidence"
SOURCE_LEDGER_SPAN = ("P0R06311", "P0R06323")
SOURCE_FORMULA = "L_Geometric proportional to -xi R Psi* Psi"


@dataclass(frozen=True, slots=True)
class DarkSectorConfig:
    """Finite simulator settings for dark-sector fixtures."""

    conformal_rescaling_weight: float = 0.34
    ethical_functional_weight: float = 0.33
    entropy_reset_weight: float = 0.33
    mmc_threshold: float = 0.70
    dark_energy_threshold: float = 0.70
    interaction_threshold: float = 0.70
    reservoir_threshold: float = 0.68
    spec_bundle_path: Path | None = None

    def __post_init__(self) -> None:
        for name in (
            "mmc_threshold",
            "dark_energy_threshold",
            "interaction_threshold",
            "reservoir_threshold",
        ):
            _require_positive(name, float(getattr(self, name)))
        for name in (
            "conformal_rescaling_weight",
            "ethical_functional_weight",
            "entropy_reset_weight",
        ):
            _require_non_negative(name, float(getattr(self, name)))


@dataclass(frozen=True, slots=True)
class PsiDmInteractionValidationResult:
    """Psi-DM interaction mechanism result."""

    spec_key: str
    validation_protocol: str
    hardware_status: str
    source_ledger_ids: tuple[str, ...]
    source_formulae: tuple[str, ...]
    interaction_score: float
    claim_boundary: str
    null_controls: MappingProxyType[str, float]
    problem_metadata: MappingProxyType[str, Any]


@dataclass(frozen=True, slots=True)
class DarkSectorFixtureResult:
    """Combined dark-sector fixture result."""

    spec_keys: tuple[str, str, str, str, str]
    hardware_status: str
    mmc_score: float
    dark_energy_score: float
    psi_dm_candidate: str
    interaction: PsiDmInteractionValidationResult
    reservoir_score: float
    config_thresholds: MappingProxyType[str, float]
    source_ledger_span: tuple[str, str]
    claim_boundary: str
    problem_metadata: MappingProxyType[str, Any]


def mmc_information_preservation_score(
    *,
    conformal_rescaling: float,
    ethical_functional_conserved: float,
    entropy_reset: float,
    config: DarkSectorConfig,
) -> float:
    """Score the source MMC operator and information-preservation labels."""
    values = np.asarray(
        [conformal_rescaling, ethical_functional_conserved, entropy_reset],
        dtype=np.float64,
    )
    _require_unit_interval("MMC information-preservation inputs", values)
    weights = np.asarray(
        [
            config.conformal_rescaling_weight,
            config.ethical_functional_weight,
            config.entropy_reset_weight,
        ],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
        raise ValueError("MMC weights must be finite and non-negative")
    total = float(np.sum(weights))
    if total <= 0.0:
        raise ValueError("at least one MMC weight must be positive")
    return float(np.dot(weights, values) / total)


def dark_energy_context_score(
    *,
    lambda_potential: float,
    rg_flow_pressure: float,
    cosmic_attractor_drive: float,
    config: DarkSectorConfig,
) -> float:
    """Return bounded DE-as-teleological-potential context score."""
    values = np.asarray(
        [lambda_potential, rg_flow_pressure, cosmic_attractor_drive],
        dtype=np.float64,
    )
    _require_unit_interval("dark-energy context inputs", values)
    return float(np.dot(np.asarray([0.34, 0.33, 0.33], dtype=np.float64), values))


def psi_dm_candidate_label(
    *,
    ssb: bool,
    alp_bec: bool,
    q_ball: bool,
    nonlinear_potential: bool,
) -> str:
    """Classify whether source-listed psi-DM candidate channels are complete."""
    if ssb and nonlinear_potential and (alp_bec or q_ball):
        return "coherent_psi_field_dark_matter_candidate"
    return "incomplete_psi_dm_candidate_boundary"


def geometric_coupling_score(
    *,
    stress_energy_tensor: float,
    curvature_coupling: float,
    weak_ordinary_matter_coupling: float,
    config: DarkSectorConfig,
) -> float:
    """Score source interaction channels while rewarding weak ordinary-matter coupling."""
    values = np.asarray(
        [stress_energy_tensor, curvature_coupling, weak_ordinary_matter_coupling],
        dtype=np.float64,
    )
    _require_unit_interval("psi-DM interaction inputs", values)
    weak_boundary = 1.0 - weak_ordinary_matter_coupling
    return float(0.40 * stress_energy_tensor + 0.40 * curvature_coupling + 0.20 * weak_boundary)


def cosmic_reservoir_score(
    *,
    structure_scaffolding: float,
    halo_coherence: float,
    l8_phase_locking: float,
    l12_gaian_sync: float,
    config: DarkSectorConfig,
) -> float:
    """Score source labels for structure scaffolding and L8/L12 reservoir coupling."""
    values = np.asarray(
        [structure_scaffolding, halo_coherence, l8_phase_locking, l12_gaian_sync],
        dtype=np.float64,
    )
    _require_unit_interval("cosmic-reservoir inputs", values)
    return float(np.dot(np.asarray([0.25, 0.30, 0.225, 0.225], dtype=np.float64), values))


def validate_psi_dm_interaction_fixture(
    config: DarkSectorConfig | None = None,
) -> PsiDmInteractionValidationResult:
    """Run the psi-DM interaction mechanism fixture."""
    cfg = config or DarkSectorConfig()
    spec = load_dark_sector_validation_spec(
        "dark_sector.psi_dm_interaction_mechanisms",
        spec_bundle_path=cfg.spec_bundle_path,
    )
    interaction_score = geometric_coupling_score(
        stress_energy_tensor=0.84,
        curvature_coupling=0.81,
        weak_ordinary_matter_coupling=0.18,
        config=cfg,
    )
    controls = {
        "missing_geometric_coupling_rejection_label": _missing_geometric_coupling_label(),
        "missing_weak_coupling_boundary_rejection_label": _missing_weak_boundary_label(),
        "unsupported_dark_matter_evidence_rejection_label": 1.0,
    }
    return PsiDmInteractionValidationResult(
        spec_key="dark_sector.psi_dm_interaction_mechanisms",
        validation_protocol=str(spec["validation_protocol"]),
        hardware_status=str(spec["hardware_status"]),
        source_ledger_ids=tuple(str(item) for item in spec["source_ledger_ids"]),
        source_formulae=tuple(str(item) for item in spec["source_formulae"]),
        interaction_score=interaction_score,
        claim_boundary=CLAIM_BOUNDARY,
        null_controls=MappingProxyType(controls),
        problem_metadata=MappingProxyType(
            {
                "interaction_threshold": cfg.interaction_threshold,
                "source_formulae": (SOURCE_FORMULA,),
                "source_ledger_span": SOURCE_LEDGER_SPAN,
                "claim_boundary": CLAIM_BOUNDARY,
            }
        ),
    )


def validate_dark_sector_fixture(
    config: DarkSectorConfig | None = None,
) -> DarkSectorFixtureResult:
    """Run the combined dark-sector fixture."""
    cfg = config or DarkSectorConfig()
    keys = (
        "dark_sector.mmc_operator_information_preservation",
        "dark_sector.dark_energy_teleological_potential_boundary",
        "dark_sector.psi_dark_matter_hypothesis_boundary",
        "dark_sector.psi_dm_interaction_mechanisms",
        "dark_sector.cosmic_coherence_reservoir_boundary",
    )
    specs = tuple(
        load_dark_sector_validation_spec(key, spec_bundle_path=cfg.spec_bundle_path)
        for key in keys
    )
    mmc = mmc_information_preservation_score(
        conformal_rescaling=0.82,
        ethical_functional_conserved=0.91,
        entropy_reset=0.74,
        config=cfg,
    )
    dark_energy = dark_energy_context_score(
        lambda_potential=0.78,
        rg_flow_pressure=0.73,
        cosmic_attractor_drive=0.76,
        config=cfg,
    )
    candidate = psi_dm_candidate_label(
        ssb=True,
        alp_bec=True,
        q_ball=True,
        nonlinear_potential=True,
    )
    interaction = validate_psi_dm_interaction_fixture(cfg)
    reservoir = cosmic_reservoir_score(
        structure_scaffolding=0.77,
        halo_coherence=0.83,
        l8_phase_locking=0.71,
        l12_gaian_sync=0.69,
        config=cfg,
    )
    return DarkSectorFixtureResult(
        spec_keys=keys,
        hardware_status=str(specs[0]["hardware_status"]),
        mmc_score=mmc,
        dark_energy_score=dark_energy,
        psi_dm_candidate=candidate,
        interaction=interaction,
        reservoir_score=reservoir,
        config_thresholds=MappingProxyType(
            {
                "mmc_threshold": cfg.mmc_threshold,
                "dark_energy_threshold": cfg.dark_energy_threshold,
                "interaction_threshold": cfg.interaction_threshold,
                "reservoir_threshold": cfg.reservoir_threshold,
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


def _require_non_negative(name: str, value: float) -> None:
    if not np.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")


def _require_unit_interval(name: str, values: np.ndarray) -> None:
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must be finite")
    if np.any(values < 0.0) or np.any(values > 1.0):
        raise ValueError(f"{name} must be in [0, 1]")


def _missing_geometric_coupling_label() -> float:
    try:
        geometric_coupling_score(
            stress_energy_tensor=0.84,
            curvature_coupling=-0.01,
            weak_ordinary_matter_coupling=0.18,
            config=DarkSectorConfig(),
        )
    except ValueError as exc:
        return float("in [0, 1]" in str(exc))
    return 0.0


def _missing_weak_boundary_label() -> float:
    try:
        geometric_coupling_score(
            stress_energy_tensor=0.84,
            curvature_coupling=0.81,
            weak_ordinary_matter_coupling=1.2,
            config=DarkSectorConfig(),
        )
    except ValueError as exc:
        return float("in [0, 1]" in str(exc))
    return 0.0


__all__ = [
    "CLAIM_BOUNDARY",
    "DarkSectorConfig",
    "DarkSectorFixtureResult",
    "PsiDmInteractionValidationResult",
    "cosmic_reservoir_score",
    "dark_energy_context_score",
    "geometric_coupling_score",
    "mmc_information_preservation_score",
    "psi_dm_candidate_label",
    "validate_dark_sector_fixture",
    "validate_psi_dm_interaction_fixture",
]
