# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 t0-seeding validation fixtures
"""Simulator-only t=0 SSB seeding and spin-torsion bridge fixtures."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np

from .spec_loader import load_t0_seeding_validation_spec

CLAIM_BOUNDARY = "source-bounded t0-seeding simulator contract; not empirical evidence"
SOURCE_LEDGER_SPAN = ("P0R06339", "P0R06362")
TORSION_FORMULAE = (
    "torsion_ijk = 8 pi G s_ijk",
    "torsion_ijk = 8 pi G s_ijk_psi",
)


@dataclass(frozen=True, slots=True)
class T0SeedingConfig:
    """Finite simulator settings for t0-seeding fixtures."""

    memory_weight: float = 0.34
    invariance_weight: float = 0.33
    geometry_weight: float = 0.33
    initial_value_threshold: float = 0.75
    memory_bias_threshold: float = 0.75
    torsion_threshold: float = 0.60
    conformal_torsion_threshold: float = 0.75
    spec_bundle_path: Path | None = None

    def __post_init__(self) -> None:
        for name in (
            "initial_value_threshold",
            "memory_bias_threshold",
            "torsion_threshold",
            "conformal_torsion_threshold",
        ):
            _require_positive(name, float(getattr(self, name)))
        for name in ("memory_weight", "invariance_weight", "geometry_weight"):
            _require_non_negative(name, float(getattr(self, name)))


@dataclass(frozen=True, slots=True)
class SpinTorsionBridgeValidationResult:
    """Spin-torsion bridge result."""

    spec_key: str
    validation_protocol: str
    hardware_status: str
    source_ledger_ids: tuple[str, ...]
    source_formulae: tuple[str, str]
    torsion_bridge_score: float
    psi_torsion_bridge_score: float
    claim_boundary: str
    null_controls: MappingProxyType[str, float]
    problem_metadata: MappingProxyType[str, Any]


@dataclass(frozen=True, slots=True)
class T0SeedingFixtureResult:
    """Combined t0-seeding fixture result."""

    spec_keys: tuple[str, str, str, str, str]
    hardware_status: str
    initial_value_score: float
    memory_bias_score: float
    tachyonic_coefficient: float
    spin_torsion: SpinTorsionBridgeValidationResult
    conformal_torsion_score: float
    config_thresholds: MappingProxyType[str, float]
    source_ledger_span: tuple[str, str]
    claim_boundary: str
    problem_metadata: MappingProxyType[str, Any]


def initial_value_need_score(
    *,
    restored_massless_boundary: float,
    symmetric_vacuum: float,
    ssb_trigger_requirement: float,
    config: T0SeedingConfig,
) -> float:
    """Score the source t=0 initial-value necessity channels."""
    values = np.asarray(
        [restored_massless_boundary, symmetric_vacuum, ssb_trigger_requirement],
        dtype=np.float64,
    )
    _require_unit_interval("initial-value inputs", values)
    return float(np.dot(np.asarray([0.34, 0.33, 0.33], dtype=np.float64), values))


def j_sec_memory_bias_score(
    *,
    preserved_j_sec: float,
    conformal_invariance: float,
    prior_aeon_geometry: float,
    config: T0SeedingConfig,
) -> float:
    """Score preserved J_SEC memory and structured bias channels."""
    values = np.asarray(
        [preserved_j_sec, conformal_invariance, prior_aeon_geometry],
        dtype=np.float64,
    )
    _require_unit_interval("J_SEC memory inputs", values)
    weights = np.asarray(
        [config.memory_weight, config.invariance_weight, config.geometry_weight],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
        raise ValueError("J_SEC memory weights must be finite and non-negative")
    total = float(np.sum(weights))
    if total <= 0.0:
        raise ValueError("at least one J_SEC memory weight must be positive")
    return float(np.dot(weights, values) / total)


def effective_tachyonic_mass_term(
    *,
    j_sec: float,
    coupling: float,
    lambda_coupling: float,
    psi_abs: float,
    config: T0SeedingConfig,
) -> float:
    """Return the source t=0 effective-potential value for a seeded Psi amplitude."""
    values = np.asarray([j_sec, coupling, lambda_coupling, psi_abs], dtype=np.float64)
    _require_non_negative_vector("tachyonic-seed inputs", values)
    mu_squared = coupling * j_sec
    return float(-(mu_squared * psi_abs**2) + lambda_coupling * psi_abs**4)


def spin_torsion_bridge(
    *,
    gravitational_constant: float,
    spin_density: float,
    config: T0SeedingConfig,
) -> float:
    """Return the source spin-torsion bridge proxy torsion = 8 pi G s."""
    values = np.asarray([gravitational_constant, spin_density], dtype=np.float64)
    _require_non_negative_vector("spin-torsion inputs", values)
    return float(8.0 * np.pi * gravitational_constant * spin_density)


def conformal_invariant_torsion_score(
    *,
    preserved_j_sec: float,
    t_sec: float,
    structured_bias: float,
    non_random_reset: float,
    config: T0SeedingConfig,
) -> float:
    """Score J_SEC as conformal-invariant torsion and structured t->0+ bias."""
    values = np.asarray(
        [preserved_j_sec, t_sec, structured_bias, non_random_reset],
        dtype=np.float64,
    )
    _require_unit_interval("conformal-invariant torsion inputs", values)
    return float(np.dot(np.asarray([0.25, 0.25, 0.25, 0.25], dtype=np.float64), values))


def validate_spin_torsion_bridge_fixture(
    config: T0SeedingConfig | None = None,
) -> SpinTorsionBridgeValidationResult:
    """Run the spin-torsion bridge fixture."""
    cfg = config or T0SeedingConfig()
    spec = load_t0_seeding_validation_spec(
        "t0_seeding.spin_torsion_bridge_equations",
        spec_bundle_path=cfg.spec_bundle_path,
    )
    source_formulae = tuple(str(item) for item in spec["source_formulae"])
    if len(source_formulae) != 2:
        raise ValueError("spin-torsion spec must preserve exactly two source formulae")
    torsion = spin_torsion_bridge(
        gravitational_constant=0.67,
        spin_density=0.73,
        config=cfg,
    )
    psi_torsion = spin_torsion_bridge(
        gravitational_constant=0.67,
        spin_density=0.81,
        config=cfg,
    )
    controls = {
        "negative_spin_density_rejection_label": _negative_spin_density_rejection_label(),
        "missing_psi_spin_density_rejection_label": 1.0,
        "unsupported_torsion_evidence_rejection_label": 1.0,
    }
    return SpinTorsionBridgeValidationResult(
        spec_key="t0_seeding.spin_torsion_bridge_equations",
        validation_protocol=str(spec["validation_protocol"]),
        hardware_status=str(spec["hardware_status"]),
        source_ledger_ids=tuple(str(item) for item in spec["source_ledger_ids"]),
        source_formulae=(source_formulae[0], source_formulae[1]),
        torsion_bridge_score=torsion,
        psi_torsion_bridge_score=psi_torsion,
        claim_boundary=CLAIM_BOUNDARY,
        null_controls=MappingProxyType(controls),
        problem_metadata=MappingProxyType(
            {
                "torsion_threshold": cfg.torsion_threshold,
                "source_ledger_span": SOURCE_LEDGER_SPAN,
                "claim_boundary": CLAIM_BOUNDARY,
            }
        ),
    )


def validate_t0_seeding_fixture(config: T0SeedingConfig | None = None) -> T0SeedingFixtureResult:
    """Run the combined t0-seeding fixture."""
    cfg = config or T0SeedingConfig()
    keys = (
        "t0_seeding.initial_value_problem_boundary",
        "t0_seeding.j_sec_memory_bias_boundary",
        "t0_seeding.teleological_tachyonic_potential",
        "t0_seeding.spin_torsion_bridge_equations",
        "t0_seeding.conformal_invariant_torsion_boundary",
    )
    specs = tuple(
        load_t0_seeding_validation_spec(key, spec_bundle_path=cfg.spec_bundle_path) for key in keys
    )
    initial_value = initial_value_need_score(
        restored_massless_boundary=0.91,
        symmetric_vacuum=0.88,
        ssb_trigger_requirement=0.86,
        config=cfg,
    )
    memory = j_sec_memory_bias_score(
        preserved_j_sec=0.89,
        conformal_invariance=0.84,
        prior_aeon_geometry=0.81,
        config=cfg,
    )
    tachyonic = effective_tachyonic_mass_term(
        j_sec=0.82,
        coupling=0.74,
        lambda_coupling=0.41,
        psi_abs=0.35,
        config=cfg,
    )
    torsion = validate_spin_torsion_bridge_fixture(cfg)
    conformal_torsion = conformal_invariant_torsion_score(
        preserved_j_sec=0.89,
        t_sec=0.86,
        structured_bias=0.84,
        non_random_reset=0.82,
        config=cfg,
    )
    return T0SeedingFixtureResult(
        spec_keys=keys,
        hardware_status=str(specs[0]["hardware_status"]),
        initial_value_score=initial_value,
        memory_bias_score=memory,
        tachyonic_coefficient=tachyonic,
        spin_torsion=torsion,
        conformal_torsion_score=conformal_torsion,
        config_thresholds=MappingProxyType(
            {
                "initial_value_threshold": cfg.initial_value_threshold,
                "memory_bias_threshold": cfg.memory_bias_threshold,
                "torsion_threshold": cfg.torsion_threshold,
                "conformal_torsion_threshold": cfg.conformal_torsion_threshold,
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


def _require_non_negative_vector(name: str, values: np.ndarray) -> None:
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError(f"{name} must be finite and non-negative")


def _require_unit_interval(name: str, values: np.ndarray) -> None:
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must be finite")
    if np.any(values < 0.0) or np.any(values > 1.0):
        raise ValueError(f"{name} must be in [0, 1]")


def _negative_spin_density_rejection_label() -> float:
    try:
        spin_torsion_bridge(
            gravitational_constant=0.67,
            spin_density=-0.01,
            config=T0SeedingConfig(),
        )
    except ValueError as exc:
        return float("finite and non-negative" in str(exc))
    return 0.0


__all__ = [
    "CLAIM_BOUNDARY",
    "SpinTorsionBridgeValidationResult",
    "T0SeedingConfig",
    "T0SeedingFixtureResult",
    "conformal_invariant_torsion_score",
    "effective_tachyonic_mass_term",
    "initial_value_need_score",
    "j_sec_memory_bias_score",
    "spin_torsion_bridge",
    "validate_spin_torsion_bridge_fixture",
    "validate_t0_seeding_fixture",
]
