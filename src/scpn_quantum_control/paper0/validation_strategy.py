# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 validation-strategy fixtures
"""Executable validation-roadmap contract for Paper 0 Applied SCPN records."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from .spec_loader import load_validation_strategy_spec

CLAIM_BOUNDARY = "validation-roadmap contract; not empirical evidence"
STAGE_ORDER = ("Stage I", "Stage II", "Stage III")
REQUIRED_DOMAINS = (
    "pathology",
    "societal_phase_transitions_l11",
    "ethical_governance_l15",
    "alignment_objective",
    "quasicriticality_universal",
    "biological_quantum_interface_l1_l2",
    "geometry_of_qualia_l5",
    "upde_multi_scale_pac",
    "hpc_transfer_entropy",
    "gaian_coupling_l12",
    "teleology_l15_l8",
)


@dataclass(frozen=True, slots=True)
class ValidationDomainTarget:
    """Single validation-roadmap target promoted from Paper 0 Applied SCPN."""

    domain: str
    stage: str
    target_type: str
    method: str
    claim_boundary: str = CLAIM_BOUNDARY


@dataclass(frozen=True, slots=True)
class ValidationStrategyConfig:
    """Finite validation-roadmap settings for Paper 0 Applied SCPN records."""

    domains: tuple[str, ...] = REQUIRED_DOMAINS
    stages: tuple[str, ...] = STAGE_ORDER
    spec_bundle_path: Path | None = None

    def __post_init__(self) -> None:
        if not self.stages:
            raise ValueError("at least one validation stage is required")
        unknown_stages = [stage for stage in self.stages if stage not in STAGE_ORDER]
        if unknown_stages:
            raise ValueError(f"unknown validation stage entries: {unknown_stages}")
        if len(set(self.domains)) != len(self.domains):
            raise ValueError("duplicate domain entries are not allowed")
        missing = sorted(set(REQUIRED_DOMAINS) - set(self.domains))
        if missing:
            raise ValueError(f"missing required source domain entries: {missing}")


@dataclass(frozen=True, slots=True)
class ValidationStrategyFixtureResult:
    """Combined Paper 0 validation-strategy fixture result."""

    spec_keys: tuple[str, ...]
    validation_protocols: tuple[str, ...]
    hardware_status: str
    domains: tuple[str, ...]
    stages: tuple[str, ...]
    domain_count: int
    stage_count: int
    stage_order_valid: bool
    claim_boundary: str
    null_controls: MappingProxyType[str, float]
    problem_metadata: MappingProxyType[str, Any]


def stage_order_index(stage: str) -> int:
    """Return the zero-based order index for a named validation stage."""
    try:
        return STAGE_ORDER.index(stage)
    except ValueError as exc:
        raise ValueError(f"unknown validation stage: {stage}") from exc


def validate_stage_order(stages: tuple[str, ...]) -> bool:
    """Return whether stages are non-empty and strictly ordered."""
    if not stages:
        raise ValueError("at least one validation stage is required")
    indices = [stage_order_index(stage) for stage in stages]
    return bool(indices == sorted(indices) and len(set(indices)) == len(indices))


def validation_domain_coverage(
    config: ValidationStrategyConfig | None = None,
) -> dict[str, ValidationDomainTarget]:
    """Return source-covered validation domains and their target methods."""
    cfg = config or ValidationStrategyConfig()
    targets = {
        "pathology": ValidationDomainTarget(
            domain="pathology",
            stage="Cross-cutting",
            target_type="systems_state",
            method="criticality/free-energy/MS-QEC target classification",
        ),
        "societal_phase_transitions_l11": ValidationDomainTarget(
            domain="societal_phase_transitions_l11",
            stage="Cross-cutting",
            target_type="spin_glass_dynamics",
            method="L11 spin-glass and quasicritical sigma target classification",
        ),
        "ethical_governance_l15": ValidationDomainTarget(
            domain="ethical_governance_l15",
            stage="Cross-cutting",
            target_type="ethical_lagrangian_cef",
            method="SEC, ethical Lagrangian, and CEF target classification",
        ),
        "alignment_objective": ValidationDomainTarget(
            domain="alignment_objective",
            stage="Cross-cutting",
            target_type="ethical_functional_embedding",
            method="ethical-functional objective target classification",
        ),
        "quasicriticality_universal": ValidationDomainTarget(
            domain="quasicriticality_universal",
            stage="Stage I",
            target_type="foundational_quasicriticality",
            method="universal quasicriticality target classification",
        ),
        "biological_quantum_interface_l1_l2": ValidationDomainTarget(
            domain="biological_quantum_interface_l1_l2",
            stage="Stage I",
            target_type="biological_quantum_interface",
            method="QEC-gap, synaptotagmin modulation, and CIGD target classification",
        ),
        "geometry_of_qualia_l5": ValidationDomainTarget(
            domain="geometry_of_qualia_l5",
            stage="Stage I",
            target_type="pta_tda_geometry",
            method="PTA/TDA target classification",
        ),
        "upde_multi_scale_pac": ValidationDomainTarget(
            domain="upde_multi_scale_pac",
            stage="Stage II",
            target_type="upde_mechanism",
            method="multi-scale PAC target classification",
        ),
        "hpc_transfer_entropy": ValidationDomainTarget(
            domain="hpc_transfer_entropy",
            stage="Stage II",
            target_type="hpc_mechanism",
            method="transfer-entropy target classification",
        ),
        "gaian_coupling_l12": ValidationDomainTarget(
            domain="gaian_coupling_l12",
            stage="Stage III",
            target_type="gaian_coupling",
            method="Schumann-resonance target classification",
        ),
        "teleology_l15_l8": ValidationDomainTarget(
            domain="teleology_l15_l8",
            stage="Stage III",
            target_type="teleology_rg_flow",
            method="L15/L8 RG-flow target classification",
        ),
    }
    return {domain: targets[domain] for domain in cfg.domains}


def validate_validation_strategy_fixture(
    config: ValidationStrategyConfig | None = None,
) -> ValidationStrategyFixtureResult:
    """Run the source-anchored validation-roadmap contract fixture."""
    cfg = config or ValidationStrategyConfig()
    specs = [
        load_validation_strategy_spec(
            "applied.validation_strategy.pathology_and_societal_phase_targets",
            spec_bundle_path=cfg.spec_bundle_path,
        ),
        load_validation_strategy_spec(
            "applied.validation_strategy.ethical_governance_alignment_targets",
            spec_bundle_path=cfg.spec_bundle_path,
        ),
        load_validation_strategy_spec(
            "applied.validation_strategy.stage_i_foundations",
            spec_bundle_path=cfg.spec_bundle_path,
        ),
        load_validation_strategy_spec(
            "applied.validation_strategy.stage_ii_iii_mechanisms_and_high_level",
            spec_bundle_path=cfg.spec_bundle_path,
        ),
    ]
    coverage = validation_domain_coverage(cfg)
    controls = {
        "duplicate_domain_rejection_label": _duplicate_domain_rejection_label(),
        "unknown_stage_rejection_label": _unknown_stage_rejection_label(),
        "missing_domain_rejection_label": _missing_domain_rejection_label(),
        "stage_order_valid_label": float(validate_stage_order(cfg.stages)),
    }
    return ValidationStrategyFixtureResult(
        spec_keys=tuple(str(spec["key"]) for spec in specs),
        validation_protocols=tuple(str(spec["validation_protocol"]) for spec in specs),
        hardware_status=str(specs[0]["hardware_status"]),
        domains=tuple(coverage),
        stages=cfg.stages,
        domain_count=len(coverage),
        stage_count=len(cfg.stages),
        stage_order_valid=validate_stage_order(cfg.stages),
        claim_boundary=CLAIM_BOUNDARY,
        null_controls=MappingProxyType(controls),
        problem_metadata=MappingProxyType(
            {
                "source_ledger_ids": tuple(str(item) for item in specs[0]["source_ledger_ids"]),
                "roadmap_only": True,
                "claim_boundary": CLAIM_BOUNDARY,
            }
        ),
    )


def _duplicate_domain_rejection_label() -> float:
    try:
        ValidationStrategyConfig(domains=("pathology", "pathology"))
    except ValueError as exc:
        return float("duplicate domain" in str(exc))
    return 0.0


def _unknown_stage_rejection_label() -> float:
    try:
        stage_order_index("Stage IV")
    except ValueError as exc:
        return float("unknown validation stage" in str(exc))
    return 0.0


def _missing_domain_rejection_label() -> float:
    try:
        ValidationStrategyConfig(domains=("pathology",))
    except ValueError as exc:
        return float("required source domain" in str(exc))
    return 0.0


__all__ = [
    "CLAIM_BOUNDARY",
    "REQUIRED_DOMAINS",
    "STAGE_ORDER",
    "ValidationDomainTarget",
    "ValidationStrategyConfig",
    "ValidationStrategyFixtureResult",
    "stage_order_index",
    "validate_stage_order",
    "validate_validation_strategy_fixture",
    "validation_domain_coverage",
]
