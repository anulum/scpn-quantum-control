# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 SSB hierarchy genesis validation
"""Source-accounting checks for Paper 0 SSB hierarchy-genesis records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

CLAIM_BOUNDARY = "source-bounded SSB hierarchy-genesis bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"
SOURCE_LEDGER_SPAN = ("P0R01693", "P0R01713")


@dataclass(frozen=True, slots=True)
class SSBHierarchyGenesisConfig:
    """Configuration for the SSB hierarchy-genesis fixture."""

    expected_source_record_count: int = 21
    expected_component_count: int = 3
    next_source_boundary: str = "P0R01714"

    def __post_init__(self) -> None:
        if self.expected_source_record_count != 21:
            raise ValueError("expected_source_record_count must equal 21")
        if self.expected_component_count != 3:
            raise ValueError("expected_component_count must equal 3")
        if self.next_source_boundary != "P0R01714":
            raise ValueError("next_source_boundary must equal P0R01714")


@dataclass(frozen=True, slots=True)
class SSBHierarchyGenesisFixtureResult:
    """Result for the Paper 0 SSB hierarchy-genesis fixture."""

    source_ledger_span: tuple[str, str]
    hardware_status: str
    claim_boundary: str
    components: dict[str, str]
    labels: dict[str, str]
    source_record_count: int
    component_count: int
    next_source_boundary: str
    null_controls: dict[str, float]
    problem_metadata: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-ready result dictionary."""
        return asdict(self)


def classify_ssb_hierarchy_genesis_component(component: str) -> str:
    """Classify source-defined SSB hierarchy-genesis components."""
    mapping = {
        "architecture_cascade": "ssb_architecture_cascade_source_boundary",
        "conformal_torsion_seeding": "conformal_torsion_seeding_source_boundary",
        "three_strike_explanation": "three_strike_explanatory_analogy_boundary",
    }
    try:
        return mapping[component]
    except KeyError as exc:
        raise ValueError("unknown SSB hierarchy-genesis component") from exc


def ssb_hierarchy_genesis_labels() -> dict[str, str]:
    """Return source-bounded labels for the SSB hierarchy-genesis slice."""
    return {
        "section": "The Genesis of the Hierarchy: A Cascade of Sequential Symmetry Breaking",
        "architecture": "15-layer SCPN as sequential SSB remnant",
        "torsion": "V_eff(|Psi|, t -> 0+) = -mu^2(T_SEC) |Psi|^2 + lambda |Psi|^4",
        "breaks": "laws, individuals, actuality",
        "next_boundary": "Meta-Framework Integrations",
    }


def validate_ssb_hierarchy_genesis_fixture(
    config: SSBHierarchyGenesisConfig | None = None,
) -> SSBHierarchyGenesisFixtureResult:
    """Validate source accounting for the SSB hierarchy-genesis slice."""
    cfg = config or SSBHierarchyGenesisConfig()
    components = (
        "architecture_cascade",
        "conformal_torsion_seeding",
        "three_strike_explanation",
    )

    return SSBHierarchyGenesisFixtureResult(
        source_ledger_span=SOURCE_LEDGER_SPAN,
        hardware_status=HARDWARE_STATUS,
        claim_boundary=CLAIM_BOUNDARY,
        components={
            component: classify_ssb_hierarchy_genesis_component(component)
            for component in components
        },
        labels=ssb_hierarchy_genesis_labels(),
        source_record_count=cfg.expected_source_record_count,
        component_count=cfg.expected_component_count,
        next_source_boundary=cfg.next_source_boundary,
        null_controls={
            "architecture_cascade_is_not_measured_layer_hierarchy": 1.0,
            "sec_torsion_seeding_is_not_observational_cosmology": 1.0,
            "three_strike_analogy_is_not_physical_derivation": 1.0,
        },
        problem_metadata={
            "source_ledger_span": SOURCE_LEDGER_SPAN,
            "source_ledger_ids": tuple(f"P0R{number:05d}" for number in range(1693, 1714)),
            "claim_boundary": CLAIM_BOUNDARY,
            "protocol_state": "source_ssb_hierarchy_genesis_only_no_experiment",
        },
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "HARDWARE_STATUS",
    "SOURCE_LEDGER_SPAN",
    "SSBHierarchyGenesisConfig",
    "SSBHierarchyGenesisFixtureResult",
    "classify_ssb_hierarchy_genesis_component",
    "ssb_hierarchy_genesis_labels",
    "validate_ssb_hierarchy_genesis_fixture",
]
