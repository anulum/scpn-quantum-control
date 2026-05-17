# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 phenomenological Lagrangian validation
"""Source-accounting checks for Paper 0 phenomenological Lagrangian records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

CLAIM_BOUNDARY = "source-bounded phenomenological Lagrangian scaffold; not derived gauge theory"
HARDWARE_STATUS = "source_methodology_no_experiment"
SOURCE_LEDGER_SPAN = ("P0R01333", "P0R01383")


@dataclass(frozen=True, slots=True)
class PhenomenologicalLagrangianConfig:
    """Configuration for the phenomenological Lagrangian fixture."""

    expected_source_record_count: int = 51
    expected_component_count: int = 5
    next_source_boundary: str = "P0R01384"

    def __post_init__(self) -> None:
        if self.expected_source_record_count != 51:
            raise ValueError("expected_source_record_count must equal 51")
        if self.expected_component_count != 5:
            raise ValueError("expected_component_count must equal 5")
        if self.next_source_boundary != "P0R01384":
            raise ValueError("next_source_boundary must equal P0R01384")


@dataclass(frozen=True, slots=True)
class PhenomenologicalLagrangianFixtureResult:
    """Result for the Paper 0 phenomenological Lagrangian fixture."""

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


def classify_phenomenological_lagrangian_component(component: str) -> str:
    """Classify source-defined phenomenological Lagrangian components."""
    mapping = {
        "section_opening_dual_coupling": "phenomenological_scaffold_dual_coupling_boundary",
        "predictive_coding_free_energy": "heuristic_free_energy_mapping_boundary",
        "black_box_interaction": "black_box_h_int_limitation_boundary",
        "master_interaction_terms": "early_master_interaction_equation_boundary",
        "architecture_stationary_action": "stationary_action_upde_scaffold_boundary",
    }
    try:
        return mapping[component]
    except KeyError as exc:
        raise ValueError("unknown phenomenological Lagrangian component") from exc


def phenomenological_lagrangian_labels() -> dict[str, str]:
    """Return source-bounded labels for the phenomenological Lagrangian slice."""
    return {
        "section": "The Phenomenological Formulation: An Evolutionary Starting Point",
        "total_lagrangian": "L_Total = L_Psi + L_Physical + L_Int",
        "interaction_split": "L_Int = L_Geometric + L_Informational",
        "upde_scaffold": "delta S_Master / delta theta_iL = 0 -> d theta_iL / dt = omega_iL + sum K_ij sin(Delta theta) + ...",
        "next_boundary": "Deriving the Master Interaction Lagrangian",
    }


def validate_phenomenological_lagrangian_fixture(
    config: PhenomenologicalLagrangianConfig | None = None,
) -> PhenomenologicalLagrangianFixtureResult:
    """Validate source accounting for the phenomenological Lagrangian slice."""
    cfg = config or PhenomenologicalLagrangianConfig()
    components = (
        "section_opening_dual_coupling",
        "predictive_coding_free_energy",
        "black_box_interaction",
        "master_interaction_terms",
        "architecture_stationary_action",
    )

    return PhenomenologicalLagrangianFixtureResult(
        source_ledger_span=SOURCE_LEDGER_SPAN,
        hardware_status=HARDWARE_STATUS,
        claim_boundary=CLAIM_BOUNDARY,
        components={
            component: classify_phenomenological_lagrangian_component(component)
            for component in components
        },
        labels=phenomenological_lagrangian_labels(),
        source_record_count=cfg.expected_source_record_count,
        component_count=cfg.expected_component_count,
        next_source_boundary=cfg.next_source_boundary,
        null_controls={
            "phenomenological_scaffold_is_not_final_gauge_derivation": 1.0,
            "black_box_h_int_must_not_satisfy_symmetry_derived_boundary": 1.0,
            "stationary_action_scaffold_is_not_complete_upde_proof": 1.0,
        },
        problem_metadata={
            "source_ledger_span": SOURCE_LEDGER_SPAN,
            "source_ledger_ids": tuple(f"P0R{number:05d}" for number in range(1333, 1384)),
            "claim_boundary": CLAIM_BOUNDARY,
            "protocol_state": "source_phenomenological_lagrangian_scaffold_only_no_experiment",
        },
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "HARDWARE_STATUS",
    "SOURCE_LEDGER_SPAN",
    "PhenomenologicalLagrangianConfig",
    "PhenomenologicalLagrangianFixtureResult",
    "classify_phenomenological_lagrangian_component",
    "phenomenological_lagrangian_labels",
    "validate_phenomenological_lagrangian_fixture",
]
