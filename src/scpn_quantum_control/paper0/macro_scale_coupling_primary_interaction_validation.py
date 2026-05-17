# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 Macro-Scale Coupling (Primary Interaction): validation
"""Source-accounting checks for Paper 0 Macro-Scale Coupling (Primary Interaction): records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

CLAIM_BOUNDARY = "source-bounded macro scale coupling primary interaction source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"
SOURCE_LEDGER_SPAN = ("P0R02107", "P0R02127")


@dataclass(frozen=True, slots=True)
class MacroScaleCouplingPrimaryInteractionConfig:
    """Configuration for this Paper 0 source-accounting fixture."""

    expected_source_record_count: int = 21
    expected_component_count: int = 4
    next_source_boundary: str = "P0R02128"

    def __post_init__(self) -> None:
        if self.expected_source_record_count != 21:
            raise ValueError("expected_source_record_count must equal 21")
        if self.expected_component_count != 4:
            raise ValueError("expected_component_count must equal 4")
        if self.next_source_boundary != "P0R02128":
            raise ValueError("next_source_boundary must equal P0R02128")


@dataclass(frozen=True, slots=True)
class MacroScaleCouplingPrimaryInteractionFixtureResult:
    """Result for this Paper 0 source-accounting fixture."""

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


def classify_macro_scale_coupling_primary_interaction_component(component: str) -> str:
    """Classify source-defined components."""
    mapping = {
        "macro_scale_coupling_primary_interaction": "macro_scale_coupling_primary_interaction_source_boundary",
        "meso_scale_transduction": "meso_scale_transduction_source_boundary",
        "quantum_scale_coupling_secondary_interaction": "quantum_scale_coupling_secondary_interaction_source_boundary",
        "domain_i_biological_substrate_layers_1_4": "domain_i_biological_substrate_layers_1_4_source_boundary",
    }
    try:
        return mapping[component]
    except KeyError as exc:
        raise ValueError("unknown macro_scale_coupling_primary_interaction component") from exc


def macro_scale_coupling_primary_interaction_labels() -> dict[str, str]:
    """Return source-bounded labels for this slice."""
    return {
        "section": "Macro-Scale Coupling (Primary Interaction):",
        "source_span": "P0R02107-P0R02127",
        "component_count": "4",
        "next_boundary": "P0R02128",
        "component_1": "Macro-Scale Coupling (Primary Interaction):",
        "component_2": "Meso-Scale Transduction:",
        "component_3": "Quantum-Scale Coupling (Secondary Interaction):",
        "component_4": "Domain I: Biological Substrate (Layers 1-4):",
    }


def validate_macro_scale_coupling_primary_interaction_fixture(
    config: MacroScaleCouplingPrimaryInteractionConfig | None = None,
) -> MacroScaleCouplingPrimaryInteractionFixtureResult:
    """Validate source accounting for this Paper 0 slice."""
    cfg = config or MacroScaleCouplingPrimaryInteractionConfig()
    components = (
        "macro_scale_coupling_primary_interaction",
        "meso_scale_transduction",
        "quantum_scale_coupling_secondary_interaction",
        "domain_i_biological_substrate_layers_1_4",
    )
    return MacroScaleCouplingPrimaryInteractionFixtureResult(
        source_ledger_span=SOURCE_LEDGER_SPAN,
        hardware_status=HARDWARE_STATUS,
        claim_boundary=CLAIM_BOUNDARY,
        components={
            component: classify_macro_scale_coupling_primary_interaction_component(component)
            for component in components
        },
        labels=macro_scale_coupling_primary_interaction_labels(),
        source_record_count=cfg.expected_source_record_count,
        component_count=cfg.expected_component_count,
        next_source_boundary=cfg.next_source_boundary,
        null_controls={
            "macro_scale_coupling_primary_interaction_is_not_empirical_validation_evidence": 1.0,
            "meso_scale_transduction_is_not_empirical_validation_evidence": 1.0,
            "quantum_scale_coupling_secondary_interaction_is_not_empirical_validation_evidence": 1.0,
            "domain_i_biological_substrate_layers_1_4_is_not_empirical_validation_evidence": 1.0,
        },
        problem_metadata={
            "source_ledger_span": SOURCE_LEDGER_SPAN,
            "source_ledger_ids": tuple(f"P0R{number:05d}" for number in range(2107, 2128)),
            "claim_boundary": CLAIM_BOUNDARY,
            "protocol_state": "source_macro_scale_coupling_primary_interaction_only_no_experiment",
        },
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "HARDWARE_STATUS",
    "SOURCE_LEDGER_SPAN",
    "MacroScaleCouplingPrimaryInteractionConfig",
    "MacroScaleCouplingPrimaryInteractionFixtureResult",
    "classify_macro_scale_coupling_primary_interaction_component",
    "macro_scale_coupling_primary_interaction_labels",
    "validate_macro_scale_coupling_primary_interaction_fixture",
]
