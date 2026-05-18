# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 III. The Developmental and Plasticity Landscape (L3) validation
"""Source-accounting checks for Paper 0 III. The Developmental and Plasticity Landscape (L3) records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

CLAIM_BOUNDARY = "source-bounded iii the developmental and plasticity landscape l3 source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"
SOURCE_LEDGER_SPAN = ("P0R04478", "P0R04487")


@dataclass(frozen=True, slots=True)
class IiiTheDevelopmentalAndPlasticityLandscapeL3Config:
    """Configuration for this Paper 0 source-accounting fixture."""

    expected_source_record_count: int = 10
    expected_component_count: int = 3
    next_source_boundary: str = "P0R04488"

    def __post_init__(self) -> None:
        if self.expected_source_record_count != 10:
            raise ValueError("expected_source_record_count must equal 10")
        if self.expected_component_count != 3:
            raise ValueError("expected_component_count must equal 3")
        if self.next_source_boundary != "P0R04488":
            raise ValueError("next_source_boundary must equal P0R04488")


@dataclass(frozen=True, slots=True)
class IiiTheDevelopmentalAndPlasticityLandscapeL3FixtureResult:
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


def classify_iii_the_developmental_and_plasticity_landscape_l3_component(component: str) -> str:
    """Classify source-defined components."""
    mapping = {
        "iii_the_developmental_and_plasticity_landscape_l3": "iii_the_developmental_and_plasticity_landscape_l3_source_boundary",
        "iv_the_dynamic_core_synchronisation_criticality_and_the_connectome_l4": "iv_the_dynamic_core_synchronisation_criticality_and_the_connectome_l4_source_boundary",
        "1_the_upde_in_the_brain_the_neural_symphony": "1_the_upde_in_the_brain_the_neural_symphony_source_boundary",
    }
    try:
        return mapping[component]
    except KeyError as exc:
        raise ValueError(
            "unknown iii_the_developmental_and_plasticity_landscape_l3 component"
        ) from exc


def iii_the_developmental_and_plasticity_landscape_l3_labels() -> dict[str, str]:
    """Return source-bounded labels for this slice."""
    return {
        "section": "III. The Developmental and Plasticity Landscape (L3)",
        "source_span": "P0R04478-P0R04487",
        "component_count": "3",
        "next_boundary": "P0R04488",
        "component_1": "III. The Developmental and Plasticity Landscape (L3)",
        "component_2": "IV. The Dynamic Core: Synchronisation, Criticality, and the Connectome (L4)",
        "component_3": "1. The UPDE in the Brain (The Neural Symphony):",
    }


def validate_iii_the_developmental_and_plasticity_landscape_l3_fixture(
    config: IiiTheDevelopmentalAndPlasticityLandscapeL3Config | None = None,
) -> IiiTheDevelopmentalAndPlasticityLandscapeL3FixtureResult:
    """Validate source accounting for this Paper 0 slice."""
    cfg = config or IiiTheDevelopmentalAndPlasticityLandscapeL3Config()
    components = (
        "iii_the_developmental_and_plasticity_landscape_l3",
        "iv_the_dynamic_core_synchronisation_criticality_and_the_connectome_l4",
        "1_the_upde_in_the_brain_the_neural_symphony",
    )
    return IiiTheDevelopmentalAndPlasticityLandscapeL3FixtureResult(
        source_ledger_span=SOURCE_LEDGER_SPAN,
        hardware_status=HARDWARE_STATUS,
        claim_boundary=CLAIM_BOUNDARY,
        components={
            component: classify_iii_the_developmental_and_plasticity_landscape_l3_component(
                component
            )
            for component in components
        },
        labels=iii_the_developmental_and_plasticity_landscape_l3_labels(),
        source_record_count=cfg.expected_source_record_count,
        component_count=cfg.expected_component_count,
        next_source_boundary=cfg.next_source_boundary,
        null_controls={
            "iii_the_developmental_and_plasticity_landscape_l3_is_not_empirical_validation_evidence": 1.0,
            "iv_the_dynamic_core_synchronisation_criticality_and_the_connectome_l4_is_not_empirical_validation_evidence": 1.0,
            "1_the_upde_in_the_brain_the_neural_symphony_is_not_empirical_validation_evidence": 1.0,
        },
        problem_metadata={
            "source_ledger_span": SOURCE_LEDGER_SPAN,
            "source_ledger_ids": tuple(f"P0R{number:05d}" for number in range(4478, 4488)),
            "claim_boundary": CLAIM_BOUNDARY,
            "protocol_state": "source_iii_the_developmental_and_plasticity_landscape_l3_only_no_experiment",
        },
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "HARDWARE_STATUS",
    "SOURCE_LEDGER_SPAN",
    "IiiTheDevelopmentalAndPlasticityLandscapeL3Config",
    "IiiTheDevelopmentalAndPlasticityLandscapeL3FixtureResult",
    "classify_iii_the_developmental_and_plasticity_landscape_l3_component",
    "iii_the_developmental_and_plasticity_landscape_l3_labels",
    "validate_iii_the_developmental_and_plasticity_landscape_l3_fixture",
]
