# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 The Mechanism of Influence: validation
"""Source-accounting checks for Paper 0 The Mechanism of Influence: records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

CLAIM_BOUNDARY = (
    "source-bounded the mechanism of influence source-accounting bridge; not validation evidence"
)
HARDWARE_STATUS = "source_methodology_no_experiment"
SOURCE_LEDGER_SPAN = ("P0R02616", "P0R02623")


@dataclass(frozen=True, slots=True)
class TheMechanismOfInfluenceConfig:
    """Configuration for this Paper 0 source-accounting fixture."""

    expected_source_record_count: int = 8
    expected_component_count: int = 4
    next_source_boundary: str = "P0R02624"

    def __post_init__(self) -> None:
        if self.expected_source_record_count != 8:
            raise ValueError("expected_source_record_count must equal 8")
        if self.expected_component_count != 4:
            raise ValueError("expected_component_count must equal 4")
        if self.next_source_boundary != "P0R02624":
            raise ValueError("next_source_boundary must equal P0R02624")


@dataclass(frozen=True, slots=True)
class TheMechanismOfInfluenceFixtureResult:
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


def classify_the_mechanism_of_influence_component(component: str) -> str:
    """Classify source-defined components."""
    mapping = {
        "the_mechanism_of_influence": "the_mechanism_of_influence_source_boundary",
        "the_unified_phase_dynamics_equation_upde_the_spine_of_the_scpn": "the_unified_phase_dynamics_equation_upde_the_spine_of_the_scpn_source_boundary",
        "the_upde_formalism": "the_upde_formalism_source_boundary",
        "components_of_the_upde": "components_of_the_upde_source_boundary",
    }
    try:
        return mapping[component]
    except KeyError as exc:
        raise ValueError("unknown the_mechanism_of_influence component") from exc


def the_mechanism_of_influence_labels() -> dict[str, str]:
    """Return source-bounded labels for this slice."""
    return {
        "section": "The Mechanism of Influence:",
        "source_span": "P0R02616-P0R02623",
        "component_count": "4",
        "next_boundary": "P0R02624",
        "component_1": "The Mechanism of Influence:",
        "component_2": "The Unified Phase Dynamics Equation (UPDE) - The Spine of the SCPN",
        "component_3": "The UPDE Formalism:",
        "component_4": "Components of the UPDE:",
    }


def validate_the_mechanism_of_influence_fixture(
    config: TheMechanismOfInfluenceConfig | None = None,
) -> TheMechanismOfInfluenceFixtureResult:
    """Validate source accounting for this Paper 0 slice."""
    cfg = config or TheMechanismOfInfluenceConfig()
    components = (
        "the_mechanism_of_influence",
        "the_unified_phase_dynamics_equation_upde_the_spine_of_the_scpn",
        "the_upde_formalism",
        "components_of_the_upde",
    )
    return TheMechanismOfInfluenceFixtureResult(
        source_ledger_span=SOURCE_LEDGER_SPAN,
        hardware_status=HARDWARE_STATUS,
        claim_boundary=CLAIM_BOUNDARY,
        components={
            component: classify_the_mechanism_of_influence_component(component)
            for component in components
        },
        labels=the_mechanism_of_influence_labels(),
        source_record_count=cfg.expected_source_record_count,
        component_count=cfg.expected_component_count,
        next_source_boundary=cfg.next_source_boundary,
        null_controls={
            "the_mechanism_of_influence_is_not_empirical_validation_evidence": 1.0,
            "the_unified_phase_dynamics_equation_upde_the_spine_of_the_scpn_is_not_empirical_validation_evidence": 1.0,
            "the_upde_formalism_is_not_empirical_validation_evidence": 1.0,
            "components_of_the_upde_is_not_empirical_validation_evidence": 1.0,
        },
        problem_metadata={
            "source_ledger_span": SOURCE_LEDGER_SPAN,
            "source_ledger_ids": tuple(f"P0R{number:05d}" for number in range(2616, 2624)),
            "claim_boundary": CLAIM_BOUNDARY,
            "protocol_state": "source_the_mechanism_of_influence_only_no_experiment",
        },
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "HARDWARE_STATUS",
    "SOURCE_LEDGER_SPAN",
    "TheMechanismOfInfluenceConfig",
    "TheMechanismOfInfluenceFixtureResult",
    "classify_the_mechanism_of_influence_component",
    "the_mechanism_of_influence_labels",
    "validate_the_mechanism_of_influence_fixture",
]
