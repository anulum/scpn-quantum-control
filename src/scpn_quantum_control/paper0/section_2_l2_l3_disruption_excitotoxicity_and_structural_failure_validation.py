# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 2. L2/L3 Disruption (Excitotoxicity and Structural Failure): validation
"""Source-accounting checks for Paper 0 2. L2/L3 Disruption (Excitotoxicity and Structural Failure): records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

CLAIM_BOUNDARY = "source-bounded section 2 l2 l3 disruption excitotoxicity and structural failure source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"
SOURCE_LEDGER_SPAN = ("P0R05058", "P0R05065")


@dataclass(frozen=True, slots=True)
class Section2L2L3DisruptionExcitotoxicityAndStructuralFailureConfig:
    """Configuration for this Paper 0 source-accounting fixture."""

    expected_source_record_count: int = 8
    expected_component_count: int = 2
    next_source_boundary: str = "P0R05066"

    def __post_init__(self) -> None:
        if self.expected_source_record_count != 8:
            raise ValueError("expected_source_record_count must equal 8")
        if self.expected_component_count != 2:
            raise ValueError("expected_component_count must equal 2")
        if self.next_source_boundary != "P0R05066":
            raise ValueError("next_source_boundary must equal P0R05066")


@dataclass(frozen=True, slots=True)
class Section2L2L3DisruptionExcitotoxicityAndStructuralFailureFixtureResult:
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


def classify_section_2_l2_l3_disruption_excitotoxicity_and_structural_failure_component(
    component: str,
) -> str:
    """Classify source-defined components."""
    mapping = {
        "2_l2_l3_disruption_excitotoxicity_and_structural_failure": "2_l2_l3_disruption_excitotoxicity_and_structural_failure_source_boundary",
        "3_l4_disruption_dyscritia_and_desynchronization": "3_l4_disruption_dyscritia_and_desynchronization_source_boundary",
    }
    try:
        return mapping[component]
    except KeyError as exc:
        raise ValueError(
            "unknown section_2_l2_l3_disruption_excitotoxicity_and_structural_failure component"
        ) from exc


def section_2_l2_l3_disruption_excitotoxicity_and_structural_failure_labels() -> dict[str, str]:
    """Return source-bounded labels for this slice."""
    return {
        "section": "2. L2/L3 Disruption (Excitotoxicity and Structural Failure):",
        "source_span": "P0R05058-P0R05065",
        "component_count": "2",
        "next_boundary": "P0R05066",
        "component_1": "2. L2/L3 Disruption (Excitotoxicity and Structural Failure):",
        "component_2": "3. L4 Disruption (Dyscritia and Desynchronization):",
    }


def validate_section_2_l2_l3_disruption_excitotoxicity_and_structural_failure_fixture(
    config: Section2L2L3DisruptionExcitotoxicityAndStructuralFailureConfig | None = None,
) -> Section2L2L3DisruptionExcitotoxicityAndStructuralFailureFixtureResult:
    """Validate source accounting for this Paper 0 slice."""
    cfg = config or Section2L2L3DisruptionExcitotoxicityAndStructuralFailureConfig()
    components = (
        "2_l2_l3_disruption_excitotoxicity_and_structural_failure",
        "3_l4_disruption_dyscritia_and_desynchronization",
    )
    return Section2L2L3DisruptionExcitotoxicityAndStructuralFailureFixtureResult(
        source_ledger_span=SOURCE_LEDGER_SPAN,
        hardware_status=HARDWARE_STATUS,
        claim_boundary=CLAIM_BOUNDARY,
        components={
            component: classify_section_2_l2_l3_disruption_excitotoxicity_and_structural_failure_component(
                component
            )
            for component in components
        },
        labels=section_2_l2_l3_disruption_excitotoxicity_and_structural_failure_labels(),
        source_record_count=cfg.expected_source_record_count,
        component_count=cfg.expected_component_count,
        next_source_boundary=cfg.next_source_boundary,
        null_controls={
            "2_l2_l3_disruption_excitotoxicity_and_structural_failure_is_not_empirical_validation_evidence": 1.0,
            "3_l4_disruption_dyscritia_and_desynchronization_is_not_empirical_validation_evidence": 1.0,
        },
        problem_metadata={
            "source_ledger_span": SOURCE_LEDGER_SPAN,
            "source_ledger_ids": tuple(f"P0R{number:05d}" for number in range(5058, 5066)),
            "claim_boundary": CLAIM_BOUNDARY,
            "protocol_state": "source_section_2_l2_l3_disruption_excitotoxicity_and_structural_failure_only_no_experiment",
        },
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "HARDWARE_STATUS",
    "SOURCE_LEDGER_SPAN",
    "Section2L2L3DisruptionExcitotoxicityAndStructuralFailureConfig",
    "Section2L2L3DisruptionExcitotoxicityAndStructuralFailureFixtureResult",
    "classify_section_2_l2_l3_disruption_excitotoxicity_and_structural_failure_component",
    "section_2_l2_l3_disruption_excitotoxicity_and_structural_failure_labels",
    "validate_section_2_l2_l3_disruption_excitotoxicity_and_structural_failure_fixture",
]
