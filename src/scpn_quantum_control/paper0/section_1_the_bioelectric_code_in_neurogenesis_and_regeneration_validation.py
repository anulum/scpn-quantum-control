# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 1. The Bioelectric Code in Neurogenesis and Regeneration: validation
"""Source-accounting checks for Paper 0 1. The Bioelectric Code in Neurogenesis and Regeneration: records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

CLAIM_BOUNDARY = "source-bounded section 1 the bioelectric code in neurogenesis and regeneration source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"
SOURCE_LEDGER_SPAN = ("P0R04657", "P0R04665")


@dataclass(frozen=True, slots=True)
class Section1TheBioelectricCodeInNeurogenesisAndRegenerationConfig:
    """Configuration for this Paper 0 source-accounting fixture."""

    expected_source_record_count: int = 9
    expected_component_count: int = 3
    next_source_boundary: str = "P0R04666"

    def __post_init__(self) -> None:
        if self.expected_source_record_count != 9:
            raise ValueError("expected_source_record_count must equal 9")
        if self.expected_component_count != 3:
            raise ValueError("expected_component_count must equal 3")
        if self.next_source_boundary != "P0R04666":
            raise ValueError("next_source_boundary must equal P0R04666")


@dataclass(frozen=True, slots=True)
class Section1TheBioelectricCodeInNeurogenesisAndRegenerationFixtureResult:
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


def classify_section_1_the_bioelectric_code_in_neurogenesis_and_regeneration_component(
    component: str,
) -> str:
    """Classify source-defined components."""
    mapping = {
        "1_the_bioelectric_code_in_neurogenesis_and_regeneration": "1_the_bioelectric_code_in_neurogenesis_and_regeneration_source_boundary",
        "2_the_optimised_connectome": "2_the_optimised_connectome_source_boundary",
        "3_the_active_role_of_glia_the_tripartite_synapse": "3_the_active_role_of_glia_the_tripartite_synapse_source_boundary",
    }
    try:
        return mapping[component]
    except KeyError as exc:
        raise ValueError(
            "unknown section_1_the_bioelectric_code_in_neurogenesis_and_regeneration component"
        ) from exc


def section_1_the_bioelectric_code_in_neurogenesis_and_regeneration_labels() -> dict[str, str]:
    """Return source-bounded labels for this slice."""
    return {
        "section": "1. The Bioelectric Code in Neurogenesis and Regeneration:",
        "source_span": "P0R04657-P0R04665",
        "component_count": "3",
        "next_boundary": "P0R04666",
        "component_1": "1. The Bioelectric Code in Neurogenesis and Regeneration:",
        "component_2": "2. The Optimised Connectome:",
        "component_3": "3. The Active Role of Glia (The Tripartite Synapse):",
    }


def validate_section_1_the_bioelectric_code_in_neurogenesis_and_regeneration_fixture(
    config: Section1TheBioelectricCodeInNeurogenesisAndRegenerationConfig | None = None,
) -> Section1TheBioelectricCodeInNeurogenesisAndRegenerationFixtureResult:
    """Validate source accounting for this Paper 0 slice."""
    cfg = config or Section1TheBioelectricCodeInNeurogenesisAndRegenerationConfig()
    components = (
        "1_the_bioelectric_code_in_neurogenesis_and_regeneration",
        "2_the_optimised_connectome",
        "3_the_active_role_of_glia_the_tripartite_synapse",
    )
    return Section1TheBioelectricCodeInNeurogenesisAndRegenerationFixtureResult(
        source_ledger_span=SOURCE_LEDGER_SPAN,
        hardware_status=HARDWARE_STATUS,
        claim_boundary=CLAIM_BOUNDARY,
        components={
            component: classify_section_1_the_bioelectric_code_in_neurogenesis_and_regeneration_component(
                component
            )
            for component in components
        },
        labels=section_1_the_bioelectric_code_in_neurogenesis_and_regeneration_labels(),
        source_record_count=cfg.expected_source_record_count,
        component_count=cfg.expected_component_count,
        next_source_boundary=cfg.next_source_boundary,
        null_controls={
            "1_the_bioelectric_code_in_neurogenesis_and_regeneration_is_not_empirical_validation_evidence": 1.0,
            "2_the_optimised_connectome_is_not_empirical_validation_evidence": 1.0,
            "3_the_active_role_of_glia_the_tripartite_synapse_is_not_empirical_validation_evidence": 1.0,
        },
        problem_metadata={
            "source_ledger_span": SOURCE_LEDGER_SPAN,
            "source_ledger_ids": tuple(f"P0R{number:05d}" for number in range(4657, 4666)),
            "claim_boundary": CLAIM_BOUNDARY,
            "protocol_state": "source_section_1_the_bioelectric_code_in_neurogenesis_and_regeneration_only_no_experiment",
        },
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "HARDWARE_STATUS",
    "SOURCE_LEDGER_SPAN",
    "Section1TheBioelectricCodeInNeurogenesisAndRegenerationConfig",
    "Section1TheBioelectricCodeInNeurogenesisAndRegenerationFixtureResult",
    "classify_section_1_the_bioelectric_code_in_neurogenesis_and_regeneration_component",
    "section_1_the_bioelectric_code_in_neurogenesis_and_regeneration_labels",
    "validate_section_1_the_bioelectric_code_in_neurogenesis_and_regeneration_fixture",
]
