# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 Application to Immune Enzymes validation
"""Source-accounting checks for Paper 0 Application to Immune Enzymes records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

CLAIM_BOUNDARY = "source-bounded application to immune enzymes source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"
SOURCE_LEDGER_SPAN = ("P0R05517", "P0R05527")


@dataclass(frozen=True, slots=True)
class ApplicationToImmuneEnzymesConfig:
    """Configuration for this Paper 0 source-accounting fixture."""

    expected_source_record_count: int = 11
    expected_component_count: int = 3
    next_source_boundary: str = "P0R05528"

    def __post_init__(self) -> None:
        if self.expected_source_record_count != 11:
            raise ValueError("expected_source_record_count must equal 11")
        if self.expected_component_count != 3:
            raise ValueError("expected_component_count must equal 3")
        if self.next_source_boundary != "P0R05528":
            raise ValueError("next_source_boundary must equal P0R05528")


@dataclass(frozen=True, slots=True)
class ApplicationToImmuneEnzymesFixtureResult:
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


def classify_application_to_immune_enzymes_component(component: str) -> str:
    """Classify source-defined components."""
    mapping = {
        "application_to_immune_enzymes": "application_to_immune_enzymes_source_boundary",
        "the_biophysics_of_coherence_a_scale_invariant_cybernetic_thread": "the_biophysics_of_coherence_a_scale_invariant_cybernetic_thread_source_boundary",
        "micro_scale_homeostasis_glial_control_of_neuronal_criticality": "micro_scale_homeostasis_glial_control_of_neuronal_criticality_source_boundary",
    }
    try:
        return mapping[component]
    except KeyError as exc:
        raise ValueError("unknown application_to_immune_enzymes component") from exc


def application_to_immune_enzymes_labels() -> dict[str, str]:
    """Return source-bounded labels for this slice."""
    return {
        "section": "Application to Immune Enzymes",
        "source_span": "P0R05517-P0R05527",
        "component_count": "3",
        "next_boundary": "P0R05528",
        "component_1": "Application to Immune Enzymes",
        "component_2": "The Biophysics of Coherence: A Scale-Invariant Cybernetic Thread",
        "component_3": "Micro-Scale Homeostasis: Glial Control of Neuronal Criticality",
    }


def validate_application_to_immune_enzymes_fixture(
    config: ApplicationToImmuneEnzymesConfig | None = None,
) -> ApplicationToImmuneEnzymesFixtureResult:
    """Validate source accounting for this Paper 0 slice."""
    cfg = config or ApplicationToImmuneEnzymesConfig()
    components = (
        "application_to_immune_enzymes",
        "the_biophysics_of_coherence_a_scale_invariant_cybernetic_thread",
        "micro_scale_homeostasis_glial_control_of_neuronal_criticality",
    )
    return ApplicationToImmuneEnzymesFixtureResult(
        source_ledger_span=SOURCE_LEDGER_SPAN,
        hardware_status=HARDWARE_STATUS,
        claim_boundary=CLAIM_BOUNDARY,
        components={
            component: classify_application_to_immune_enzymes_component(component)
            for component in components
        },
        labels=application_to_immune_enzymes_labels(),
        source_record_count=cfg.expected_source_record_count,
        component_count=cfg.expected_component_count,
        next_source_boundary=cfg.next_source_boundary,
        null_controls={
            "application_to_immune_enzymes_is_not_empirical_validation_evidence": 1.0,
            "the_biophysics_of_coherence_a_scale_invariant_cybernetic_thread_is_not_empirical_validation_evidence": 1.0,
            "micro_scale_homeostasis_glial_control_of_neuronal_criticality_is_not_empirical_validation_evidence": 1.0,
        },
        problem_metadata={
            "source_ledger_span": SOURCE_LEDGER_SPAN,
            "source_ledger_ids": tuple(f"P0R{number:05d}" for number in range(5517, 5528)),
            "claim_boundary": CLAIM_BOUNDARY,
            "protocol_state": "source_application_to_immune_enzymes_only_no_experiment",
        },
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "HARDWARE_STATUS",
    "SOURCE_LEDGER_SPAN",
    "ApplicationToImmuneEnzymesConfig",
    "ApplicationToImmuneEnzymesFixtureResult",
    "classify_application_to_immune_enzymes_component",
    "application_to_immune_enzymes_labels",
    "validate_application_to_immune_enzymes_fixture",
]
