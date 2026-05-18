# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 Explicit Identification of Terms: validation
"""Source-accounting checks for Paper 0 Explicit Identification of Terms: records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

CLAIM_BOUNDARY = "source-bounded explicit identification of terms source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"
SOURCE_LEDGER_SPAN = ("P0R04283", "P0R04290")


@dataclass(frozen=True, slots=True)
class ExplicitIdentificationOfTermsConfig:
    """Configuration for this Paper 0 source-accounting fixture."""

    expected_source_record_count: int = 8
    expected_component_count: int = 3
    next_source_boundary: str = "P0R04291"

    def __post_init__(self) -> None:
        if self.expected_source_record_count != 8:
            raise ValueError("expected_source_record_count must equal 8")
        if self.expected_component_count != 3:
            raise ValueError("expected_component_count must equal 3")
        if self.next_source_boundary != "P0R04291":
            raise ValueError("next_source_boundary must equal P0R04291")


@dataclass(frozen=True, slots=True)
class ExplicitIdentificationOfTermsFixtureResult:
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


def classify_explicit_identification_of_terms_component(component: str) -> str:
    """Classify source-defined components."""
    mapping = {
        "explicit_identification_of_terms": "explicit_identification_of_terms_source_boundary",
        "the_nature_of_the_interaction": "the_nature_of_the_interaction_source_boundary",
        "the_psi_field_electromagnetic_interface_the_role_of_axion_like_particles": "the_psi_field_electromagnetic_interface_the_role_of_axion_like_particles_source_boundary",
    }
    try:
        return mapping[component]
    except KeyError as exc:
        raise ValueError("unknown explicit_identification_of_terms component") from exc


def explicit_identification_of_terms_labels() -> dict[str, str]:
    """Return source-bounded labels for this slice."""
    return {
        "section": "Explicit Identification of Terms:",
        "source_span": "P0R04283-P0R04290",
        "component_count": "3",
        "next_boundary": "P0R04291",
        "component_1": "Explicit Identification of Terms:",
        "component_2": "The Nature of the Interaction:",
        "component_3": "The Psi-Field-Electromagnetic Interface: The Role of Axion-Like Particles (ALPs)",
    }


def validate_explicit_identification_of_terms_fixture(
    config: ExplicitIdentificationOfTermsConfig | None = None,
) -> ExplicitIdentificationOfTermsFixtureResult:
    """Validate source accounting for this Paper 0 slice."""
    cfg = config or ExplicitIdentificationOfTermsConfig()
    components = (
        "explicit_identification_of_terms",
        "the_nature_of_the_interaction",
        "the_psi_field_electromagnetic_interface_the_role_of_axion_like_particles",
    )
    return ExplicitIdentificationOfTermsFixtureResult(
        source_ledger_span=SOURCE_LEDGER_SPAN,
        hardware_status=HARDWARE_STATUS,
        claim_boundary=CLAIM_BOUNDARY,
        components={
            component: classify_explicit_identification_of_terms_component(component)
            for component in components
        },
        labels=explicit_identification_of_terms_labels(),
        source_record_count=cfg.expected_source_record_count,
        component_count=cfg.expected_component_count,
        next_source_boundary=cfg.next_source_boundary,
        null_controls={
            "explicit_identification_of_terms_is_not_empirical_validation_evidence": 1.0,
            "the_nature_of_the_interaction_is_not_empirical_validation_evidence": 1.0,
            "the_psi_field_electromagnetic_interface_the_role_of_axion_like_particles_is_not_empirical_validation_evidence": 1.0,
        },
        problem_metadata={
            "source_ledger_span": SOURCE_LEDGER_SPAN,
            "source_ledger_ids": tuple(f"P0R{number:05d}" for number in range(4283, 4291)),
            "claim_boundary": CLAIM_BOUNDARY,
            "protocol_state": "source_explicit_identification_of_terms_only_no_experiment",
        },
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "HARDWARE_STATUS",
    "SOURCE_LEDGER_SPAN",
    "ExplicitIdentificationOfTermsConfig",
    "ExplicitIdentificationOfTermsFixtureResult",
    "classify_explicit_identification_of_terms_component",
    "explicit_identification_of_terms_labels",
    "validate_explicit_identification_of_terms_fixture",
]
