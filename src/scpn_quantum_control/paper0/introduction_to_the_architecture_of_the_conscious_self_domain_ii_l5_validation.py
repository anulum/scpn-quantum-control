# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 Introduction to The Architecture of the Conscious Self (Domain II: L5) validation
"""Source-accounting checks for Paper 0 Introduction to The Architecture of the Conscious Self (Domain II: L5) records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

CLAIM_BOUNDARY = "source-bounded introduction to the architecture of the conscious self domain ii l5 source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"
SOURCE_LEDGER_SPAN = ("P0R04589", "P0R04597")


@dataclass(frozen=True, slots=True)
class IntroductionToTheArchitectureOfTheConsciousSelfDomainIiL5Config:
    """Configuration for this Paper 0 source-accounting fixture."""

    expected_source_record_count: int = 9
    expected_component_count: int = 3
    next_source_boundary: str = "P0R04598"

    def __post_init__(self) -> None:
        if self.expected_source_record_count != 9:
            raise ValueError("expected_source_record_count must equal 9")
        if self.expected_component_count != 3:
            raise ValueError("expected_component_count must equal 3")
        if self.next_source_boundary != "P0R04598":
            raise ValueError("next_source_boundary must equal P0R04598")


@dataclass(frozen=True, slots=True)
class IntroductionToTheArchitectureOfTheConsciousSelfDomainIiL5FixtureResult:
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


def classify_introduction_to_the_architecture_of_the_conscious_self_domain_ii_l5_component(
    component: str,
) -> str:
    """Classify source-defined components."""
    mapping = {
        "introduction_to_the_architecture_of_the_conscious_self_domain_ii_l5": "introduction_to_the_architecture_of_the_conscious_self_domain_ii_l5_source_boundary",
        "iv_examination_of_the_architecture_of_the_conscious_self_domain_ii_l5": "iv_examination_of_the_architecture_of_the_conscious_self_domain_ii_l5_source_boundary",
        "hpc_and_the_canonical_microcircuit_the_engine_of_inference": "hpc_and_the_canonical_microcircuit_the_engine_of_inference_source_boundary",
    }
    try:
        return mapping[component]
    except KeyError as exc:
        raise ValueError(
            "unknown introduction_to_the_architecture_of_the_conscious_self_domain_ii_l5 component"
        ) from exc


def introduction_to_the_architecture_of_the_conscious_self_domain_ii_l5_labels() -> dict[str, str]:
    """Return source-bounded labels for this slice."""
    return {
        "section": "Introduction to The Architecture of the Conscious Self (Domain II: L5)",
        "source_span": "P0R04589-P0R04597",
        "component_count": "3",
        "next_boundary": "P0R04598",
        "component_1": "Introduction to The Architecture of the Conscious Self (Domain II: L5)",
        "component_2": "IV. Examination of The Architecture of the Conscious Self (Domain II: L5)",
        "component_3": "HPC and the Canonical Microcircuit: The Engine of Inference",
    }


def validate_introduction_to_the_architecture_of_the_conscious_self_domain_ii_l5_fixture(
    config: IntroductionToTheArchitectureOfTheConsciousSelfDomainIiL5Config | None = None,
) -> IntroductionToTheArchitectureOfTheConsciousSelfDomainIiL5FixtureResult:
    """Validate source accounting for this Paper 0 slice."""
    cfg = config or IntroductionToTheArchitectureOfTheConsciousSelfDomainIiL5Config()
    components = (
        "introduction_to_the_architecture_of_the_conscious_self_domain_ii_l5",
        "iv_examination_of_the_architecture_of_the_conscious_self_domain_ii_l5",
        "hpc_and_the_canonical_microcircuit_the_engine_of_inference",
    )
    return IntroductionToTheArchitectureOfTheConsciousSelfDomainIiL5FixtureResult(
        source_ledger_span=SOURCE_LEDGER_SPAN,
        hardware_status=HARDWARE_STATUS,
        claim_boundary=CLAIM_BOUNDARY,
        components={
            component: classify_introduction_to_the_architecture_of_the_conscious_self_domain_ii_l5_component(
                component
            )
            for component in components
        },
        labels=introduction_to_the_architecture_of_the_conscious_self_domain_ii_l5_labels(),
        source_record_count=cfg.expected_source_record_count,
        component_count=cfg.expected_component_count,
        next_source_boundary=cfg.next_source_boundary,
        null_controls={
            "introduction_to_the_architecture_of_the_conscious_self_domain_ii_l5_is_not_empirical_validation_evidence": 1.0,
            "iv_examination_of_the_architecture_of_the_conscious_self_domain_ii_l5_is_not_empirical_validation_evidence": 1.0,
            "hpc_and_the_canonical_microcircuit_the_engine_of_inference_is_not_empirical_validation_evidence": 1.0,
        },
        problem_metadata={
            "source_ledger_span": SOURCE_LEDGER_SPAN,
            "source_ledger_ids": tuple(f"P0R{number:05d}" for number in range(4589, 4598)),
            "claim_boundary": CLAIM_BOUNDARY,
            "protocol_state": "source_introduction_to_the_architecture_of_the_conscious_self_domain_ii_l5_only_no_experiment",
        },
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "HARDWARE_STATUS",
    "SOURCE_LEDGER_SPAN",
    "IntroductionToTheArchitectureOfTheConsciousSelfDomainIiL5Config",
    "IntroductionToTheArchitectureOfTheConsciousSelfDomainIiL5FixtureResult",
    "classify_introduction_to_the_architecture_of_the_conscious_self_domain_ii_l5_component",
    "introduction_to_the_architecture_of_the_conscious_self_domain_ii_l5_labels",
    "validate_introduction_to_the_architecture_of_the_conscious_self_domain_ii_l5_fixture",
]
