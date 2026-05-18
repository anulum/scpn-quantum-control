# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 2. A Path Integral Formulation of Causal Path Entropy validation
"""Source-accounting checks for Paper 0 2. A Path Integral Formulation of Causal Path Entropy records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

CLAIM_BOUNDARY = "source-bounded section 2 a path integral formulation of causal path entropy source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"
SOURCE_LEDGER_SPAN = ("P0R03737", "P0R03752")


@dataclass(frozen=True, slots=True)
class Section2APathIntegralFormulationOfCausalPathEntropyConfig:
    """Configuration for this Paper 0 source-accounting fixture."""

    expected_source_record_count: int = 16
    expected_component_count: int = 1
    next_source_boundary: str = "P0R03753"

    def __post_init__(self) -> None:
        if self.expected_source_record_count != 16:
            raise ValueError("expected_source_record_count must equal 16")
        if self.expected_component_count != 1:
            raise ValueError("expected_component_count must equal 1")
        if self.next_source_boundary != "P0R03753":
            raise ValueError("next_source_boundary must equal P0R03753")


@dataclass(frozen=True, slots=True)
class Section2APathIntegralFormulationOfCausalPathEntropyFixtureResult:
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


def classify_section_2_a_path_integral_formulation_of_causal_path_entropy_component(
    component: str,
) -> str:
    """Classify source-defined components."""
    mapping = {
        "2_a_path_integral_formulation_of_causal_path_entropy": "2_a_path_integral_formulation_of_causal_path_entropy_source_boundary"
    }
    try:
        return mapping[component]
    except KeyError as exc:
        raise ValueError(
            "unknown section_2_a_path_integral_formulation_of_causal_path_entropy component"
        ) from exc


def section_2_a_path_integral_formulation_of_causal_path_entropy_labels() -> dict[str, str]:
    """Return source-bounded labels for this slice."""
    return {
        "section": "2. A Path Integral Formulation of Causal Path Entropy",
        "source_span": "P0R03737-P0R03752",
        "component_count": "1",
        "next_boundary": "P0R03753",
        "component_1": "2. A Path Integral Formulation of Causal Path Entropy",
    }


def validate_section_2_a_path_integral_formulation_of_causal_path_entropy_fixture(
    config: Section2APathIntegralFormulationOfCausalPathEntropyConfig | None = None,
) -> Section2APathIntegralFormulationOfCausalPathEntropyFixtureResult:
    """Validate source accounting for this Paper 0 slice."""
    cfg = config or Section2APathIntegralFormulationOfCausalPathEntropyConfig()
    components = ("2_a_path_integral_formulation_of_causal_path_entropy",)
    return Section2APathIntegralFormulationOfCausalPathEntropyFixtureResult(
        source_ledger_span=SOURCE_LEDGER_SPAN,
        hardware_status=HARDWARE_STATUS,
        claim_boundary=CLAIM_BOUNDARY,
        components={
            component: classify_section_2_a_path_integral_formulation_of_causal_path_entropy_component(
                component
            )
            for component in components
        },
        labels=section_2_a_path_integral_formulation_of_causal_path_entropy_labels(),
        source_record_count=cfg.expected_source_record_count,
        component_count=cfg.expected_component_count,
        next_source_boundary=cfg.next_source_boundary,
        null_controls={
            "2_a_path_integral_formulation_of_causal_path_entropy_is_not_empirical_validation_evidence": 1.0
        },
        problem_metadata={
            "source_ledger_span": SOURCE_LEDGER_SPAN,
            "source_ledger_ids": tuple(f"P0R{number:05d}" for number in range(3737, 3753)),
            "claim_boundary": CLAIM_BOUNDARY,
            "protocol_state": "source_section_2_a_path_integral_formulation_of_causal_path_entropy_only_no_experiment",
        },
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "HARDWARE_STATUS",
    "SOURCE_LEDGER_SPAN",
    "Section2APathIntegralFormulationOfCausalPathEntropyConfig",
    "Section2APathIntegralFormulationOfCausalPathEntropyFixtureResult",
    "classify_section_2_a_path_integral_formulation_of_causal_path_entropy_component",
    "section_2_a_path_integral_formulation_of_causal_path_entropy_labels",
    "validate_section_2_a_path_integral_formulation_of_causal_path_entropy_fixture",
]
