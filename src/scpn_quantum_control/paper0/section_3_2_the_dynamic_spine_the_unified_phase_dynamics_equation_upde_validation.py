# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 3.2 The Dynamic Spine: The Unified Phase Dynamics Equation (UPDE) validation
"""Source-accounting checks for Paper 0 3.2 The Dynamic Spine: The Unified Phase Dynamics Equation (UPDE) records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

CLAIM_BOUNDARY = "source-bounded section 3 2 the dynamic spine the unified phase dynamics equation upde source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"
SOURCE_LEDGER_SPAN = ("P0R02580", "P0R02599")


@dataclass(frozen=True, slots=True)
class Section32TheDynamicSpineTheUnifiedPhaseDynamicsEquationUpdeConfig:
    """Configuration for this Paper 0 source-accounting fixture."""

    expected_source_record_count: int = 20
    expected_component_count: int = 2
    next_source_boundary: str = "P0R02600"

    def __post_init__(self) -> None:
        if self.expected_source_record_count != 20:
            raise ValueError("expected_source_record_count must equal 20")
        if self.expected_component_count != 2:
            raise ValueError("expected_component_count must equal 2")
        if self.next_source_boundary != "P0R02600":
            raise ValueError("next_source_boundary must equal P0R02600")


@dataclass(frozen=True, slots=True)
class Section32TheDynamicSpineTheUnifiedPhaseDynamicsEquationUpdeFixtureResult:
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


def classify_section_3_2_the_dynamic_spine_the_unified_phase_dynamics_equation_upde_component(
    component: str,
) -> str:
    """Classify source-defined components."""
    mapping = {
        "3_2_the_dynamic_spine_the_unified_phase_dynamics_equation_upde": "3_2_the_dynamic_spine_the_unified_phase_dynamics_equation_upde_source_boundary",
        "the_unified_phase_dynamics_equation_upde": "the_unified_phase_dynamics_equation_upde_source_boundary",
    }
    try:
        return mapping[component]
    except KeyError as exc:
        raise ValueError(
            "unknown section_3_2_the_dynamic_spine_the_unified_phase_dynamics_equation_upde component"
        ) from exc


def section_3_2_the_dynamic_spine_the_unified_phase_dynamics_equation_upde_labels() -> dict[
    str, str
]:
    """Return source-bounded labels for this slice."""
    return {
        "section": "3.2 The Dynamic Spine: The Unified Phase Dynamics Equation (UPDE)",
        "source_span": "P0R02580-P0R02599",
        "component_count": "2",
        "next_boundary": "P0R02600",
        "component_1": "3.2 The Dynamic Spine: The Unified Phase Dynamics Equation (UPDE)",
        "component_2": "The Unified Phase Dynamics Equation (UPDE)",
    }


def validate_section_3_2_the_dynamic_spine_the_unified_phase_dynamics_equation_upde_fixture(
    config: Section32TheDynamicSpineTheUnifiedPhaseDynamicsEquationUpdeConfig | None = None,
) -> Section32TheDynamicSpineTheUnifiedPhaseDynamicsEquationUpdeFixtureResult:
    """Validate source accounting for this Paper 0 slice."""
    cfg = config or Section32TheDynamicSpineTheUnifiedPhaseDynamicsEquationUpdeConfig()
    components = (
        "3_2_the_dynamic_spine_the_unified_phase_dynamics_equation_upde",
        "the_unified_phase_dynamics_equation_upde",
    )
    return Section32TheDynamicSpineTheUnifiedPhaseDynamicsEquationUpdeFixtureResult(
        source_ledger_span=SOURCE_LEDGER_SPAN,
        hardware_status=HARDWARE_STATUS,
        claim_boundary=CLAIM_BOUNDARY,
        components={
            component: classify_section_3_2_the_dynamic_spine_the_unified_phase_dynamics_equation_upde_component(
                component
            )
            for component in components
        },
        labels=section_3_2_the_dynamic_spine_the_unified_phase_dynamics_equation_upde_labels(),
        source_record_count=cfg.expected_source_record_count,
        component_count=cfg.expected_component_count,
        next_source_boundary=cfg.next_source_boundary,
        null_controls={
            "3_2_the_dynamic_spine_the_unified_phase_dynamics_equation_upde_is_not_empirical_validation_evidence": 1.0,
            "the_unified_phase_dynamics_equation_upde_is_not_empirical_validation_evidence": 1.0,
        },
        problem_metadata={
            "source_ledger_span": SOURCE_LEDGER_SPAN,
            "source_ledger_ids": tuple(f"P0R{number:05d}" for number in range(2580, 2600)),
            "claim_boundary": CLAIM_BOUNDARY,
            "protocol_state": "source_section_3_2_the_dynamic_spine_the_unified_phase_dynamics_equation_upde_only_no_experiment",
        },
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "HARDWARE_STATUS",
    "SOURCE_LEDGER_SPAN",
    "Section32TheDynamicSpineTheUnifiedPhaseDynamicsEquationUpdeConfig",
    "Section32TheDynamicSpineTheUnifiedPhaseDynamicsEquationUpdeFixtureResult",
    "classify_section_3_2_the_dynamic_spine_the_unified_phase_dynamics_equation_upde_component",
    "section_3_2_the_dynamic_spine_the_unified_phase_dynamics_equation_upde_labels",
    "validate_section_3_2_the_dynamic_spine_the_unified_phase_dynamics_equation_upde_fixture",
]
