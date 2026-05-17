# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 Overarching Dynamic Principles and the Mathematical Spine validation
"""Source-accounting checks for Paper 0 Overarching Dynamic Principles and the Mathematical Spine records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

CLAIM_BOUNDARY = "source-bounded overarching dynamic principles and the mathematical spine source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"
SOURCE_LEDGER_SPAN = ("P0R02502", "P0R02512")


@dataclass(frozen=True, slots=True)
class OverarchingDynamicPrinciplesAndTheMathematicalSpineConfig:
    """Configuration for this Paper 0 source-accounting fixture."""

    expected_source_record_count: int = 11
    expected_component_count: int = 2
    next_source_boundary: str = "P0R02513"

    def __post_init__(self) -> None:
        if self.expected_source_record_count != 11:
            raise ValueError("expected_source_record_count must equal 11")
        if self.expected_component_count != 2:
            raise ValueError("expected_component_count must equal 2")
        if self.next_source_boundary != "P0R02513":
            raise ValueError("next_source_boundary must equal P0R02513")


@dataclass(frozen=True, slots=True)
class OverarchingDynamicPrinciplesAndTheMathematicalSpineFixtureResult:
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


def classify_overarching_dynamic_principles_and_the_mathematical_spine_component(
    component: str,
) -> str:
    """Classify source-defined components."""
    mapping = {
        "overarching_dynamic_principles_and_the_mathematical_spine": "overarching_dynamic_principles_and_the_mathematical_spine_source_boundary",
        "i_the_unified_phase_dynamics_equation_upde_the_spine_of_the_scpn": "i_the_unified_phase_dynamics_equation_upde_the_spine_of_the_scpn_source_boundary",
    }
    try:
        return mapping[component]
    except KeyError as exc:
        raise ValueError(
            "unknown overarching_dynamic_principles_and_the_mathematical_spine component"
        ) from exc


def overarching_dynamic_principles_and_the_mathematical_spine_labels() -> dict[str, str]:
    """Return source-bounded labels for this slice."""
    return {
        "section": "Overarching Dynamic Principles and the Mathematical Spine",
        "source_span": "P0R02502-P0R02512",
        "component_count": "2",
        "next_boundary": "P0R02513",
        "component_1": "Overarching Dynamic Principles and the Mathematical Spine",
        "component_2": "I. The Unified Phase Dynamics Equation (UPDE) - The Spine of the SCPN",
    }


def validate_overarching_dynamic_principles_and_the_mathematical_spine_fixture(
    config: OverarchingDynamicPrinciplesAndTheMathematicalSpineConfig | None = None,
) -> OverarchingDynamicPrinciplesAndTheMathematicalSpineFixtureResult:
    """Validate source accounting for this Paper 0 slice."""
    cfg = config or OverarchingDynamicPrinciplesAndTheMathematicalSpineConfig()
    components = (
        "overarching_dynamic_principles_and_the_mathematical_spine",
        "i_the_unified_phase_dynamics_equation_upde_the_spine_of_the_scpn",
    )
    return OverarchingDynamicPrinciplesAndTheMathematicalSpineFixtureResult(
        source_ledger_span=SOURCE_LEDGER_SPAN,
        hardware_status=HARDWARE_STATUS,
        claim_boundary=CLAIM_BOUNDARY,
        components={
            component: classify_overarching_dynamic_principles_and_the_mathematical_spine_component(
                component
            )
            for component in components
        },
        labels=overarching_dynamic_principles_and_the_mathematical_spine_labels(),
        source_record_count=cfg.expected_source_record_count,
        component_count=cfg.expected_component_count,
        next_source_boundary=cfg.next_source_boundary,
        null_controls={
            "overarching_dynamic_principles_and_the_mathematical_spine_is_not_empirical_validation_evidence": 1.0,
            "i_the_unified_phase_dynamics_equation_upde_the_spine_of_the_scpn_is_not_empirical_validation_evidence": 1.0,
        },
        problem_metadata={
            "source_ledger_span": SOURCE_LEDGER_SPAN,
            "source_ledger_ids": tuple(f"P0R{number:05d}" for number in range(2502, 2513)),
            "claim_boundary": CLAIM_BOUNDARY,
            "protocol_state": "source_overarching_dynamic_principles_and_the_mathematical_spine_only_no_experiment",
        },
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "HARDWARE_STATUS",
    "SOURCE_LEDGER_SPAN",
    "OverarchingDynamicPrinciplesAndTheMathematicalSpineConfig",
    "OverarchingDynamicPrinciplesAndTheMathematicalSpineFixtureResult",
    "classify_overarching_dynamic_principles_and_the_mathematical_spine_component",
    "overarching_dynamic_principles_and_the_mathematical_spine_labels",
    "validate_overarching_dynamic_principles_and_the_mathematical_spine_fixture",
]
