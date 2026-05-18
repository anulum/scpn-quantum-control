# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 3.1. Noether's Theorem on the Qualia Fiber validation
"""Source-accounting checks for Paper 0 3.1. Noether's Theorem on the Qualia Fiber records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

CLAIM_BOUNDARY = "source-bounded section 3 1 noether s theorem on the qualia fiber source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"
SOURCE_LEDGER_SPAN = ("P0R03630", "P0R03637")


@dataclass(frozen=True, slots=True)
class Section31NoetherSTheoremOnTheQualiaFiberConfig:
    """Configuration for this Paper 0 source-accounting fixture."""

    expected_source_record_count: int = 8
    expected_component_count: int = 2
    next_source_boundary: str = "P0R03638"

    def __post_init__(self) -> None:
        if self.expected_source_record_count != 8:
            raise ValueError("expected_source_record_count must equal 8")
        if self.expected_component_count != 2:
            raise ValueError("expected_component_count must equal 2")
        if self.next_source_boundary != "P0R03638":
            raise ValueError("next_source_boundary must equal P0R03638")


@dataclass(frozen=True, slots=True)
class Section31NoetherSTheoremOnTheQualiaFiberFixtureResult:
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


def classify_section_3_1_noether_s_theorem_on_the_qualia_fiber_component(component: str) -> str:
    """Classify source-defined components."""
    mapping = {
        "3_1_noether_s_theorem_on_the_qualia_fiber": "3_1_noether_s_theorem_on_the_qualia_fiber_source_boundary",
        "3_2_defining_coherence_complexity_and_qualia_capacity_as_physical_charge": "3_2_defining_coherence_complexity_and_qualia_capacity_as_physical_charge_source_boundary",
    }
    try:
        return mapping[component]
    except KeyError as exc:
        raise ValueError(
            "unknown section_3_1_noether_s_theorem_on_the_qualia_fiber component"
        ) from exc


def section_3_1_noether_s_theorem_on_the_qualia_fiber_labels() -> dict[str, str]:
    """Return source-bounded labels for this slice."""
    return {
        "section": "3.1. Noether's Theorem on the Qualia Fiber",
        "source_span": "P0R03630-P0R03637",
        "component_count": "2",
        "next_boundary": "P0R03638",
        "component_1": "3.1. Noether's Theorem on the Qualia Fiber",
        "component_2": "3.2. Defining Coherence, Complexity, and Qualia Capacity as Physical Charges",
    }


def validate_section_3_1_noether_s_theorem_on_the_qualia_fiber_fixture(
    config: Section31NoetherSTheoremOnTheQualiaFiberConfig | None = None,
) -> Section31NoetherSTheoremOnTheQualiaFiberFixtureResult:
    """Validate source accounting for this Paper 0 slice."""
    cfg = config or Section31NoetherSTheoremOnTheQualiaFiberConfig()
    components = (
        "3_1_noether_s_theorem_on_the_qualia_fiber",
        "3_2_defining_coherence_complexity_and_qualia_capacity_as_physical_charge",
    )
    return Section31NoetherSTheoremOnTheQualiaFiberFixtureResult(
        source_ledger_span=SOURCE_LEDGER_SPAN,
        hardware_status=HARDWARE_STATUS,
        claim_boundary=CLAIM_BOUNDARY,
        components={
            component: classify_section_3_1_noether_s_theorem_on_the_qualia_fiber_component(
                component
            )
            for component in components
        },
        labels=section_3_1_noether_s_theorem_on_the_qualia_fiber_labels(),
        source_record_count=cfg.expected_source_record_count,
        component_count=cfg.expected_component_count,
        next_source_boundary=cfg.next_source_boundary,
        null_controls={
            "3_1_noether_s_theorem_on_the_qualia_fiber_is_not_empirical_validation_evidence": 1.0,
            "3_2_defining_coherence_complexity_and_qualia_capacity_as_physical_charge_is_not_empirical_validation_evidence": 1.0,
        },
        problem_metadata={
            "source_ledger_span": SOURCE_LEDGER_SPAN,
            "source_ledger_ids": tuple(f"P0R{number:05d}" for number in range(3630, 3638)),
            "claim_boundary": CLAIM_BOUNDARY,
            "protocol_state": "source_section_3_1_noether_s_theorem_on_the_qualia_fiber_only_no_experiment",
        },
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "HARDWARE_STATUS",
    "SOURCE_LEDGER_SPAN",
    "Section31NoetherSTheoremOnTheQualiaFiberConfig",
    "Section31NoetherSTheoremOnTheQualiaFiberFixtureResult",
    "classify_section_3_1_noether_s_theorem_on_the_qualia_fiber_component",
    "section_3_1_noether_s_theorem_on_the_qualia_fiber_labels",
    "validate_section_3_1_noether_s_theorem_on_the_qualia_fiber_fixture",
]
