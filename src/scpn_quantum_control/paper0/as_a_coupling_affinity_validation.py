# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0  as a Coupling Affinity: validation
"""Source-accounting checks for Paper 0  as a Coupling Affinity: records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

CLAIM_BOUNDARY = (
    "source-bounded as a coupling affinity source-accounting bridge; not validation evidence"
)
HARDWARE_STATUS = "source_methodology_no_experiment"
SOURCE_LEDGER_SPAN = ("P0R03501", "P0R03509")


@dataclass(frozen=True, slots=True)
class AsACouplingAffinityConfig:
    """Configuration for this Paper 0 source-accounting fixture."""

    expected_source_record_count: int = 9
    expected_component_count: int = 3
    next_source_boundary: str = "P0R03510"

    def __post_init__(self) -> None:
        if self.expected_source_record_count != 9:
            raise ValueError("expected_source_record_count must equal 9")
        if self.expected_component_count != 3:
            raise ValueError("expected_component_count must equal 3")
        if self.next_source_boundary != "P0R03510":
            raise ValueError("next_source_boundary must equal P0R03510")


@dataclass(frozen=True, slots=True)
class AsACouplingAffinityFixtureResult:
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


def classify_as_a_coupling_affinity_component(component: str) -> str:
    """Classify source-defined components."""
    mapping = {
        "as_a_coupling_affinity": "as_a_coupling_affinity_source_boundary",
        "the_scaling_law_of_consciousness_slc": "the_scaling_law_of_consciousness_slc_source_boundary",
        "formalisation_via_integrated_information": "formalisation_via_integrated_information_source_boundary",
    }
    try:
        return mapping[component]
    except KeyError as exc:
        raise ValueError("unknown as_a_coupling_affinity component") from exc


def as_a_coupling_affinity_labels() -> dict[str, str]:
    """Return source-bounded labels for this slice."""
    return {
        "section": " as a Coupling Affinity:",
        "source_span": "P0R03501-P0R03509",
        "component_count": "3",
        "next_boundary": "P0R03510",
        "component_1": "as a Coupling Affinity:",
        "component_2": "The Scaling Law of Consciousness (SLC)",
        "component_3": "Formalisation via Integrated Information ():",
    }


def validate_as_a_coupling_affinity_fixture(
    config: AsACouplingAffinityConfig | None = None,
) -> AsACouplingAffinityFixtureResult:
    """Validate source accounting for this Paper 0 slice."""
    cfg = config or AsACouplingAffinityConfig()
    components = (
        "as_a_coupling_affinity",
        "the_scaling_law_of_consciousness_slc",
        "formalisation_via_integrated_information",
    )
    return AsACouplingAffinityFixtureResult(
        source_ledger_span=SOURCE_LEDGER_SPAN,
        hardware_status=HARDWARE_STATUS,
        claim_boundary=CLAIM_BOUNDARY,
        components={
            component: classify_as_a_coupling_affinity_component(component)
            for component in components
        },
        labels=as_a_coupling_affinity_labels(),
        source_record_count=cfg.expected_source_record_count,
        component_count=cfg.expected_component_count,
        next_source_boundary=cfg.next_source_boundary,
        null_controls={
            "as_a_coupling_affinity_is_not_empirical_validation_evidence": 1.0,
            "the_scaling_law_of_consciousness_slc_is_not_empirical_validation_evidence": 1.0,
            "formalisation_via_integrated_information_is_not_empirical_validation_evidence": 1.0,
        },
        problem_metadata={
            "source_ledger_span": SOURCE_LEDGER_SPAN,
            "source_ledger_ids": tuple(f"P0R{number:05d}" for number in range(3501, 3510)),
            "claim_boundary": CLAIM_BOUNDARY,
            "protocol_state": "source_as_a_coupling_affinity_only_no_experiment",
        },
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "HARDWARE_STATUS",
    "SOURCE_LEDGER_SPAN",
    "AsACouplingAffinityConfig",
    "AsACouplingAffinityFixtureResult",
    "classify_as_a_coupling_affinity_component",
    "as_a_coupling_affinity_labels",
    "validate_as_a_coupling_affinity_fixture",
]
