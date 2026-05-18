# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 2. The Synaptic Junction and Downward Causation (L2): validation
"""Source-accounting checks for Paper 0 2. The Synaptic Junction and Downward Causation (L2): records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

CLAIM_BOUNDARY = "source-bounded section 2 the synaptic junction and downward causation l2 source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"
SOURCE_LEDGER_SPAN = ("P0R04470", "P0R04477")


@dataclass(frozen=True, slots=True)
class Section2TheSynapticJunctionAndDownwardCausationL2Config:
    """Configuration for this Paper 0 source-accounting fixture."""

    expected_source_record_count: int = 8
    expected_component_count: int = 1
    next_source_boundary: str = "P0R04478"

    def __post_init__(self) -> None:
        if self.expected_source_record_count != 8:
            raise ValueError("expected_source_record_count must equal 8")
        if self.expected_component_count != 1:
            raise ValueError("expected_component_count must equal 1")
        if self.next_source_boundary != "P0R04478":
            raise ValueError("next_source_boundary must equal P0R04478")


@dataclass(frozen=True, slots=True)
class Section2TheSynapticJunctionAndDownwardCausationL2FixtureResult:
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


def classify_section_2_the_synaptic_junction_and_downward_causation_l2_component(
    component: str,
) -> str:
    """Classify source-defined components."""
    mapping = {
        "2_the_synaptic_junction_and_downward_causation_l2": "2_the_synaptic_junction_and_downward_causation_l2_source_boundary"
    }
    try:
        return mapping[component]
    except KeyError as exc:
        raise ValueError(
            "unknown section_2_the_synaptic_junction_and_downward_causation_l2 component"
        ) from exc


def section_2_the_synaptic_junction_and_downward_causation_l2_labels() -> dict[str, str]:
    """Return source-bounded labels for this slice."""
    return {
        "section": "2. The Synaptic Junction and Downward Causation (L2):",
        "source_span": "P0R04470-P0R04477",
        "component_count": "1",
        "next_boundary": "P0R04478",
        "component_1": "2. The Synaptic Junction and Downward Causation (L2):",
    }


def validate_section_2_the_synaptic_junction_and_downward_causation_l2_fixture(
    config: Section2TheSynapticJunctionAndDownwardCausationL2Config | None = None,
) -> Section2TheSynapticJunctionAndDownwardCausationL2FixtureResult:
    """Validate source accounting for this Paper 0 slice."""
    cfg = config or Section2TheSynapticJunctionAndDownwardCausationL2Config()
    components = ("2_the_synaptic_junction_and_downward_causation_l2",)
    return Section2TheSynapticJunctionAndDownwardCausationL2FixtureResult(
        source_ledger_span=SOURCE_LEDGER_SPAN,
        hardware_status=HARDWARE_STATUS,
        claim_boundary=CLAIM_BOUNDARY,
        components={
            component: classify_section_2_the_synaptic_junction_and_downward_causation_l2_component(
                component
            )
            for component in components
        },
        labels=section_2_the_synaptic_junction_and_downward_causation_l2_labels(),
        source_record_count=cfg.expected_source_record_count,
        component_count=cfg.expected_component_count,
        next_source_boundary=cfg.next_source_boundary,
        null_controls={
            "2_the_synaptic_junction_and_downward_causation_l2_is_not_empirical_validation_evidence": 1.0
        },
        problem_metadata={
            "source_ledger_span": SOURCE_LEDGER_SPAN,
            "source_ledger_ids": tuple(f"P0R{number:05d}" for number in range(4470, 4478)),
            "claim_boundary": CLAIM_BOUNDARY,
            "protocol_state": "source_section_2_the_synaptic_junction_and_downward_causation_l2_only_no_experiment",
        },
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "HARDWARE_STATUS",
    "SOURCE_LEDGER_SPAN",
    "Section2TheSynapticJunctionAndDownwardCausationL2Config",
    "Section2TheSynapticJunctionAndDownwardCausationL2FixtureResult",
    "classify_section_2_the_synaptic_junction_and_downward_causation_l2_component",
    "section_2_the_synaptic_junction_and_downward_causation_l2_labels",
    "validate_section_2_the_synaptic_junction_and_downward_causation_l2_fixture",
]
