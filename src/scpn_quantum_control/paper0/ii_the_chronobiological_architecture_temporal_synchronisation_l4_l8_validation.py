# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 II. The Chronobiological Architecture: Temporal Synchronisation (L4/L8) validation
"""Source-accounting checks for Paper 0 II. The Chronobiological Architecture: Temporal Synchronisation (L4/L8) records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

CLAIM_BOUNDARY = "source-bounded ii the chronobiological architecture temporal synchronisation l4 l8 source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"
SOURCE_LEDGER_SPAN = ("P0R04975", "P0R04982")


@dataclass(frozen=True, slots=True)
class IiTheChronobiologicalArchitectureTemporalSynchronisationL4L8Config:
    """Configuration for this Paper 0 source-accounting fixture."""

    expected_source_record_count: int = 8
    expected_component_count: int = 3
    next_source_boundary: str = "P0R04983"

    def __post_init__(self) -> None:
        if self.expected_source_record_count != 8:
            raise ValueError("expected_source_record_count must equal 8")
        if self.expected_component_count != 3:
            raise ValueError("expected_component_count must equal 3")
        if self.next_source_boundary != "P0R04983":
            raise ValueError("next_source_boundary must equal P0R04983")


@dataclass(frozen=True, slots=True)
class IiTheChronobiologicalArchitectureTemporalSynchronisationL4L8FixtureResult:
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


def classify_ii_the_chronobiological_architecture_temporal_synchronisation_l4_l8_component(
    component: str,
) -> str:
    """Classify source-defined components."""
    mapping = {
        "ii_the_chronobiological_architecture_temporal_synchronisation_l4_l8": "ii_the_chronobiological_architecture_temporal_synchronisation_l4_l8_source_boundary",
        "1_the_master_clock_scn_and_circadian_rhythms": "1_the_master_clock_scn_and_circadian_rhythms_source_boundary",
        "2_coupling_to_the_universal_tact_l8_interface": "2_coupling_to_the_universal_tact_l8_interface_source_boundary",
    }
    try:
        return mapping[component]
    except KeyError as exc:
        raise ValueError(
            "unknown ii_the_chronobiological_architecture_temporal_synchronisation_l4_l8 component"
        ) from exc


def ii_the_chronobiological_architecture_temporal_synchronisation_l4_l8_labels() -> dict[str, str]:
    """Return source-bounded labels for this slice."""
    return {
        "section": "II. The Chronobiological Architecture: Temporal Synchronisation (L4/L8)",
        "source_span": "P0R04975-P0R04982",
        "component_count": "3",
        "next_boundary": "P0R04983",
        "component_1": "II. The Chronobiological Architecture: Temporal Synchronisation (L4/L8)",
        "component_2": "1. The Master Clock (SCN) and Circadian Rhythms:",
        "component_3": "2. Coupling to the Universal Tact (L8 Interface):",
    }


def validate_ii_the_chronobiological_architecture_temporal_synchronisation_l4_l8_fixture(
    config: IiTheChronobiologicalArchitectureTemporalSynchronisationL4L8Config | None = None,
) -> IiTheChronobiologicalArchitectureTemporalSynchronisationL4L8FixtureResult:
    """Validate source accounting for this Paper 0 slice."""
    cfg = config or IiTheChronobiologicalArchitectureTemporalSynchronisationL4L8Config()
    components = (
        "ii_the_chronobiological_architecture_temporal_synchronisation_l4_l8",
        "1_the_master_clock_scn_and_circadian_rhythms",
        "2_coupling_to_the_universal_tact_l8_interface",
    )
    return IiTheChronobiologicalArchitectureTemporalSynchronisationL4L8FixtureResult(
        source_ledger_span=SOURCE_LEDGER_SPAN,
        hardware_status=HARDWARE_STATUS,
        claim_boundary=CLAIM_BOUNDARY,
        components={
            component: classify_ii_the_chronobiological_architecture_temporal_synchronisation_l4_l8_component(
                component
            )
            for component in components
        },
        labels=ii_the_chronobiological_architecture_temporal_synchronisation_l4_l8_labels(),
        source_record_count=cfg.expected_source_record_count,
        component_count=cfg.expected_component_count,
        next_source_boundary=cfg.next_source_boundary,
        null_controls={
            "ii_the_chronobiological_architecture_temporal_synchronisation_l4_l8_is_not_empirical_validation_evidence": 1.0,
            "1_the_master_clock_scn_and_circadian_rhythms_is_not_empirical_validation_evidence": 1.0,
            "2_coupling_to_the_universal_tact_l8_interface_is_not_empirical_validation_evidence": 1.0,
        },
        problem_metadata={
            "source_ledger_span": SOURCE_LEDGER_SPAN,
            "source_ledger_ids": tuple(f"P0R{number:05d}" for number in range(4975, 4983)),
            "claim_boundary": CLAIM_BOUNDARY,
            "protocol_state": "source_ii_the_chronobiological_architecture_temporal_synchronisation_l4_l8_only_no_experiment",
        },
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "HARDWARE_STATUS",
    "SOURCE_LEDGER_SPAN",
    "IiTheChronobiologicalArchitectureTemporalSynchronisationL4L8Config",
    "IiTheChronobiologicalArchitectureTemporalSynchronisationL4L8FixtureResult",
    "classify_ii_the_chronobiological_architecture_temporal_synchronisation_l4_l8_component",
    "ii_the_chronobiological_architecture_temporal_synchronisation_l4_l8_labels",
    "validate_ii_the_chronobiological_architecture_temporal_synchronisation_l4_l8_fixture",
]
