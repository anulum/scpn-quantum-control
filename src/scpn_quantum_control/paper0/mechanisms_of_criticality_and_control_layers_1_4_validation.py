# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 Mechanisms of Criticality and Control (Layers 1-4) validation
"""Source-accounting checks for Paper 0 Mechanisms of Criticality and Control (Layers 1-4) records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

CLAIM_BOUNDARY = "source-bounded mechanisms of criticality and control layers 1 4 source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"
SOURCE_LEDGER_SPAN = ("P0R05113", "P0R05123")


@dataclass(frozen=True, slots=True)
class MechanismsOfCriticalityAndControlLayers14Config:
    """Configuration for this Paper 0 source-accounting fixture."""

    expected_source_record_count: int = 11
    expected_component_count: int = 2
    next_source_boundary: str = "P0R05124"

    def __post_init__(self) -> None:
        if self.expected_source_record_count != 11:
            raise ValueError("expected_source_record_count must equal 11")
        if self.expected_component_count != 2:
            raise ValueError("expected_component_count must equal 2")
        if self.next_source_boundary != "P0R05124":
            raise ValueError("next_source_boundary must equal P0R05124")


@dataclass(frozen=True, slots=True)
class MechanismsOfCriticalityAndControlLayers14FixtureResult:
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


def classify_mechanisms_of_criticality_and_control_layers_1_4_component(component: str) -> str:
    """Classify source-defined components."""
    mapping = {
        "mechanisms_of_criticality_and_control_layers_1_4": "mechanisms_of_criticality_and_control_layers_1_4_source_boundary",
        "p0r05120": "p0r05120_source_boundary",
    }
    try:
        return mapping[component]
    except KeyError as exc:
        raise ValueError(
            "unknown mechanisms_of_criticality_and_control_layers_1_4 component"
        ) from exc


def mechanisms_of_criticality_and_control_layers_1_4_labels() -> dict[str, str]:
    """Return source-bounded labels for this slice."""
    return {
        "section": "Mechanisms of Criticality and Control (Layers 1-4)",
        "source_span": "P0R05113-P0R05123",
        "component_count": "2",
        "next_boundary": "P0R05124",
        "component_1": "Mechanisms of Criticality and Control (Layers 1-4)",
        "component_2": "P0R05120",
    }


def validate_mechanisms_of_criticality_and_control_layers_1_4_fixture(
    config: MechanismsOfCriticalityAndControlLayers14Config | None = None,
) -> MechanismsOfCriticalityAndControlLayers14FixtureResult:
    """Validate source accounting for this Paper 0 slice."""
    cfg = config or MechanismsOfCriticalityAndControlLayers14Config()
    components = ("mechanisms_of_criticality_and_control_layers_1_4", "p0r05120")
    return MechanismsOfCriticalityAndControlLayers14FixtureResult(
        source_ledger_span=SOURCE_LEDGER_SPAN,
        hardware_status=HARDWARE_STATUS,
        claim_boundary=CLAIM_BOUNDARY,
        components={
            component: classify_mechanisms_of_criticality_and_control_layers_1_4_component(
                component
            )
            for component in components
        },
        labels=mechanisms_of_criticality_and_control_layers_1_4_labels(),
        source_record_count=cfg.expected_source_record_count,
        component_count=cfg.expected_component_count,
        next_source_boundary=cfg.next_source_boundary,
        null_controls={
            "mechanisms_of_criticality_and_control_layers_1_4_is_not_empirical_validation_evidence": 1.0,
            "p0r05120_is_not_empirical_validation_evidence": 1.0,
        },
        problem_metadata={
            "source_ledger_span": SOURCE_LEDGER_SPAN,
            "source_ledger_ids": tuple(f"P0R{number:05d}" for number in range(5113, 5124)),
            "claim_boundary": CLAIM_BOUNDARY,
            "protocol_state": "source_mechanisms_of_criticality_and_control_layers_1_4_only_no_experiment",
        },
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "HARDWARE_STATUS",
    "SOURCE_LEDGER_SPAN",
    "MechanismsOfCriticalityAndControlLayers14Config",
    "MechanismsOfCriticalityAndControlLayers14FixtureResult",
    "classify_mechanisms_of_criticality_and_control_layers_1_4_component",
    "mechanisms_of_criticality_and_control_layers_1_4_labels",
    "validate_mechanisms_of_criticality_and_control_layers_1_4_fixture",
]
