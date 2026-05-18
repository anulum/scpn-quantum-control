# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 The Microbiome as a Foundational Control Layer validation
"""Source-accounting checks for Paper 0 The Microbiome as a Foundational Control Layer records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

CLAIM_BOUNDARY = "source-bounded the microbiome as a foundational control layer source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"
SOURCE_LEDGER_SPAN = ("P0R05479", "P0R05492")


@dataclass(frozen=True, slots=True)
class TheMicrobiomeAsAFoundationalControlLayerConfig:
    """Configuration for this Paper 0 source-accounting fixture."""

    expected_source_record_count: int = 14
    expected_component_count: int = 2
    next_source_boundary: str = "P0R05493"

    def __post_init__(self) -> None:
        if self.expected_source_record_count != 14:
            raise ValueError("expected_source_record_count must equal 14")
        if self.expected_component_count != 2:
            raise ValueError("expected_component_count must equal 2")
        if self.next_source_boundary != "P0R05493":
            raise ValueError("next_source_boundary must equal P0R05493")


@dataclass(frozen=True, slots=True)
class TheMicrobiomeAsAFoundationalControlLayerFixtureResult:
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


def classify_the_microbiome_as_a_foundational_control_layer_component(component: str) -> str:
    """Classify source-defined components."""
    mapping = {
        "the_microbiome_as_a_foundational_control_layer": "the_microbiome_as_a_foundational_control_layer_source_boundary",
        "a_two_timescale_model_of_glial_neuronal_coupling": "a_two_timescale_model_of_glial_neuronal_coupling_source_boundary",
    }
    try:
        return mapping[component]
    except KeyError as exc:
        raise ValueError(
            "unknown the_microbiome_as_a_foundational_control_layer component"
        ) from exc


def the_microbiome_as_a_foundational_control_layer_labels() -> dict[str, str]:
    """Return source-bounded labels for this slice."""
    return {
        "section": "The Microbiome as a Foundational Control Layer",
        "source_span": "P0R05479-P0R05492",
        "component_count": "2",
        "next_boundary": "P0R05493",
        "component_1": "The Microbiome as a Foundational Control Layer",
        "component_2": "A Two-Timescale Model of Glial-Neuronal Coupling",
    }


def validate_the_microbiome_as_a_foundational_control_layer_fixture(
    config: TheMicrobiomeAsAFoundationalControlLayerConfig | None = None,
) -> TheMicrobiomeAsAFoundationalControlLayerFixtureResult:
    """Validate source accounting for this Paper 0 slice."""
    cfg = config or TheMicrobiomeAsAFoundationalControlLayerConfig()
    components = (
        "the_microbiome_as_a_foundational_control_layer",
        "a_two_timescale_model_of_glial_neuronal_coupling",
    )
    return TheMicrobiomeAsAFoundationalControlLayerFixtureResult(
        source_ledger_span=SOURCE_LEDGER_SPAN,
        hardware_status=HARDWARE_STATUS,
        claim_boundary=CLAIM_BOUNDARY,
        components={
            component: classify_the_microbiome_as_a_foundational_control_layer_component(component)
            for component in components
        },
        labels=the_microbiome_as_a_foundational_control_layer_labels(),
        source_record_count=cfg.expected_source_record_count,
        component_count=cfg.expected_component_count,
        next_source_boundary=cfg.next_source_boundary,
        null_controls={
            "the_microbiome_as_a_foundational_control_layer_is_not_empirical_validation_evidence": 1.0,
            "a_two_timescale_model_of_glial_neuronal_coupling_is_not_empirical_validation_evidence": 1.0,
        },
        problem_metadata={
            "source_ledger_span": SOURCE_LEDGER_SPAN,
            "source_ledger_ids": tuple(f"P0R{number:05d}" for number in range(5479, 5493)),
            "claim_boundary": CLAIM_BOUNDARY,
            "protocol_state": "source_the_microbiome_as_a_foundational_control_layer_only_no_experiment",
        },
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "HARDWARE_STATUS",
    "SOURCE_LEDGER_SPAN",
    "TheMicrobiomeAsAFoundationalControlLayerConfig",
    "TheMicrobiomeAsAFoundationalControlLayerFixtureResult",
    "classify_the_microbiome_as_a_foundational_control_layer_component",
    "the_microbiome_as_a_foundational_control_layer_labels",
    "validate_the_microbiome_as_a_foundational_control_layer_fixture",
]
