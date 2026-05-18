# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 The Mechanism of Interaction: validation
"""Source-accounting checks for Paper 0 The Mechanism of Interaction: records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

CLAIM_BOUNDARY = (
    "source-bounded the mechanism of interaction source-accounting bridge; not validation evidence"
)
HARDWARE_STATUS = "source_methodology_no_experiment"
SOURCE_LEDGER_SPAN = ("P0R03148", "P0R03173")


@dataclass(frozen=True, slots=True)
class TheMechanismOfInteractionConfig:
    """Configuration for this Paper 0 source-accounting fixture."""

    expected_source_record_count: int = 26
    expected_component_count: int = 2
    next_source_boundary: str = "P0R03174"

    def __post_init__(self) -> None:
        if self.expected_source_record_count != 26:
            raise ValueError("expected_source_record_count must equal 26")
        if self.expected_component_count != 2:
            raise ValueError("expected_component_count must equal 2")
        if self.next_source_boundary != "P0R03174":
            raise ValueError("next_source_boundary must equal P0R03174")


@dataclass(frozen=True, slots=True)
class TheMechanismOfInteractionFixtureResult:
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


def classify_the_mechanism_of_interaction_component(component: str) -> str:
    """Classify source-defined components."""
    mapping = {
        "the_mechanism_of_interaction": "the_mechanism_of_interaction_source_boundary",
        "stabiliser_transfer_lemma_sketch_ms_qec_bridge_stabiliser_transfer_acros": "stabiliser_transfer_lemma_sketch_ms_qec_bridge_stabiliser_transfer_acros_source_boundary",
    }
    try:
        return mapping[component]
    except KeyError as exc:
        raise ValueError("unknown the_mechanism_of_interaction component") from exc


def the_mechanism_of_interaction_labels() -> dict[str, str]:
    """Return source-bounded labels for this slice."""
    return {
        "section": "The Mechanism of Interaction:",
        "source_span": "P0R03148-P0R03173",
        "component_count": "2",
        "next_boundary": "P0R03174",
        "component_1": "The Mechanism of Interaction:",
        "component_2": "Stabiliser Transfer Lemma (Sketch) - MS-QEC Bridge: Stabiliser Transfer across L9->L10",
    }


def validate_the_mechanism_of_interaction_fixture(
    config: TheMechanismOfInteractionConfig | None = None,
) -> TheMechanismOfInteractionFixtureResult:
    """Validate source accounting for this Paper 0 slice."""
    cfg = config or TheMechanismOfInteractionConfig()
    components = (
        "the_mechanism_of_interaction",
        "stabiliser_transfer_lemma_sketch_ms_qec_bridge_stabiliser_transfer_acros",
    )
    return TheMechanismOfInteractionFixtureResult(
        source_ledger_span=SOURCE_LEDGER_SPAN,
        hardware_status=HARDWARE_STATUS,
        claim_boundary=CLAIM_BOUNDARY,
        components={
            component: classify_the_mechanism_of_interaction_component(component)
            for component in components
        },
        labels=the_mechanism_of_interaction_labels(),
        source_record_count=cfg.expected_source_record_count,
        component_count=cfg.expected_component_count,
        next_source_boundary=cfg.next_source_boundary,
        null_controls={
            "the_mechanism_of_interaction_is_not_empirical_validation_evidence": 1.0,
            "stabiliser_transfer_lemma_sketch_ms_qec_bridge_stabiliser_transfer_acros_is_not_empirical_validation_evidence": 1.0,
        },
        problem_metadata={
            "source_ledger_span": SOURCE_LEDGER_SPAN,
            "source_ledger_ids": tuple(f"P0R{number:05d}" for number in range(3148, 3174)),
            "claim_boundary": CLAIM_BOUNDARY,
            "protocol_state": "source_the_mechanism_of_interaction_only_no_experiment",
        },
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "HARDWARE_STATUS",
    "SOURCE_LEDGER_SPAN",
    "TheMechanismOfInteractionConfig",
    "TheMechanismOfInteractionFixtureResult",
    "classify_the_mechanism_of_interaction_component",
    "the_mechanism_of_interaction_labels",
    "validate_the_mechanism_of_interaction_fixture",
]
