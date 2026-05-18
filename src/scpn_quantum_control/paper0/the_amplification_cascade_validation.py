# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 The Amplification Cascade: validation
"""Source-accounting checks for Paper 0 The Amplification Cascade: records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

CLAIM_BOUNDARY = (
    "source-bounded the amplification cascade source-accounting bridge; not validation evidence"
)
HARDWARE_STATUS = "source_methodology_no_experiment"
SOURCE_LEDGER_SPAN = ("P0R05323", "P0R05330")


@dataclass(frozen=True, slots=True)
class TheAmplificationCascadeConfig:
    """Configuration for this Paper 0 source-accounting fixture."""

    expected_source_record_count: int = 8
    expected_component_count: int = 2
    next_source_boundary: str = "P0R05331"

    def __post_init__(self) -> None:
        if self.expected_source_record_count != 8:
            raise ValueError("expected_source_record_count must equal 8")
        if self.expected_component_count != 2:
            raise ValueError("expected_component_count must equal 2")
        if self.next_source_boundary != "P0R05331":
            raise ValueError("next_source_boundary must equal P0R05331")


@dataclass(frozen=True, slots=True)
class TheAmplificationCascadeFixtureResult:
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


def classify_the_amplification_cascade_component(component: str) -> str:
    """Classify source-defined components."""
    mapping = {
        "the_amplification_cascade": "the_amplification_cascade_source_boundary",
        "the_morphogenetic_blueprint_layers_3_4_here_we_explore_the_genomic_epige": "the_morphogenetic_blueprint_layers_3_4_here_we_explore_the_genomic_epige_source_boundary",
    }
    try:
        return mapping[component]
    except KeyError as exc:
        raise ValueError("unknown the_amplification_cascade component") from exc


def the_amplification_cascade_labels() -> dict[str, str]:
    """Return source-bounded labels for this slice."""
    return {
        "section": "The Amplification Cascade:",
        "source_span": "P0R05323-P0R05330",
        "component_count": "2",
        "next_boundary": "P0R05331",
        "component_1": "The Amplification Cascade:",
        "component_2": "The Morphogenetic Blueprint (Layers 3-4): Here, we explore the Genomic-Epigenomic-Morphogenetic (Layer 3) layer, framing DNA as a resonant fractal antenna that reads and writes information via CISS and bioelectric codes. This blueprint guides the formation of the organism, whose tissues then learn to oscillate in unison in the Cellular-Tissue Synchronisation (Layer 4) layer, where coherent rhythms emerge from networks operating at quasicriticality.",
    }


def validate_the_amplification_cascade_fixture(
    config: TheAmplificationCascadeConfig | None = None,
) -> TheAmplificationCascadeFixtureResult:
    """Validate source accounting for this Paper 0 slice."""
    cfg = config or TheAmplificationCascadeConfig()
    components = (
        "the_amplification_cascade",
        "the_morphogenetic_blueprint_layers_3_4_here_we_explore_the_genomic_epige",
    )
    return TheAmplificationCascadeFixtureResult(
        source_ledger_span=SOURCE_LEDGER_SPAN,
        hardware_status=HARDWARE_STATUS,
        claim_boundary=CLAIM_BOUNDARY,
        components={
            component: classify_the_amplification_cascade_component(component)
            for component in components
        },
        labels=the_amplification_cascade_labels(),
        source_record_count=cfg.expected_source_record_count,
        component_count=cfg.expected_component_count,
        next_source_boundary=cfg.next_source_boundary,
        null_controls={
            "the_amplification_cascade_is_not_empirical_validation_evidence": 1.0,
            "the_morphogenetic_blueprint_layers_3_4_here_we_explore_the_genomic_epige_is_not_empirical_validation_evidence": 1.0,
        },
        problem_metadata={
            "source_ledger_span": SOURCE_LEDGER_SPAN,
            "source_ledger_ids": tuple(f"P0R{number:05d}" for number in range(5323, 5331)),
            "claim_boundary": CLAIM_BOUNDARY,
            "protocol_state": "source_the_amplification_cascade_only_no_experiment",
        },
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "HARDWARE_STATUS",
    "SOURCE_LEDGER_SPAN",
    "TheAmplificationCascadeConfig",
    "TheAmplificationCascadeFixtureResult",
    "classify_the_amplification_cascade_component",
    "the_amplification_cascade_labels",
    "validate_the_amplification_cascade_fixture",
]
