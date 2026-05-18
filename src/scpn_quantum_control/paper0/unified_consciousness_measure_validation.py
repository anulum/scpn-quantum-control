# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 Unified Consciousness Measure: validation
"""Source-accounting checks for Paper 0 Unified Consciousness Measure: records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

CLAIM_BOUNDARY = "source-bounded unified consciousness measure source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"
SOURCE_LEDGER_SPAN = ("P0R03564", "P0R03580")


@dataclass(frozen=True, slots=True)
class UnifiedConsciousnessMeasureConfig:
    """Configuration for this Paper 0 source-accounting fixture."""

    expected_source_record_count: int = 17
    expected_component_count: int = 2
    next_source_boundary: str = "P0R03581"

    def __post_init__(self) -> None:
        if self.expected_source_record_count != 17:
            raise ValueError("expected_source_record_count must equal 17")
        if self.expected_component_count != 2:
            raise ValueError("expected_component_count must equal 2")
        if self.next_source_boundary != "P0R03581":
            raise ValueError("next_source_boundary must equal P0R03581")


@dataclass(frozen=True, slots=True)
class UnifiedConsciousnessMeasureFixtureResult:
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


def classify_unified_consciousness_measure_component(component: str) -> str:
    """Classify source-defined components."""
    mapping = {
        "unified_consciousness_measure": "unified_consciousness_measure_source_boundary",
        "iit_axioms_in_scpn": "iit_axioms_in_scpn_source_boundary",
    }
    try:
        return mapping[component]
    except KeyError as exc:
        raise ValueError("unknown unified_consciousness_measure component") from exc


def unified_consciousness_measure_labels() -> dict[str, str]:
    """Return source-bounded labels for this slice."""
    return {
        "section": "Unified Consciousness Measure:",
        "source_span": "P0R03564-P0R03580",
        "component_count": "2",
        "next_boundary": "P0R03581",
        "component_1": "Unified Consciousness Measure:",
        "component_2": "IIT Axioms in SCPN:",
    }


def validate_unified_consciousness_measure_fixture(
    config: UnifiedConsciousnessMeasureConfig | None = None,
) -> UnifiedConsciousnessMeasureFixtureResult:
    """Validate source accounting for this Paper 0 slice."""
    cfg = config or UnifiedConsciousnessMeasureConfig()
    components = ("unified_consciousness_measure", "iit_axioms_in_scpn")
    return UnifiedConsciousnessMeasureFixtureResult(
        source_ledger_span=SOURCE_LEDGER_SPAN,
        hardware_status=HARDWARE_STATUS,
        claim_boundary=CLAIM_BOUNDARY,
        components={
            component: classify_unified_consciousness_measure_component(component)
            for component in components
        },
        labels=unified_consciousness_measure_labels(),
        source_record_count=cfg.expected_source_record_count,
        component_count=cfg.expected_component_count,
        next_source_boundary=cfg.next_source_boundary,
        null_controls={
            "unified_consciousness_measure_is_not_empirical_validation_evidence": 1.0,
            "iit_axioms_in_scpn_is_not_empirical_validation_evidence": 1.0,
        },
        problem_metadata={
            "source_ledger_span": SOURCE_LEDGER_SPAN,
            "source_ledger_ids": tuple(f"P0R{number:05d}" for number in range(3564, 3581)),
            "claim_boundary": CLAIM_BOUNDARY,
            "protocol_state": "source_unified_consciousness_measure_only_no_experiment",
        },
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "HARDWARE_STATUS",
    "SOURCE_LEDGER_SPAN",
    "UnifiedConsciousnessMeasureConfig",
    "UnifiedConsciousnessMeasureFixtureResult",
    "classify_unified_consciousness_measure_component",
    "unified_consciousness_measure_labels",
    "validate_unified_consciousness_measure_fixture",
]
