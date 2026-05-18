# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0  Gaia & Biosphere Intelligence validation
"""Source-accounting checks for Paper 0  Gaia & Biosphere Intelligence records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

CLAIM_BOUNDARY = (
    "source-bounded gaia biosphere intelligence source-accounting bridge; not validation evidence"
)
HARDWARE_STATUS = "source_methodology_no_experiment"
SOURCE_LEDGER_SPAN = ("P0R05910", "P0R05918")


@dataclass(frozen=True, slots=True)
class GaiaBiosphereIntelligenceConfig:
    """Configuration for this Paper 0 source-accounting fixture."""

    expected_source_record_count: int = 9
    expected_component_count: int = 2
    next_source_boundary: str = "P0R05919"

    def __post_init__(self) -> None:
        if self.expected_source_record_count != 9:
            raise ValueError("expected_source_record_count must equal 9")
        if self.expected_component_count != 2:
            raise ValueError("expected_component_count must equal 2")
        if self.next_source_boundary != "P0R05919":
            raise ValueError("next_source_boundary must equal P0R05919")


@dataclass(frozen=True, slots=True)
class GaiaBiosphereIntelligenceFixtureResult:
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


def classify_gaia_biosphere_intelligence_component(component: str) -> str:
    """Classify source-defined components."""
    mapping = {
        "gaia_biosphere_intelligence": "gaia_biosphere_intelligence_source_boundary",
        "metaphysical_foundational_crossovers": "metaphysical_foundational_crossovers_source_boundary",
    }
    try:
        return mapping[component]
    except KeyError as exc:
        raise ValueError("unknown gaia_biosphere_intelligence component") from exc


def gaia_biosphere_intelligence_labels() -> dict[str, str]:
    """Return source-bounded labels for this slice."""
    return {
        "section": " Gaia & Biosphere Intelligence",
        "source_span": "P0R05910-P0R05918",
        "component_count": "2",
        "next_boundary": "P0R05919",
        "component_1": "Gaia & Biosphere Intelligence",
        "component_2": "Metaphysical / Foundational Crossovers",
    }


def validate_gaia_biosphere_intelligence_fixture(
    config: GaiaBiosphereIntelligenceConfig | None = None,
) -> GaiaBiosphereIntelligenceFixtureResult:
    """Validate source accounting for this Paper 0 slice."""
    cfg = config or GaiaBiosphereIntelligenceConfig()
    components = ("gaia_biosphere_intelligence", "metaphysical_foundational_crossovers")
    return GaiaBiosphereIntelligenceFixtureResult(
        source_ledger_span=SOURCE_LEDGER_SPAN,
        hardware_status=HARDWARE_STATUS,
        claim_boundary=CLAIM_BOUNDARY,
        components={
            component: classify_gaia_biosphere_intelligence_component(component)
            for component in components
        },
        labels=gaia_biosphere_intelligence_labels(),
        source_record_count=cfg.expected_source_record_count,
        component_count=cfg.expected_component_count,
        next_source_boundary=cfg.next_source_boundary,
        null_controls={
            "gaia_biosphere_intelligence_is_not_empirical_validation_evidence": 1.0,
            "metaphysical_foundational_crossovers_is_not_empirical_validation_evidence": 1.0,
        },
        problem_metadata={
            "source_ledger_span": SOURCE_LEDGER_SPAN,
            "source_ledger_ids": tuple(f"P0R{number:05d}" for number in range(5910, 5919)),
            "claim_boundary": CLAIM_BOUNDARY,
            "protocol_state": "source_gaia_biosphere_intelligence_only_no_experiment",
        },
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "HARDWARE_STATUS",
    "SOURCE_LEDGER_SPAN",
    "GaiaBiosphereIntelligenceConfig",
    "GaiaBiosphereIntelligenceFixtureResult",
    "classify_gaia_biosphere_intelligence_component",
    "gaia_biosphere_intelligence_labels",
    "validate_gaia_biosphere_intelligence_fixture",
]
