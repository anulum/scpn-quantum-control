# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 Experimental Design: Coherence vs. Engagement Optimisation validation
"""Source-accounting checks for Paper 0 Experimental Design: Coherence vs. Engagement Optimisation records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

CLAIM_BOUNDARY = "source-bounded experimental design coherence vs engagement optimisation source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"
SOURCE_LEDGER_SPAN = ("P0R05256", "P0R05263")


@dataclass(frozen=True, slots=True)
class ExperimentalDesignCoherenceVsEngagementOptimisationConfig:
    """Configuration for this Paper 0 source-accounting fixture."""

    expected_source_record_count: int = 8
    expected_component_count: int = 1
    next_source_boundary: str = "P0R05264"

    def __post_init__(self) -> None:
        if self.expected_source_record_count != 8:
            raise ValueError("expected_source_record_count must equal 8")
        if self.expected_component_count != 1:
            raise ValueError("expected_component_count must equal 1")
        if self.next_source_boundary != "P0R05264":
            raise ValueError("next_source_boundary must equal P0R05264")


@dataclass(frozen=True, slots=True)
class ExperimentalDesignCoherenceVsEngagementOptimisationFixtureResult:
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


def classify_experimental_design_coherence_vs_engagement_optimisation_component(
    component: str,
) -> str:
    """Classify source-defined components."""
    mapping = {
        "experimental_design_coherence_vs_engagement_optimisation": "experimental_design_coherence_vs_engagement_optimisation_source_boundary"
    }
    try:
        return mapping[component]
    except KeyError as exc:
        raise ValueError(
            "unknown experimental_design_coherence_vs_engagement_optimisation component"
        ) from exc


def experimental_design_coherence_vs_engagement_optimisation_labels() -> dict[str, str]:
    """Return source-bounded labels for this slice."""
    return {
        "section": "Experimental Design: Coherence vs. Engagement Optimisation",
        "source_span": "P0R05256-P0R05263",
        "component_count": "1",
        "next_boundary": "P0R05264",
        "component_1": "Experimental Design: Coherence vs. Engagement Optimisation",
    }


def validate_experimental_design_coherence_vs_engagement_optimisation_fixture(
    config: ExperimentalDesignCoherenceVsEngagementOptimisationConfig | None = None,
) -> ExperimentalDesignCoherenceVsEngagementOptimisationFixtureResult:
    """Validate source accounting for this Paper 0 slice."""
    cfg = config or ExperimentalDesignCoherenceVsEngagementOptimisationConfig()
    components = ("experimental_design_coherence_vs_engagement_optimisation",)
    return ExperimentalDesignCoherenceVsEngagementOptimisationFixtureResult(
        source_ledger_span=SOURCE_LEDGER_SPAN,
        hardware_status=HARDWARE_STATUS,
        claim_boundary=CLAIM_BOUNDARY,
        components={
            component: classify_experimental_design_coherence_vs_engagement_optimisation_component(
                component
            )
            for component in components
        },
        labels=experimental_design_coherence_vs_engagement_optimisation_labels(),
        source_record_count=cfg.expected_source_record_count,
        component_count=cfg.expected_component_count,
        next_source_boundary=cfg.next_source_boundary,
        null_controls={
            "experimental_design_coherence_vs_engagement_optimisation_is_not_empirical_validation_evidence": 1.0
        },
        problem_metadata={
            "source_ledger_span": SOURCE_LEDGER_SPAN,
            "source_ledger_ids": tuple(f"P0R{number:05d}" for number in range(5256, 5264)),
            "claim_boundary": CLAIM_BOUNDARY,
            "protocol_state": "source_experimental_design_coherence_vs_engagement_optimisation_only_no_experiment",
        },
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "HARDWARE_STATUS",
    "SOURCE_LEDGER_SPAN",
    "ExperimentalDesignCoherenceVsEngagementOptimisationConfig",
    "ExperimentalDesignCoherenceVsEngagementOptimisationFixtureResult",
    "classify_experimental_design_coherence_vs_engagement_optimisation_component",
    "experimental_design_coherence_vs_engagement_optimisation_labels",
    "validate_experimental_design_coherence_vs_engagement_optimisation_fixture",
]
