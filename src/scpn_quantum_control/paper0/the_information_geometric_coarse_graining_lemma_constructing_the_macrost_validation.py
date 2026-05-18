# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 The Information-Geometric Coarse-Graining Lemma: Constructing the Macrostates Manifold validation
"""Source-accounting checks for Paper 0 The Information-Geometric Coarse-Graining Lemma: Constructing the Macrostates Manifold records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

CLAIM_BOUNDARY = "source-bounded the information geometric coarse graining lemma constructing the macrost source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"
SOURCE_LEDGER_SPAN = ("P0R04151", "P0R04167")


@dataclass(frozen=True, slots=True)
class TheInformationGeometricCoarseGrainingLemmaConstructingTheMacrostConfig:
    """Configuration for this Paper 0 source-accounting fixture."""

    expected_source_record_count: int = 17
    expected_component_count: int = 1
    next_source_boundary: str = "P0R04168"

    def __post_init__(self) -> None:
        if self.expected_source_record_count != 17:
            raise ValueError("expected_source_record_count must equal 17")
        if self.expected_component_count != 1:
            raise ValueError("expected_component_count must equal 1")
        if self.next_source_boundary != "P0R04168":
            raise ValueError("next_source_boundary must equal P0R04168")


@dataclass(frozen=True, slots=True)
class TheInformationGeometricCoarseGrainingLemmaConstructingTheMacrostFixtureResult:
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


def classify_the_information_geometric_coarse_graining_lemma_constructing_the_macrost_component(
    component: str,
) -> str:
    """Classify source-defined components."""
    mapping = {
        "the_information_geometric_coarse_graining_lemma_constructing_the_macrost": "the_information_geometric_coarse_graining_lemma_constructing_the_macrost_source_boundary"
    }
    try:
        return mapping[component]
    except KeyError as exc:
        raise ValueError(
            "unknown the_information_geometric_coarse_graining_lemma_constructing_the_macrost component"
        ) from exc


def the_information_geometric_coarse_graining_lemma_constructing_the_macrost_labels() -> dict[
    str, str
]:
    """Return source-bounded labels for this slice."""
    return {
        "section": "The Information-Geometric Coarse-Graining Lemma: Constructing the Macrostates Manifold",
        "source_span": "P0R04151-P0R04167",
        "component_count": "1",
        "next_boundary": "P0R04168",
        "component_1": "The Information-Geometric Coarse-Graining Lemma: Constructing the Macrostates Manifold",
    }


def validate_the_information_geometric_coarse_graining_lemma_constructing_the_macrost_fixture(
    config: TheInformationGeometricCoarseGrainingLemmaConstructingTheMacrostConfig | None = None,
) -> TheInformationGeometricCoarseGrainingLemmaConstructingTheMacrostFixtureResult:
    """Validate source accounting for this Paper 0 slice."""
    cfg = config or TheInformationGeometricCoarseGrainingLemmaConstructingTheMacrostConfig()
    components = ("the_information_geometric_coarse_graining_lemma_constructing_the_macrost",)
    return TheInformationGeometricCoarseGrainingLemmaConstructingTheMacrostFixtureResult(
        source_ledger_span=SOURCE_LEDGER_SPAN,
        hardware_status=HARDWARE_STATUS,
        claim_boundary=CLAIM_BOUNDARY,
        components={
            component: classify_the_information_geometric_coarse_graining_lemma_constructing_the_macrost_component(
                component
            )
            for component in components
        },
        labels=the_information_geometric_coarse_graining_lemma_constructing_the_macrost_labels(),
        source_record_count=cfg.expected_source_record_count,
        component_count=cfg.expected_component_count,
        next_source_boundary=cfg.next_source_boundary,
        null_controls={
            "the_information_geometric_coarse_graining_lemma_constructing_the_macrost_is_not_empirical_validation_evidence": 1.0
        },
        problem_metadata={
            "source_ledger_span": SOURCE_LEDGER_SPAN,
            "source_ledger_ids": tuple(f"P0R{number:05d}" for number in range(4151, 4168)),
            "claim_boundary": CLAIM_BOUNDARY,
            "protocol_state": "source_the_information_geometric_coarse_graining_lemma_constructing_the_macrost_only_no_experiment",
        },
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "HARDWARE_STATUS",
    "SOURCE_LEDGER_SPAN",
    "TheInformationGeometricCoarseGrainingLemmaConstructingTheMacrostConfig",
    "TheInformationGeometricCoarseGrainingLemmaConstructingTheMacrostFixtureResult",
    "classify_the_information_geometric_coarse_graining_lemma_constructing_the_macrost_component",
    "the_information_geometric_coarse_graining_lemma_constructing_the_macrost_labels",
    "validate_the_information_geometric_coarse_graining_lemma_constructing_the_macrost_fixture",
]
