# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 Topological Invariants (bk): Determine the structure and richness of the qualia (Betti numbers). validation
"""Source-accounting checks for Paper 0 Topological Invariants (bk): Determine the structure and richness of the qualia (Betti numbers). records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

CLAIM_BOUNDARY = "source-bounded topological invariants bk determine the structure and richness of the qu source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"
SOURCE_LEDGER_SPAN = ("P0R06023", "P0R06030")


@dataclass(frozen=True, slots=True)
class TopologicalInvariantsBkDetermineTheStructureAndRichnessOfTheQuConfig:
    """Configuration for this Paper 0 source-accounting fixture."""

    expected_source_record_count: int = 8
    expected_component_count: int = 3
    next_source_boundary: str = "P0R06031"

    def __post_init__(self) -> None:
        if self.expected_source_record_count != 8:
            raise ValueError("expected_source_record_count must equal 8")
        if self.expected_component_count != 3:
            raise ValueError("expected_component_count must equal 3")
        if self.next_source_boundary != "P0R06031":
            raise ValueError("next_source_boundary must equal P0R06031")


@dataclass(frozen=True, slots=True)
class TopologicalInvariantsBkDetermineTheStructureAndRichnessOfTheQuFixtureResult:
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


def classify_topological_invariants_bk_determine_the_structure_and_richness_of_the_qu_component(
    component: str,
) -> str:
    """Classify source-defined components."""
    mapping = {
        "topological_invariants_bk_determine_the_structure_and_richness_of_the_qu": "topological_invariants_bk_determine_the_structure_and_richness_of_the_qu_source_boundary",
        "v_the_scpn_evolutionary_synthesis": "v_the_scpn_evolutionary_synthesis_source_boundary",
        "1_the_adaptive_potential_landscape_apl": "1_the_adaptive_potential_landscape_apl_source_boundary",
    }
    try:
        return mapping[component]
    except KeyError as exc:
        raise ValueError(
            "unknown topological_invariants_bk_determine_the_structure_and_richness_of_the_qu component"
        ) from exc


def topological_invariants_bk_determine_the_structure_and_richness_of_the_qu_labels() -> dict[
    str, str
]:
    """Return source-bounded labels for this slice."""
    return {
        "section": "Topological Invariants (bk): Determine the structure and richness of the qualia (Betti numbers).",
        "source_span": "P0R06023-P0R06030",
        "component_count": "3",
        "next_boundary": "P0R06031",
        "component_1": "Topological Invariants (bk): Determine the structure and richness of the qualia (Betti numbers).",
        "component_2": "V. The SCPN Evolutionary Synthesis",
        "component_3": "1. The Adaptive Potential Landscape (APL):",
    }


def validate_topological_invariants_bk_determine_the_structure_and_richness_of_the_qu_fixture(
    config: TopologicalInvariantsBkDetermineTheStructureAndRichnessOfTheQuConfig | None = None,
) -> TopologicalInvariantsBkDetermineTheStructureAndRichnessOfTheQuFixtureResult:
    """Validate source accounting for this Paper 0 slice."""
    cfg = config or TopologicalInvariantsBkDetermineTheStructureAndRichnessOfTheQuConfig()
    components = (
        "topological_invariants_bk_determine_the_structure_and_richness_of_the_qu",
        "v_the_scpn_evolutionary_synthesis",
        "1_the_adaptive_potential_landscape_apl",
    )
    return TopologicalInvariantsBkDetermineTheStructureAndRichnessOfTheQuFixtureResult(
        source_ledger_span=SOURCE_LEDGER_SPAN,
        hardware_status=HARDWARE_STATUS,
        claim_boundary=CLAIM_BOUNDARY,
        components={
            component: classify_topological_invariants_bk_determine_the_structure_and_richness_of_the_qu_component(
                component
            )
            for component in components
        },
        labels=topological_invariants_bk_determine_the_structure_and_richness_of_the_qu_labels(),
        source_record_count=cfg.expected_source_record_count,
        component_count=cfg.expected_component_count,
        next_source_boundary=cfg.next_source_boundary,
        null_controls={
            "topological_invariants_bk_determine_the_structure_and_richness_of_the_qu_is_not_empirical_validation_evidence": 1.0,
            "v_the_scpn_evolutionary_synthesis_is_not_empirical_validation_evidence": 1.0,
            "1_the_adaptive_potential_landscape_apl_is_not_empirical_validation_evidence": 1.0,
        },
        problem_metadata={
            "source_ledger_span": SOURCE_LEDGER_SPAN,
            "source_ledger_ids": tuple(f"P0R{number:05d}" for number in range(6023, 6031)),
            "claim_boundary": CLAIM_BOUNDARY,
            "protocol_state": "source_topological_invariants_bk_determine_the_structure_and_richness_of_the_qu_only_no_experiment",
        },
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "HARDWARE_STATUS",
    "SOURCE_LEDGER_SPAN",
    "TopologicalInvariantsBkDetermineTheStructureAndRichnessOfTheQuConfig",
    "TopologicalInvariantsBkDetermineTheStructureAndRichnessOfTheQuFixtureResult",
    "classify_topological_invariants_bk_determine_the_structure_and_richness_of_the_qu_component",
    "topological_invariants_bk_determine_the_structure_and_richness_of_the_qu_labels",
    "validate_topological_invariants_bk_determine_the_structure_and_richness_of_the_qu_fixture",
]
