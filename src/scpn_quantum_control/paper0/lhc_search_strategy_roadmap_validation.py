# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 LHC search-strategy roadmap validation
"""Source-accounting checks for Paper 0 LHC search-strategy roadmap records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

CLAIM_BOUNDARY = "source-bounded LHC search-strategy roadmap bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"
SOURCE_LEDGER_SPAN = ("P0R01684", "P0R01692")


@dataclass(frozen=True, slots=True)
class LHCSearchStrategyRoadmapConfig:
    """Configuration for the LHC search-strategy roadmap fixture."""

    expected_source_record_count: int = 9
    expected_component_count: int = 3
    next_source_boundary: str = "P0R01693"

    def __post_init__(self) -> None:
        if self.expected_source_record_count != 9:
            raise ValueError("expected_source_record_count must equal 9")
        if self.expected_component_count != 3:
            raise ValueError("expected_component_count must equal 3")
        if self.next_source_boundary != "P0R01693":
            raise ValueError("next_source_boundary must equal P0R01693")


@dataclass(frozen=True, slots=True)
class LHCSearchStrategyRoadmapFixtureResult:
    """Result for the Paper 0 LHC search-strategy roadmap fixture."""

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


def classify_lhc_search_strategy_roadmap_component(component: str) -> str:
    """Classify source-defined LHC search-strategy roadmap components."""
    mapping = {
        "search_signature_overview": "lhc_search_signature_overview_boundary",
        "table_roadmap": "table_2_experimental_roadmap_source_boundary",
        "ssb_cascade_transition": "ssb_cascade_section_transition_boundary",
    }
    try:
        return mapping[component]
    except KeyError as exc:
        raise ValueError("unknown LHC search-strategy roadmap component") from exc


def lhc_search_strategy_roadmap_labels() -> dict[str, str]:
    """Return source-bounded labels for the LHC search-strategy roadmap slice."""
    return {
        "section": "Phenomenology and Search Strategies at the LHC",
        "table": "TBL003 Proposed Experimental Search Parameters for the Psi-Higgs Boson",
        "channels": "exotic Higgs decays, resonant production, cascade decays, invisible decays",
        "next_boundary": "The Genesis of the Hierarchy: A Cascade of Sequential Symmetry Breaking",
    }


def validate_lhc_search_strategy_roadmap_fixture(
    config: LHCSearchStrategyRoadmapConfig | None = None,
) -> LHCSearchStrategyRoadmapFixtureResult:
    """Validate source accounting for the LHC search-strategy roadmap slice."""
    cfg = config or LHCSearchStrategyRoadmapConfig()
    components = (
        "search_signature_overview",
        "table_roadmap",
        "ssb_cascade_transition",
    )

    return LHCSearchStrategyRoadmapFixtureResult(
        source_ledger_span=SOURCE_LEDGER_SPAN,
        hardware_status=HARDWARE_STATUS,
        claim_boundary=CLAIM_BOUNDARY,
        components={
            component: classify_lhc_search_strategy_roadmap_component(component)
            for component in components
        },
        labels=lhc_search_strategy_roadmap_labels(),
        source_record_count=cfg.expected_source_record_count,
        component_count=cfg.expected_component_count,
        next_source_boundary=cfg.next_source_boundary,
        null_controls={
            "lhc_search_channels_are_not_observed_psi_higgs_events": 1.0,
            "tbl003_is_source_roadmap_not_experimental_result": 1.0,
            "ssb_cascade_transition_has_no_empirical_claim": 1.0,
        },
        problem_metadata={
            "source_ledger_span": SOURCE_LEDGER_SPAN,
            "source_ledger_ids": tuple(f"P0R{number:05d}" for number in range(1684, 1693)),
            "claim_boundary": CLAIM_BOUNDARY,
            "protocol_state": "source_lhc_search_strategy_roadmap_only_no_experiment",
        },
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "HARDWARE_STATUS",
    "SOURCE_LEDGER_SPAN",
    "LHCSearchStrategyRoadmapConfig",
    "LHCSearchStrategyRoadmapFixtureResult",
    "classify_lhc_search_strategy_roadmap_component",
    "lhc_search_strategy_roadmap_labels",
    "validate_lhc_search_strategy_roadmap_fixture",
]
