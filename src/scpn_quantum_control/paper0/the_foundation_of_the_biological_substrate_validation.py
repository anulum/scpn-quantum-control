# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 The Foundation of the Biological Substrate validation
"""Source-accounting checks for Paper 0 The Foundation of the Biological Substrate records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

CLAIM_BOUNDARY = "source-bounded the foundation of the biological substrate source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"
SOURCE_LEDGER_SPAN = ("P0R05306", "P0R05313")


@dataclass(frozen=True, slots=True)
class TheFoundationOfTheBiologicalSubstrateConfig:
    """Configuration for this Paper 0 source-accounting fixture."""

    expected_source_record_count: int = 8
    expected_component_count: int = 3
    next_source_boundary: str = "P0R05314"

    def __post_init__(self) -> None:
        if self.expected_source_record_count != 8:
            raise ValueError("expected_source_record_count must equal 8")
        if self.expected_component_count != 3:
            raise ValueError("expected_component_count must equal 3")
        if self.next_source_boundary != "P0R05314":
            raise ValueError("next_source_boundary must equal P0R05314")


@dataclass(frozen=True, slots=True)
class TheFoundationOfTheBiologicalSubstrateFixtureResult:
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


def classify_the_foundation_of_the_biological_substrate_component(component: str) -> str:
    """Classify source-defined components."""
    mapping = {
        "the_foundation_of_the_biological_substrate": "the_foundation_of_the_biological_substrate_source_boundary",
        "i_the_qed_of_water_coherence_domains_cds": "i_the_qed_of_water_coherence_domains_cds_source_boundary",
        "ii_the_emergence_of_life_abiogenesis_within_the_scpn": "ii_the_emergence_of_life_abiogenesis_within_the_scpn_source_boundary",
    }
    try:
        return mapping[component]
    except KeyError as exc:
        raise ValueError("unknown the_foundation_of_the_biological_substrate component") from exc


def the_foundation_of_the_biological_substrate_labels() -> dict[str, str]:
    """Return source-bounded labels for this slice."""
    return {
        "section": "The Foundation of the Biological Substrate",
        "source_span": "P0R05306-P0R05313",
        "component_count": "3",
        "next_boundary": "P0R05314",
        "component_1": "The Foundation of the Biological Substrate",
        "component_2": "I. The QED of Water: Coherence Domains (CDs)",
        "component_3": "II. The Emergence of Life: Abiogenesis within the SCPN",
    }


def validate_the_foundation_of_the_biological_substrate_fixture(
    config: TheFoundationOfTheBiologicalSubstrateConfig | None = None,
) -> TheFoundationOfTheBiologicalSubstrateFixtureResult:
    """Validate source accounting for this Paper 0 slice."""
    cfg = config or TheFoundationOfTheBiologicalSubstrateConfig()
    components = (
        "the_foundation_of_the_biological_substrate",
        "i_the_qed_of_water_coherence_domains_cds",
        "ii_the_emergence_of_life_abiogenesis_within_the_scpn",
    )
    return TheFoundationOfTheBiologicalSubstrateFixtureResult(
        source_ledger_span=SOURCE_LEDGER_SPAN,
        hardware_status=HARDWARE_STATUS,
        claim_boundary=CLAIM_BOUNDARY,
        components={
            component: classify_the_foundation_of_the_biological_substrate_component(component)
            for component in components
        },
        labels=the_foundation_of_the_biological_substrate_labels(),
        source_record_count=cfg.expected_source_record_count,
        component_count=cfg.expected_component_count,
        next_source_boundary=cfg.next_source_boundary,
        null_controls={
            "the_foundation_of_the_biological_substrate_is_not_empirical_validation_evidence": 1.0,
            "i_the_qed_of_water_coherence_domains_cds_is_not_empirical_validation_evidence": 1.0,
            "ii_the_emergence_of_life_abiogenesis_within_the_scpn_is_not_empirical_validation_evidence": 1.0,
        },
        problem_metadata={
            "source_ledger_span": SOURCE_LEDGER_SPAN,
            "source_ledger_ids": tuple(f"P0R{number:05d}" for number in range(5306, 5314)),
            "claim_boundary": CLAIM_BOUNDARY,
            "protocol_state": "source_the_foundation_of_the_biological_substrate_only_no_experiment",
        },
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "HARDWARE_STATUS",
    "SOURCE_LEDGER_SPAN",
    "TheFoundationOfTheBiologicalSubstrateConfig",
    "TheFoundationOfTheBiologicalSubstrateFixtureResult",
    "classify_the_foundation_of_the_biological_substrate_component",
    "the_foundation_of_the_biological_substrate_labels",
    "validate_the_foundation_of_the_biological_substrate_fixture",
]
