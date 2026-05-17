# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0  validation
"""Source-accounting checks for Paper 0  records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

CLAIM_BOUNDARY = "source-bounded paper0 slice source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"
SOURCE_LEDGER_SPAN = ("P0R01959", "P0R01992")


@dataclass(frozen=True, slots=True)
class Paper0SliceConfig:
    """Configuration for this Paper 0 source-accounting fixture."""

    expected_source_record_count: int = 34
    expected_component_count: int = 3
    next_source_boundary: str = "P0R01993"

    def __post_init__(self) -> None:
        if self.expected_source_record_count != 34:
            raise ValueError("expected_source_record_count must equal 34")
        if self.expected_component_count != 3:
            raise ValueError("expected_component_count must equal 3")
        if self.next_source_boundary != "P0R01993":
            raise ValueError("next_source_boundary must equal P0R01993")


@dataclass(frozen=True, slots=True)
class Paper0SliceFixtureResult:
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


def classify_paper0_slice_component(component: str) -> str:
    """Classify source-defined components."""
    mapping = {
        "source_component": "source_component_source_boundary",
        "p0r01965": "p0r01965_source_boundary",
        "2_6_8_formal_derivation_of_the_hierarchy_via_bulk_brane_overlap_integral": "2_6_8_formal_derivation_of_the_hierarchy_via_bulk_brane_overlap_integral_source_boundary",
    }
    try:
        return mapping[component]
    except KeyError as exc:
        raise ValueError("unknown paper0_slice component") from exc


def paper0_slice_labels() -> dict[str, str]:
    """Return source-bounded labels for this slice."""
    return {
        "section": "",
        "source_span": "P0R01959-P0R01992",
        "component_count": "3",
        "next_boundary": "P0R01993",
        "component_1": "",
        "component_2": "P0R01965",
        "component_3": "2.6.8 Formal Derivation of the Hierarchy via Bulk-Brane Overlap Integrals",
    }


def validate_paper0_slice_fixture(
    config: Paper0SliceConfig | None = None,
) -> Paper0SliceFixtureResult:
    """Validate source accounting for this Paper 0 slice."""
    cfg = config or Paper0SliceConfig()
    components = (
        "source_component",
        "p0r01965",
        "2_6_8_formal_derivation_of_the_hierarchy_via_bulk_brane_overlap_integral",
    )
    return Paper0SliceFixtureResult(
        source_ledger_span=SOURCE_LEDGER_SPAN,
        hardware_status=HARDWARE_STATUS,
        claim_boundary=CLAIM_BOUNDARY,
        components={
            component: classify_paper0_slice_component(component) for component in components
        },
        labels=paper0_slice_labels(),
        source_record_count=cfg.expected_source_record_count,
        component_count=cfg.expected_component_count,
        next_source_boundary=cfg.next_source_boundary,
        null_controls={
            "source_component_is_not_empirical_validation_evidence": 1.0,
            "p0r01965_is_not_empirical_validation_evidence": 1.0,
            "2_6_8_formal_derivation_of_the_hierarchy_via_bulk_brane_overlap_integral_is_not_empirical_validation_evidence": 1.0,
        },
        problem_metadata={
            "source_ledger_span": SOURCE_LEDGER_SPAN,
            "source_ledger_ids": tuple(f"P0R{number:05d}" for number in range(1959, 1993)),
            "claim_boundary": CLAIM_BOUNDARY,
            "protocol_state": "source_paper0_slice_only_no_experiment",
        },
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "HARDWARE_STATUS",
    "SOURCE_LEDGER_SPAN",
    "Paper0SliceConfig",
    "Paper0SliceFixtureResult",
    "classify_paper0_slice_component",
    "paper0_slice_labels",
    "validate_paper0_slice_fixture",
]
