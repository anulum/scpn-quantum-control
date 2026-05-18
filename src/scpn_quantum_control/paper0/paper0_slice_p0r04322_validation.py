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

CLAIM_BOUNDARY = (
    "source-bounded paper0 slice p0r04322 source-accounting bridge; not validation evidence"
)
HARDWARE_STATUS = "source_methodology_no_experiment"
SOURCE_LEDGER_SPAN = ("P0R04322", "P0R04329")


@dataclass(frozen=True, slots=True)
class Paper0SliceP0r04322Config:
    """Configuration for this Paper 0 source-accounting fixture."""

    expected_source_record_count: int = 8
    expected_component_count: int = 8
    next_source_boundary: str = "P0R04330"

    def __post_init__(self) -> None:
        if self.expected_source_record_count != 8:
            raise ValueError("expected_source_record_count must equal 8")
        if self.expected_component_count != 8:
            raise ValueError("expected_component_count must equal 8")
        if self.next_source_boundary != "P0R04330":
            raise ValueError("next_source_boundary must equal P0R04330")


@dataclass(frozen=True, slots=True)
class Paper0SliceP0r04322FixtureResult:
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


def classify_paper0_slice_p0r04322_component(component: str) -> str:
    """Classify source-defined components."""
    mapping = {
        "p0r04322": "p0r04322_source_boundary",
        "p0r04323": "p0r04323_source_boundary",
        "p0r04324": "p0r04324_source_boundary",
        "p0r04325": "p0r04325_source_boundary",
        "p0r04326": "p0r04326_source_boundary",
        "p0r04327": "p0r04327_source_boundary",
        "p0r04328": "p0r04328_source_boundary",
        "p0r04329": "p0r04329_source_boundary",
    }
    try:
        return mapping[component]
    except KeyError as exc:
        raise ValueError("unknown paper0_slice_p0r04322 component") from exc


def paper0_slice_p0r04322_labels() -> dict[str, str]:
    """Return source-bounded labels for this slice."""
    return {
        "section": "",
        "source_span": "P0R04322-P0R04329",
        "component_count": "8",
        "next_boundary": "P0R04330",
        "component_1": "P0R04322",
        "component_2": "P0R04323",
        "component_3": "P0R04324",
        "component_4": "P0R04325",
        "component_5": "P0R04326",
        "component_6": "P0R04327",
        "component_7": "P0R04328",
        "component_8": "P0R04329",
    }


def validate_paper0_slice_p0r04322_fixture(
    config: Paper0SliceP0r04322Config | None = None,
) -> Paper0SliceP0r04322FixtureResult:
    """Validate source accounting for this Paper 0 slice."""
    cfg = config or Paper0SliceP0r04322Config()
    components = (
        "p0r04322",
        "p0r04323",
        "p0r04324",
        "p0r04325",
        "p0r04326",
        "p0r04327",
        "p0r04328",
        "p0r04329",
    )
    return Paper0SliceP0r04322FixtureResult(
        source_ledger_span=SOURCE_LEDGER_SPAN,
        hardware_status=HARDWARE_STATUS,
        claim_boundary=CLAIM_BOUNDARY,
        components={
            component: classify_paper0_slice_p0r04322_component(component)
            for component in components
        },
        labels=paper0_slice_p0r04322_labels(),
        source_record_count=cfg.expected_source_record_count,
        component_count=cfg.expected_component_count,
        next_source_boundary=cfg.next_source_boundary,
        null_controls={
            "p0r04322_is_not_empirical_validation_evidence": 1.0,
            "p0r04323_is_not_empirical_validation_evidence": 1.0,
            "p0r04324_is_not_empirical_validation_evidence": 1.0,
            "p0r04325_is_not_empirical_validation_evidence": 1.0,
            "p0r04326_is_not_empirical_validation_evidence": 1.0,
            "p0r04327_is_not_empirical_validation_evidence": 1.0,
            "p0r04328_is_not_empirical_validation_evidence": 1.0,
            "p0r04329_is_not_empirical_validation_evidence": 1.0,
        },
        problem_metadata={
            "source_ledger_span": SOURCE_LEDGER_SPAN,
            "source_ledger_ids": tuple(f"P0R{number:05d}" for number in range(4322, 4330)),
            "claim_boundary": CLAIM_BOUNDARY,
            "protocol_state": "source_paper0_slice_p0r04322_only_no_experiment",
        },
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "HARDWARE_STATUS",
    "SOURCE_LEDGER_SPAN",
    "Paper0SliceP0r04322Config",
    "Paper0SliceP0r04322FixtureResult",
    "classify_paper0_slice_p0r04322_component",
    "paper0_slice_p0r04322_labels",
    "validate_paper0_slice_p0r04322_fixture",
]
