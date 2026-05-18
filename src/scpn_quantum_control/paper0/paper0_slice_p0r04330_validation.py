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
    "source-bounded paper0 slice p0r04330 source-accounting bridge; not validation evidence"
)
HARDWARE_STATUS = "source_methodology_no_experiment"
SOURCE_LEDGER_SPAN = ("P0R04330", "P0R04337")


@dataclass(frozen=True, slots=True)
class Paper0SliceP0r04330Config:
    """Configuration for this Paper 0 source-accounting fixture."""

    expected_source_record_count: int = 8
    expected_component_count: int = 8
    next_source_boundary: str = "P0R04338"

    def __post_init__(self) -> None:
        if self.expected_source_record_count != 8:
            raise ValueError("expected_source_record_count must equal 8")
        if self.expected_component_count != 8:
            raise ValueError("expected_component_count must equal 8")
        if self.next_source_boundary != "P0R04338":
            raise ValueError("next_source_boundary must equal P0R04338")


@dataclass(frozen=True, slots=True)
class Paper0SliceP0r04330FixtureResult:
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


def classify_paper0_slice_p0r04330_component(component: str) -> str:
    """Classify source-defined components."""
    mapping = {
        "p0r04330": "p0r04330_source_boundary",
        "p0r04331": "p0r04331_source_boundary",
        "p0r04332": "p0r04332_source_boundary",
        "p0r04333": "p0r04333_source_boundary",
        "p0r04334": "p0r04334_source_boundary",
        "p0r04335": "p0r04335_source_boundary",
        "p0r04336": "p0r04336_source_boundary",
        "p0r04337": "p0r04337_source_boundary",
    }
    try:
        return mapping[component]
    except KeyError as exc:
        raise ValueError("unknown paper0_slice_p0r04330 component") from exc


def paper0_slice_p0r04330_labels() -> dict[str, str]:
    """Return source-bounded labels for this slice."""
    return {
        "section": "",
        "source_span": "P0R04330-P0R04337",
        "component_count": "8",
        "next_boundary": "P0R04338",
        "component_1": "P0R04330",
        "component_2": "P0R04331",
        "component_3": "P0R04332",
        "component_4": "P0R04333",
        "component_5": "P0R04334",
        "component_6": "P0R04335",
        "component_7": "P0R04336",
        "component_8": "P0R04337",
    }


def validate_paper0_slice_p0r04330_fixture(
    config: Paper0SliceP0r04330Config | None = None,
) -> Paper0SliceP0r04330FixtureResult:
    """Validate source accounting for this Paper 0 slice."""
    cfg = config or Paper0SliceP0r04330Config()
    components = (
        "p0r04330",
        "p0r04331",
        "p0r04332",
        "p0r04333",
        "p0r04334",
        "p0r04335",
        "p0r04336",
        "p0r04337",
    )
    return Paper0SliceP0r04330FixtureResult(
        source_ledger_span=SOURCE_LEDGER_SPAN,
        hardware_status=HARDWARE_STATUS,
        claim_boundary=CLAIM_BOUNDARY,
        components={
            component: classify_paper0_slice_p0r04330_component(component)
            for component in components
        },
        labels=paper0_slice_p0r04330_labels(),
        source_record_count=cfg.expected_source_record_count,
        component_count=cfg.expected_component_count,
        next_source_boundary=cfg.next_source_boundary,
        null_controls={
            "p0r04330_is_not_empirical_validation_evidence": 1.0,
            "p0r04331_is_not_empirical_validation_evidence": 1.0,
            "p0r04332_is_not_empirical_validation_evidence": 1.0,
            "p0r04333_is_not_empirical_validation_evidence": 1.0,
            "p0r04334_is_not_empirical_validation_evidence": 1.0,
            "p0r04335_is_not_empirical_validation_evidence": 1.0,
            "p0r04336_is_not_empirical_validation_evidence": 1.0,
            "p0r04337_is_not_empirical_validation_evidence": 1.0,
        },
        problem_metadata={
            "source_ledger_span": SOURCE_LEDGER_SPAN,
            "source_ledger_ids": tuple(f"P0R{number:05d}" for number in range(4330, 4338)),
            "claim_boundary": CLAIM_BOUNDARY,
            "protocol_state": "source_paper0_slice_p0r04330_only_no_experiment",
        },
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "HARDWARE_STATUS",
    "SOURCE_LEDGER_SPAN",
    "Paper0SliceP0r04330Config",
    "Paper0SliceP0r04330FixtureResult",
    "classify_paper0_slice_p0r04330_component",
    "paper0_slice_p0r04330_labels",
    "validate_paper0_slice_p0r04330_fixture",
]
