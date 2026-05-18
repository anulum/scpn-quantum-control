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
    "source-bounded paper0 slice p0r02923 source-accounting bridge; not validation evidence"
)
HARDWARE_STATUS = "source_methodology_no_experiment"
SOURCE_LEDGER_SPAN = ("P0R02923", "P0R02930")


@dataclass(frozen=True, slots=True)
class Paper0SliceP0r02923Config:
    """Configuration for this Paper 0 source-accounting fixture."""

    expected_source_record_count: int = 8
    expected_component_count: int = 8
    next_source_boundary: str = "P0R02931"

    def __post_init__(self) -> None:
        if self.expected_source_record_count != 8:
            raise ValueError("expected_source_record_count must equal 8")
        if self.expected_component_count != 8:
            raise ValueError("expected_component_count must equal 8")
        if self.next_source_boundary != "P0R02931":
            raise ValueError("next_source_boundary must equal P0R02931")


@dataclass(frozen=True, slots=True)
class Paper0SliceP0r02923FixtureResult:
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


def classify_paper0_slice_p0r02923_component(component: str) -> str:
    """Classify source-defined components."""
    mapping = {
        "p0r02923": "p0r02923_source_boundary",
        "p0r02924": "p0r02924_source_boundary",
        "p0r02925": "p0r02925_source_boundary",
        "p0r02926": "p0r02926_source_boundary",
        "p0r02927": "p0r02927_source_boundary",
        "p0r02928": "p0r02928_source_boundary",
        "p0r02929": "p0r02929_source_boundary",
        "p0r02930": "p0r02930_source_boundary",
    }
    try:
        return mapping[component]
    except KeyError as exc:
        raise ValueError("unknown paper0_slice_p0r02923 component") from exc


def paper0_slice_p0r02923_labels() -> dict[str, str]:
    """Return source-bounded labels for this slice."""
    return {
        "section": "",
        "source_span": "P0R02923-P0R02930",
        "component_count": "8",
        "next_boundary": "P0R02931",
        "component_1": "P0R02923",
        "component_2": "P0R02924",
        "component_3": "P0R02925",
        "component_4": "P0R02926",
        "component_5": "P0R02927",
        "component_6": "P0R02928",
        "component_7": "P0R02929",
        "component_8": "P0R02930",
    }


def validate_paper0_slice_p0r02923_fixture(
    config: Paper0SliceP0r02923Config | None = None,
) -> Paper0SliceP0r02923FixtureResult:
    """Validate source accounting for this Paper 0 slice."""
    cfg = config or Paper0SliceP0r02923Config()
    components = (
        "p0r02923",
        "p0r02924",
        "p0r02925",
        "p0r02926",
        "p0r02927",
        "p0r02928",
        "p0r02929",
        "p0r02930",
    )
    return Paper0SliceP0r02923FixtureResult(
        source_ledger_span=SOURCE_LEDGER_SPAN,
        hardware_status=HARDWARE_STATUS,
        claim_boundary=CLAIM_BOUNDARY,
        components={
            component: classify_paper0_slice_p0r02923_component(component)
            for component in components
        },
        labels=paper0_slice_p0r02923_labels(),
        source_record_count=cfg.expected_source_record_count,
        component_count=cfg.expected_component_count,
        next_source_boundary=cfg.next_source_boundary,
        null_controls={
            "p0r02923_is_not_empirical_validation_evidence": 1.0,
            "p0r02924_is_not_empirical_validation_evidence": 1.0,
            "p0r02925_is_not_empirical_validation_evidence": 1.0,
            "p0r02926_is_not_empirical_validation_evidence": 1.0,
            "p0r02927_is_not_empirical_validation_evidence": 1.0,
            "p0r02928_is_not_empirical_validation_evidence": 1.0,
            "p0r02929_is_not_empirical_validation_evidence": 1.0,
            "p0r02930_is_not_empirical_validation_evidence": 1.0,
        },
        problem_metadata={
            "source_ledger_span": SOURCE_LEDGER_SPAN,
            "source_ledger_ids": tuple(f"P0R{number:05d}" for number in range(2923, 2931)),
            "claim_boundary": CLAIM_BOUNDARY,
            "protocol_state": "source_paper0_slice_p0r02923_only_no_experiment",
        },
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "HARDWARE_STATUS",
    "SOURCE_LEDGER_SPAN",
    "Paper0SliceP0r02923Config",
    "Paper0SliceP0r02923FixtureResult",
    "classify_paper0_slice_p0r02923_component",
    "paper0_slice_p0r02923_labels",
    "validate_paper0_slice_p0r02923_fixture",
]
