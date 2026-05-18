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
    "source-bounded paper0 slice p0r04310 source-accounting bridge; not validation evidence"
)
HARDWARE_STATUS = "source_methodology_no_experiment"
SOURCE_LEDGER_SPAN = ("P0R04310", "P0R04321")


@dataclass(frozen=True, slots=True)
class Paper0SliceP0r04310Config:
    """Configuration for this Paper 0 source-accounting fixture."""

    expected_source_record_count: int = 12
    expected_component_count: int = 2
    next_source_boundary: str = "P0R04322"

    def __post_init__(self) -> None:
        if self.expected_source_record_count != 12:
            raise ValueError("expected_source_record_count must equal 12")
        if self.expected_component_count != 2:
            raise ValueError("expected_component_count must equal 2")
        if self.next_source_boundary != "P0R04322":
            raise ValueError("next_source_boundary must equal P0R04322")


@dataclass(frozen=True, slots=True)
class Paper0SliceP0r04310FixtureResult:
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


def classify_paper0_slice_p0r04310_component(component: str) -> str:
    """Classify source-defined components."""
    mapping = {
        "p0r04310": "p0r04310_source_boundary",
        "the_two_scalar_sector_and_the_pseudoscalar_coupling": "the_two_scalar_sector_and_the_pseudoscalar_coupling_source_boundary",
    }
    try:
        return mapping[component]
    except KeyError as exc:
        raise ValueError("unknown paper0_slice_p0r04310 component") from exc


def paper0_slice_p0r04310_labels() -> dict[str, str]:
    """Return source-bounded labels for this slice."""
    return {
        "section": "",
        "source_span": "P0R04310-P0R04321",
        "component_count": "2",
        "next_boundary": "P0R04322",
        "component_1": "P0R04310",
        "component_2": "The Two-Scalar Sector and the Pseudoscalar Coupling",
    }


def validate_paper0_slice_p0r04310_fixture(
    config: Paper0SliceP0r04310Config | None = None,
) -> Paper0SliceP0r04310FixtureResult:
    """Validate source accounting for this Paper 0 slice."""
    cfg = config or Paper0SliceP0r04310Config()
    components = ("p0r04310", "the_two_scalar_sector_and_the_pseudoscalar_coupling")
    return Paper0SliceP0r04310FixtureResult(
        source_ledger_span=SOURCE_LEDGER_SPAN,
        hardware_status=HARDWARE_STATUS,
        claim_boundary=CLAIM_BOUNDARY,
        components={
            component: classify_paper0_slice_p0r04310_component(component)
            for component in components
        },
        labels=paper0_slice_p0r04310_labels(),
        source_record_count=cfg.expected_source_record_count,
        component_count=cfg.expected_component_count,
        next_source_boundary=cfg.next_source_boundary,
        null_controls={
            "p0r04310_is_not_empirical_validation_evidence": 1.0,
            "the_two_scalar_sector_and_the_pseudoscalar_coupling_is_not_empirical_validation_evidence": 1.0,
        },
        problem_metadata={
            "source_ledger_span": SOURCE_LEDGER_SPAN,
            "source_ledger_ids": tuple(f"P0R{number:05d}" for number in range(4310, 4322)),
            "claim_boundary": CLAIM_BOUNDARY,
            "protocol_state": "source_paper0_slice_p0r04310_only_no_experiment",
        },
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "HARDWARE_STATUS",
    "SOURCE_LEDGER_SPAN",
    "Paper0SliceP0r04310Config",
    "Paper0SliceP0r04310FixtureResult",
    "classify_paper0_slice_p0r04310_component",
    "paper0_slice_p0r04310_labels",
    "validate_paper0_slice_p0r04310_fixture",
]
