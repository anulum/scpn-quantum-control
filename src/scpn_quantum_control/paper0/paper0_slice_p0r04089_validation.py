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
    "source-bounded paper0 slice p0r04089 source-accounting bridge; not validation evidence"
)
HARDWARE_STATUS = "source_methodology_no_experiment"
SOURCE_LEDGER_SPAN = ("P0R04089", "P0R04097")


@dataclass(frozen=True, slots=True)
class Paper0SliceP0r04089Config:
    """Configuration for this Paper 0 source-accounting fixture."""

    expected_source_record_count: int = 9
    expected_component_count: int = 8
    next_source_boundary: str = "P0R04098"

    def __post_init__(self) -> None:
        if self.expected_source_record_count != 9:
            raise ValueError("expected_source_record_count must equal 9")
        if self.expected_component_count != 8:
            raise ValueError("expected_component_count must equal 8")
        if self.next_source_boundary != "P0R04098":
            raise ValueError("next_source_boundary must equal P0R04098")


@dataclass(frozen=True, slots=True)
class Paper0SliceP0r04089FixtureResult:
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


def classify_paper0_slice_p0r04089_component(component: str) -> str:
    """Classify source-defined components."""
    mapping = {
        "p0r04089": "p0r04089_source_boundary",
        "p0r04090": "p0r04090_source_boundary",
        "p0r04091": "p0r04091_source_boundary",
        "p0r04092": "p0r04092_source_boundary",
        "p0r04093": "p0r04093_source_boundary",
        "p0r04094": "p0r04094_source_boundary",
        "p0r04095": "p0r04095_source_boundary",
        "p0r04096": "p0r04096_source_boundary",
    }
    try:
        return mapping[component]
    except KeyError as exc:
        raise ValueError("unknown paper0_slice_p0r04089 component") from exc


def paper0_slice_p0r04089_labels() -> dict[str, str]:
    """Return source-bounded labels for this slice."""
    return {
        "section": "",
        "source_span": "P0R04089-P0R04097",
        "component_count": "8",
        "next_boundary": "P0R04098",
        "component_1": "P0R04089",
        "component_2": "P0R04090",
        "component_3": "P0R04091",
        "component_4": "P0R04092",
        "component_5": "P0R04093",
        "component_6": "P0R04094",
        "component_7": "P0R04095",
        "component_8": "P0R04096",
    }


def validate_paper0_slice_p0r04089_fixture(
    config: Paper0SliceP0r04089Config | None = None,
) -> Paper0SliceP0r04089FixtureResult:
    """Validate source accounting for this Paper 0 slice."""
    cfg = config or Paper0SliceP0r04089Config()
    components = (
        "p0r04089",
        "p0r04090",
        "p0r04091",
        "p0r04092",
        "p0r04093",
        "p0r04094",
        "p0r04095",
        "p0r04096",
    )
    return Paper0SliceP0r04089FixtureResult(
        source_ledger_span=SOURCE_LEDGER_SPAN,
        hardware_status=HARDWARE_STATUS,
        claim_boundary=CLAIM_BOUNDARY,
        components={
            component: classify_paper0_slice_p0r04089_component(component)
            for component in components
        },
        labels=paper0_slice_p0r04089_labels(),
        source_record_count=cfg.expected_source_record_count,
        component_count=cfg.expected_component_count,
        next_source_boundary=cfg.next_source_boundary,
        null_controls={
            "p0r04089_is_not_empirical_validation_evidence": 1.0,
            "p0r04090_is_not_empirical_validation_evidence": 1.0,
            "p0r04091_is_not_empirical_validation_evidence": 1.0,
            "p0r04092_is_not_empirical_validation_evidence": 1.0,
            "p0r04093_is_not_empirical_validation_evidence": 1.0,
            "p0r04094_is_not_empirical_validation_evidence": 1.0,
            "p0r04095_is_not_empirical_validation_evidence": 1.0,
            "p0r04096_is_not_empirical_validation_evidence": 1.0,
        },
        problem_metadata={
            "source_ledger_span": SOURCE_LEDGER_SPAN,
            "source_ledger_ids": tuple(f"P0R{number:05d}" for number in range(4089, 4098)),
            "claim_boundary": CLAIM_BOUNDARY,
            "protocol_state": "source_paper0_slice_p0r04089_only_no_experiment",
        },
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "HARDWARE_STATUS",
    "SOURCE_LEDGER_SPAN",
    "Paper0SliceP0r04089Config",
    "Paper0SliceP0r04089FixtureResult",
    "classify_paper0_slice_p0r04089_component",
    "paper0_slice_p0r04089_labels",
    "validate_paper0_slice_p0r04089_fixture",
]
