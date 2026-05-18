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
    "source-bounded paper0 slice p0r04338 source-accounting bridge; not validation evidence"
)
HARDWARE_STATUS = "source_methodology_no_experiment"
SOURCE_LEDGER_SPAN = ("P0R04338", "P0R04347")


@dataclass(frozen=True, slots=True)
class Paper0SliceP0r04338Config:
    """Configuration for this Paper 0 source-accounting fixture."""

    expected_source_record_count: int = 10
    expected_component_count: int = 2
    next_source_boundary: str = "P0R04348"

    def __post_init__(self) -> None:
        if self.expected_source_record_count != 10:
            raise ValueError("expected_source_record_count must equal 10")
        if self.expected_component_count != 2:
            raise ValueError("expected_component_count must equal 2")
        if self.next_source_boundary != "P0R04348":
            raise ValueError("next_source_boundary must equal P0R04348")


@dataclass(frozen=True, slots=True)
class Paper0SliceP0r04338FixtureResult:
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


def classify_paper0_slice_p0r04338_component(component: str) -> str:
    """Classify source-defined components."""
    mapping = {
        "p0r04338": "p0r04338_source_boundary",
        "mechanism_the_primakoff_effect_in_the_brain": "mechanism_the_primakoff_effect_in_the_brain_source_boundary",
    }
    try:
        return mapping[component]
    except KeyError as exc:
        raise ValueError("unknown paper0_slice_p0r04338 component") from exc


def paper0_slice_p0r04338_labels() -> dict[str, str]:
    """Return source-bounded labels for this slice."""
    return {
        "section": "",
        "source_span": "P0R04338-P0R04347",
        "component_count": "2",
        "next_boundary": "P0R04348",
        "component_1": "P0R04338",
        "component_2": "Mechanism: The Primakoff Effect in the Brain",
    }


def validate_paper0_slice_p0r04338_fixture(
    config: Paper0SliceP0r04338Config | None = None,
) -> Paper0SliceP0r04338FixtureResult:
    """Validate source accounting for this Paper 0 slice."""
    cfg = config or Paper0SliceP0r04338Config()
    components = ("p0r04338", "mechanism_the_primakoff_effect_in_the_brain")
    return Paper0SliceP0r04338FixtureResult(
        source_ledger_span=SOURCE_LEDGER_SPAN,
        hardware_status=HARDWARE_STATUS,
        claim_boundary=CLAIM_BOUNDARY,
        components={
            component: classify_paper0_slice_p0r04338_component(component)
            for component in components
        },
        labels=paper0_slice_p0r04338_labels(),
        source_record_count=cfg.expected_source_record_count,
        component_count=cfg.expected_component_count,
        next_source_boundary=cfg.next_source_boundary,
        null_controls={
            "p0r04338_is_not_empirical_validation_evidence": 1.0,
            "mechanism_the_primakoff_effect_in_the_brain_is_not_empirical_validation_evidence": 1.0,
        },
        problem_metadata={
            "source_ledger_span": SOURCE_LEDGER_SPAN,
            "source_ledger_ids": tuple(f"P0R{number:05d}" for number in range(4338, 4348)),
            "claim_boundary": CLAIM_BOUNDARY,
            "protocol_state": "source_paper0_slice_p0r04338_only_no_experiment",
        },
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "HARDWARE_STATUS",
    "SOURCE_LEDGER_SPAN",
    "Paper0SliceP0r04338Config",
    "Paper0SliceP0r04338FixtureResult",
    "classify_paper0_slice_p0r04338_component",
    "paper0_slice_p0r04338_labels",
    "validate_paper0_slice_p0r04338_fixture",
]
