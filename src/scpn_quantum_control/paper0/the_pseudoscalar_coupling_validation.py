# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 The Pseudoscalar Coupling validation
"""Source-accounting checks for Paper 0 The Pseudoscalar Coupling records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

CLAIM_BOUNDARY = (
    "source-bounded the pseudoscalar coupling source-accounting bridge; not validation evidence"
)
HARDWARE_STATUS = "source_methodology_no_experiment"
SOURCE_LEDGER_SPAN = ("P0R04291", "P0R04309")


@dataclass(frozen=True, slots=True)
class ThePseudoscalarCouplingConfig:
    """Configuration for this Paper 0 source-accounting fixture."""

    expected_source_record_count: int = 19
    expected_component_count: int = 1
    next_source_boundary: str = "P0R04310"

    def __post_init__(self) -> None:
        if self.expected_source_record_count != 19:
            raise ValueError("expected_source_record_count must equal 19")
        if self.expected_component_count != 1:
            raise ValueError("expected_component_count must equal 1")
        if self.next_source_boundary != "P0R04310":
            raise ValueError("next_source_boundary must equal P0R04310")


@dataclass(frozen=True, slots=True)
class ThePseudoscalarCouplingFixtureResult:
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


def classify_the_pseudoscalar_coupling_component(component: str) -> str:
    """Classify source-defined components."""
    mapping = {"the_pseudoscalar_coupling": "the_pseudoscalar_coupling_source_boundary"}
    try:
        return mapping[component]
    except KeyError as exc:
        raise ValueError("unknown the_pseudoscalar_coupling component") from exc


def the_pseudoscalar_coupling_labels() -> dict[str, str]:
    """Return source-bounded labels for this slice."""
    return {
        "section": "The Pseudoscalar Coupling",
        "source_span": "P0R04291-P0R04309",
        "component_count": "1",
        "next_boundary": "P0R04310",
        "component_1": "The Pseudoscalar Coupling",
    }


def validate_the_pseudoscalar_coupling_fixture(
    config: ThePseudoscalarCouplingConfig | None = None,
) -> ThePseudoscalarCouplingFixtureResult:
    """Validate source accounting for this Paper 0 slice."""
    cfg = config or ThePseudoscalarCouplingConfig()
    components = ("the_pseudoscalar_coupling",)
    return ThePseudoscalarCouplingFixtureResult(
        source_ledger_span=SOURCE_LEDGER_SPAN,
        hardware_status=HARDWARE_STATUS,
        claim_boundary=CLAIM_BOUNDARY,
        components={
            component: classify_the_pseudoscalar_coupling_component(component)
            for component in components
        },
        labels=the_pseudoscalar_coupling_labels(),
        source_record_count=cfg.expected_source_record_count,
        component_count=cfg.expected_component_count,
        next_source_boundary=cfg.next_source_boundary,
        null_controls={"the_pseudoscalar_coupling_is_not_empirical_validation_evidence": 1.0},
        problem_metadata={
            "source_ledger_span": SOURCE_LEDGER_SPAN,
            "source_ledger_ids": tuple(f"P0R{number:05d}" for number in range(4291, 4310)),
            "claim_boundary": CLAIM_BOUNDARY,
            "protocol_state": "source_the_pseudoscalar_coupling_only_no_experiment",
        },
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "HARDWARE_STATUS",
    "SOURCE_LEDGER_SPAN",
    "ThePseudoscalarCouplingConfig",
    "ThePseudoscalarCouplingFixtureResult",
    "classify_the_pseudoscalar_coupling_component",
    "the_pseudoscalar_coupling_labels",
    "validate_the_pseudoscalar_coupling_fixture",
]
