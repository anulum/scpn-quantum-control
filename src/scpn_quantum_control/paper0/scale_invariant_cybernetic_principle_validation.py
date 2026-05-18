# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 Scale-Invariant Cybernetic Principle validation
"""Source-accounting checks for Paper 0 Scale-Invariant Cybernetic Principle records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

CLAIM_BOUNDARY = "source-bounded scale invariant cybernetic principle source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"
SOURCE_LEDGER_SPAN = ("P0R05493", "P0R05507")


@dataclass(frozen=True, slots=True)
class ScaleInvariantCyberneticPrincipleConfig:
    """Configuration for this Paper 0 source-accounting fixture."""

    expected_source_record_count: int = 15
    expected_component_count: int = 2
    next_source_boundary: str = "P0R05508"

    def __post_init__(self) -> None:
        if self.expected_source_record_count != 15:
            raise ValueError("expected_source_record_count must equal 15")
        if self.expected_component_count != 2:
            raise ValueError("expected_component_count must equal 2")
        if self.next_source_boundary != "P0R05508":
            raise ValueError("next_source_boundary must equal P0R05508")


@dataclass(frozen=True, slots=True)
class ScaleInvariantCyberneticPrincipleFixtureResult:
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


def classify_scale_invariant_cybernetic_principle_component(component: str) -> str:
    """Classify source-defined components."""
    mapping = {
        "scale_invariant_cybernetic_principle": "scale_invariant_cybernetic_principle_source_boundary",
        "citations": "citations_source_boundary",
    }
    try:
        return mapping[component]
    except KeyError as exc:
        raise ValueError("unknown scale_invariant_cybernetic_principle component") from exc


def scale_invariant_cybernetic_principle_labels() -> dict[str, str]:
    """Return source-bounded labels for this slice."""
    return {
        "section": "Scale-Invariant Cybernetic Principle",
        "source_span": "P0R05493-P0R05507",
        "component_count": "2",
        "next_boundary": "P0R05508",
        "component_1": "Scale-Invariant Cybernetic Principle",
        "component_2": "Citations:",
    }


def validate_scale_invariant_cybernetic_principle_fixture(
    config: ScaleInvariantCyberneticPrincipleConfig | None = None,
) -> ScaleInvariantCyberneticPrincipleFixtureResult:
    """Validate source accounting for this Paper 0 slice."""
    cfg = config or ScaleInvariantCyberneticPrincipleConfig()
    components = ("scale_invariant_cybernetic_principle", "citations")
    return ScaleInvariantCyberneticPrincipleFixtureResult(
        source_ledger_span=SOURCE_LEDGER_SPAN,
        hardware_status=HARDWARE_STATUS,
        claim_boundary=CLAIM_BOUNDARY,
        components={
            component: classify_scale_invariant_cybernetic_principle_component(component)
            for component in components
        },
        labels=scale_invariant_cybernetic_principle_labels(),
        source_record_count=cfg.expected_source_record_count,
        component_count=cfg.expected_component_count,
        next_source_boundary=cfg.next_source_boundary,
        null_controls={
            "scale_invariant_cybernetic_principle_is_not_empirical_validation_evidence": 1.0,
            "citations_is_not_empirical_validation_evidence": 1.0,
        },
        problem_metadata={
            "source_ledger_span": SOURCE_LEDGER_SPAN,
            "source_ledger_ids": tuple(f"P0R{number:05d}" for number in range(5493, 5508)),
            "claim_boundary": CLAIM_BOUNDARY,
            "protocol_state": "source_scale_invariant_cybernetic_principle_only_no_experiment",
        },
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "HARDWARE_STATUS",
    "SOURCE_LEDGER_SPAN",
    "ScaleInvariantCyberneticPrincipleConfig",
    "ScaleInvariantCyberneticPrincipleFixtureResult",
    "classify_scale_invariant_cybernetic_principle_component",
    "scale_invariant_cybernetic_principle_labels",
    "validate_scale_invariant_cybernetic_principle_fixture",
]
