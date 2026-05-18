# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0  Thermodynamics & Entropy validation
"""Source-accounting checks for Paper 0  Thermodynamics & Entropy records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

CLAIM_BOUNDARY = (
    "source-bounded thermodynamics entropy source-accounting bridge; not validation evidence"
)
HARDWARE_STATUS = "source_methodology_no_experiment"
SOURCE_LEDGER_SPAN = ("P0R05721", "P0R05729")


@dataclass(frozen=True, slots=True)
class ThermodynamicsEntropyConfig:
    """Configuration for this Paper 0 source-accounting fixture."""

    expected_source_record_count: int = 9
    expected_component_count: int = 2
    next_source_boundary: str = "P0R05730"

    def __post_init__(self) -> None:
        if self.expected_source_record_count != 9:
            raise ValueError("expected_source_record_count must equal 9")
        if self.expected_component_count != 2:
            raise ValueError("expected_component_count must equal 2")
        if self.next_source_boundary != "P0R05730":
            raise ValueError("next_source_boundary must equal P0R05730")


@dataclass(frozen=True, slots=True)
class ThermodynamicsEntropyFixtureResult:
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


def classify_thermodynamics_entropy_component(component: str) -> str:
    """Classify source-defined components."""
    mapping = {
        "thermodynamics_entropy": "thermodynamics_entropy_source_boundary",
        "gauge_field_theory_foundations": "gauge_field_theory_foundations_source_boundary",
    }
    try:
        return mapping[component]
    except KeyError as exc:
        raise ValueError("unknown thermodynamics_entropy component") from exc


def thermodynamics_entropy_labels() -> dict[str, str]:
    """Return source-bounded labels for this slice."""
    return {
        "section": " Thermodynamics & Entropy",
        "source_span": "P0R05721-P0R05729",
        "component_count": "2",
        "next_boundary": "P0R05730",
        "component_1": "Thermodynamics & Entropy",
        "component_2": "Gauge & Field Theory Foundations",
    }


def validate_thermodynamics_entropy_fixture(
    config: ThermodynamicsEntropyConfig | None = None,
) -> ThermodynamicsEntropyFixtureResult:
    """Validate source accounting for this Paper 0 slice."""
    cfg = config or ThermodynamicsEntropyConfig()
    components = ("thermodynamics_entropy", "gauge_field_theory_foundations")
    return ThermodynamicsEntropyFixtureResult(
        source_ledger_span=SOURCE_LEDGER_SPAN,
        hardware_status=HARDWARE_STATUS,
        claim_boundary=CLAIM_BOUNDARY,
        components={
            component: classify_thermodynamics_entropy_component(component)
            for component in components
        },
        labels=thermodynamics_entropy_labels(),
        source_record_count=cfg.expected_source_record_count,
        component_count=cfg.expected_component_count,
        next_source_boundary=cfg.next_source_boundary,
        null_controls={
            "thermodynamics_entropy_is_not_empirical_validation_evidence": 1.0,
            "gauge_field_theory_foundations_is_not_empirical_validation_evidence": 1.0,
        },
        problem_metadata={
            "source_ledger_span": SOURCE_LEDGER_SPAN,
            "source_ledger_ids": tuple(f"P0R{number:05d}" for number in range(5721, 5730)),
            "claim_boundary": CLAIM_BOUNDARY,
            "protocol_state": "source_thermodynamics_entropy_only_no_experiment",
        },
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "HARDWARE_STATUS",
    "SOURCE_LEDGER_SPAN",
    "ThermodynamicsEntropyConfig",
    "ThermodynamicsEntropyFixtureResult",
    "classify_thermodynamics_entropy_component",
    "thermodynamics_entropy_labels",
    "validate_thermodynamics_entropy_fixture",
]
