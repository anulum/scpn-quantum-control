# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 Quasicriticality and Self-Organisation validation
"""Source-accounting checks for Paper 0 Quasicriticality and Self-Organisation records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

CLAIM_BOUNDARY = "source-bounded quasicriticality and self organisation source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"
SOURCE_LEDGER_SPAN = ("P0R02831", "P0R02838")


@dataclass(frozen=True, slots=True)
class QuasicriticalityAndSelfOrganisationConfig:
    """Configuration for this Paper 0 source-accounting fixture."""

    expected_source_record_count: int = 8
    expected_component_count: int = 2
    next_source_boundary: str = "P0R02839"

    def __post_init__(self) -> None:
        if self.expected_source_record_count != 8:
            raise ValueError("expected_source_record_count must equal 8")
        if self.expected_component_count != 2:
            raise ValueError("expected_component_count must equal 2")
        if self.next_source_boundary != "P0R02839":
            raise ValueError("next_source_boundary must equal P0R02839")


@dataclass(frozen=True, slots=True)
class QuasicriticalityAndSelfOrganisationFixtureResult:
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


def classify_quasicriticality_and_self_organisation_component(component: str) -> str:
    """Classify source-defined components."""
    mapping = {
        "quasicriticality_and_self_organisation": "quasicriticality_and_self_organisation_source_boundary",
        "meta_framework_integrations": "meta_framework_integrations_source_boundary",
    }
    try:
        return mapping[component]
    except KeyError as exc:
        raise ValueError("unknown quasicriticality_and_self_organisation component") from exc


def quasicriticality_and_self_organisation_labels() -> dict[str, str]:
    """Return source-bounded labels for this slice."""
    return {
        "section": "Quasicriticality and Self-Organisation",
        "source_span": "P0R02831-P0R02838",
        "component_count": "2",
        "next_boundary": "P0R02839",
        "component_1": "Quasicriticality and Self-Organisation",
        "component_2": "Meta-Framework Integrations",
    }


def validate_quasicriticality_and_self_organisation_fixture(
    config: QuasicriticalityAndSelfOrganisationConfig | None = None,
) -> QuasicriticalityAndSelfOrganisationFixtureResult:
    """Validate source accounting for this Paper 0 slice."""
    cfg = config or QuasicriticalityAndSelfOrganisationConfig()
    components = ("quasicriticality_and_self_organisation", "meta_framework_integrations")
    return QuasicriticalityAndSelfOrganisationFixtureResult(
        source_ledger_span=SOURCE_LEDGER_SPAN,
        hardware_status=HARDWARE_STATUS,
        claim_boundary=CLAIM_BOUNDARY,
        components={
            component: classify_quasicriticality_and_self_organisation_component(component)
            for component in components
        },
        labels=quasicriticality_and_self_organisation_labels(),
        source_record_count=cfg.expected_source_record_count,
        component_count=cfg.expected_component_count,
        next_source_boundary=cfg.next_source_boundary,
        null_controls={
            "quasicriticality_and_self_organisation_is_not_empirical_validation_evidence": 1.0,
            "meta_framework_integrations_is_not_empirical_validation_evidence": 1.0,
        },
        problem_metadata={
            "source_ledger_span": SOURCE_LEDGER_SPAN,
            "source_ledger_ids": tuple(f"P0R{number:05d}" for number in range(2831, 2839)),
            "claim_boundary": CLAIM_BOUNDARY,
            "protocol_state": "source_quasicriticality_and_self_organisation_only_no_experiment",
        },
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "HARDWARE_STATUS",
    "SOURCE_LEDGER_SPAN",
    "QuasicriticalityAndSelfOrganisationConfig",
    "QuasicriticalityAndSelfOrganisationFixtureResult",
    "classify_quasicriticality_and_self_organisation_component",
    "quasicriticality_and_self_organisation_labels",
    "validate_quasicriticality_and_self_organisation_fixture",
]
