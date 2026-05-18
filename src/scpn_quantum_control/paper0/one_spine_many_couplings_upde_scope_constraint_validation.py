# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 One Spine, Many Couplings - UPDE Scope Constraint validation
"""Source-accounting checks for Paper 0 One Spine, Many Couplings - UPDE Scope Constraint records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

CLAIM_BOUNDARY = "source-bounded one spine many couplings upde scope constraint source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"
SOURCE_LEDGER_SPAN = ("P0R02682", "P0R02745")


@dataclass(frozen=True, slots=True)
class OneSpineManyCouplingsUpdeScopeConstraintConfig:
    """Configuration for this Paper 0 source-accounting fixture."""

    expected_source_record_count: int = 64
    expected_component_count: int = 1
    next_source_boundary: str = "P0R02746"

    def __post_init__(self) -> None:
        if self.expected_source_record_count != 64:
            raise ValueError("expected_source_record_count must equal 64")
        if self.expected_component_count != 1:
            raise ValueError("expected_component_count must equal 1")
        if self.next_source_boundary != "P0R02746":
            raise ValueError("next_source_boundary must equal P0R02746")


@dataclass(frozen=True, slots=True)
class OneSpineManyCouplingsUpdeScopeConstraintFixtureResult:
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


def classify_one_spine_many_couplings_upde_scope_constraint_component(component: str) -> str:
    """Classify source-defined components."""
    mapping = {
        "one_spine_many_couplings_upde_scope_constraint": "one_spine_many_couplings_upde_scope_constraint_source_boundary"
    }
    try:
        return mapping[component]
    except KeyError as exc:
        raise ValueError(
            "unknown one_spine_many_couplings_upde_scope_constraint component"
        ) from exc


def one_spine_many_couplings_upde_scope_constraint_labels() -> dict[str, str]:
    """Return source-bounded labels for this slice."""
    return {
        "section": "One Spine, Many Couplings - UPDE Scope Constraint",
        "source_span": "P0R02682-P0R02745",
        "component_count": "1",
        "next_boundary": "P0R02746",
        "component_1": "One Spine, Many Couplings - UPDE Scope Constraint",
    }


def validate_one_spine_many_couplings_upde_scope_constraint_fixture(
    config: OneSpineManyCouplingsUpdeScopeConstraintConfig | None = None,
) -> OneSpineManyCouplingsUpdeScopeConstraintFixtureResult:
    """Validate source accounting for this Paper 0 slice."""
    cfg = config or OneSpineManyCouplingsUpdeScopeConstraintConfig()
    components = ("one_spine_many_couplings_upde_scope_constraint",)
    return OneSpineManyCouplingsUpdeScopeConstraintFixtureResult(
        source_ledger_span=SOURCE_LEDGER_SPAN,
        hardware_status=HARDWARE_STATUS,
        claim_boundary=CLAIM_BOUNDARY,
        components={
            component: classify_one_spine_many_couplings_upde_scope_constraint_component(component)
            for component in components
        },
        labels=one_spine_many_couplings_upde_scope_constraint_labels(),
        source_record_count=cfg.expected_source_record_count,
        component_count=cfg.expected_component_count,
        next_source_boundary=cfg.next_source_boundary,
        null_controls={
            "one_spine_many_couplings_upde_scope_constraint_is_not_empirical_validation_evidence": 1.0
        },
        problem_metadata={
            "source_ledger_span": SOURCE_LEDGER_SPAN,
            "source_ledger_ids": tuple(f"P0R{number:05d}" for number in range(2682, 2746)),
            "claim_boundary": CLAIM_BOUNDARY,
            "protocol_state": "source_one_spine_many_couplings_upde_scope_constraint_only_no_experiment",
        },
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "HARDWARE_STATUS",
    "SOURCE_LEDGER_SPAN",
    "OneSpineManyCouplingsUpdeScopeConstraintConfig",
    "OneSpineManyCouplingsUpdeScopeConstraintFixtureResult",
    "classify_one_spine_many_couplings_upde_scope_constraint_component",
    "one_spine_many_couplings_upde_scope_constraint_labels",
    "validate_one_spine_many_couplings_upde_scope_constraint_fixture",
]
