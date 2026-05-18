# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 L15 Reformulation: The SEC Objective Functional (Decision-Theoretic Form) (revision 11.00) validation
"""Source-accounting checks for Paper 0 L15 Reformulation: The SEC Objective Functional (Decision-Theoretic Form) (revision 11.00) records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

CLAIM_BOUNDARY = "source-bounded l15 reformulation the sec objective functional decision theoretic form r source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"
SOURCE_LEDGER_SPAN = ("P0R03981", "P0R04000")


@dataclass(frozen=True, slots=True)
class L15ReformulationTheSecObjectiveFunctionalDecisionTheoreticFormRConfig:
    """Configuration for this Paper 0 source-accounting fixture."""

    expected_source_record_count: int = 20
    expected_component_count: int = 2
    next_source_boundary: str = "P0R04001"

    def __post_init__(self) -> None:
        if self.expected_source_record_count != 20:
            raise ValueError("expected_source_record_count must equal 20")
        if self.expected_component_count != 2:
            raise ValueError("expected_component_count must equal 2")
        if self.next_source_boundary != "P0R04001":
            raise ValueError("next_source_boundary must equal P0R04001")


@dataclass(frozen=True, slots=True)
class L15ReformulationTheSecObjectiveFunctionalDecisionTheoreticFormRFixtureResult:
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


def classify_l15_reformulation_the_sec_objective_functional_decision_theoretic_form_r_component(
    component: str,
) -> str:
    """Classify source-defined components."""
    mapping = {
        "l15_reformulation_the_sec_objective_functional_decision_theoretic_form_r": "l15_reformulation_the_sec_objective_functional_decision_theoretic_form_r_source_boundary",
        "definition_canonical_form": "definition_canonical_form_source_boundary",
    }
    try:
        return mapping[component]
    except KeyError as exc:
        raise ValueError(
            "unknown l15_reformulation_the_sec_objective_functional_decision_theoretic_form_r component"
        ) from exc


def l15_reformulation_the_sec_objective_functional_decision_theoretic_form_r_labels() -> dict[
    str, str
]:
    """Return source-bounded labels for this slice."""
    return {
        "section": "L15 Reformulation: The SEC Objective Functional (Decision-Theoretic Form) (revision 11.00)",
        "source_span": "P0R03981-P0R04000",
        "component_count": "2",
        "next_boundary": "P0R04001",
        "component_1": "L15 Reformulation: The SEC Objective Functional (Decision-Theoretic Form) (revision 11.00)",
        "component_2": "Definition (canonical form).",
    }


def validate_l15_reformulation_the_sec_objective_functional_decision_theoretic_form_r_fixture(
    config: L15ReformulationTheSecObjectiveFunctionalDecisionTheoreticFormRConfig | None = None,
) -> L15ReformulationTheSecObjectiveFunctionalDecisionTheoreticFormRFixtureResult:
    """Validate source accounting for this Paper 0 slice."""
    cfg = config or L15ReformulationTheSecObjectiveFunctionalDecisionTheoreticFormRConfig()
    components = (
        "l15_reformulation_the_sec_objective_functional_decision_theoretic_form_r",
        "definition_canonical_form",
    )
    return L15ReformulationTheSecObjectiveFunctionalDecisionTheoreticFormRFixtureResult(
        source_ledger_span=SOURCE_LEDGER_SPAN,
        hardware_status=HARDWARE_STATUS,
        claim_boundary=CLAIM_BOUNDARY,
        components={
            component: classify_l15_reformulation_the_sec_objective_functional_decision_theoretic_form_r_component(
                component
            )
            for component in components
        },
        labels=l15_reformulation_the_sec_objective_functional_decision_theoretic_form_r_labels(),
        source_record_count=cfg.expected_source_record_count,
        component_count=cfg.expected_component_count,
        next_source_boundary=cfg.next_source_boundary,
        null_controls={
            "l15_reformulation_the_sec_objective_functional_decision_theoretic_form_r_is_not_empirical_validation_evidence": 1.0,
            "definition_canonical_form_is_not_empirical_validation_evidence": 1.0,
        },
        problem_metadata={
            "source_ledger_span": SOURCE_LEDGER_SPAN,
            "source_ledger_ids": tuple(f"P0R{number:05d}" for number in range(3981, 4001)),
            "claim_boundary": CLAIM_BOUNDARY,
            "protocol_state": "source_l15_reformulation_the_sec_objective_functional_decision_theoretic_form_r_only_no_experiment",
        },
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "HARDWARE_STATUS",
    "SOURCE_LEDGER_SPAN",
    "L15ReformulationTheSecObjectiveFunctionalDecisionTheoreticFormRConfig",
    "L15ReformulationTheSecObjectiveFunctionalDecisionTheoreticFormRFixtureResult",
    "classify_l15_reformulation_the_sec_objective_functional_decision_theoretic_form_r_component",
    "l15_reformulation_the_sec_objective_functional_decision_theoretic_form_r_labels",
    "validate_l15_reformulation_the_sec_objective_functional_decision_theoretic_form_r_fixture",
]
