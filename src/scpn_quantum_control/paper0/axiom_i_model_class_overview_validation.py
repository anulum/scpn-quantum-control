# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 Axiom I model-class overview validation
"""Source-accounting checks for Paper 0 Axiom I model-class overview records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

CLAIM_BOUNDARY = "source-bounded Axiom I model-class overview; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"
SOURCE_LEDGER_SPAN = ("P0R00703", "P0R00716")


@dataclass(frozen=True, slots=True)
class AxiomIModelClassOverviewConfig:
    """Configuration for the Axiom I model-class overview fixture."""

    expected_selection_criterion_count: int = 3
    expected_blank_separator_count: int = 1
    next_source_boundary: str = "P0R00717"

    def __post_init__(self) -> None:
        if self.expected_selection_criterion_count != 3:
            raise ValueError("expected_selection_criterion_count must equal 3")
        if self.expected_blank_separator_count != 1:
            raise ValueError("expected_blank_separator_count must equal 1")
        if self.next_source_boundary != "P0R00717":
            raise ValueError("next_source_boundary must equal P0R00717")


@dataclass(frozen=True, slots=True)
class AxiomIModelClassOverviewFixtureResult:
    """Result for the Paper 0 Axiom I model-class overview fixture."""

    source_ledger_span: tuple[str, str]
    hardware_status: str
    claim_boundary: str
    selection_criteria: dict[str, str]
    model_class_choices: dict[str, str]
    labels: dict[str, str]
    selection_criterion_count: int
    model_class_choice_count: int
    blank_separator_count: int
    next_source_boundary: str
    null_controls: dict[str, float]
    problem_metadata: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-ready result dictionary."""
        return asdict(self)


def classify_selection_criterion(criterion: str) -> str:
    """Classify one source-defined model-class selection criterion."""
    mapping = {
        "spin0": "irreducible_spin0_degree_of_freedom",
        "phase": "intentional_phase_variable",
        "soliton": "stable_finite_energy_self_structure",
    }
    try:
        return mapping[criterion]
    except KeyError as exc:
        raise ValueError("unknown model-class criterion") from exc


def classify_model_class_choice(choice: str) -> str:
    """Classify one source-defined selected or rejected model-class role."""
    mapping = {
        "complex_scalar": "minimal_spin0_amplitude_phase_carrier",
        "local_u1": "local_gauge_phase_agency_via_infoton",
        "ssb_potential": "mexican_hat_stability_mechanism",
        "rejected_alternatives": "real_global_vector_spinor_alternatives_rejected",
    }
    try:
        return mapping[choice]
    except KeyError as exc:
        raise ValueError("unknown model-class choice") from exc


def axiom_i_model_class_overview_labels() -> dict[str, str]:
    """Return source-bounded labels for the Axiom I model-class overview slice."""
    return {
        "section": "Model-Class Justification: From Axiom to Lagrangian",
        "selected_family": "complex scalar field with local U(1) and SSB",
        "next_boundary": "Meta-Framework Integrations",
    }


def validate_axiom_i_model_class_overview_fixture(
    config: AxiomIModelClassOverviewConfig | None = None,
) -> AxiomIModelClassOverviewFixtureResult:
    """Validate source accounting for the Axiom I model-class overview slice."""
    cfg = config or AxiomIModelClassOverviewConfig()
    selection_criteria = ("spin0", "phase", "soliton")
    model_class_choices = ("complex_scalar", "local_u1", "ssb_potential", "rejected_alternatives")

    return AxiomIModelClassOverviewFixtureResult(
        source_ledger_span=SOURCE_LEDGER_SPAN,
        hardware_status=HARDWARE_STATUS,
        claim_boundary=CLAIM_BOUNDARY,
        selection_criteria={
            criterion: classify_selection_criterion(criterion) for criterion in selection_criteria
        },
        model_class_choices={
            choice: classify_model_class_choice(choice) for choice in model_class_choices
        },
        labels=axiom_i_model_class_overview_labels(),
        selection_criterion_count=cfg.expected_selection_criterion_count,
        model_class_choice_count=len(model_class_choices),
        blank_separator_count=cfg.expected_blank_separator_count,
        next_source_boundary=cfg.next_source_boundary,
        null_controls={
            "model_class_overview_is_not_empirical_validation": 1.0,
            "rejected_alternatives_require_downstream_falsification": 1.0,
            "pedagogical_metaphors_are_not_model_terms": 1.0,
        },
        problem_metadata={
            "source_ledger_span": SOURCE_LEDGER_SPAN,
            "source_ledger_ids": tuple(f"P0R{number:05d}" for number in range(703, 717)),
            "claim_boundary": CLAIM_BOUNDARY,
            "protocol_state": "source_model_class_overview_only_no_experiment",
        },
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "HARDWARE_STATUS",
    "SOURCE_LEDGER_SPAN",
    "AxiomIModelClassOverviewConfig",
    "AxiomIModelClassOverviewFixtureResult",
    "axiom_i_model_class_overview_labels",
    "classify_model_class_choice",
    "classify_selection_criterion",
    "validate_axiom_i_model_class_overview_fixture",
]
