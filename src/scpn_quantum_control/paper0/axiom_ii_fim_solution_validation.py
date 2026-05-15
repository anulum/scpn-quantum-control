# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 Axiom II FIM solution validation
"""Source-accounting checks for Paper 0 Axiom II FIM-solution records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

CLAIM_BOUNDARY = "source-bounded Axiom II FIM-solution map; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"
SOURCE_LEDGER_SPAN = ("P0R00775", "P0R00781")


@dataclass(frozen=True, slots=True)
class AxiomIIFIMSolutionConfig:
    """Configuration for the Axiom II FIM-solution fixture."""

    expected_source_record_count: int = 7
    expected_physical_statement_count: int = 2
    next_source_boundary: str = "P0R00782"

    def __post_init__(self) -> None:
        if self.expected_source_record_count != 7:
            raise ValueError("expected_source_record_count must equal 7")
        if self.expected_physical_statement_count != 2:
            raise ValueError("expected_physical_statement_count must equal 2")
        if self.next_source_boundary != "P0R00782":
            raise ValueError("next_source_boundary must equal P0R00782")


@dataclass(frozen=True, slots=True)
class AxiomIIFIMSolutionFixtureResult:
    """Result for the Paper 0 Axiom II FIM-solution fixture."""

    source_ledger_span: tuple[str, str]
    hardware_status: str
    claim_boundary: str
    components: dict[str, str]
    labels: dict[str, str]
    source_record_count: int
    metric_definition_count: int
    physical_statement_count: int
    synthesis_statement_count: int
    next_source_boundary: str
    null_controls: dict[str, float]
    problem_metadata: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-ready result dictionary."""
        return asdict(self)


def classify_fim_solution_component(component: str) -> str:
    """Classify source-defined Axiom II FIM-solution components."""
    mapping = {
        "metric_definition": "fim_statistical_manifold_metric",
        "informational_interaction": "infoton_propagates_through_information_geometry",
        "complexity_coupling": "coupling_strength_tracks_informational_complexity",
        "fep_hpc_upde_synthesis": "shared_information_geometry_language",
    }
    try:
        return mapping[component]
    except KeyError as exc:
        raise ValueError("unknown FIM-solution component") from exc


def axiom_ii_fim_solution_labels() -> dict[str, str]:
    """Return source-bounded labels for the Axiom II FIM-solution slice."""
    return {
        "section": "The Fisher Information Metric (FIM) as the Solution",
        "metric": "natural unique Riemannian metric on a statistical manifold",
        "next_boundary": "Formal Consequence: The Informational Lagrangian",
    }


def validate_axiom_ii_fim_solution_fixture(
    config: AxiomIIFIMSolutionConfig | None = None,
) -> AxiomIIFIMSolutionFixtureResult:
    """Validate source accounting for the Axiom II FIM-solution slice."""
    cfg = config or AxiomIIFIMSolutionConfig()
    components = (
        "metric_definition",
        "informational_interaction",
        "complexity_coupling",
        "fep_hpc_upde_synthesis",
    )

    return AxiomIIFIMSolutionFixtureResult(
        source_ledger_span=SOURCE_LEDGER_SPAN,
        hardware_status=HARDWARE_STATUS,
        claim_boundary=CLAIM_BOUNDARY,
        components={
            component: classify_fim_solution_component(component) for component in components
        },
        labels=axiom_ii_fim_solution_labels(),
        source_record_count=cfg.expected_source_record_count,
        metric_definition_count=1,
        physical_statement_count=cfg.expected_physical_statement_count,
        synthesis_statement_count=2,
        next_source_boundary=cfg.next_source_boundary,
        null_controls={
            "fim_natural_metric_statement_is_source_claim_not_proof": 1.0,
            "complexity_coupling_requires_downstream_operational_metric": 1.0,
            "fep_hpc_upde_synthesis_is_not_empirical_validation": 1.0,
        },
        problem_metadata={
            "source_ledger_span": SOURCE_LEDGER_SPAN,
            "source_ledger_ids": tuple(f"P0R{number:05d}" for number in range(775, 782)),
            "claim_boundary": CLAIM_BOUNDARY,
            "protocol_state": "source_axiom_ii_fim_solution_only_no_experiment",
        },
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "HARDWARE_STATUS",
    "SOURCE_LEDGER_SPAN",
    "AxiomIIFIMSolutionConfig",
    "AxiomIIFIMSolutionFixtureResult",
    "axiom_ii_fim_solution_labels",
    "classify_fim_solution_component",
    "validate_axiom_ii_fim_solution_fixture",
]
