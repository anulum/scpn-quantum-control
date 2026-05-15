# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 Axiom II infoton geometry validation
"""Source-accounting checks for Paper 0 Axiom II infoton-geometry records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

CLAIM_BOUNDARY = "source-bounded Axiom II infoton-geometry map; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"
SOURCE_LEDGER_SPAN = ("P0R00770", "P0R00774")


@dataclass(frozen=True, slots=True)
class AxiomIIInfotonGeometryConfig:
    """Configuration for the Axiom II infoton-geometry fixture."""

    expected_source_record_count: int = 5
    expected_gauge_necessity_count: int = 1
    next_source_boundary: str = "P0R00775"

    def __post_init__(self) -> None:
        if self.expected_source_record_count != 5:
            raise ValueError("expected_source_record_count must equal 5")
        if self.expected_gauge_necessity_count != 1:
            raise ValueError("expected_gauge_necessity_count must equal 1")
        if self.next_source_boundary != "P0R00775":
            raise ValueError("next_source_boundary must equal P0R00775")


@dataclass(frozen=True, slots=True)
class AxiomIIInfotonGeometryFixtureResult:
    """Result for the Paper 0 Axiom II infoton-geometry fixture."""

    source_ledger_span: tuple[str, str]
    hardware_status: str
    claim_boundary: str
    components: dict[str, str]
    labels: dict[str, str]
    source_record_count: int
    gauge_necessity_count: int
    baseline_lagrangian_count: int
    fim_claim_count: int
    next_source_boundary: str
    null_controls: dict[str, float]
    problem_metadata: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-ready result dictionary."""
        return asdict(self)


def classify_infoton_geometry_component(component: str) -> str:
    """Classify source-defined Axiom II infoton-geometry components."""
    mapping = {
        "problem_heading": "infoton_geometry_problem",
        "gauge_necessity": "u1_local_complex_field_requires_spin1_infoton",
        "spacetime_baseline": "standard_em_spacetime_metric_kinetic_term",
        "fim_dynamics": "infoton_dynamics_governed_by_fim",
    }
    try:
        return mapping[component]
    except KeyError as exc:
        raise ValueError("unknown infoton-geometry component") from exc


def axiom_ii_infoton_geometry_labels() -> dict[str, str]:
    """Return source-bounded labels for the Axiom II infoton-geometry slice."""
    return {
        "section": 'The Central Problem: The Geometry of the "Infoton"',
        "baseline_lagrangian": "L_EM = -1/4 F_mu_nu F^mu_nu",
        "next_boundary": "The Fisher Information Metric (FIM) as the Solution",
    }


def validate_axiom_ii_infoton_geometry_fixture(
    config: AxiomIIInfotonGeometryConfig | None = None,
) -> AxiomIIInfotonGeometryFixtureResult:
    """Validate source accounting for the Axiom II infoton-geometry slice."""
    cfg = config or AxiomIIInfotonGeometryConfig()
    components = (
        "problem_heading",
        "gauge_necessity",
        "spacetime_baseline",
        "fim_dynamics",
    )

    return AxiomIIInfotonGeometryFixtureResult(
        source_ledger_span=SOURCE_LEDGER_SPAN,
        hardware_status=HARDWARE_STATUS,
        claim_boundary=CLAIM_BOUNDARY,
        components={
            component: classify_infoton_geometry_component(component) for component in components
        },
        labels=axiom_ii_infoton_geometry_labels(),
        source_record_count=cfg.expected_source_record_count,
        gauge_necessity_count=cfg.expected_gauge_necessity_count,
        baseline_lagrangian_count=1,
        fim_claim_count=1,
        next_source_boundary=cfg.next_source_boundary,
        null_controls={
            "gauge_necessity_is_source_derivation_pointer_not_rederived_here": 1.0,
            "standard_em_lagrangian_is_baseline_not_scpn_result": 1.0,
            "fim_dynamics_claim_requires_downstream_validation": 1.0,
        },
        problem_metadata={
            "source_ledger_span": SOURCE_LEDGER_SPAN,
            "source_ledger_ids": tuple(f"P0R{number:05d}" for number in range(770, 775)),
            "claim_boundary": CLAIM_BOUNDARY,
            "protocol_state": "source_axiom_ii_infoton_geometry_only_no_experiment",
        },
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "HARDWARE_STATUS",
    "SOURCE_LEDGER_SPAN",
    "AxiomIIInfotonGeometryConfig",
    "AxiomIIInfotonGeometryFixtureResult",
    "axiom_ii_infoton_geometry_labels",
    "classify_infoton_geometry_component",
    "validate_axiom_ii_infoton_geometry_fixture",
]
