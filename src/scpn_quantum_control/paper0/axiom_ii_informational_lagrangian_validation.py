# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 Axiom II informational Lagrangian validation
"""Source-accounting checks for Paper 0 Axiom II informational-Lagrangian records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

CLAIM_BOUNDARY = "source-bounded Axiom II informational-Lagrangian map; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"
SOURCE_LEDGER_SPAN = ("P0R00782", "P0R00790")


@dataclass(frozen=True, slots=True)
class AxiomIIInformationalLagrangianConfig:
    """Configuration for the Axiom II informational-Lagrangian fixture."""

    expected_source_record_count: int = 9
    expected_gauge_equation_count: int = 2
    next_source_boundary: str = "P0R00791"

    def __post_init__(self) -> None:
        if self.expected_source_record_count != 9:
            raise ValueError("expected_source_record_count must equal 9")
        if self.expected_gauge_equation_count != 2:
            raise ValueError("expected_gauge_equation_count must equal 2")
        if self.next_source_boundary != "P0R00791":
            raise ValueError("next_source_boundary must equal P0R00791")


@dataclass(frozen=True, slots=True)
class AxiomIIInformationalLagrangianFixtureResult:
    """Result for the Paper 0 Axiom II informational-Lagrangian fixture."""

    source_ledger_span: tuple[str, str]
    hardware_status: str
    claim_boundary: str
    components: dict[str, str]
    labels: dict[str, str]
    source_record_count: int
    gauge_equation_count: int
    standard_gauge_baseline_count: int
    scpn_gauge_count: int
    pullback_protocol_count: int
    next_source_boundary: str
    null_controls: dict[str, float]
    problem_metadata: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-ready result dictionary."""
        return asdict(self)


def classify_informational_lagrangian_component(component: str) -> str:
    """Classify source-defined Axiom II informational-Lagrangian components."""
    mapping = {
        "kinetic_term_modification": ("infoton_kinetic_term_uses_pulled_back_information_metric"),
        "standard_gauge_baseline": "standard_gauge_lagrangian_spacetime_metric_baseline",
        "scpn_gauge_lagrangian": "scpn_gauge_lagrangian_information_metric_substitution",
        "operational_pullback_protocol": "chapter6_pullback_protocol_falsifiability_bridge",
    }
    try:
        return mapping[component]
    except KeyError as exc:
        raise ValueError("unknown informational-Lagrangian component") from exc


def axiom_ii_informational_lagrangian_labels() -> dict[str, str]:
    """Return source-bounded labels for the Axiom II informational-Lagrangian slice."""
    return {
        "section": "Formal Consequence: The Informational Lagrangian",
        "standard_lagrangian": (
            "L_gauge = -1/4 g^{mu alpha} g^{nu beta} F_{mu nu} F_{alpha beta}"
        ),
        "scpn_lagrangian": (
            "L_gauge = -1/4 tilde_g_F^{mu alpha} tilde_g_F^{nu beta} F_{mu nu} F_{alpha beta}"
        ),
        "next_boundary": "Axiom III: The Drive of Teleological Optimisation",
    }


def validate_axiom_ii_informational_lagrangian_fixture(
    config: AxiomIIInformationalLagrangianConfig | None = None,
) -> AxiomIIInformationalLagrangianFixtureResult:
    """Validate source accounting for the Axiom II informational-Lagrangian slice."""
    cfg = config or AxiomIIInformationalLagrangianConfig()
    components = (
        "kinetic_term_modification",
        "standard_gauge_baseline",
        "scpn_gauge_lagrangian",
        "operational_pullback_protocol",
    )

    return AxiomIIInformationalLagrangianFixtureResult(
        source_ledger_span=SOURCE_LEDGER_SPAN,
        hardware_status=HARDWARE_STATUS,
        claim_boundary=CLAIM_BOUNDARY,
        components={
            component: classify_informational_lagrangian_component(component)
            for component in components
        },
        labels=axiom_ii_informational_lagrangian_labels(),
        source_record_count=cfg.expected_source_record_count,
        gauge_equation_count=cfg.expected_gauge_equation_count,
        standard_gauge_baseline_count=3,
        scpn_gauge_count=3,
        pullback_protocol_count=1,
        next_source_boundary=cfg.next_source_boundary,
        null_controls={
            "informational_lagrangian_is_source_formula_not_empirical_evidence": 1.0,
            "pulled_back_fim_requires_chapter6_operational_protocol": 1.0,
            "falsifiability_claim_requires_downstream_bridge_tests": 1.0,
        },
        problem_metadata={
            "source_ledger_span": SOURCE_LEDGER_SPAN,
            "source_ledger_ids": tuple(f"P0R{number:05d}" for number in range(782, 791)),
            "claim_boundary": CLAIM_BOUNDARY,
            "protocol_state": "source_axiom_ii_informational_lagrangian_only_no_experiment",
        },
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "HARDWARE_STATUS",
    "SOURCE_LEDGER_SPAN",
    "AxiomIIInformationalLagrangianConfig",
    "AxiomIIInformationalLagrangianFixtureResult",
    "axiom_ii_informational_lagrangian_labels",
    "classify_informational_lagrangian_component",
    "validate_axiom_ii_informational_lagrangian_fixture",
]
