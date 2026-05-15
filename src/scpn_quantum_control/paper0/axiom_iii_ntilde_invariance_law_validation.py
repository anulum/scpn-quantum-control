# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 Axiom III Ntilde invariance-law validation
"""Source-accounting checks for Paper 0 Axiom III Ntilde-invariance-law records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

CLAIM_BOUNDARY = "source-bounded Axiom III Ntilde-invariance-law map; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"
SOURCE_LEDGER_SPAN = ("P0R00800", "P0R00810")


@dataclass(frozen=True, slots=True)
class AxiomIIINtildeInvarianceLawConfig:
    """Configuration for the Axiom III Ntilde-invariance-law fixture."""

    expected_source_record_count: int = 11
    expected_variable_definition_count: int = 3
    next_source_boundary: str = "P0R00811"

    def __post_init__(self) -> None:
        if self.expected_source_record_count != 11:
            raise ValueError("expected_source_record_count must equal 11")
        if self.expected_variable_definition_count != 3:
            raise ValueError("expected_variable_definition_count must equal 3")
        if self.next_source_boundary != "P0R00811":
            raise ValueError("next_source_boundary must equal P0R00811")


@dataclass(frozen=True, slots=True)
class AxiomIIINtildeInvarianceLawFixtureResult:
    """Result for the Paper 0 Axiom III Ntilde-invariance-law fixture."""

    source_ledger_span: tuple[str, str]
    hardware_status: str
    claim_boundary: str
    components: dict[str, str]
    labels: dict[str, str]
    source_record_count: int
    invariant_definition_count: int
    variable_definition_count: int
    threshold_equation_count: int
    reversible_limit_count: int
    next_source_boundary: str
    null_controls: dict[str, float]
    problem_metadata: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-ready result dictionary."""
        return asdict(self)


def classify_ntilde_invariance_law_component(component: str) -> str:
    """Classify source-defined Axiom III Ntilde-invariance-law components."""
    mapping = {
        "physical_law_identification": (
            "teleological_drive_identified_with_dimensionless_ntilde_invariant"
        ),
        "invariant_ratio_equation": "ntilde_power_over_reversible_information_processing_cost",
        "variable_definitions": "power_information_rate_and_reversible_cost_per_bit",
        "unity_threshold_limit": "ntilde_unity_reversible_efficiency_limit",
    }
    try:
        return mapping[component]
    except KeyError as exc:
        raise ValueError("unknown Ntilde-invariance-law component") from exc


def axiom_iii_ntilde_invariance_law_labels() -> dict[str, str]:
    """Return source-bounded labels for the Axiom III Ntilde-invariance-law slice."""
    return {
        "section": "Formal Physical Definition: The tilde_N_t Invariance Law",
        "invariant": (
            "tilde_N_t = P / (epsilon_b dot_I) = (E/t) / ((Delta F_rev / Delta I) dot_I)"
        ),
        "threshold": "tilde_N_t -> 1",
        "next_boundary": "Equivalence of SEC and the tilde_N_t = 1 State",
    }


def validate_axiom_iii_ntilde_invariance_law_fixture(
    config: AxiomIIINtildeInvarianceLawConfig | None = None,
) -> AxiomIIINtildeInvarianceLawFixtureResult:
    """Validate source accounting for the Axiom III Ntilde-invariance-law slice."""
    cfg = config or AxiomIIINtildeInvarianceLawConfig()
    components = (
        "physical_law_identification",
        "invariant_ratio_equation",
        "variable_definitions",
        "unity_threshold_limit",
    )

    return AxiomIIINtildeInvarianceLawFixtureResult(
        source_ledger_span=SOURCE_LEDGER_SPAN,
        hardware_status=HARDWARE_STATUS,
        claim_boundary=CLAIM_BOUNDARY,
        components={
            component: classify_ntilde_invariance_law_component(component)
            for component in components
        },
        labels=axiom_iii_ntilde_invariance_law_labels(),
        source_record_count=cfg.expected_source_record_count,
        invariant_definition_count=3,
        variable_definition_count=cfg.expected_variable_definition_count,
        threshold_equation_count=1,
        reversible_limit_count=1,
        next_source_boundary=cfg.next_source_boundary,
        null_controls={
            "ntilde_invariance_law_is_source_claim_not_empirical_evidence": 1.0,
            "petrasek_2025_reference_requires_bibliographic_trace": 1.0,
            "unity_threshold_requires_downstream_operational_validation": 1.0,
        },
        problem_metadata={
            "source_ledger_span": SOURCE_LEDGER_SPAN,
            "source_ledger_ids": tuple(f"P0R{number:05d}" for number in range(800, 811)),
            "claim_boundary": CLAIM_BOUNDARY,
            "protocol_state": "source_axiom_iii_ntilde_invariance_law_only_no_experiment",
        },
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "HARDWARE_STATUS",
    "SOURCE_LEDGER_SPAN",
    "AxiomIIINtildeInvarianceLawConfig",
    "AxiomIIINtildeInvarianceLawFixtureResult",
    "axiom_iii_ntilde_invariance_law_labels",
    "classify_ntilde_invariance_law_component",
    "validate_axiom_iii_ntilde_invariance_law_fixture",
]
