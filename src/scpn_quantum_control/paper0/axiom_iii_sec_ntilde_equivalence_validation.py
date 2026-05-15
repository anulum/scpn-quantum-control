# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 Axiom III SEC-Ntilde equivalence validation
"""Source-accounting checks for Paper 0 Axiom III SEC-Ntilde-equivalence records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

CLAIM_BOUNDARY = "source-bounded Axiom III SEC-Ntilde-equivalence map; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"
SOURCE_LEDGER_SPAN = ("P0R00811", "P0R00817")


@dataclass(frozen=True, slots=True)
class AxiomIIISECNtildeEquivalenceConfig:
    """Configuration for the Axiom III SEC-Ntilde-equivalence fixture."""

    expected_source_record_count: int = 7
    expected_blank_terminal_record_count: int = 1
    next_source_boundary: str = "P0R00818"

    def __post_init__(self) -> None:
        if self.expected_source_record_count != 7:
            raise ValueError("expected_source_record_count must equal 7")
        if self.expected_blank_terminal_record_count != 1:
            raise ValueError("expected_blank_terminal_record_count must equal 1")
        if self.next_source_boundary != "P0R00818":
            raise ValueError("next_source_boundary must equal P0R00818")


@dataclass(frozen=True, slots=True)
class AxiomIIISECNtildeEquivalenceFixtureResult:
    """Result for the Paper 0 Axiom III SEC-Ntilde-equivalence fixture."""

    source_ledger_span: tuple[str, str]
    hardware_status: str
    claim_boundary: str
    components: dict[str, str]
    labels: dict[str, str]
    source_record_count: int
    equivalence_claim_count: int
    architecture_target_count: int
    efficiency_claim_count: int
    blank_terminal_record_count: int
    next_source_boundary: str
    null_controls: dict[str, float]
    problem_metadata: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-ready result dictionary."""
        return asdict(self)


def classify_sec_ntilde_equivalence_component(component: str) -> str:
    """Classify source-defined Axiom III SEC-Ntilde-equivalence components."""
    mapping = {
        "equivalence_heading": "sec_ntilde_unity_equivalence_subsection",
        "macroscopic_realisation": "sec_as_macroscopic_realisation_of_ntilde_unity",
        "quasicritical_efficiency": "ntilde_unity_as_quasicritical_efficiency_target",
        "causal_imperative_architecture": ("causal_imperative_and_15_layer_locking_architecture"),
    }
    try:
        return mapping[component]
    except KeyError as exc:
        raise ValueError("unknown SEC-Ntilde-equivalence component") from exc


def axiom_iii_sec_ntilde_equivalence_labels() -> dict[str, str]:
    """Return source-bounded labels for the Axiom III SEC-Ntilde-equivalence slice."""
    return {
        "section": "Equivalence of SEC and the tilde_N_t = 1 State",
        "equivalence": "SEC is macroscopic physical realisation of tilde_N_t = 1",
        "source_integrity": "P0R00816 truncated; P0R00817 blank",
        "next_boundary": "1.4 Tripartite Ontology: The Substance of Information",
    }


def validate_axiom_iii_sec_ntilde_equivalence_fixture(
    config: AxiomIIISECNtildeEquivalenceConfig | None = None,
) -> AxiomIIISECNtildeEquivalenceFixtureResult:
    """Validate source accounting for the Axiom III SEC-Ntilde-equivalence slice."""
    cfg = config or AxiomIIISECNtildeEquivalenceConfig()
    components = (
        "equivalence_heading",
        "macroscopic_realisation",
        "quasicritical_efficiency",
        "causal_imperative_architecture",
    )

    return AxiomIIISECNtildeEquivalenceFixtureResult(
        source_ledger_span=SOURCE_LEDGER_SPAN,
        hardware_status=HARDWARE_STATUS,
        claim_boundary=CLAIM_BOUNDARY,
        components={
            component: classify_sec_ntilde_equivalence_component(component)
            for component in components
        },
        labels=axiom_iii_sec_ntilde_equivalence_labels(),
        source_record_count=cfg.expected_source_record_count,
        equivalence_claim_count=2,
        architecture_target_count=2,
        efficiency_claim_count=2,
        blank_terminal_record_count=cfg.expected_blank_terminal_record_count,
        next_source_boundary=cfg.next_source_boundary,
        null_controls={
            "sec_ntilde_equivalence_is_source_claim_not_empirical_evidence": 1.0,
            "truncated_p0r00816_requires_source_audit_before_completion_claim": 1.0,
            "blank_p0r00817_is_preserved_not_silently_omitted": 1.0,
        },
        problem_metadata={
            "source_ledger_span": SOURCE_LEDGER_SPAN,
            "source_ledger_ids": tuple(f"P0R{number:05d}" for number in range(811, 818)),
            "claim_boundary": CLAIM_BOUNDARY,
            "protocol_state": "source_axiom_iii_sec_ntilde_equivalence_only_no_experiment",
        },
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "HARDWARE_STATUS",
    "SOURCE_LEDGER_SPAN",
    "AxiomIIISECNtildeEquivalenceConfig",
    "AxiomIIISECNtildeEquivalenceFixtureResult",
    "axiom_iii_sec_ntilde_equivalence_labels",
    "classify_sec_ntilde_equivalence_component",
    "validate_axiom_iii_sec_ntilde_equivalence_fixture",
]
