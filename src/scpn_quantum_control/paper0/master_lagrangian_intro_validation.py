# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 master Lagrangian intro validation
"""Source-accounting checks for Paper 0 master-Lagrangian introduction records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

CLAIM_BOUNDARY = "source-bounded master-Lagrangian introduction; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"
SOURCE_LEDGER_SPAN = ("P0R00987", "P0R01017")


@dataclass(frozen=True, slots=True)
class MasterLagrangianIntroConfig:
    """Configuration for the master-Lagrangian-introduction fixture."""

    expected_source_record_count: int = 31
    expected_blank_record_count: int = 2
    expected_meta_framework_record_count: int = 16
    next_source_boundary: str = "P0R01018"

    def __post_init__(self) -> None:
        if self.expected_source_record_count != 31:
            raise ValueError("expected_source_record_count must equal 31")
        if self.expected_blank_record_count != 2:
            raise ValueError("expected_blank_record_count must equal 2")
        if self.expected_meta_framework_record_count != 16:
            raise ValueError("expected_meta_framework_record_count must equal 16")
        if self.next_source_boundary != "P0R01018":
            raise ValueError("next_source_boundary must equal P0R01018")


@dataclass(frozen=True, slots=True)
class MasterLagrangianIntroFixtureResult:
    """Result for the Paper 0 master-Lagrangian-introduction fixture."""

    source_ledger_span: tuple[str, str]
    hardware_status: str
    claim_boundary: str
    components: dict[str, str]
    labels: dict[str, str]
    source_record_count: int
    blank_record_count: int
    introduction_record_count: int
    meta_framework_record_count: int
    gauge_inference_record_count: int
    psis_coupling_record_count: int
    next_source_boundary: str
    null_controls: dict[str, float]
    problem_metadata: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-ready result dictionary."""
        return asdict(self)


def classify_master_lagrangian_intro_component(component: str) -> str:
    """Classify source-defined master-Lagrangian-introduction components."""
    mapping = {
        "part_ii_boundary": "physical_sector_and_master_lagrangian_section_boundary",
        "first_principles_framing": "phenomenological_to_first_principles_claim_boundary",
        "two_stream_derivation": (
            "u1_informational_and_curved_spacetime_geometric_derivation_claims"
        ),
        "explanatory_analogies": "lay_analogies_preserved_not_validation_evidence",
        "gauge_inference_integration": (
            "gauge_invariance_and_infoton_prediction_error_inference_mapping"
        ),
        "psis_coupling_gauge_interpretation": (
            "h_int_gauge_interaction_noether_current_and_dual_coupling_mapping"
        ),
    }
    try:
        return mapping[component]
    except KeyError as exc:
        raise ValueError("unknown master-Lagrangian-intro component") from exc


def master_lagrangian_intro_labels() -> dict[str, str]:
    """Return source-bounded labels for the master-Lagrangian-intro slice."""
    return {
        "section": "2.1 Master Interaction Lagrangian: Derivation from First Principles",
        "informational": "local U(1) gauge invariance of complex scalar Psi",
        "mediator": "infoton gauge boson A_mu",
        "current": "J_mu = i(Psi* partial_mu Psi - Psi partial_mu Psi*)",
        "coupling": "H_int = -lambda * Psis * sigma with lambda = g",
        "next_boundary": "A Gauge-Principle Derivation of the Psi-Field",
    }


def validate_master_lagrangian_intro_fixture(
    config: MasterLagrangianIntroConfig | None = None,
) -> MasterLagrangianIntroFixtureResult:
    """Validate source accounting for the master-Lagrangian-introduction slice."""
    cfg = config or MasterLagrangianIntroConfig()
    components = (
        "part_ii_boundary",
        "first_principles_framing",
        "two_stream_derivation",
        "explanatory_analogies",
        "gauge_inference_integration",
        "psis_coupling_gauge_interpretation",
    )

    return MasterLagrangianIntroFixtureResult(
        source_ledger_span=SOURCE_LEDGER_SPAN,
        hardware_status=HARDWARE_STATUS,
        claim_boundary=CLAIM_BOUNDARY,
        components={
            component: classify_master_lagrangian_intro_component(component)
            for component in components
        },
        labels=master_lagrangian_intro_labels(),
        source_record_count=cfg.expected_source_record_count,
        blank_record_count=cfg.expected_blank_record_count,
        introduction_record_count=13,
        meta_framework_record_count=cfg.expected_meta_framework_record_count,
        gauge_inference_record_count=6,
        psis_coupling_record_count=9,
        next_source_boundary=cfg.next_source_boundary,
        null_controls={
            "master_lagrangian_intro_is_source_claim_not_empirical_evidence": 1.0,
            "first_principles_language_is_not_proof_without_derivation_fixture": 1.0,
            "blank_records_p0r00988_p0r01001_are_preserved": 1.0,
        },
        problem_metadata={
            "source_ledger_span": SOURCE_LEDGER_SPAN,
            "source_ledger_ids": tuple(f"P0R{number:05d}" for number in range(987, 1018)),
            "claim_boundary": CLAIM_BOUNDARY,
            "protocol_state": "source_master_lagrangian_intro_only_no_experiment",
        },
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "HARDWARE_STATUS",
    "SOURCE_LEDGER_SPAN",
    "MasterLagrangianIntroConfig",
    "MasterLagrangianIntroFixtureResult",
    "classify_master_lagrangian_intro_component",
    "master_lagrangian_intro_labels",
    "validate_master_lagrangian_intro_fixture",
]
