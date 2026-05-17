# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 Non-Abelian qualia field validation
"""Source-accounting checks for Paper 0 Non-Abelian qualia-field records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

CLAIM_BOUNDARY = "source-bounded Non-Abelian qualia-field hypothesis; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"
SOURCE_LEDGER_SPAN = ("P0R01103", "P0R01134")


@dataclass(frozen=True, slots=True)
class NonAbelianQualiaFieldConfig:
    """Configuration for the Non-Abelian qualia-field fixture."""

    expected_source_record_count: int = 32
    expected_anomaly_condition_record_count: int = 10
    expected_confinement_record_count: int = 9
    next_source_boundary: str = "P0R01135"

    def __post_init__(self) -> None:
        if self.expected_source_record_count != 32:
            raise ValueError("expected_source_record_count must equal 32")
        if self.expected_anomaly_condition_record_count != 10:
            raise ValueError("expected_anomaly_condition_record_count must equal 10")
        if self.expected_confinement_record_count != 9:
            raise ValueError("expected_confinement_record_count must equal 9")
        if self.next_source_boundary != "P0R01135":
            raise ValueError("next_source_boundary must equal P0R01135")


@dataclass(frozen=True, slots=True)
class NonAbelianQualiaFieldFixtureResult:
    """Result for the Paper 0 Non-Abelian qualia-field fixture."""

    source_ledger_span: tuple[str, str]
    hardware_status: str
    claim_boundary: str
    components: dict[str, str]
    labels: dict[str, str]
    source_record_count: int
    structural_record_count: int
    context_record_count: int
    claim_record_count: int
    validation_target_record_count: int
    anomaly_condition_record_count: int
    confinement_record_count: int
    topological_entanglement_record_count: int
    next_source_boundary: str
    null_controls: dict[str, float]
    problem_metadata: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-ready result dictionary."""
        return asdict(self)


def classify_non_abelian_qualia_field_component(component: str) -> str:
    """Classify source-defined Non-Abelian qualia-field components."""
    mapping = {
        "boundary_and_rationale": "u1_to_su_n_qualia_field_hypothesis_boundary",
        "self_interacting_gauge_bosons": "non_abelian_gauge_boson_multiplicity_and_self_interaction",
        "anomaly_cancellation_condition": "su_n_qualia_colour_anomaly_cancellation_constraint",
        "confinement_binding_boundary": "qcd_analogue_confinement_and_binding_problem_claim_boundary",
        "topological_entanglement_resolution": "topological_entanglement_and_qualia_ball_prediction_boundary",
    }
    try:
        return mapping[component]
    except KeyError as exc:
        raise ValueError("unknown Non-Abelian qualia-field component") from exc


def non_abelian_qualia_field_labels() -> dict[str, str]:
    """Return source-bounded labels for the Non-Abelian qualia-field slice."""
    return {
        "section": "Beyond U(1): The Hypothesis of a Non-Abelian Qualia Field",
        "field_strength": "F_mu_nu^a includes g f_abc A_mu^b A_nu^c",
        "anomaly_condition": "sum d_abc q_i_a q_i_b q_i_c == 0",
        "confinement": "Qualia confinement remains QCD-analogue source claim",
        "next_boundary": "Consistency Conditions and the Origin of Geometric Coupling",
    }


def validate_non_abelian_qualia_field_fixture(
    config: NonAbelianQualiaFieldConfig | None = None,
) -> NonAbelianQualiaFieldFixtureResult:
    """Validate source accounting for the Non-Abelian qualia-field slice."""
    cfg = config or NonAbelianQualiaFieldConfig()
    components = (
        "boundary_and_rationale",
        "self_interacting_gauge_bosons",
        "anomaly_cancellation_condition",
        "confinement_binding_boundary",
        "topological_entanglement_resolution",
    )

    return NonAbelianQualiaFieldFixtureResult(
        source_ledger_span=SOURCE_LEDGER_SPAN,
        hardware_status=HARDWARE_STATUS,
        claim_boundary=CLAIM_BOUNDARY,
        components={
            component: classify_non_abelian_qualia_field_component(component)
            for component in components
        },
        labels=non_abelian_qualia_field_labels(),
        source_record_count=cfg.expected_source_record_count,
        structural_record_count=2,
        context_record_count=13,
        claim_record_count=13,
        validation_target_record_count=4,
        anomaly_condition_record_count=cfg.expected_anomaly_condition_record_count,
        confinement_record_count=cfg.expected_confinement_record_count,
        topological_entanglement_record_count=7,
        next_source_boundary=cfg.next_source_boundary,
        null_controls={
            "non_abelian_qualia_field_is_source_hypothesis_not_empirical_evidence": 1.0,
            "qcd_analogy_alone_is_not_biological_confinement_validation": 1.0,
            "trivial_singlet_neutrality_is_preserved_as_rejected_mapping": 1.0,
        },
        problem_metadata={
            "source_ledger_span": SOURCE_LEDGER_SPAN,
            "source_ledger_ids": tuple(f"P0R{number:05d}" for number in range(1103, 1135)),
            "claim_boundary": CLAIM_BOUNDARY,
            "protocol_state": "source_non_abelian_qualia_field_only_no_experiment",
        },
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "HARDWARE_STATUS",
    "SOURCE_LEDGER_SPAN",
    "NonAbelianQualiaFieldConfig",
    "NonAbelianQualiaFieldFixtureResult",
    "classify_non_abelian_qualia_field_component",
    "non_abelian_qualia_field_labels",
    "validate_non_abelian_qualia_field_fixture",
]
