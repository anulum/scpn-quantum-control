# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 terminology bridge validation
"""Source-accounting checks for Paper 0 terminology-bridge records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

CLAIM_BOUNDARY = "source-bounded terminology bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"
SOURCE_LEDGER_SPAN = ("P0R00610", "P0R00634")


@dataclass(frozen=True, slots=True)
class TerminologyBridgeConfig:
    """Configuration for the terminology-bridge fixture."""

    expected_mainstream_anchor_count: int = 4
    expected_analogy_boundary_count: int = 2
    next_source_boundary: str = "P0R00635"

    def __post_init__(self) -> None:
        if self.expected_mainstream_anchor_count != 4:
            raise ValueError("expected_mainstream_anchor_count must equal 4")
        if self.expected_analogy_boundary_count != 2:
            raise ValueError("expected_analogy_boundary_count must equal 2")
        if self.next_source_boundary != "P0R00635":
            raise ValueError("next_source_boundary must equal P0R00635")


@dataclass(frozen=True, slots=True)
class TerminologyBridgeFixtureResult:
    """Result for the Paper 0 terminology-bridge fixture."""

    source_ledger_span: tuple[str, str]
    hardware_status: str
    claim_boundary: str
    mainstream_anchors: dict[str, str]
    pela_boundaries: dict[str, str]
    labels: dict[str, str]
    mainstream_anchor_count: int
    analogy_boundary_count: int
    next_source_boundary: str
    null_controls: dict[str, float]
    problem_metadata: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-ready result dictionary."""
        return asdict(self)


def classify_mainstream_anchor(term: str) -> str:
    """Classify the source-defined mainstream anchor for a core term."""
    mapping = {
        "psi_field": "field_theory_section_of_fibre_bundle",
        "upde": "nonlinear_coupled_oscillator_model",
        "geometric_qualia": "topology_metric_state_manifold_tda",
        "pela": "yang_mills_like_regulariser_not_literal_gauge_law",
    }
    try:
        return mapping[term]
    except KeyError as exc:
        raise ValueError("unknown terminology anchor") from exc


def classify_pela_boundary(boundary: str) -> str:
    """Classify PELA/Yang-Mills analogy boundaries from source records."""
    mapping = {
        "role": "supervisory_optimisation_prior",
        "h_int_effect": "sets_boundary_conditions_or_tunes_parameters",
        "gauge_status": "analogy_not_deductive_derivation",
        "simulation_status": "stress_testable_control_functional",
    }
    try:
        return mapping[boundary]
    except KeyError as exc:
        raise ValueError("unknown PELA boundary") from exc


def terminology_bridge_labels() -> dict[str, str]:
    """Return source-bounded terminology bridge labels."""
    return {
        "bridge": "Terminology Bridge",
        "predictive_coding": "precision-weighted priors",
        "h_int": "H_int = -lambda * Psi_s * sigma",
        "sigma_target": "topological invariants or geometric properties",
    }


def validate_terminology_bridge_fixture(
    config: TerminologyBridgeConfig | None = None,
) -> TerminologyBridgeFixtureResult:
    """Validate source accounting for the terminology bridge slice."""
    cfg = config or TerminologyBridgeConfig()
    anchor_keys = ("psi_field", "upde", "geometric_qualia", "pela")
    boundary_keys = ("role", "h_int_effect", "gauge_status", "simulation_status")
    anchors = {key: classify_mainstream_anchor(key) for key in anchor_keys}
    boundaries = {key: classify_pela_boundary(key) for key in boundary_keys}

    return TerminologyBridgeFixtureResult(
        source_ledger_span=SOURCE_LEDGER_SPAN,
        hardware_status=HARDWARE_STATUS,
        claim_boundary=CLAIM_BOUNDARY,
        mainstream_anchors=anchors,
        pela_boundaries=boundaries,
        labels=terminology_bridge_labels(),
        mainstream_anchor_count=cfg.expected_mainstream_anchor_count,
        analogy_boundary_count=cfg.expected_analogy_boundary_count,
        next_source_boundary=cfg.next_source_boundary,
        null_controls={
            "anchor_map_is_not_validation_evidence": 1.0,
            "yang_mills_similarity_is_not_deductive_equivalence": 1.0,
            "pela_does_not_add_force_term_to_h_int": 1.0,
        },
        problem_metadata={
            "source_ledger_span": SOURCE_LEDGER_SPAN,
            "source_ledger_ids": tuple(f"P0R{number:05d}" for number in range(610, 635)),
            "claim_boundary": CLAIM_BOUNDARY,
            "protocol_state": "source_anchor_map_only_no_experiment",
        },
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "HARDWARE_STATUS",
    "SOURCE_LEDGER_SPAN",
    "TerminologyBridgeConfig",
    "TerminologyBridgeFixtureResult",
    "classify_mainstream_anchor",
    "classify_pela_boundary",
    "terminology_bridge_labels",
    "validate_terminology_bridge_fixture",
]
