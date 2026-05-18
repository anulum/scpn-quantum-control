# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 V. The Interface Problem Synthesis (Mind-Body-Field) validation
"""Source-accounting checks for Paper 0 V. The Interface Problem Synthesis (Mind-Body-Field) records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

CLAIM_BOUNDARY = "source-bounded v the interface problem synthesis mind body field source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"
SOURCE_LEDGER_SPAN = ("P0R03241", "P0R03249")


@dataclass(frozen=True, slots=True)
class VTheInterfaceProblemSynthesisMindBodyFieldConfig:
    """Configuration for this Paper 0 source-accounting fixture."""

    expected_source_record_count: int = 9
    expected_component_count: int = 4
    next_source_boundary: str = "P0R03250"

    def __post_init__(self) -> None:
        if self.expected_source_record_count != 9:
            raise ValueError("expected_source_record_count must equal 9")
        if self.expected_component_count != 4:
            raise ValueError("expected_component_count must equal 4")
        if self.next_source_boundary != "P0R03250":
            raise ValueError("next_source_boundary must equal P0R03250")


@dataclass(frozen=True, slots=True)
class VTheInterfaceProblemSynthesisMindBodyFieldFixtureResult:
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


def classify_v_the_interface_problem_synthesis_mind_body_field_component(component: str) -> str:
    """Classify source-defined components."""
    mapping = {
        "v_the_interface_problem_synthesis_mind_body_field": "v_the_interface_problem_synthesis_mind_body_field_source_boundary",
        "downward_causation_the_psi_field_biases_physical_dynamics_via_qze_stabil": "downward_causation_the_psi_field_biases_physical_dynamics_via_qze_stabil_source_boundary",
        "upward_causation_the_physical_substrate_structures_the_psi_field_by_enco": "upward_causation_the_physical_substrate_structures_the_psi_field_by_enco_source_boundary",
        "vi_formalising_emergence_phase_transitions_and_ginzburg_landau_theory": "vi_formalising_emergence_phase_transitions_and_ginzburg_landau_theory_source_boundary",
    }
    try:
        return mapping[component]
    except KeyError as exc:
        raise ValueError(
            "unknown v_the_interface_problem_synthesis_mind_body_field component"
        ) from exc


def v_the_interface_problem_synthesis_mind_body_field_labels() -> dict[str, str]:
    """Return source-bounded labels for this slice."""
    return {
        "section": "V. The Interface Problem Synthesis (Mind-Body-Field)",
        "source_span": "P0R03241-P0R03249",
        "component_count": "4",
        "next_boundary": "P0R03250",
        "component_1": "V. The Interface Problem Synthesis (Mind-Body-Field)",
        "component_2": "Downward Causation: The Psi-field biases physical dynamics via QZE stabilisation (Attention) and guiding self-organisation at criticality.",
        "component_3": "Upward Causation: The physical substrate structures the Psi-field by encoding information (Topological Defects) and constraining the field geometry (L5 Manifold).",
        "component_4": "VI. Formalising Emergence: Phase Transitions and Ginzburg-Landau Theory",
    }


def validate_v_the_interface_problem_synthesis_mind_body_field_fixture(
    config: VTheInterfaceProblemSynthesisMindBodyFieldConfig | None = None,
) -> VTheInterfaceProblemSynthesisMindBodyFieldFixtureResult:
    """Validate source accounting for this Paper 0 slice."""
    cfg = config or VTheInterfaceProblemSynthesisMindBodyFieldConfig()
    components = (
        "v_the_interface_problem_synthesis_mind_body_field",
        "downward_causation_the_psi_field_biases_physical_dynamics_via_qze_stabil",
        "upward_causation_the_physical_substrate_structures_the_psi_field_by_enco",
        "vi_formalising_emergence_phase_transitions_and_ginzburg_landau_theory",
    )
    return VTheInterfaceProblemSynthesisMindBodyFieldFixtureResult(
        source_ledger_span=SOURCE_LEDGER_SPAN,
        hardware_status=HARDWARE_STATUS,
        claim_boundary=CLAIM_BOUNDARY,
        components={
            component: classify_v_the_interface_problem_synthesis_mind_body_field_component(
                component
            )
            for component in components
        },
        labels=v_the_interface_problem_synthesis_mind_body_field_labels(),
        source_record_count=cfg.expected_source_record_count,
        component_count=cfg.expected_component_count,
        next_source_boundary=cfg.next_source_boundary,
        null_controls={
            "v_the_interface_problem_synthesis_mind_body_field_is_not_empirical_validation_evidence": 1.0,
            "downward_causation_the_psi_field_biases_physical_dynamics_via_qze_stabil_is_not_empirical_validation_evidence": 1.0,
            "upward_causation_the_physical_substrate_structures_the_psi_field_by_enco_is_not_empirical_validation_evidence": 1.0,
            "vi_formalising_emergence_phase_transitions_and_ginzburg_landau_theory_is_not_empirical_validation_evidence": 1.0,
        },
        problem_metadata={
            "source_ledger_span": SOURCE_LEDGER_SPAN,
            "source_ledger_ids": tuple(f"P0R{number:05d}" for number in range(3241, 3250)),
            "claim_boundary": CLAIM_BOUNDARY,
            "protocol_state": "source_v_the_interface_problem_synthesis_mind_body_field_only_no_experiment",
        },
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "HARDWARE_STATUS",
    "SOURCE_LEDGER_SPAN",
    "VTheInterfaceProblemSynthesisMindBodyFieldConfig",
    "VTheInterfaceProblemSynthesisMindBodyFieldFixtureResult",
    "classify_v_the_interface_problem_synthesis_mind_body_field_component",
    "v_the_interface_problem_synthesis_mind_body_field_labels",
    "validate_v_the_interface_problem_synthesis_mind_body_field_fixture",
]
