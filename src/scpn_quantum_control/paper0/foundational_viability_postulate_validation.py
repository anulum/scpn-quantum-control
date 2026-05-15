# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 foundational viability postulate validation
"""Source-accounting checks for Paper 0 foundational viability postulates."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

CLAIM_BOUNDARY = "source-bounded foundational viability postulate; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"
SOURCE_LEDGER_SPAN = ("P0R00464", "P0R00505")
COUPLING_EQUATION = "H_int = -lambda * Psi_s * sigma"


@dataclass(frozen=True, slots=True)
class FoundationalViabilityPostulateConfig:
    """Configuration for the foundational viability postulate fixture."""

    next_source_boundary: str = "P0R00506"

    def __post_init__(self) -> None:
        if self.next_source_boundary != "P0R00506":
            raise ValueError("next_source_boundary must equal P0R00506")


@dataclass(frozen=True, slots=True)
class FoundationalViabilityPostulateFixtureResult:
    """Result for the Paper 0 foundational viability postulate fixture."""

    source_ledger_span: tuple[str, str]
    hardware_status: str
    claim_boundary: str
    viability_pillars: dict[str, str]
    physical_postulate_components: dict[str, str]
    quantum_numbers: dict[str, str]
    pillar_count: int
    physics_postulate_count: int
    coupling_equation: str
    next_source_boundary: str
    null_controls: dict[str, float]
    problem_metadata: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-ready result dictionary."""
        return asdict(self)


def classify_viability_pillar(pillar: str) -> str:
    """Classify the three source viability pillars."""
    mapping = {
        "ontological_postulate": "psi_field_primitive_ontology",
        "derived_interactions": "u1_fim_interaction_derivation",
        "multiscale_architecture": "hierarchy_rg_bidirectional_causality",
    }
    try:
        return mapping[pillar]
    except KeyError as exc:
        raise ValueError("unknown foundational viability pillar") from exc


def classify_physical_postulate_component(component: str) -> str:
    """Classify source physical-postulate components."""
    mapping = {
        "complex_scalar": "standard_qft_scalar_field",
        "spin": "spin_0_bosonic_quanta",
        "phase_symmetry": "global_u1_phase_symmetry",
        "fim_coupling": "informational_geometry_coupling",
    }
    try:
        return mapping[component]
    except KeyError as exc:
        raise ValueError("unknown physical postulate component") from exc


def psi_field_quantum_numbers() -> dict[str, str]:
    """Return the source-bounded Psi-field quantum-number formalisation."""
    return {
        "spin": "0",
        "statistics": "bosonic",
        "symmetry": "global U(1) phase",
        "decomposition": "Psi = |Psi| e^{i theta}",
    }


def validate_foundational_viability_postulate_fixture(
    config: FoundationalViabilityPostulateConfig | None = None,
) -> FoundationalViabilityPostulateFixtureResult:
    """Validate source accounting for the foundational viability postulate run."""
    cfg = config or FoundationalViabilityPostulateConfig()
    pillars = (
        "ontological_postulate",
        "derived_interactions",
        "multiscale_architecture",
    )
    components = ("complex_scalar", "spin", "phase_symmetry", "fim_coupling")
    viability_pillars = {pillar: classify_viability_pillar(pillar) for pillar in pillars}
    physical_components = {
        component: classify_physical_postulate_component(component) for component in components
    }

    return FoundationalViabilityPostulateFixtureResult(
        source_ledger_span=SOURCE_LEDGER_SPAN,
        hardware_status=HARDWARE_STATUS,
        claim_boundary=CLAIM_BOUNDARY,
        viability_pillars=viability_pillars,
        physical_postulate_components=physical_components,
        quantum_numbers=psi_field_quantum_numbers(),
        pillar_count=len(viability_pillars),
        physics_postulate_count=len(physical_components),
        coupling_equation=COUPLING_EQUATION,
        next_source_boundary=cfg.next_source_boundary,
        null_controls={
            "internal_consistency_is_not_empirical_validation": 1.0,
            "gauge_derivation_requires_later_section_boundary": 1.0,
            "complex_scalar_formalisation_not_detection_label": 1.0,
        },
        problem_metadata={
            "source_ledger_span": SOURCE_LEDGER_SPAN,
            "source_ledger_ids": tuple(f"P0R{number:05d}" for number in range(464, 506)),
            "claim_boundary": CLAIM_BOUNDARY,
            "protocol_state": "source_internal_consistency_map_only_no_experiment",
        },
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "COUPLING_EQUATION",
    "HARDWARE_STATUS",
    "SOURCE_LEDGER_SPAN",
    "FoundationalViabilityPostulateConfig",
    "FoundationalViabilityPostulateFixtureResult",
    "classify_physical_postulate_component",
    "classify_viability_pillar",
    "psi_field_quantum_numbers",
    "validate_foundational_viability_postulate_fixture",
]
