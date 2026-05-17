# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 Psi-Higgs new-scalar-particle validation
"""Source-accounting checks for Paper 0 Psi-Higgs new-scalar-particle records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

CLAIM_BOUNDARY = "source-bounded Psi-Higgs new-scalar-particle bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"
SOURCE_LEDGER_SPAN = ("P0R01638", "P0R01646")


@dataclass(frozen=True, slots=True)
class PsiHiggsNewScalarParticleConfig:
    """Configuration for the Psi-Higgs new-scalar-particle fixture."""

    expected_source_record_count: int = 9
    expected_component_count: int = 3
    next_source_boundary: str = "P0R01647"

    def __post_init__(self) -> None:
        if self.expected_source_record_count != 9:
            raise ValueError("expected_source_record_count must equal 9")
        if self.expected_component_count != 3:
            raise ValueError("expected_component_count must equal 3")
        if self.next_source_boundary != "P0R01647":
            raise ValueError("next_source_boundary must equal P0R01647")


@dataclass(frozen=True, slots=True)
class PsiHiggsNewScalarParticleFixtureResult:
    """Result for the Paper 0 Psi-Higgs new-scalar-particle fixture."""

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


def classify_psi_higgs_new_scalar_particle_component(component: str) -> str:
    """Classify source-defined Psi-Higgs new-scalar-particle components."""
    mapping = {
        "scalar_remnant_identity": "psi_higgs_scalar_remnant_identity_boundary",
        "potential_mass_term": "psi_higgs_potential_mass_term_source_boundary",
        "mass_and_detection_boundary": "psi_higgs_mass_relation_future_discovery_boundary",
    }
    try:
        return mapping[component]
    except KeyError as exc:
        raise ValueError("unknown Psi-Higgs new-scalar-particle component") from exc


def psi_higgs_new_scalar_particle_labels() -> dict[str, str]:
    """Return source-bounded labels for the Psi-Higgs new-scalar-particle slice."""
    return {
        "section": "The Psi-Higgs Boson: A New Scalar Particle",
        "identity": "h(x) radial fluctuation of the Psi-field",
        "mass_term": "L_mass,h = -lambda v^2 h^2",
        "mass": "m_h = sqrt(2 lambda) v",
        "next_boundary": "Experimental Signatures and Search Strategies",
    }


def validate_psi_higgs_new_scalar_particle_fixture(
    config: PsiHiggsNewScalarParticleConfig | None = None,
) -> PsiHiggsNewScalarParticleFixtureResult:
    """Validate source accounting for the Psi-Higgs new-scalar-particle slice."""
    cfg = config or PsiHiggsNewScalarParticleConfig()
    components = (
        "scalar_remnant_identity",
        "potential_mass_term",
        "mass_and_detection_boundary",
    )

    return PsiHiggsNewScalarParticleFixtureResult(
        source_ledger_span=SOURCE_LEDGER_SPAN,
        hardware_status=HARDWARE_STATUS,
        claim_boundary=CLAIM_BOUNDARY,
        components={
            component: classify_psi_higgs_new_scalar_particle_component(component)
            for component in components
        },
        labels=psi_higgs_new_scalar_particle_labels(),
        source_record_count=cfg.expected_source_record_count,
        component_count=cfg.expected_component_count,
        next_source_boundary=cfg.next_source_boundary,
        null_controls={
            "scalar_remnant_identity_is_not_particle_detection": 1.0,
            "potential_mass_term_is_not_collider_mass_reconstruction": 1.0,
            "future_discovery_clause_is_not_current_evidence": 1.0,
        },
        problem_metadata={
            "source_ledger_span": SOURCE_LEDGER_SPAN,
            "source_ledger_ids": tuple(f"P0R{number:05d}" for number in range(1638, 1647)),
            "claim_boundary": CLAIM_BOUNDARY,
            "protocol_state": "source_psi_higgs_new_scalar_particle_only_no_experiment",
        },
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "HARDWARE_STATUS",
    "SOURCE_LEDGER_SPAN",
    "PsiHiggsNewScalarParticleConfig",
    "PsiHiggsNewScalarParticleFixtureResult",
    "classify_psi_higgs_new_scalar_particle_component",
    "psi_higgs_new_scalar_particle_labels",
    "validate_psi_higgs_new_scalar_particle_fixture",
]
