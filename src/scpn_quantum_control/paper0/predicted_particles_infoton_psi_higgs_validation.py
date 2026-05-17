# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 predicted particles validation
"""Source-accounting checks for Paper 0 infoton and Psi-Higgs prediction records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

CLAIM_BOUNDARY = "source-bounded infoton and Psi-Higgs prediction bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"
SOURCE_LEDGER_SPAN = ("P0R01597", "P0R01622")


@dataclass(frozen=True, slots=True)
class PredictedParticlesInfotonPsiHiggsConfig:
    """Configuration for the infoton and Psi-Higgs prediction fixture."""

    expected_source_record_count: int = 26
    expected_component_count: int = 4
    next_source_boundary: str = "P0R01623"

    def __post_init__(self) -> None:
        if self.expected_source_record_count != 26:
            raise ValueError("expected_source_record_count must equal 26")
        if self.expected_component_count != 4:
            raise ValueError("expected_component_count must equal 4")
        if self.next_source_boundary != "P0R01623":
            raise ValueError("next_source_boundary must equal P0R01623")


@dataclass(frozen=True, slots=True)
class PredictedParticlesInfotonPsiHiggsFixtureResult:
    """Result for the Paper 0 infoton and Psi-Higgs prediction fixture."""

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


def classify_predicted_particles_infoton_psi_higgs_component(component: str) -> str:
    """Classify source-defined infoton and Psi-Higgs prediction components."""
    mapping = {
        "particle_prediction_opening": "u1_ssb_infoton_psi_higgs_prediction_boundary",
        "search_strategy_summary": "collider_cosmology_search_strategy_claim_boundary",
        "active_inference_mapping": "active_inference_particle_role_mapping_boundary",
        "h_int_falsifiability_bridge": "h_int_parameter_falsifiability_bridge_boundary",
    }
    try:
        return mapping[component]
    except KeyError as exc:
        raise ValueError("unknown predicted-particles component") from exc


def predicted_particles_infoton_psi_higgs_labels() -> dict[str, str]:
    """Return source-bounded labels for the predicted-particles slice."""
    return {
        "section": "Predicted Particles: The Infoton and the Psi-Higgs Boson",
        "infoton_mass": "m_A = g v",
        "psi_higgs_mass": "m_h = sqrt(2 lambda) v",
        "search_channels": "LHC and gravitational-wave signatures",
        "interaction": "H_int = -lambda * Psi_s * sigma",
        "next_boundary": "Derivation of the Infoton's Properties",
    }


def validate_predicted_particles_infoton_psi_higgs_fixture(
    config: PredictedParticlesInfotonPsiHiggsConfig | None = None,
) -> PredictedParticlesInfotonPsiHiggsFixtureResult:
    """Validate source accounting for the infoton and Psi-Higgs prediction slice."""
    cfg = config or PredictedParticlesInfotonPsiHiggsConfig()
    components = (
        "particle_prediction_opening",
        "search_strategy_summary",
        "active_inference_mapping",
        "h_int_falsifiability_bridge",
    )

    return PredictedParticlesInfotonPsiHiggsFixtureResult(
        source_ledger_span=SOURCE_LEDGER_SPAN,
        hardware_status=HARDWARE_STATUS,
        claim_boundary=CLAIM_BOUNDARY,
        components={
            component: classify_predicted_particles_infoton_psi_higgs_component(component)
            for component in components
        },
        labels=predicted_particles_infoton_psi_higgs_labels(),
        source_record_count=cfg.expected_source_record_count,
        component_count=cfg.expected_component_count,
        next_source_boundary=cfg.next_source_boundary,
        null_controls={
            "predicted_particles_are_not_observed_discoveries": 1.0,
            "lhc_and_lisa_search_channels_are_not_completed_evidence": 1.0,
            "active_inference_particle_mapping_is_not_neural_measurement": 1.0,
            "h_int_falsifiability_bridge_is_not_empirical_detection": 1.0,
        },
        problem_metadata={
            "source_ledger_span": SOURCE_LEDGER_SPAN,
            "source_ledger_ids": tuple(f"P0R{number:05d}" for number in range(1597, 1623)),
            "claim_boundary": CLAIM_BOUNDARY,
            "protocol_state": "source_predicted_particles_infoton_psi_higgs_only_no_experiment",
        },
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "HARDWARE_STATUS",
    "SOURCE_LEDGER_SPAN",
    "PredictedParticlesInfotonPsiHiggsConfig",
    "PredictedParticlesInfotonPsiHiggsFixtureResult",
    "classify_predicted_particles_infoton_psi_higgs_component",
    "predicted_particles_infoton_psi_higgs_labels",
    "validate_predicted_particles_infoton_psi_higgs_fixture",
]
