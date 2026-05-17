# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 Psi-Higgs LHC phenomenology validation
"""Source-accounting checks for Paper 0 Psi-Higgs LHC phenomenology records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

CLAIM_BOUNDARY = "source-bounded Psi-Higgs LHC phenomenology bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"
SOURCE_LEDGER_SPAN = ("P0R01655", "P0R01668")


@dataclass(frozen=True, slots=True)
class PsiHiggsLHCPhenomenologyConfig:
    """Configuration for the Psi-Higgs LHC phenomenology fixture."""

    expected_source_record_count: int = 14
    expected_component_count: int = 3
    next_source_boundary: str = "P0R01669"

    def __post_init__(self) -> None:
        if self.expected_source_record_count != 14:
            raise ValueError("expected_source_record_count must equal 14")
        if self.expected_component_count != 3:
            raise ValueError("expected_component_count must equal 3")
        if self.next_source_boundary != "P0R01669":
            raise ValueError("next_source_boundary must equal P0R01669")


@dataclass(frozen=True, slots=True)
class PsiHiggsLHCPhenomenologyFixtureResult:
    """Result for the Paper 0 Psi-Higgs LHC phenomenology fixture."""

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


def classify_psi_higgs_lhc_phenomenology_component(component: str) -> str:
    """Classify source-defined Psi-Higgs LHC phenomenology components."""
    mapping = {
        "phenomenology_bridge": "psi_higgs_lhc_phenomenology_bridge_boundary",
        "scalar_mixing_mechanism": "psi_higgs_sm_higgs_scalar_mixing_boundary",
        "scalar_potential_and_cross_term": "higgs_portal_potential_cross_term_boundary",
    }
    try:
        return mapping[component]
    except KeyError as exc:
        raise ValueError("unknown Psi-Higgs LHC phenomenology component") from exc


def psi_higgs_lhc_phenomenology_labels() -> dict[str, str]:
    """Return source-bounded labels for the Psi-Higgs LHC phenomenology slice."""
    return {
        "section": "The Psi-Higgs Boson: Phenomenology and Experimental Signatures at the LHC",
        "mechanism": "Psi-Higgs mechanism and scalar mixing",
        "portal": "V_mix = lambda_mix (H^dagger H) |Psi|^2",
        "cross_term": "lambda_mix v_h v_psi h_bare h_Psi,bare",
        "next_boundary": "Mass Eigenstates and the Mixing Angle (theta)",
    }


def validate_psi_higgs_lhc_phenomenology_fixture(
    config: PsiHiggsLHCPhenomenologyConfig | None = None,
) -> PsiHiggsLHCPhenomenologyFixtureResult:
    """Validate source accounting for the Psi-Higgs LHC phenomenology slice."""
    cfg = config or PsiHiggsLHCPhenomenologyConfig()
    components = (
        "phenomenology_bridge",
        "scalar_mixing_mechanism",
        "scalar_potential_and_cross_term",
    )

    return PsiHiggsLHCPhenomenologyFixtureResult(
        source_ledger_span=SOURCE_LEDGER_SPAN,
        hardware_status=HARDWARE_STATUS,
        claim_boundary=CLAIM_BOUNDARY,
        components={
            component: classify_psi_higgs_lhc_phenomenology_component(component)
            for component in components
        },
        labels=psi_higgs_lhc_phenomenology_labels(),
        source_record_count=cfg.expected_source_record_count,
        component_count=cfg.expected_component_count,
        next_source_boundary=cfg.next_source_boundary,
        null_controls={
            "lhc_phenomenology_bridge_is_not_observed_lhc_signal": 1.0,
            "scalar_mixing_claim_is_not_measured_higgs_admixture": 1.0,
            "higgs_portal_potential_is_not_fitted_collider_model": 1.0,
        },
        problem_metadata={
            "source_ledger_span": SOURCE_LEDGER_SPAN,
            "source_ledger_ids": tuple(f"P0R{number:05d}" for number in range(1655, 1669)),
            "claim_boundary": CLAIM_BOUNDARY,
            "protocol_state": "source_psi_higgs_lhc_phenomenology_only_no_experiment",
        },
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "HARDWARE_STATUS",
    "SOURCE_LEDGER_SPAN",
    "PsiHiggsLHCPhenomenologyConfig",
    "PsiHiggsLHCPhenomenologyFixtureResult",
    "classify_psi_higgs_lhc_phenomenology_component",
    "psi_higgs_lhc_phenomenology_labels",
    "validate_psi_higgs_lhc_phenomenology_fixture",
]
