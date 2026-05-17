# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 infoton-properties derivation validation
"""Source-accounting checks for Paper 0 infoton-properties derivation records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

CLAIM_BOUNDARY = "source-bounded infoton-properties derivation bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"
SOURCE_LEDGER_SPAN = ("P0R01623", "P0R01637")


@dataclass(frozen=True, slots=True)
class DerivationInfotonPropertiesConfig:
    """Configuration for the infoton-properties derivation fixture."""

    expected_source_record_count: int = 15
    expected_component_count: int = 4
    next_source_boundary: str = "P0R01638"

    def __post_init__(self) -> None:
        if self.expected_source_record_count != 15:
            raise ValueError("expected_source_record_count must equal 15")
        if self.expected_component_count != 4:
            raise ValueError("expected_component_count must equal 4")
        if self.next_source_boundary != "P0R01638":
            raise ValueError("next_source_boundary must equal P0R01638")


@dataclass(frozen=True, slots=True)
class DerivationInfotonPropertiesFixtureResult:
    """Result for the Paper 0 infoton-properties derivation fixture."""

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


def classify_derivation_infoton_properties_component(component: str) -> str:
    """Classify source-defined infoton-properties derivation components."""
    mapping = {
        "lagrangian_and_potential": "u1_lagrangian_potential_derivation_boundary",
        "vev_and_goldstone_absorption": "vev_goldstone_absorption_source_boundary",
        "mass_identification": "infoton_mass_identification_source_boundary",
        "range_consequence": "short_range_informational_force_consequence_boundary",
    }
    try:
        return mapping[component]
    except KeyError as exc:
        raise ValueError("unknown infoton-properties derivation component") from exc


def derivation_infoton_properties_labels() -> dict[str, str]:
    """Return source-bounded labels for the infoton-properties derivation slice."""
    return {
        "section": "Derivation of the Infoton's Properties",
        "lagrangian": "L = |D_mu Psi|^2 - V(|Psi|) - 1/4 F_mu_nu F^mu_nu",
        "potential": "V(|Psi|) = -mu^2 |Psi|^2 + lambda |Psi|^4",
        "mass": "m_A = g v",
        "range": "lambda_range ~= hbar / (m_A c)",
        "next_boundary": "The Psi-Higgs Boson: A New Scalar Particle",
    }


def validate_derivation_infoton_properties_fixture(
    config: DerivationInfotonPropertiesConfig | None = None,
) -> DerivationInfotonPropertiesFixtureResult:
    """Validate source accounting for the infoton-properties derivation slice."""
    cfg = config or DerivationInfotonPropertiesConfig()
    components = (
        "lagrangian_and_potential",
        "vev_and_goldstone_absorption",
        "mass_identification",
        "range_consequence",
    )

    return DerivationInfotonPropertiesFixtureResult(
        source_ledger_span=SOURCE_LEDGER_SPAN,
        hardware_status=HARDWARE_STATUS,
        claim_boundary=CLAIM_BOUNDARY,
        components={
            component: classify_derivation_infoton_properties_component(component)
            for component in components
        },
        labels=derivation_infoton_properties_labels(),
        source_record_count=cfg.expected_source_record_count,
        component_count=cfg.expected_component_count,
        next_source_boundary=cfg.next_source_boundary,
        null_controls={
            "lagrangian_derivation_is_not_detector_evidence": 1.0,
            "goldstone_absorption_is_source_derivation_not_observed_event": 1.0,
            "infoton_mass_relation_is_not_measured_mass": 1.0,
            "force_range_relation_is_not_measured_force_profile": 1.0,
        },
        problem_metadata={
            "source_ledger_span": SOURCE_LEDGER_SPAN,
            "source_ledger_ids": tuple(f"P0R{number:05d}" for number in range(1623, 1638)),
            "claim_boundary": CLAIM_BOUNDARY,
            "protocol_state": "source_derivation_infoton_properties_only_no_experiment",
        },
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "HARDWARE_STATUS",
    "SOURCE_LEDGER_SPAN",
    "DerivationInfotonPropertiesConfig",
    "DerivationInfotonPropertiesFixtureResult",
    "classify_derivation_infoton_properties_component",
    "derivation_infoton_properties_labels",
    "validate_derivation_infoton_properties_fixture",
]
