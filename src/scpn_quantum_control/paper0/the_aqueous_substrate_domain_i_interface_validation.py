# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 The Aqueous Substrate (Domain I Interface) validation
"""Source-accounting checks for Paper 0 The Aqueous Substrate (Domain I Interface) records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

CLAIM_BOUNDARY = "source-bounded the aqueous substrate domain i interface source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"
SOURCE_LEDGER_SPAN = ("P0R05331", "P0R05346")


@dataclass(frozen=True, slots=True)
class TheAqueousSubstrateDomainIInterfaceConfig:
    """Configuration for this Paper 0 source-accounting fixture."""

    expected_source_record_count: int = 16
    expected_component_count: int = 4
    next_source_boundary: str = "P0R05347"

    def __post_init__(self) -> None:
        if self.expected_source_record_count != 16:
            raise ValueError("expected_source_record_count must equal 16")
        if self.expected_component_count != 4:
            raise ValueError("expected_component_count must equal 4")
        if self.next_source_boundary != "P0R05347":
            raise ValueError("next_source_boundary must equal P0R05347")


@dataclass(frozen=True, slots=True)
class TheAqueousSubstrateDomainIInterfaceFixtureResult:
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


def classify_the_aqueous_substrate_domain_i_interface_component(component: str) -> str:
    """Classify source-defined components."""
    mapping = {
        "the_aqueous_substrate_domain_i_interface": "the_aqueous_substrate_domain_i_interface_source_boundary",
        "coherence_domains_cds_predicted_by_qed_interfacial_water_forms_cds_where": "coherence_domains_cds_predicted_by_qed_interfacial_water_forms_cds_where_source_boundary",
        "integration_in_l1_cds_shield_microtubule_qubits_and_support_frhlich_cond": "integration_in_l1_cds_shield_microtubule_qubits_and_support_frhlich_cond_source_boundary",
        "the_genesis_of_life_abiogenesis_as_a_guided_phase_transition": "the_genesis_of_life_abiogenesis_as_a_guided_phase_transition_source_boundary",
    }
    try:
        return mapping[component]
    except KeyError as exc:
        raise ValueError("unknown the_aqueous_substrate_domain_i_interface component") from exc


def the_aqueous_substrate_domain_i_interface_labels() -> dict[str, str]:
    """Return source-bounded labels for this slice."""
    return {
        "section": "The Aqueous Substrate (Domain I Interface)",
        "source_span": "P0R05331-P0R05346",
        "component_count": "4",
        "next_boundary": "P0R05347",
        "component_1": "The Aqueous Substrate (Domain I Interface)",
        "component_2": "Coherence Domains (CDs): Predicted by QED, Interfacial water forms CDs where molecules oscillate in phase. This facilitates quasi-superconductivity (proton hopping) and stabilises quantum states.",
        "component_3": "Integration: In L1, CDs shield microtubule qubits and support Frhlich condensation. In L3/L4, bioelectric codes are mediated by proton currents (IProton) within this network. The network of CDs acts as a dynamic memory medium enabling rapid signalling.",
        "component_4": "The Genesis of Life - Abiogenesis as a Guided Phase Transition",
    }


def validate_the_aqueous_substrate_domain_i_interface_fixture(
    config: TheAqueousSubstrateDomainIInterfaceConfig | None = None,
) -> TheAqueousSubstrateDomainIInterfaceFixtureResult:
    """Validate source accounting for this Paper 0 slice."""
    cfg = config or TheAqueousSubstrateDomainIInterfaceConfig()
    components = (
        "the_aqueous_substrate_domain_i_interface",
        "coherence_domains_cds_predicted_by_qed_interfacial_water_forms_cds_where",
        "integration_in_l1_cds_shield_microtubule_qubits_and_support_frhlich_cond",
        "the_genesis_of_life_abiogenesis_as_a_guided_phase_transition",
    )
    return TheAqueousSubstrateDomainIInterfaceFixtureResult(
        source_ledger_span=SOURCE_LEDGER_SPAN,
        hardware_status=HARDWARE_STATUS,
        claim_boundary=CLAIM_BOUNDARY,
        components={
            component: classify_the_aqueous_substrate_domain_i_interface_component(component)
            for component in components
        },
        labels=the_aqueous_substrate_domain_i_interface_labels(),
        source_record_count=cfg.expected_source_record_count,
        component_count=cfg.expected_component_count,
        next_source_boundary=cfg.next_source_boundary,
        null_controls={
            "the_aqueous_substrate_domain_i_interface_is_not_empirical_validation_evidence": 1.0,
            "coherence_domains_cds_predicted_by_qed_interfacial_water_forms_cds_where_is_not_empirical_validation_evidence": 1.0,
            "integration_in_l1_cds_shield_microtubule_qubits_and_support_frhlich_cond_is_not_empirical_validation_evidence": 1.0,
            "the_genesis_of_life_abiogenesis_as_a_guided_phase_transition_is_not_empirical_validation_evidence": 1.0,
        },
        problem_metadata={
            "source_ledger_span": SOURCE_LEDGER_SPAN,
            "source_ledger_ids": tuple(f"P0R{number:05d}" for number in range(5331, 5347)),
            "claim_boundary": CLAIM_BOUNDARY,
            "protocol_state": "source_the_aqueous_substrate_domain_i_interface_only_no_experiment",
        },
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "HARDWARE_STATUS",
    "SOURCE_LEDGER_SPAN",
    "TheAqueousSubstrateDomainIInterfaceConfig",
    "TheAqueousSubstrateDomainIInterfaceFixtureResult",
    "classify_the_aqueous_substrate_domain_i_interface_component",
    "the_aqueous_substrate_domain_i_interface_labels",
    "validate_the_aqueous_substrate_domain_i_interface_fixture",
]
