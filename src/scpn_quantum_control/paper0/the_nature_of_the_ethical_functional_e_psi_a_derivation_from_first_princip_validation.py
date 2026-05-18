# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 The Nature of the Ethical Functional E[Psi]: A Derivation from First Principles validation
"""Source-accounting checks for Paper 0 The Nature of the Ethical Functional E[Psi]: A Derivation from First Principles records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

CLAIM_BOUNDARY = "source-bounded the nature of the ethical functional e psi a derivation from first princip source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"
SOURCE_LEDGER_SPAN = ("P0R03603", "P0R03611")


@dataclass(frozen=True, slots=True)
class TheNatureOfTheEthicalFunctionalEPsiADerivationFromFirstPrincipConfig:
    """Configuration for this Paper 0 source-accounting fixture."""

    expected_source_record_count: int = 9
    expected_component_count: int = 2
    next_source_boundary: str = "P0R03612"

    def __post_init__(self) -> None:
        if self.expected_source_record_count != 9:
            raise ValueError("expected_source_record_count must equal 9")
        if self.expected_component_count != 2:
            raise ValueError("expected_component_count must equal 2")
        if self.next_source_boundary != "P0R03612":
            raise ValueError("next_source_boundary must equal P0R03612")


@dataclass(frozen=True, slots=True)
class TheNatureOfTheEthicalFunctionalEPsiADerivationFromFirstPrincipFixtureResult:
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


def classify_the_nature_of_the_ethical_functional_e_psi_a_derivation_from_first_princip_component(
    component: str,
) -> str:
    """Classify source-defined components."""
    mapping = {
        "the_nature_of_the_ethical_functional_e_psi_a_derivation_from_first_princ": "the_nature_of_the_ethical_functional_e_psi_a_derivation_from_first_princ_source_boundary",
        "section_1_avoiding_the_category_error_from_philosophy_to_physics": "section_1_avoiding_the_category_error_from_philosophy_to_physics_source_boundary",
    }
    try:
        return mapping[component]
    except KeyError as exc:
        raise ValueError(
            "unknown the_nature_of_the_ethical_functional_e_psi_a_derivation_from_first_princip component"
        ) from exc


def the_nature_of_the_ethical_functional_e_psi_a_derivation_from_first_princip_labels() -> dict[
    str, str
]:
    """Return source-bounded labels for this slice."""
    return {
        "section": "The Nature of the Ethical Functional E[Psi]: A Derivation from First Principles",
        "source_span": "P0R03603-P0R03611",
        "component_count": "2",
        "next_boundary": "P0R03612",
        "component_1": "The Nature of the Ethical Functional E[Psi]: A Derivation from First Principles",
        "component_2": "Section 1: Avoiding the Category Error: From Philosophy to Physics",
    }


def validate_the_nature_of_the_ethical_functional_e_psi_a_derivation_from_first_princip_fixture(
    config: TheNatureOfTheEthicalFunctionalEPsiADerivationFromFirstPrincipConfig | None = None,
) -> TheNatureOfTheEthicalFunctionalEPsiADerivationFromFirstPrincipFixtureResult:
    """Validate source accounting for this Paper 0 slice."""
    cfg = config or TheNatureOfTheEthicalFunctionalEPsiADerivationFromFirstPrincipConfig()
    components = (
        "the_nature_of_the_ethical_functional_e_psi_a_derivation_from_first_princ",
        "section_1_avoiding_the_category_error_from_philosophy_to_physics",
    )
    return TheNatureOfTheEthicalFunctionalEPsiADerivationFromFirstPrincipFixtureResult(
        source_ledger_span=SOURCE_LEDGER_SPAN,
        hardware_status=HARDWARE_STATUS,
        claim_boundary=CLAIM_BOUNDARY,
        components={
            component: classify_the_nature_of_the_ethical_functional_e_psi_a_derivation_from_first_princip_component(
                component
            )
            for component in components
        },
        labels=the_nature_of_the_ethical_functional_e_psi_a_derivation_from_first_princip_labels(),
        source_record_count=cfg.expected_source_record_count,
        component_count=cfg.expected_component_count,
        next_source_boundary=cfg.next_source_boundary,
        null_controls={
            "the_nature_of_the_ethical_functional_e_psi_a_derivation_from_first_princ_is_not_empirical_validation_evidence": 1.0,
            "section_1_avoiding_the_category_error_from_philosophy_to_physics_is_not_empirical_validation_evidence": 1.0,
        },
        problem_metadata={
            "source_ledger_span": SOURCE_LEDGER_SPAN,
            "source_ledger_ids": tuple(f"P0R{number:05d}" for number in range(3603, 3612)),
            "claim_boundary": CLAIM_BOUNDARY,
            "protocol_state": "source_the_nature_of_the_ethical_functional_e_psi_a_derivation_from_first_princip_only_no_experiment",
        },
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "HARDWARE_STATUS",
    "SOURCE_LEDGER_SPAN",
    "TheNatureOfTheEthicalFunctionalEPsiADerivationFromFirstPrincipConfig",
    "TheNatureOfTheEthicalFunctionalEPsiADerivationFromFirstPrincipFixtureResult",
    "classify_the_nature_of_the_ethical_functional_e_psi_a_derivation_from_first_princip_component",
    "the_nature_of_the_ethical_functional_e_psi_a_derivation_from_first_princip_labels",
    "validate_the_nature_of_the_ethical_functional_e_psi_a_derivation_from_first_princip_fixture",
]
