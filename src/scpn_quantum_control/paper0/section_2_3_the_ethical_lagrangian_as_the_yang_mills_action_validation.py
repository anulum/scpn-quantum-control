# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 2.3. The Ethical Lagrangian as the Yang-Mills Action validation
"""Source-accounting checks for Paper 0 2.3. The Ethical Lagrangian as the Yang-Mills Action records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

CLAIM_BOUNDARY = "source-bounded section 2 3 the ethical lagrangian as the yang mills action source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"
SOURCE_LEDGER_SPAN = ("P0R03622", "P0R03629")


@dataclass(frozen=True, slots=True)
class Section23TheEthicalLagrangianAsTheYangMillsActionConfig:
    """Configuration for this Paper 0 source-accounting fixture."""

    expected_source_record_count: int = 8
    expected_component_count: int = 2
    next_source_boundary: str = "P0R03630"

    def __post_init__(self) -> None:
        if self.expected_source_record_count != 8:
            raise ValueError("expected_source_record_count must equal 8")
        if self.expected_component_count != 2:
            raise ValueError("expected_component_count must equal 2")
        if self.next_source_boundary != "P0R03630":
            raise ValueError("next_source_boundary must equal P0R03630")


@dataclass(frozen=True, slots=True)
class Section23TheEthicalLagrangianAsTheYangMillsActionFixtureResult:
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


def classify_section_2_3_the_ethical_lagrangian_as_the_yang_mills_action_component(
    component: str,
) -> str:
    """Classify source-defined components."""
    mapping = {
        "2_3_the_ethical_lagrangian_as_the_yang_mills_action": "2_3_the_ethical_lagrangian_as_the_yang_mills_action_source_boundary",
        "section_3_the_conserved_ethical_charge_and_its_physical_basis": "section_3_the_conserved_ethical_charge_and_its_physical_basis_source_boundary",
    }
    try:
        return mapping[component]
    except KeyError as exc:
        raise ValueError(
            "unknown section_2_3_the_ethical_lagrangian_as_the_yang_mills_action component"
        ) from exc


def section_2_3_the_ethical_lagrangian_as_the_yang_mills_action_labels() -> dict[str, str]:
    """Return source-bounded labels for this slice."""
    return {
        "section": "2.3. The Ethical Lagrangian as the Yang-Mills Action",
        "source_span": "P0R03622-P0R03629",
        "component_count": "2",
        "next_boundary": "P0R03630",
        "component_1": "2.3. The Ethical Lagrangian as the Yang-Mills Action",
        "component_2": 'Section 3: The Conserved "Ethical Charge" and its Physical Basis',
    }


def validate_section_2_3_the_ethical_lagrangian_as_the_yang_mills_action_fixture(
    config: Section23TheEthicalLagrangianAsTheYangMillsActionConfig | None = None,
) -> Section23TheEthicalLagrangianAsTheYangMillsActionFixtureResult:
    """Validate source accounting for this Paper 0 slice."""
    cfg = config or Section23TheEthicalLagrangianAsTheYangMillsActionConfig()
    components = (
        "2_3_the_ethical_lagrangian_as_the_yang_mills_action",
        "section_3_the_conserved_ethical_charge_and_its_physical_basis",
    )
    return Section23TheEthicalLagrangianAsTheYangMillsActionFixtureResult(
        source_ledger_span=SOURCE_LEDGER_SPAN,
        hardware_status=HARDWARE_STATUS,
        claim_boundary=CLAIM_BOUNDARY,
        components={
            component: classify_section_2_3_the_ethical_lagrangian_as_the_yang_mills_action_component(
                component
            )
            for component in components
        },
        labels=section_2_3_the_ethical_lagrangian_as_the_yang_mills_action_labels(),
        source_record_count=cfg.expected_source_record_count,
        component_count=cfg.expected_component_count,
        next_source_boundary=cfg.next_source_boundary,
        null_controls={
            "2_3_the_ethical_lagrangian_as_the_yang_mills_action_is_not_empirical_validation_evidence": 1.0,
            "section_3_the_conserved_ethical_charge_and_its_physical_basis_is_not_empirical_validation_evidence": 1.0,
        },
        problem_metadata={
            "source_ledger_span": SOURCE_LEDGER_SPAN,
            "source_ledger_ids": tuple(f"P0R{number:05d}" for number in range(3622, 3630)),
            "claim_boundary": CLAIM_BOUNDARY,
            "protocol_state": "source_section_2_3_the_ethical_lagrangian_as_the_yang_mills_action_only_no_experiment",
        },
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "HARDWARE_STATUS",
    "SOURCE_LEDGER_SPAN",
    "Section23TheEthicalLagrangianAsTheYangMillsActionConfig",
    "Section23TheEthicalLagrangianAsTheYangMillsActionFixtureResult",
    "classify_section_2_3_the_ethical_lagrangian_as_the_yang_mills_action_component",
    "section_2_3_the_ethical_lagrangian_as_the_yang_mills_action_labels",
    "validate_section_2_3_the_ethical_lagrangian_as_the_yang_mills_action_fixture",
]
