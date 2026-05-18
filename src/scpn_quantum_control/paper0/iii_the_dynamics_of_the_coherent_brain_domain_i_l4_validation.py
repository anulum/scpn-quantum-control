# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 III. The Dynamics of the Coherent Brain (Domain I: L4) validation
"""Source-accounting checks for Paper 0 III. The Dynamics of the Coherent Brain (Domain I: L4) records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

CLAIM_BOUNDARY = "source-bounded iii the dynamics of the coherent brain domain i l4 source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"
SOURCE_LEDGER_SPAN = ("P0R04666", "P0R04673")


@dataclass(frozen=True, slots=True)
class IiiTheDynamicsOfTheCoherentBrainDomainIL4Config:
    """Configuration for this Paper 0 source-accounting fixture."""

    expected_source_record_count: int = 8
    expected_component_count: int = 3
    next_source_boundary: str = "P0R04674"

    def __post_init__(self) -> None:
        if self.expected_source_record_count != 8:
            raise ValueError("expected_source_record_count must equal 8")
        if self.expected_component_count != 3:
            raise ValueError("expected_component_count must equal 3")
        if self.next_source_boundary != "P0R04674":
            raise ValueError("next_source_boundary must equal P0R04674")


@dataclass(frozen=True, slots=True)
class IiiTheDynamicsOfTheCoherentBrainDomainIL4FixtureResult:
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


def classify_iii_the_dynamics_of_the_coherent_brain_domain_i_l4_component(component: str) -> str:
    """Classify source-defined components."""
    mapping = {
        "iii_the_dynamics_of_the_coherent_brain_domain_i_l4": "iii_the_dynamics_of_the_coherent_brain_domain_i_l4_source_boundary",
        "1_the_upde_and_oscillatory_hierarchies": "1_the_upde_and_oscillatory_hierarchies_source_boundary",
        "2_the_mechanisms_of_self_organised_criticality_soc": "2_the_mechanisms_of_self_organised_criticality_soc_source_boundary",
    }
    try:
        return mapping[component]
    except KeyError as exc:
        raise ValueError(
            "unknown iii_the_dynamics_of_the_coherent_brain_domain_i_l4 component"
        ) from exc


def iii_the_dynamics_of_the_coherent_brain_domain_i_l4_labels() -> dict[str, str]:
    """Return source-bounded labels for this slice."""
    return {
        "section": "III. The Dynamics of the Coherent Brain (Domain I: L4)",
        "source_span": "P0R04666-P0R04673",
        "component_count": "3",
        "next_boundary": "P0R04674",
        "component_1": "III. The Dynamics of the Coherent Brain (Domain I: L4)",
        "component_2": "1. The UPDE and Oscillatory Hierarchies:",
        "component_3": "2. The Mechanisms of Self-Organised Criticality (SOC):",
    }


def validate_iii_the_dynamics_of_the_coherent_brain_domain_i_l4_fixture(
    config: IiiTheDynamicsOfTheCoherentBrainDomainIL4Config | None = None,
) -> IiiTheDynamicsOfTheCoherentBrainDomainIL4FixtureResult:
    """Validate source accounting for this Paper 0 slice."""
    cfg = config or IiiTheDynamicsOfTheCoherentBrainDomainIL4Config()
    components = (
        "iii_the_dynamics_of_the_coherent_brain_domain_i_l4",
        "1_the_upde_and_oscillatory_hierarchies",
        "2_the_mechanisms_of_self_organised_criticality_soc",
    )
    return IiiTheDynamicsOfTheCoherentBrainDomainIL4FixtureResult(
        source_ledger_span=SOURCE_LEDGER_SPAN,
        hardware_status=HARDWARE_STATUS,
        claim_boundary=CLAIM_BOUNDARY,
        components={
            component: classify_iii_the_dynamics_of_the_coherent_brain_domain_i_l4_component(
                component
            )
            for component in components
        },
        labels=iii_the_dynamics_of_the_coherent_brain_domain_i_l4_labels(),
        source_record_count=cfg.expected_source_record_count,
        component_count=cfg.expected_component_count,
        next_source_boundary=cfg.next_source_boundary,
        null_controls={
            "iii_the_dynamics_of_the_coherent_brain_domain_i_l4_is_not_empirical_validation_evidence": 1.0,
            "1_the_upde_and_oscillatory_hierarchies_is_not_empirical_validation_evidence": 1.0,
            "2_the_mechanisms_of_self_organised_criticality_soc_is_not_empirical_validation_evidence": 1.0,
        },
        problem_metadata={
            "source_ledger_span": SOURCE_LEDGER_SPAN,
            "source_ledger_ids": tuple(f"P0R{number:05d}" for number in range(4666, 4674)),
            "claim_boundary": CLAIM_BOUNDARY,
            "protocol_state": "source_iii_the_dynamics_of_the_coherent_brain_domain_i_l4_only_no_experiment",
        },
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "HARDWARE_STATUS",
    "SOURCE_LEDGER_SPAN",
    "IiiTheDynamicsOfTheCoherentBrainDomainIL4Config",
    "IiiTheDynamicsOfTheCoherentBrainDomainIL4FixtureResult",
    "classify_iii_the_dynamics_of_the_coherent_brain_domain_i_l4_component",
    "iii_the_dynamics_of_the_coherent_brain_domain_i_l4_labels",
    "validate_iii_the_dynamics_of_the_coherent_brain_domain_i_l4_fixture",
]
