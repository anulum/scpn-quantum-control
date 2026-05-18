# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 Metastability and Chimaera States: The Nuance of Quasicriticality validation
"""Source-accounting checks for Paper 0 Metastability and Chimaera States: The Nuance of Quasicriticality records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

CLAIM_BOUNDARY = "source-bounded metastability and chimaera states the nuance of quasicriticality source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"
SOURCE_LEDGER_SPAN = ("P0R04581", "P0R04588")


@dataclass(frozen=True, slots=True)
class MetastabilityAndChimaeraStatesTheNuanceOfQuasicriticalityConfig:
    """Configuration for this Paper 0 source-accounting fixture."""

    expected_source_record_count: int = 8
    expected_component_count: int = 2
    next_source_boundary: str = "P0R04589"

    def __post_init__(self) -> None:
        if self.expected_source_record_count != 8:
            raise ValueError("expected_source_record_count must equal 8")
        if self.expected_component_count != 2:
            raise ValueError("expected_component_count must equal 2")
        if self.next_source_boundary != "P0R04589":
            raise ValueError("next_source_boundary must equal P0R04589")


@dataclass(frozen=True, slots=True)
class MetastabilityAndChimaeraStatesTheNuanceOfQuasicriticalityFixtureResult:
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


def classify_metastability_and_chimaera_states_the_nuance_of_quasicriticality_component(
    component: str,
) -> str:
    """Classify source-defined components."""
    mapping = {
        "metastability_and_chimaera_states_the_nuance_of_quasicriticality": "metastability_and_chimaera_states_the_nuance_of_quasicriticality_source_boundary",
        "the_dynamic_connectome_functional_reconfiguration_on_a_static_scaffold": "the_dynamic_connectome_functional_reconfiguration_on_a_static_scaffold_source_boundary",
    }
    try:
        return mapping[component]
    except KeyError as exc:
        raise ValueError(
            "unknown metastability_and_chimaera_states_the_nuance_of_quasicriticality component"
        ) from exc


def metastability_and_chimaera_states_the_nuance_of_quasicriticality_labels() -> dict[str, str]:
    """Return source-bounded labels for this slice."""
    return {
        "section": "Metastability and Chimaera States: The Nuance of Quasicriticality",
        "source_span": "P0R04581-P0R04588",
        "component_count": "2",
        "next_boundary": "P0R04589",
        "component_1": "Metastability and Chimaera States: The Nuance of Quasicriticality",
        "component_2": "The Dynamic Connectome: Functional Reconfiguration on a Static Scaffold",
    }


def validate_metastability_and_chimaera_states_the_nuance_of_quasicriticality_fixture(
    config: MetastabilityAndChimaeraStatesTheNuanceOfQuasicriticalityConfig | None = None,
) -> MetastabilityAndChimaeraStatesTheNuanceOfQuasicriticalityFixtureResult:
    """Validate source accounting for this Paper 0 slice."""
    cfg = config or MetastabilityAndChimaeraStatesTheNuanceOfQuasicriticalityConfig()
    components = (
        "metastability_and_chimaera_states_the_nuance_of_quasicriticality",
        "the_dynamic_connectome_functional_reconfiguration_on_a_static_scaffold",
    )
    return MetastabilityAndChimaeraStatesTheNuanceOfQuasicriticalityFixtureResult(
        source_ledger_span=SOURCE_LEDGER_SPAN,
        hardware_status=HARDWARE_STATUS,
        claim_boundary=CLAIM_BOUNDARY,
        components={
            component: classify_metastability_and_chimaera_states_the_nuance_of_quasicriticality_component(
                component
            )
            for component in components
        },
        labels=metastability_and_chimaera_states_the_nuance_of_quasicriticality_labels(),
        source_record_count=cfg.expected_source_record_count,
        component_count=cfg.expected_component_count,
        next_source_boundary=cfg.next_source_boundary,
        null_controls={
            "metastability_and_chimaera_states_the_nuance_of_quasicriticality_is_not_empirical_validation_evidence": 1.0,
            "the_dynamic_connectome_functional_reconfiguration_on_a_static_scaffold_is_not_empirical_validation_evidence": 1.0,
        },
        problem_metadata={
            "source_ledger_span": SOURCE_LEDGER_SPAN,
            "source_ledger_ids": tuple(f"P0R{number:05d}" for number in range(4581, 4589)),
            "claim_boundary": CLAIM_BOUNDARY,
            "protocol_state": "source_metastability_and_chimaera_states_the_nuance_of_quasicriticality_only_no_experiment",
        },
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "HARDWARE_STATUS",
    "SOURCE_LEDGER_SPAN",
    "MetastabilityAndChimaeraStatesTheNuanceOfQuasicriticalityConfig",
    "MetastabilityAndChimaeraStatesTheNuanceOfQuasicriticalityFixtureResult",
    "classify_metastability_and_chimaera_states_the_nuance_of_quasicriticality_component",
    "metastability_and_chimaera_states_the_nuance_of_quasicriticality_labels",
    "validate_metastability_and_chimaera_states_the_nuance_of_quasicriticality_fixture",
]
