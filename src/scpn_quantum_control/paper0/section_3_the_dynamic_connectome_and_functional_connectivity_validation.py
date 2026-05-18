# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 3. The Dynamic Connectome and Functional Connectivity: validation
"""Source-accounting checks for Paper 0 3. The Dynamic Connectome and Functional Connectivity: records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

CLAIM_BOUNDARY = "source-bounded section 3 the dynamic connectome and functional connectivity source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"
SOURCE_LEDGER_SPAN = ("P0R04674", "P0R04683")


@dataclass(frozen=True, slots=True)
class Section3TheDynamicConnectomeAndFunctionalConnectivityConfig:
    """Configuration for this Paper 0 source-accounting fixture."""

    expected_source_record_count: int = 10
    expected_component_count: int = 4
    next_source_boundary: str = "P0R04684"

    def __post_init__(self) -> None:
        if self.expected_source_record_count != 10:
            raise ValueError("expected_source_record_count must equal 10")
        if self.expected_component_count != 4:
            raise ValueError("expected_component_count must equal 4")
        if self.next_source_boundary != "P0R04684":
            raise ValueError("next_source_boundary must equal P0R04684")


@dataclass(frozen=True, slots=True)
class Section3TheDynamicConnectomeAndFunctionalConnectivityFixtureResult:
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


def classify_section_3_the_dynamic_connectome_and_functional_connectivity_component(
    component: str,
) -> str:
    """Classify source-defined components."""
    mapping = {
        "3_the_dynamic_connectome_and_functional_connectivity": "3_the_dynamic_connectome_and_functional_connectivity_source_boundary",
        "iv_the_architecture_of_the_conscious_self_domain_ii_l5": "iv_the_architecture_of_the_conscious_self_domain_ii_l5_source_boundary",
        "1_hierarchical_predictive_coding_hpc_and_the_canonical_microcircuit": "1_hierarchical_predictive_coding_hpc_and_the_canonical_microcircuit_source_boundary",
        "2_the_self_the_dmn_and_the_strange_loop": "2_the_self_the_dmn_and_the_strange_loop_source_boundary",
    }
    try:
        return mapping[component]
    except KeyError as exc:
        raise ValueError(
            "unknown section_3_the_dynamic_connectome_and_functional_connectivity component"
        ) from exc


def section_3_the_dynamic_connectome_and_functional_connectivity_labels() -> dict[str, str]:
    """Return source-bounded labels for this slice."""
    return {
        "section": "3. The Dynamic Connectome and Functional Connectivity:",
        "source_span": "P0R04674-P0R04683",
        "component_count": "4",
        "next_boundary": "P0R04684",
        "component_1": "3. The Dynamic Connectome and Functional Connectivity:",
        "component_2": "IV. The Architecture of the Conscious Self (Domain II: L5)",
        "component_3": "1. Hierarchical Predictive Coding (HPC) and the Canonical Microcircuit:",
        "component_4": "2. The Self, the DMN, and the Strange Loop:",
    }


def validate_section_3_the_dynamic_connectome_and_functional_connectivity_fixture(
    config: Section3TheDynamicConnectomeAndFunctionalConnectivityConfig | None = None,
) -> Section3TheDynamicConnectomeAndFunctionalConnectivityFixtureResult:
    """Validate source accounting for this Paper 0 slice."""
    cfg = config or Section3TheDynamicConnectomeAndFunctionalConnectivityConfig()
    components = (
        "3_the_dynamic_connectome_and_functional_connectivity",
        "iv_the_architecture_of_the_conscious_self_domain_ii_l5",
        "1_hierarchical_predictive_coding_hpc_and_the_canonical_microcircuit",
        "2_the_self_the_dmn_and_the_strange_loop",
    )
    return Section3TheDynamicConnectomeAndFunctionalConnectivityFixtureResult(
        source_ledger_span=SOURCE_LEDGER_SPAN,
        hardware_status=HARDWARE_STATUS,
        claim_boundary=CLAIM_BOUNDARY,
        components={
            component: classify_section_3_the_dynamic_connectome_and_functional_connectivity_component(
                component
            )
            for component in components
        },
        labels=section_3_the_dynamic_connectome_and_functional_connectivity_labels(),
        source_record_count=cfg.expected_source_record_count,
        component_count=cfg.expected_component_count,
        next_source_boundary=cfg.next_source_boundary,
        null_controls={
            "3_the_dynamic_connectome_and_functional_connectivity_is_not_empirical_validation_evidence": 1.0,
            "iv_the_architecture_of_the_conscious_self_domain_ii_l5_is_not_empirical_validation_evidence": 1.0,
            "1_hierarchical_predictive_coding_hpc_and_the_canonical_microcircuit_is_not_empirical_validation_evidence": 1.0,
            "2_the_self_the_dmn_and_the_strange_loop_is_not_empirical_validation_evidence": 1.0,
        },
        problem_metadata={
            "source_ledger_span": SOURCE_LEDGER_SPAN,
            "source_ledger_ids": tuple(f"P0R{number:05d}" for number in range(4674, 4684)),
            "claim_boundary": CLAIM_BOUNDARY,
            "protocol_state": "source_section_3_the_dynamic_connectome_and_functional_connectivity_only_no_experiment",
        },
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "HARDWARE_STATUS",
    "SOURCE_LEDGER_SPAN",
    "Section3TheDynamicConnectomeAndFunctionalConnectivityConfig",
    "Section3TheDynamicConnectomeAndFunctionalConnectivityFixtureResult",
    "classify_section_3_the_dynamic_connectome_and_functional_connectivity_component",
    "section_3_the_dynamic_connectome_and_functional_connectivity_labels",
    "validate_section_3_the_dynamic_connectome_and_functional_connectivity_fixture",
]
