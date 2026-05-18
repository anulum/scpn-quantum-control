# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 II. Hierarchical Predictive Coding: The SCPN's Computational Algorithm validation
"""Source-accounting checks for Paper 0 II. Hierarchical Predictive Coding: The SCPN's Computational Algorithm records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

CLAIM_BOUNDARY = "source-bounded ii hierarchical predictive coding the scpn s computational algorithm source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"
SOURCE_LEDGER_SPAN = ("P0R06156", "P0R06163")


@dataclass(frozen=True, slots=True)
class IiHierarchicalPredictiveCodingTheScpnSComputationalAlgorithmConfig:
    """Configuration for this Paper 0 source-accounting fixture."""

    expected_source_record_count: int = 8
    expected_component_count: int = 2
    next_source_boundary: str = "P0R06164"

    def __post_init__(self) -> None:
        if self.expected_source_record_count != 8:
            raise ValueError("expected_source_record_count must equal 8")
        if self.expected_component_count != 2:
            raise ValueError("expected_component_count must equal 2")
        if self.next_source_boundary != "P0R06164":
            raise ValueError("next_source_boundary must equal P0R06164")


@dataclass(frozen=True, slots=True)
class IiHierarchicalPredictiveCodingTheScpnSComputationalAlgorithmFixtureResult:
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


def classify_ii_hierarchical_predictive_coding_the_scpn_s_computational_algorithm_component(
    component: str,
) -> str:
    """Classify source-defined components."""
    mapping = {
        "ii_hierarchical_predictive_coding_the_scpn_s_computational_algorithm": "ii_hierarchical_predictive_coding_the_scpn_s_computational_algorithm_source_boundary",
        "iii_the_upde_as_the_physical_implementation_of_free_energy_minimisation": "iii_the_upde_as_the_physical_implementation_of_free_energy_minimisation_source_boundary",
    }
    try:
        return mapping[component]
    except KeyError as exc:
        raise ValueError(
            "unknown ii_hierarchical_predictive_coding_the_scpn_s_computational_algorithm component"
        ) from exc


def ii_hierarchical_predictive_coding_the_scpn_s_computational_algorithm_labels() -> dict[
    str, str
]:
    """Return source-bounded labels for this slice."""
    return {
        "section": "II. Hierarchical Predictive Coding: The SCPN's Computational Algorithm",
        "source_span": "P0R06156-P0R06163",
        "component_count": "2",
        "next_boundary": "P0R06164",
        "component_1": "II. Hierarchical Predictive Coding: The SCPN's Computational Algorithm",
        "component_2": "III. The UPDE as the Physical Implementation of Free Energy Minimisation",
    }


def validate_ii_hierarchical_predictive_coding_the_scpn_s_computational_algorithm_fixture(
    config: IiHierarchicalPredictiveCodingTheScpnSComputationalAlgorithmConfig | None = None,
) -> IiHierarchicalPredictiveCodingTheScpnSComputationalAlgorithmFixtureResult:
    """Validate source accounting for this Paper 0 slice."""
    cfg = config or IiHierarchicalPredictiveCodingTheScpnSComputationalAlgorithmConfig()
    components = (
        "ii_hierarchical_predictive_coding_the_scpn_s_computational_algorithm",
        "iii_the_upde_as_the_physical_implementation_of_free_energy_minimisation",
    )
    return IiHierarchicalPredictiveCodingTheScpnSComputationalAlgorithmFixtureResult(
        source_ledger_span=SOURCE_LEDGER_SPAN,
        hardware_status=HARDWARE_STATUS,
        claim_boundary=CLAIM_BOUNDARY,
        components={
            component: classify_ii_hierarchical_predictive_coding_the_scpn_s_computational_algorithm_component(
                component
            )
            for component in components
        },
        labels=ii_hierarchical_predictive_coding_the_scpn_s_computational_algorithm_labels(),
        source_record_count=cfg.expected_source_record_count,
        component_count=cfg.expected_component_count,
        next_source_boundary=cfg.next_source_boundary,
        null_controls={
            "ii_hierarchical_predictive_coding_the_scpn_s_computational_algorithm_is_not_empirical_validation_evidence": 1.0,
            "iii_the_upde_as_the_physical_implementation_of_free_energy_minimisation_is_not_empirical_validation_evidence": 1.0,
        },
        problem_metadata={
            "source_ledger_span": SOURCE_LEDGER_SPAN,
            "source_ledger_ids": tuple(f"P0R{number:05d}" for number in range(6156, 6164)),
            "claim_boundary": CLAIM_BOUNDARY,
            "protocol_state": "source_ii_hierarchical_predictive_coding_the_scpn_s_computational_algorithm_only_no_experiment",
        },
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "HARDWARE_STATUS",
    "SOURCE_LEDGER_SPAN",
    "IiHierarchicalPredictiveCodingTheScpnSComputationalAlgorithmConfig",
    "IiHierarchicalPredictiveCodingTheScpnSComputationalAlgorithmFixtureResult",
    "classify_ii_hierarchical_predictive_coding_the_scpn_s_computational_algorithm_component",
    "ii_hierarchical_predictive_coding_the_scpn_s_computational_algorithm_labels",
    "validate_ii_hierarchical_predictive_coding_the_scpn_s_computational_algorithm_fixture",
]
