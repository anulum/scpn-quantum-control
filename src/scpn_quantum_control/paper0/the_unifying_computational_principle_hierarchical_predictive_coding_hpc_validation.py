# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 The Unifying Computational Principle: Hierarchical Predictive Coding (HPC) validation
"""Source-accounting checks for Paper 0 The Unifying Computational Principle: Hierarchical Predictive Coding (HPC) records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

CLAIM_BOUNDARY = "source-bounded the unifying computational principle hierarchical predictive coding hpc source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"
SOURCE_LEDGER_SPAN = ("P0R06147", "P0R06155")


@dataclass(frozen=True, slots=True)
class TheUnifyingComputationalPrincipleHierarchicalPredictiveCodingHpcConfig:
    """Configuration for this Paper 0 source-accounting fixture."""

    expected_source_record_count: int = 9
    expected_component_count: int = 2
    next_source_boundary: str = "P0R06156"

    def __post_init__(self) -> None:
        if self.expected_source_record_count != 9:
            raise ValueError("expected_source_record_count must equal 9")
        if self.expected_component_count != 2:
            raise ValueError("expected_component_count must equal 2")
        if self.next_source_boundary != "P0R06156":
            raise ValueError("next_source_boundary must equal P0R06156")


@dataclass(frozen=True, slots=True)
class TheUnifyingComputationalPrincipleHierarchicalPredictiveCodingHpcFixtureResult:
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


def classify_the_unifying_computational_principle_hierarchical_predictive_coding_hpc_component(
    component: str,
) -> str:
    """Classify source-defined components."""
    mapping = {
        "the_unifying_computational_principle_hierarchical_predictive_coding_hpc": "the_unifying_computational_principle_hierarchical_predictive_coding_hpc_source_boundary",
        "i_the_free_energy_principle_the_imperative_to_minimise_surprise": "i_the_free_energy_principle_the_imperative_to_minimise_surprise_source_boundary",
    }
    try:
        return mapping[component]
    except KeyError as exc:
        raise ValueError(
            "unknown the_unifying_computational_principle_hierarchical_predictive_coding_hpc component"
        ) from exc


def the_unifying_computational_principle_hierarchical_predictive_coding_hpc_labels() -> dict[
    str, str
]:
    """Return source-bounded labels for this slice."""
    return {
        "section": "The Unifying Computational Principle: Hierarchical Predictive Coding (HPC)",
        "source_span": "P0R06147-P0R06155",
        "component_count": "2",
        "next_boundary": "P0R06156",
        "component_1": "The Unifying Computational Principle: Hierarchical Predictive Coding (HPC)",
        "component_2": "I. The Free Energy Principle: The Imperative to Minimise Surprise",
    }


def validate_the_unifying_computational_principle_hierarchical_predictive_coding_hpc_fixture(
    config: TheUnifyingComputationalPrincipleHierarchicalPredictiveCodingHpcConfig | None = None,
) -> TheUnifyingComputationalPrincipleHierarchicalPredictiveCodingHpcFixtureResult:
    """Validate source accounting for this Paper 0 slice."""
    cfg = config or TheUnifyingComputationalPrincipleHierarchicalPredictiveCodingHpcConfig()
    components = (
        "the_unifying_computational_principle_hierarchical_predictive_coding_hpc",
        "i_the_free_energy_principle_the_imperative_to_minimise_surprise",
    )
    return TheUnifyingComputationalPrincipleHierarchicalPredictiveCodingHpcFixtureResult(
        source_ledger_span=SOURCE_LEDGER_SPAN,
        hardware_status=HARDWARE_STATUS,
        claim_boundary=CLAIM_BOUNDARY,
        components={
            component: classify_the_unifying_computational_principle_hierarchical_predictive_coding_hpc_component(
                component
            )
            for component in components
        },
        labels=the_unifying_computational_principle_hierarchical_predictive_coding_hpc_labels(),
        source_record_count=cfg.expected_source_record_count,
        component_count=cfg.expected_component_count,
        next_source_boundary=cfg.next_source_boundary,
        null_controls={
            "the_unifying_computational_principle_hierarchical_predictive_coding_hpc_is_not_empirical_validation_evidence": 1.0,
            "i_the_free_energy_principle_the_imperative_to_minimise_surprise_is_not_empirical_validation_evidence": 1.0,
        },
        problem_metadata={
            "source_ledger_span": SOURCE_LEDGER_SPAN,
            "source_ledger_ids": tuple(f"P0R{number:05d}" for number in range(6147, 6156)),
            "claim_boundary": CLAIM_BOUNDARY,
            "protocol_state": "source_the_unifying_computational_principle_hierarchical_predictive_coding_hpc_only_no_experiment",
        },
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "HARDWARE_STATUS",
    "SOURCE_LEDGER_SPAN",
    "TheUnifyingComputationalPrincipleHierarchicalPredictiveCodingHpcConfig",
    "TheUnifyingComputationalPrincipleHierarchicalPredictiveCodingHpcFixtureResult",
    "classify_the_unifying_computational_principle_hierarchical_predictive_coding_hpc_component",
    "the_unifying_computational_principle_hierarchical_predictive_coding_hpc_labels",
    "validate_the_unifying_computational_principle_hierarchical_predictive_coding_hpc_fixture",
]
