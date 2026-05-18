# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 IV. Learning, Memory, and Plasticity: The Adaptive Scaffold (L1-L9) validation
"""Source-accounting checks for Paper 0 IV. Learning, Memory, and Plasticity: The Adaptive Scaffold (L1-L9) records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

CLAIM_BOUNDARY = "source-bounded iv learning memory and plasticity the adaptive scaffold l1 l9 source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"
SOURCE_LEDGER_SPAN = ("P0R05001", "P0R05008")


@dataclass(frozen=True, slots=True)
class IvLearningMemoryAndPlasticityTheAdaptiveScaffoldL1L9Config:
    """Configuration for this Paper 0 source-accounting fixture."""

    expected_source_record_count: int = 8
    expected_component_count: int = 3
    next_source_boundary: str = "P0R05009"

    def __post_init__(self) -> None:
        if self.expected_source_record_count != 8:
            raise ValueError("expected_source_record_count must equal 8")
        if self.expected_component_count != 3:
            raise ValueError("expected_component_count must equal 3")
        if self.next_source_boundary != "P0R05009":
            raise ValueError("next_source_boundary must equal P0R05009")


@dataclass(frozen=True, slots=True)
class IvLearningMemoryAndPlasticityTheAdaptiveScaffoldL1L9FixtureResult:
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


def classify_iv_learning_memory_and_plasticity_the_adaptive_scaffold_l1_l9_component(
    component: str,
) -> str:
    """Classify source-defined components."""
    mapping = {
        "iv_learning_memory_and_plasticity_the_adaptive_scaffold_l1_l9": "iv_learning_memory_and_plasticity_the_adaptive_scaffold_l1_l9_source_boundary",
        "1_the_multi_scale_memory_trace_the_engram": "1_the_multi_scale_memory_trace_the_engram_source_boundary",
        "2_the_mechanism_of_learning_hpc_optimisation_and_psi_guidance": "2_the_mechanism_of_learning_hpc_optimisation_and_psi_guidance_source_boundary",
    }
    try:
        return mapping[component]
    except KeyError as exc:
        raise ValueError(
            "unknown iv_learning_memory_and_plasticity_the_adaptive_scaffold_l1_l9 component"
        ) from exc


def iv_learning_memory_and_plasticity_the_adaptive_scaffold_l1_l9_labels() -> dict[str, str]:
    """Return source-bounded labels for this slice."""
    return {
        "section": "IV. Learning, Memory, and Plasticity: The Adaptive Scaffold (L1-L9)",
        "source_span": "P0R05001-P0R05008",
        "component_count": "3",
        "next_boundary": "P0R05009",
        "component_1": "IV. Learning, Memory, and Plasticity: The Adaptive Scaffold (L1-L9)",
        "component_2": "1. The Multi-Scale Memory Trace (The Engram):",
        "component_3": "2. The Mechanism of Learning (HPC Optimisation and Psi-Guidance):",
    }


def validate_iv_learning_memory_and_plasticity_the_adaptive_scaffold_l1_l9_fixture(
    config: IvLearningMemoryAndPlasticityTheAdaptiveScaffoldL1L9Config | None = None,
) -> IvLearningMemoryAndPlasticityTheAdaptiveScaffoldL1L9FixtureResult:
    """Validate source accounting for this Paper 0 slice."""
    cfg = config or IvLearningMemoryAndPlasticityTheAdaptiveScaffoldL1L9Config()
    components = (
        "iv_learning_memory_and_plasticity_the_adaptive_scaffold_l1_l9",
        "1_the_multi_scale_memory_trace_the_engram",
        "2_the_mechanism_of_learning_hpc_optimisation_and_psi_guidance",
    )
    return IvLearningMemoryAndPlasticityTheAdaptiveScaffoldL1L9FixtureResult(
        source_ledger_span=SOURCE_LEDGER_SPAN,
        hardware_status=HARDWARE_STATUS,
        claim_boundary=CLAIM_BOUNDARY,
        components={
            component: classify_iv_learning_memory_and_plasticity_the_adaptive_scaffold_l1_l9_component(
                component
            )
            for component in components
        },
        labels=iv_learning_memory_and_plasticity_the_adaptive_scaffold_l1_l9_labels(),
        source_record_count=cfg.expected_source_record_count,
        component_count=cfg.expected_component_count,
        next_source_boundary=cfg.next_source_boundary,
        null_controls={
            "iv_learning_memory_and_plasticity_the_adaptive_scaffold_l1_l9_is_not_empirical_validation_evidence": 1.0,
            "1_the_multi_scale_memory_trace_the_engram_is_not_empirical_validation_evidence": 1.0,
            "2_the_mechanism_of_learning_hpc_optimisation_and_psi_guidance_is_not_empirical_validation_evidence": 1.0,
        },
        problem_metadata={
            "source_ledger_span": SOURCE_LEDGER_SPAN,
            "source_ledger_ids": tuple(f"P0R{number:05d}" for number in range(5001, 5009)),
            "claim_boundary": CLAIM_BOUNDARY,
            "protocol_state": "source_iv_learning_memory_and_plasticity_the_adaptive_scaffold_l1_l9_only_no_experiment",
        },
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "HARDWARE_STATUS",
    "SOURCE_LEDGER_SPAN",
    "IvLearningMemoryAndPlasticityTheAdaptiveScaffoldL1L9Config",
    "IvLearningMemoryAndPlasticityTheAdaptiveScaffoldL1L9FixtureResult",
    "classify_iv_learning_memory_and_plasticity_the_adaptive_scaffold_l1_l9_component",
    "iv_learning_memory_and_plasticity_the_adaptive_scaffold_l1_l9_labels",
    "validate_iv_learning_memory_and_plasticity_the_adaptive_scaffold_l1_l9_fixture",
]
