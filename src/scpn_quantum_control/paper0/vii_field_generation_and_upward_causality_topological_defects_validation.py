# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 VII. Field Generation and Upward Causality (Topological Defects) validation
"""Source-accounting checks for Paper 0 VII. Field Generation and Upward Causality (Topological Defects) records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

CLAIM_BOUNDARY = "source-bounded vii field generation and upward causality topological defects source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"
SOURCE_LEDGER_SPAN = ("P0R03250", "P0R03259")


@dataclass(frozen=True, slots=True)
class ViiFieldGenerationAndUpwardCausalityTopologicalDefectsConfig:
    """Configuration for this Paper 0 source-accounting fixture."""

    expected_source_record_count: int = 10
    expected_component_count: int = 4
    next_source_boundary: str = "P0R03260"

    def __post_init__(self) -> None:
        if self.expected_source_record_count != 10:
            raise ValueError("expected_source_record_count must equal 10")
        if self.expected_component_count != 4:
            raise ValueError("expected_component_count must equal 4")
        if self.next_source_boundary != "P0R03260":
            raise ValueError("next_source_boundary must equal P0R03260")


@dataclass(frozen=True, slots=True)
class ViiFieldGenerationAndUpwardCausalityTopologicalDefectsFixtureResult:
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


def classify_vii_field_generation_and_upward_causality_topological_defects_component(
    component: str,
) -> str:
    """Classify source-defined components."""
    mapping = {
        "vii_field_generation_and_upward_causality_topological_defects": "vii_field_generation_and_upward_causality_topological_defects_source_boundary",
        "viii_the_combination_problem_and_panpsychist_fusion_quantum_mereology": "viii_the_combination_problem_and_panpsychist_fusion_quantum_mereology_source_boundary",
        "integrative_mechanisms_in_short": "integrative_mechanisms_in_short_source_boundary",
        "i_the_unifying_computational_principle_hpc": "i_the_unifying_computational_principle_hpc_source_boundary",
    }
    try:
        return mapping[component]
    except KeyError as exc:
        raise ValueError(
            "unknown vii_field_generation_and_upward_causality_topological_defects component"
        ) from exc


def vii_field_generation_and_upward_causality_topological_defects_labels() -> dict[str, str]:
    """Return source-bounded labels for this slice."""
    return {
        "section": "VII. Field Generation and Upward Causality (Topological Defects)",
        "source_span": "P0R03250-P0R03259",
        "component_count": "4",
        "next_boundary": "P0R03260",
        "component_1": "VII. Field Generation and Upward Causality (Topological Defects)",
        "component_2": "VIII. The Combination Problem and Panpsychist Fusion (Quantum Mereology)",
        "component_3": "Integrative Mechanisms in short",
        "component_4": "I. The Unifying Computational Principle (HPC)",
    }


def validate_vii_field_generation_and_upward_causality_topological_defects_fixture(
    config: ViiFieldGenerationAndUpwardCausalityTopologicalDefectsConfig | None = None,
) -> ViiFieldGenerationAndUpwardCausalityTopologicalDefectsFixtureResult:
    """Validate source accounting for this Paper 0 slice."""
    cfg = config or ViiFieldGenerationAndUpwardCausalityTopologicalDefectsConfig()
    components = (
        "vii_field_generation_and_upward_causality_topological_defects",
        "viii_the_combination_problem_and_panpsychist_fusion_quantum_mereology",
        "integrative_mechanisms_in_short",
        "i_the_unifying_computational_principle_hpc",
    )
    return ViiFieldGenerationAndUpwardCausalityTopologicalDefectsFixtureResult(
        source_ledger_span=SOURCE_LEDGER_SPAN,
        hardware_status=HARDWARE_STATUS,
        claim_boundary=CLAIM_BOUNDARY,
        components={
            component: classify_vii_field_generation_and_upward_causality_topological_defects_component(
                component
            )
            for component in components
        },
        labels=vii_field_generation_and_upward_causality_topological_defects_labels(),
        source_record_count=cfg.expected_source_record_count,
        component_count=cfg.expected_component_count,
        next_source_boundary=cfg.next_source_boundary,
        null_controls={
            "vii_field_generation_and_upward_causality_topological_defects_is_not_empirical_validation_evidence": 1.0,
            "viii_the_combination_problem_and_panpsychist_fusion_quantum_mereology_is_not_empirical_validation_evidence": 1.0,
            "integrative_mechanisms_in_short_is_not_empirical_validation_evidence": 1.0,
            "i_the_unifying_computational_principle_hpc_is_not_empirical_validation_evidence": 1.0,
        },
        problem_metadata={
            "source_ledger_span": SOURCE_LEDGER_SPAN,
            "source_ledger_ids": tuple(f"P0R{number:05d}" for number in range(3250, 3260)),
            "claim_boundary": CLAIM_BOUNDARY,
            "protocol_state": "source_vii_field_generation_and_upward_causality_topological_defects_only_no_experiment",
        },
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "HARDWARE_STATUS",
    "SOURCE_LEDGER_SPAN",
    "ViiFieldGenerationAndUpwardCausalityTopologicalDefectsConfig",
    "ViiFieldGenerationAndUpwardCausalityTopologicalDefectsFixtureResult",
    "classify_vii_field_generation_and_upward_causality_topological_defects_component",
    "vii_field_generation_and_upward_causality_topological_defects_labels",
    "validate_vii_field_generation_and_upward_causality_topological_defects_fixture",
]
