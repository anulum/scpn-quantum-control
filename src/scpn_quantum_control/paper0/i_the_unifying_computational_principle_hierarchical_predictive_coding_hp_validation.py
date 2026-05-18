# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 I. The Unifying Computational Principle: Hierarchical Predictive Coding (HPC) validation
"""Source-accounting checks for Paper 0 I. The Unifying Computational Principle: Hierarchical Predictive Coding (HPC) records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

CLAIM_BOUNDARY = "source-bounded i the unifying computational principle hierarchical predictive coding hp source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"
SOURCE_LEDGER_SPAN = ("P0R03197", "P0R03207")


@dataclass(frozen=True, slots=True)
class ITheUnifyingComputationalPrincipleHierarchicalPredictiveCodingHpConfig:
    """Configuration for this Paper 0 source-accounting fixture."""

    expected_source_record_count: int = 11
    expected_component_count: int = 4
    next_source_boundary: str = "P0R03208"

    def __post_init__(self) -> None:
        if self.expected_source_record_count != 11:
            raise ValueError("expected_source_record_count must equal 11")
        if self.expected_component_count != 4:
            raise ValueError("expected_component_count must equal 4")
        if self.next_source_boundary != "P0R03208":
            raise ValueError("next_source_boundary must equal P0R03208")


@dataclass(frozen=True, slots=True)
class ITheUnifyingComputationalPrincipleHierarchicalPredictiveCodingHpFixtureResult:
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


def classify_i_the_unifying_computational_principle_hierarchical_predictive_coding_hp_component(
    component: str,
) -> str:
    """Classify source-defined components."""
    mapping = {
        "i_the_unifying_computational_principle_hierarchical_predictive_coding_hp": "i_the_unifying_computational_principle_hierarchical_predictive_coding_hp_source_boundary",
        "1_the_generative_model_downward_projection": "1_the_generative_model_downward_projection_source_boundary",
        "2_inference_and_error_upward_filtering": "2_inference_and_error_upward_filtering_source_boundary",
        "3_optimisation_free_energy_minimisation": "3_optimisation_free_energy_minimisation_source_boundary",
    }
    try:
        return mapping[component]
    except KeyError as exc:
        raise ValueError(
            "unknown i_the_unifying_computational_principle_hierarchical_predictive_coding_hp component"
        ) from exc


def i_the_unifying_computational_principle_hierarchical_predictive_coding_hp_labels() -> dict[
    str, str
]:
    """Return source-bounded labels for this slice."""
    return {
        "section": "I. The Unifying Computational Principle: Hierarchical Predictive Coding (HPC)",
        "source_span": "P0R03197-P0R03207",
        "component_count": "4",
        "next_boundary": "P0R03208",
        "component_1": "I. The Unifying Computational Principle: Hierarchical Predictive Coding (HPC)",
        "component_2": "1. The Generative Model (Downward Projection):",
        "component_3": "2. Inference and Error (Upward Filtering):",
        "component_4": "3. Optimisation (Free Energy Minimisation):",
    }


def validate_i_the_unifying_computational_principle_hierarchical_predictive_coding_hp_fixture(
    config: ITheUnifyingComputationalPrincipleHierarchicalPredictiveCodingHpConfig | None = None,
) -> ITheUnifyingComputationalPrincipleHierarchicalPredictiveCodingHpFixtureResult:
    """Validate source accounting for this Paper 0 slice."""
    cfg = config or ITheUnifyingComputationalPrincipleHierarchicalPredictiveCodingHpConfig()
    components = (
        "i_the_unifying_computational_principle_hierarchical_predictive_coding_hp",
        "1_the_generative_model_downward_projection",
        "2_inference_and_error_upward_filtering",
        "3_optimisation_free_energy_minimisation",
    )
    return ITheUnifyingComputationalPrincipleHierarchicalPredictiveCodingHpFixtureResult(
        source_ledger_span=SOURCE_LEDGER_SPAN,
        hardware_status=HARDWARE_STATUS,
        claim_boundary=CLAIM_BOUNDARY,
        components={
            component: classify_i_the_unifying_computational_principle_hierarchical_predictive_coding_hp_component(
                component
            )
            for component in components
        },
        labels=i_the_unifying_computational_principle_hierarchical_predictive_coding_hp_labels(),
        source_record_count=cfg.expected_source_record_count,
        component_count=cfg.expected_component_count,
        next_source_boundary=cfg.next_source_boundary,
        null_controls={
            "i_the_unifying_computational_principle_hierarchical_predictive_coding_hp_is_not_empirical_validation_evidence": 1.0,
            "1_the_generative_model_downward_projection_is_not_empirical_validation_evidence": 1.0,
            "2_inference_and_error_upward_filtering_is_not_empirical_validation_evidence": 1.0,
            "3_optimisation_free_energy_minimisation_is_not_empirical_validation_evidence": 1.0,
        },
        problem_metadata={
            "source_ledger_span": SOURCE_LEDGER_SPAN,
            "source_ledger_ids": tuple(f"P0R{number:05d}" for number in range(3197, 3208)),
            "claim_boundary": CLAIM_BOUNDARY,
            "protocol_state": "source_i_the_unifying_computational_principle_hierarchical_predictive_coding_hp_only_no_experiment",
        },
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "HARDWARE_STATUS",
    "SOURCE_LEDGER_SPAN",
    "ITheUnifyingComputationalPrincipleHierarchicalPredictiveCodingHpConfig",
    "ITheUnifyingComputationalPrincipleHierarchicalPredictiveCodingHpFixtureResult",
    "classify_i_the_unifying_computational_principle_hierarchical_predictive_coding_hp_component",
    "i_the_unifying_computational_principle_hierarchical_predictive_coding_hp_labels",
    "validate_i_the_unifying_computational_principle_hierarchical_predictive_coding_hp_fixture",
]
