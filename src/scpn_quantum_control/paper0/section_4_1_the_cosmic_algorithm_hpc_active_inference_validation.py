# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 4.1 The Cosmic Algorithm: HPC & Active Inference validation
"""Source-accounting checks for Paper 0 4.1 The Cosmic Algorithm: HPC & Active Inference records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

CLAIM_BOUNDARY = "source-bounded section 4 1 the cosmic algorithm hpc active inference source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"
SOURCE_LEDGER_SPAN = ("P0R03174", "P0R03196")


@dataclass(frozen=True, slots=True)
class Section41TheCosmicAlgorithmHpcActiveInferenceConfig:
    """Configuration for this Paper 0 source-accounting fixture."""

    expected_source_record_count: int = 23
    expected_component_count: int = 2
    next_source_boundary: str = "P0R03197"

    def __post_init__(self) -> None:
        if self.expected_source_record_count != 23:
            raise ValueError("expected_source_record_count must equal 23")
        if self.expected_component_count != 2:
            raise ValueError("expected_component_count must equal 2")
        if self.next_source_boundary != "P0R03197":
            raise ValueError("next_source_boundary must equal P0R03197")


@dataclass(frozen=True, slots=True)
class Section41TheCosmicAlgorithmHpcActiveInferenceFixtureResult:
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


def classify_section_4_1_the_cosmic_algorithm_hpc_active_inference_component(
    component: str,
) -> str:
    """Classify source-defined components."""
    mapping = {
        "4_1_the_cosmic_algorithm_hpc_active_inference": "4_1_the_cosmic_algorithm_hpc_active_inference_source_boundary",
        "integrative_mechanisms_the_computational_and_physical_synthesis": "integrative_mechanisms_the_computational_and_physical_synthesis_source_boundary",
    }
    try:
        return mapping[component]
    except KeyError as exc:
        raise ValueError(
            "unknown section_4_1_the_cosmic_algorithm_hpc_active_inference component"
        ) from exc


def section_4_1_the_cosmic_algorithm_hpc_active_inference_labels() -> dict[str, str]:
    """Return source-bounded labels for this slice."""
    return {
        "section": "4.1 The Cosmic Algorithm: HPC & Active Inference",
        "source_span": "P0R03174-P0R03196",
        "component_count": "2",
        "next_boundary": "P0R03197",
        "component_1": "4.1 The Cosmic Algorithm: HPC & Active Inference",
        "component_2": "Integrative Mechanisms: The Computational and Physical Synthesis",
    }


def validate_section_4_1_the_cosmic_algorithm_hpc_active_inference_fixture(
    config: Section41TheCosmicAlgorithmHpcActiveInferenceConfig | None = None,
) -> Section41TheCosmicAlgorithmHpcActiveInferenceFixtureResult:
    """Validate source accounting for this Paper 0 slice."""
    cfg = config or Section41TheCosmicAlgorithmHpcActiveInferenceConfig()
    components = (
        "4_1_the_cosmic_algorithm_hpc_active_inference",
        "integrative_mechanisms_the_computational_and_physical_synthesis",
    )
    return Section41TheCosmicAlgorithmHpcActiveInferenceFixtureResult(
        source_ledger_span=SOURCE_LEDGER_SPAN,
        hardware_status=HARDWARE_STATUS,
        claim_boundary=CLAIM_BOUNDARY,
        components={
            component: classify_section_4_1_the_cosmic_algorithm_hpc_active_inference_component(
                component
            )
            for component in components
        },
        labels=section_4_1_the_cosmic_algorithm_hpc_active_inference_labels(),
        source_record_count=cfg.expected_source_record_count,
        component_count=cfg.expected_component_count,
        next_source_boundary=cfg.next_source_boundary,
        null_controls={
            "4_1_the_cosmic_algorithm_hpc_active_inference_is_not_empirical_validation_evidence": 1.0,
            "integrative_mechanisms_the_computational_and_physical_synthesis_is_not_empirical_validation_evidence": 1.0,
        },
        problem_metadata={
            "source_ledger_span": SOURCE_LEDGER_SPAN,
            "source_ledger_ids": tuple(f"P0R{number:05d}" for number in range(3174, 3197)),
            "claim_boundary": CLAIM_BOUNDARY,
            "protocol_state": "source_section_4_1_the_cosmic_algorithm_hpc_active_inference_only_no_experiment",
        },
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "HARDWARE_STATUS",
    "SOURCE_LEDGER_SPAN",
    "Section41TheCosmicAlgorithmHpcActiveInferenceConfig",
    "Section41TheCosmicAlgorithmHpcActiveInferenceFixtureResult",
    "classify_section_4_1_the_cosmic_algorithm_hpc_active_inference_component",
    "section_4_1_the_cosmic_algorithm_hpc_active_inference_labels",
    "validate_section_4_1_the_cosmic_algorithm_hpc_active_inference_fixture",
]
