# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 IV. The Generation of Subjective Experience (Geometric Qualia) validation
"""Source-accounting checks for Paper 0 IV. The Generation of Subjective Experience (Geometric Qualia) records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

CLAIM_BOUNDARY = "source-bounded iv the generation of subjective experience geometric qualia source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"
SOURCE_LEDGER_SPAN = ("P0R06015", "P0R06022")


@dataclass(frozen=True, slots=True)
class IvTheGenerationOfSubjectiveExperienceGeometricQualiaConfig:
    """Configuration for this Paper 0 source-accounting fixture."""

    expected_source_record_count: int = 8
    expected_component_count: int = 5
    next_source_boundary: str = "P0R06023"

    def __post_init__(self) -> None:
        if self.expected_source_record_count != 8:
            raise ValueError("expected_source_record_count must equal 8")
        if self.expected_component_count != 5:
            raise ValueError("expected_component_count must equal 5")
        if self.next_source_boundary != "P0R06023":
            raise ValueError("next_source_boundary must equal P0R06023")


@dataclass(frozen=True, slots=True)
class IvTheGenerationOfSubjectiveExperienceGeometricQualiaFixtureResult:
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


def classify_iv_the_generation_of_subjective_experience_geometric_qualia_component(
    component: str,
) -> str:
    """Classify source-defined components."""
    mapping = {
        "iv_the_generation_of_subjective_experience_geometric_qualia": "iv_the_generation_of_subjective_experience_geometric_qualia_source_boundary",
        "the_ontology_of_experience": "the_ontology_of_experience_source_boundary",
        "formalisation_of_geometric_qualia": "formalisation_of_geometric_qualia_source_boundary",
        "the_metric_tensor_gmu_encodes_the_valence_the_intrinsic_curvature_ricci": "the_metric_tensor_gmu_encodes_the_valence_the_intrinsic_curvature_ricci_source_boundary",
        "the_connection_defines_the_flow_of_experience_stream_of_consciousness_vi": "the_connection_defines_the_flow_of_experience_stream_of_consciousness_vi_source_boundary",
    }
    try:
        return mapping[component]
    except KeyError as exc:
        raise ValueError(
            "unknown iv_the_generation_of_subjective_experience_geometric_qualia component"
        ) from exc


def iv_the_generation_of_subjective_experience_geometric_qualia_labels() -> dict[str, str]:
    """Return source-bounded labels for this slice."""
    return {
        "section": "IV. The Generation of Subjective Experience (Geometric Qualia)",
        "source_span": "P0R06015-P0R06022",
        "component_count": "5",
        "next_boundary": "P0R06023",
        "component_1": "IV. The Generation of Subjective Experience (Geometric Qualia)",
        "component_2": "The Ontology of Experience:",
        "component_3": "Formalisation of Geometric Qualia:",
        "component_4": "The Metric Tensor (gmu): Encodes the valence. The intrinsic curvature (Ricci scalar, R) corresponds to intensity. Intensity(Qualia)R=gmuRmu",
        "component_5": "The Connection (): Defines the flow of experience (stream of consciousness) via parallel transport.",
    }


def validate_iv_the_generation_of_subjective_experience_geometric_qualia_fixture(
    config: IvTheGenerationOfSubjectiveExperienceGeometricQualiaConfig | None = None,
) -> IvTheGenerationOfSubjectiveExperienceGeometricQualiaFixtureResult:
    """Validate source accounting for this Paper 0 slice."""
    cfg = config or IvTheGenerationOfSubjectiveExperienceGeometricQualiaConfig()
    components = (
        "iv_the_generation_of_subjective_experience_geometric_qualia",
        "the_ontology_of_experience",
        "formalisation_of_geometric_qualia",
        "the_metric_tensor_gmu_encodes_the_valence_the_intrinsic_curvature_ricci",
        "the_connection_defines_the_flow_of_experience_stream_of_consciousness_vi",
    )
    return IvTheGenerationOfSubjectiveExperienceGeometricQualiaFixtureResult(
        source_ledger_span=SOURCE_LEDGER_SPAN,
        hardware_status=HARDWARE_STATUS,
        claim_boundary=CLAIM_BOUNDARY,
        components={
            component: classify_iv_the_generation_of_subjective_experience_geometric_qualia_component(
                component
            )
            for component in components
        },
        labels=iv_the_generation_of_subjective_experience_geometric_qualia_labels(),
        source_record_count=cfg.expected_source_record_count,
        component_count=cfg.expected_component_count,
        next_source_boundary=cfg.next_source_boundary,
        null_controls={
            "iv_the_generation_of_subjective_experience_geometric_qualia_is_not_empirical_validation_evidence": 1.0,
            "the_ontology_of_experience_is_not_empirical_validation_evidence": 1.0,
            "formalisation_of_geometric_qualia_is_not_empirical_validation_evidence": 1.0,
            "the_metric_tensor_gmu_encodes_the_valence_the_intrinsic_curvature_ricci_is_not_empirical_validation_evidence": 1.0,
            "the_connection_defines_the_flow_of_experience_stream_of_consciousness_vi_is_not_empirical_validation_evidence": 1.0,
        },
        problem_metadata={
            "source_ledger_span": SOURCE_LEDGER_SPAN,
            "source_ledger_ids": tuple(f"P0R{number:05d}" for number in range(6015, 6023)),
            "claim_boundary": CLAIM_BOUNDARY,
            "protocol_state": "source_iv_the_generation_of_subjective_experience_geometric_qualia_only_no_experiment",
        },
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "HARDWARE_STATUS",
    "SOURCE_LEDGER_SPAN",
    "IvTheGenerationOfSubjectiveExperienceGeometricQualiaConfig",
    "IvTheGenerationOfSubjectiveExperienceGeometricQualiaFixtureResult",
    "classify_iv_the_generation_of_subjective_experience_geometric_qualia_component",
    "iv_the_generation_of_subjective_experience_geometric_qualia_labels",
    "validate_iv_the_generation_of_subjective_experience_geometric_qualia_fixture",
]
