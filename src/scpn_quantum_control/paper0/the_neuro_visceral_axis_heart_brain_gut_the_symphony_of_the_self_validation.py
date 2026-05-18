# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 The Neuro-Visceral Axis (Heart-Brain-Gut): The Symphony of the Self validation
"""Source-accounting checks for Paper 0 The Neuro-Visceral Axis (Heart-Brain-Gut): The Symphony of the Self records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

CLAIM_BOUNDARY = "source-bounded the neuro visceral axis heart brain gut the symphony of the self source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"
SOURCE_LEDGER_SPAN = ("P0R04607", "P0R04621")


@dataclass(frozen=True, slots=True)
class TheNeuroVisceralAxisHeartBrainGutTheSymphonyOfTheSelfConfig:
    """Configuration for this Paper 0 source-accounting fixture."""

    expected_source_record_count: int = 15
    expected_component_count: int = 3
    next_source_boundary: str = "P0R04622"

    def __post_init__(self) -> None:
        if self.expected_source_record_count != 15:
            raise ValueError("expected_source_record_count must equal 15")
        if self.expected_component_count != 3:
            raise ValueError("expected_component_count must equal 3")
        if self.next_source_boundary != "P0R04622":
            raise ValueError("next_source_boundary must equal P0R04622")


@dataclass(frozen=True, slots=True)
class TheNeuroVisceralAxisHeartBrainGutTheSymphonyOfTheSelfFixtureResult:
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


def classify_the_neuro_visceral_axis_heart_brain_gut_the_symphony_of_the_self_component(
    component: str,
) -> str:
    """Classify source-defined components."""
    mapping = {
        "the_neuro_visceral_axis_heart_brain_gut_the_symphony_of_the_self": "the_neuro_visceral_axis_heart_brain_gut_the_symphony_of_the_self_source_boundary",
        "interoceptive_inference_the_physics_of_emotion": "interoceptive_inference_the_physics_of_emotion_source_boundary",
        "psychoneuroimmunology_pni_the_decoherence_field_of_inflammation": "psychoneuroimmunology_pni_the_decoherence_field_of_inflammation_source_boundary",
    }
    try:
        return mapping[component]
    except KeyError as exc:
        raise ValueError(
            "unknown the_neuro_visceral_axis_heart_brain_gut_the_symphony_of_the_self component"
        ) from exc


def the_neuro_visceral_axis_heart_brain_gut_the_symphony_of_the_self_labels() -> dict[str, str]:
    """Return source-bounded labels for this slice."""
    return {
        "section": "The Neuro-Visceral Axis (Heart-Brain-Gut): The Symphony of the Self",
        "source_span": "P0R04607-P0R04621",
        "component_count": "3",
        "next_boundary": "P0R04622",
        "component_1": "The Neuro-Visceral Axis (Heart-Brain-Gut): The Symphony of the Self",
        "component_2": "Interoceptive Inference: The Physics of Emotion",
        "component_3": "Psychoneuroimmunology (PNI): The Decoherence Field of Inflammation",
    }


def validate_the_neuro_visceral_axis_heart_brain_gut_the_symphony_of_the_self_fixture(
    config: TheNeuroVisceralAxisHeartBrainGutTheSymphonyOfTheSelfConfig | None = None,
) -> TheNeuroVisceralAxisHeartBrainGutTheSymphonyOfTheSelfFixtureResult:
    """Validate source accounting for this Paper 0 slice."""
    cfg = config or TheNeuroVisceralAxisHeartBrainGutTheSymphonyOfTheSelfConfig()
    components = (
        "the_neuro_visceral_axis_heart_brain_gut_the_symphony_of_the_self",
        "interoceptive_inference_the_physics_of_emotion",
        "psychoneuroimmunology_pni_the_decoherence_field_of_inflammation",
    )
    return TheNeuroVisceralAxisHeartBrainGutTheSymphonyOfTheSelfFixtureResult(
        source_ledger_span=SOURCE_LEDGER_SPAN,
        hardware_status=HARDWARE_STATUS,
        claim_boundary=CLAIM_BOUNDARY,
        components={
            component: classify_the_neuro_visceral_axis_heart_brain_gut_the_symphony_of_the_self_component(
                component
            )
            for component in components
        },
        labels=the_neuro_visceral_axis_heart_brain_gut_the_symphony_of_the_self_labels(),
        source_record_count=cfg.expected_source_record_count,
        component_count=cfg.expected_component_count,
        next_source_boundary=cfg.next_source_boundary,
        null_controls={
            "the_neuro_visceral_axis_heart_brain_gut_the_symphony_of_the_self_is_not_empirical_validation_evidence": 1.0,
            "interoceptive_inference_the_physics_of_emotion_is_not_empirical_validation_evidence": 1.0,
            "psychoneuroimmunology_pni_the_decoherence_field_of_inflammation_is_not_empirical_validation_evidence": 1.0,
        },
        problem_metadata={
            "source_ledger_span": SOURCE_LEDGER_SPAN,
            "source_ledger_ids": tuple(f"P0R{number:05d}" for number in range(4607, 4622)),
            "claim_boundary": CLAIM_BOUNDARY,
            "protocol_state": "source_the_neuro_visceral_axis_heart_brain_gut_the_symphony_of_the_self_only_no_experiment",
        },
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "HARDWARE_STATUS",
    "SOURCE_LEDGER_SPAN",
    "TheNeuroVisceralAxisHeartBrainGutTheSymphonyOfTheSelfConfig",
    "TheNeuroVisceralAxisHeartBrainGutTheSymphonyOfTheSelfFixtureResult",
    "classify_the_neuro_visceral_axis_heart_brain_gut_the_symphony_of_the_self_component",
    "the_neuro_visceral_axis_heart_brain_gut_the_symphony_of_the_self_labels",
    "validate_the_neuro_visceral_axis_heart_brain_gut_the_symphony_of_the_self_fixture",
]
