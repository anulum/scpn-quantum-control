# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 3. The Quasicritical Brain: validation
"""Source-accounting checks for Paper 0 3. The Quasicritical Brain: records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

CLAIM_BOUNDARY = "source-bounded section 3 the quasicritical brain source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"
SOURCE_LEDGER_SPAN = ("P0R04499", "P0R04506")


@dataclass(frozen=True, slots=True)
class Section3TheQuasicriticalBrainConfig:
    """Configuration for this Paper 0 source-accounting fixture."""

    expected_source_record_count: int = 8
    expected_component_count: int = 3
    next_source_boundary: str = "P0R04507"

    def __post_init__(self) -> None:
        if self.expected_source_record_count != 8:
            raise ValueError("expected_source_record_count must equal 8")
        if self.expected_component_count != 3:
            raise ValueError("expected_component_count must equal 3")
        if self.next_source_boundary != "P0R04507":
            raise ValueError("next_source_boundary must equal P0R04507")


@dataclass(frozen=True, slots=True)
class Section3TheQuasicriticalBrainFixtureResult:
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


def classify_section_3_the_quasicritical_brain_component(component: str) -> str:
    """Classify source-defined components."""
    mapping = {
        "3_the_quasicritical_brain": "3_the_quasicritical_brain_source_boundary",
        "4_the_role_of_glia_astrocytes": "4_the_role_of_glia_astrocytes_source_boundary",
        "5_the_cerebellum_the_timing_and_prediction_engine_l4_l5_interface": "5_the_cerebellum_the_timing_and_prediction_engine_l4_l5_interface_source_boundary",
    }
    try:
        return mapping[component]
    except KeyError as exc:
        raise ValueError("unknown section_3_the_quasicritical_brain component") from exc


def section_3_the_quasicritical_brain_labels() -> dict[str, str]:
    """Return source-bounded labels for this slice."""
    return {
        "section": "3. The Quasicritical Brain:",
        "source_span": "P0R04499-P0R04506",
        "component_count": "3",
        "next_boundary": "P0R04507",
        "component_1": "3. The Quasicritical Brain:",
        "component_2": "4. The Role of Glia (Astrocytes):",
        "component_3": "5. The Cerebellum: The Timing and Prediction Engine (L4/L5 Interface)",
    }


def validate_section_3_the_quasicritical_brain_fixture(
    config: Section3TheQuasicriticalBrainConfig | None = None,
) -> Section3TheQuasicriticalBrainFixtureResult:
    """Validate source accounting for this Paper 0 slice."""
    cfg = config or Section3TheQuasicriticalBrainConfig()
    components = (
        "3_the_quasicritical_brain",
        "4_the_role_of_glia_astrocytes",
        "5_the_cerebellum_the_timing_and_prediction_engine_l4_l5_interface",
    )
    return Section3TheQuasicriticalBrainFixtureResult(
        source_ledger_span=SOURCE_LEDGER_SPAN,
        hardware_status=HARDWARE_STATUS,
        claim_boundary=CLAIM_BOUNDARY,
        components={
            component: classify_section_3_the_quasicritical_brain_component(component)
            for component in components
        },
        labels=section_3_the_quasicritical_brain_labels(),
        source_record_count=cfg.expected_source_record_count,
        component_count=cfg.expected_component_count,
        next_source_boundary=cfg.next_source_boundary,
        null_controls={
            "3_the_quasicritical_brain_is_not_empirical_validation_evidence": 1.0,
            "4_the_role_of_glia_astrocytes_is_not_empirical_validation_evidence": 1.0,
            "5_the_cerebellum_the_timing_and_prediction_engine_l4_l5_interface_is_not_empirical_validation_evidence": 1.0,
        },
        problem_metadata={
            "source_ledger_span": SOURCE_LEDGER_SPAN,
            "source_ledger_ids": tuple(f"P0R{number:05d}" for number in range(4499, 4507)),
            "claim_boundary": CLAIM_BOUNDARY,
            "protocol_state": "source_section_3_the_quasicritical_brain_only_no_experiment",
        },
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "HARDWARE_STATUS",
    "SOURCE_LEDGER_SPAN",
    "Section3TheQuasicriticalBrainConfig",
    "Section3TheQuasicriticalBrainFixtureResult",
    "classify_section_3_the_quasicritical_brain_component",
    "section_3_the_quasicritical_brain_labels",
    "validate_section_3_the_quasicritical_brain_fixture",
]
