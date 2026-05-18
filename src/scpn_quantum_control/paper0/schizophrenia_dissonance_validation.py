# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 Schizophrenia (Dissonance) validation
"""Source-accounting checks for Paper 0 Schizophrenia (Dissonance) records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

CLAIM_BOUNDARY = (
    "source-bounded schizophrenia dissonance source-accounting bridge; not validation evidence"
)
HARDWARE_STATUS = "source_methodology_no_experiment"
SOURCE_LEDGER_SPAN = ("P0R04630", "P0R04639")


@dataclass(frozen=True, slots=True)
class SchizophreniaDissonanceConfig:
    """Configuration for this Paper 0 source-accounting fixture."""

    expected_source_record_count: int = 10
    expected_component_count: int = 3
    next_source_boundary: str = "P0R04640"

    def __post_init__(self) -> None:
        if self.expected_source_record_count != 10:
            raise ValueError("expected_source_record_count must equal 10")
        if self.expected_component_count != 3:
            raise ValueError("expected_component_count must equal 3")
        if self.next_source_boundary != "P0R04640":
            raise ValueError("next_source_boundary must equal P0R04640")


@dataclass(frozen=True, slots=True)
class SchizophreniaDissonanceFixtureResult:
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


def classify_schizophrenia_dissonance_component(component: str) -> str:
    """Classify source-defined components."""
    mapping = {
        "schizophrenia_dissonance": "schizophrenia_dissonance_source_boundary",
        "depression_dyscritia_dissonance": "depression_dyscritia_dissonance_source_boundary",
        "alzheimer_s_disease_decoherence_cascade": "alzheimer_s_disease_decoherence_cascade_source_boundary",
    }
    try:
        return mapping[component]
    except KeyError as exc:
        raise ValueError("unknown schizophrenia_dissonance component") from exc


def schizophrenia_dissonance_labels() -> dict[str, str]:
    """Return source-bounded labels for this slice."""
    return {
        "section": "Schizophrenia (Dissonance)",
        "source_span": "P0R04630-P0R04639",
        "component_count": "3",
        "next_boundary": "P0R04640",
        "component_1": "Schizophrenia (Dissonance)",
        "component_2": "Depression (Dyscritia & Dissonance)",
        "component_3": "Alzheimer's Disease (Decoherence Cascade)",
    }


def validate_schizophrenia_dissonance_fixture(
    config: SchizophreniaDissonanceConfig | None = None,
) -> SchizophreniaDissonanceFixtureResult:
    """Validate source accounting for this Paper 0 slice."""
    cfg = config or SchizophreniaDissonanceConfig()
    components = (
        "schizophrenia_dissonance",
        "depression_dyscritia_dissonance",
        "alzheimer_s_disease_decoherence_cascade",
    )
    return SchizophreniaDissonanceFixtureResult(
        source_ledger_span=SOURCE_LEDGER_SPAN,
        hardware_status=HARDWARE_STATUS,
        claim_boundary=CLAIM_BOUNDARY,
        components={
            component: classify_schizophrenia_dissonance_component(component)
            for component in components
        },
        labels=schizophrenia_dissonance_labels(),
        source_record_count=cfg.expected_source_record_count,
        component_count=cfg.expected_component_count,
        next_source_boundary=cfg.next_source_boundary,
        null_controls={
            "schizophrenia_dissonance_is_not_empirical_validation_evidence": 1.0,
            "depression_dyscritia_dissonance_is_not_empirical_validation_evidence": 1.0,
            "alzheimer_s_disease_decoherence_cascade_is_not_empirical_validation_evidence": 1.0,
        },
        problem_metadata={
            "source_ledger_span": SOURCE_LEDGER_SPAN,
            "source_ledger_ids": tuple(f"P0R{number:05d}" for number in range(4630, 4640)),
            "claim_boundary": CLAIM_BOUNDARY,
            "protocol_state": "source_schizophrenia_dissonance_only_no_experiment",
        },
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "HARDWARE_STATUS",
    "SOURCE_LEDGER_SPAN",
    "SchizophreniaDissonanceConfig",
    "SchizophreniaDissonanceFixtureResult",
    "classify_schizophrenia_dissonance_component",
    "schizophrenia_dissonance_labels",
    "validate_schizophrenia_dissonance_fixture",
]
