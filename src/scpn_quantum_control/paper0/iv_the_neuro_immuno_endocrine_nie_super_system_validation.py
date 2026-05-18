# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 IV. The Neuro-Immuno-Endocrine (NIE) Super-System validation
"""Source-accounting checks for Paper 0 IV. The Neuro-Immuno-Endocrine (NIE) Super-System records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

CLAIM_BOUNDARY = "source-bounded iv the neuro immuno endocrine nie super system source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"
SOURCE_LEDGER_SPAN = ("P0R04921", "P0R04934")


@dataclass(frozen=True, slots=True)
class IvTheNeuroImmunoEndocrineNieSuperSystemConfig:
    """Configuration for this Paper 0 source-accounting fixture."""

    expected_source_record_count: int = 14
    expected_component_count: int = 2
    next_source_boundary: str = "P0R04935"

    def __post_init__(self) -> None:
        if self.expected_source_record_count != 14:
            raise ValueError("expected_source_record_count must equal 14")
        if self.expected_component_count != 2:
            raise ValueError("expected_component_count must equal 2")
        if self.next_source_boundary != "P0R04935":
            raise ValueError("next_source_boundary must equal P0R04935")


@dataclass(frozen=True, slots=True)
class IvTheNeuroImmunoEndocrineNieSuperSystemFixtureResult:
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


def classify_iv_the_neuro_immuno_endocrine_nie_super_system_component(component: str) -> str:
    """Classify source-defined components."""
    mapping = {
        "iv_the_neuro_immuno_endocrine_nie_super_system": "iv_the_neuro_immuno_endocrine_nie_super_system_source_boundary",
        "1_the_psychoneuroimmunology_pni_axis_and_inflammation_the_decoherence_fi": "1_the_psychoneuroimmunology_pni_axis_and_inflammation_the_decoherence_fi_source_boundary",
    }
    try:
        return mapping[component]
    except KeyError as exc:
        raise ValueError(
            "unknown iv_the_neuro_immuno_endocrine_nie_super_system component"
        ) from exc


def iv_the_neuro_immuno_endocrine_nie_super_system_labels() -> dict[str, str]:
    """Return source-bounded labels for this slice."""
    return {
        "section": "IV. The Neuro-Immuno-Endocrine (NIE) Super-System",
        "source_span": "P0R04921-P0R04934",
        "component_count": "2",
        "next_boundary": "P0R04935",
        "component_1": "IV. The Neuro-Immuno-Endocrine (NIE) Super-System",
        "component_2": "1. The Psychoneuroimmunology (PNI) Axis and Inflammation (The Decoherence Field)",
    }


def validate_iv_the_neuro_immuno_endocrine_nie_super_system_fixture(
    config: IvTheNeuroImmunoEndocrineNieSuperSystemConfig | None = None,
) -> IvTheNeuroImmunoEndocrineNieSuperSystemFixtureResult:
    """Validate source accounting for this Paper 0 slice."""
    cfg = config or IvTheNeuroImmunoEndocrineNieSuperSystemConfig()
    components = (
        "iv_the_neuro_immuno_endocrine_nie_super_system",
        "1_the_psychoneuroimmunology_pni_axis_and_inflammation_the_decoherence_fi",
    )
    return IvTheNeuroImmunoEndocrineNieSuperSystemFixtureResult(
        source_ledger_span=SOURCE_LEDGER_SPAN,
        hardware_status=HARDWARE_STATUS,
        claim_boundary=CLAIM_BOUNDARY,
        components={
            component: classify_iv_the_neuro_immuno_endocrine_nie_super_system_component(component)
            for component in components
        },
        labels=iv_the_neuro_immuno_endocrine_nie_super_system_labels(),
        source_record_count=cfg.expected_source_record_count,
        component_count=cfg.expected_component_count,
        next_source_boundary=cfg.next_source_boundary,
        null_controls={
            "iv_the_neuro_immuno_endocrine_nie_super_system_is_not_empirical_validation_evidence": 1.0,
            "1_the_psychoneuroimmunology_pni_axis_and_inflammation_the_decoherence_fi_is_not_empirical_validation_evidence": 1.0,
        },
        problem_metadata={
            "source_ledger_span": SOURCE_LEDGER_SPAN,
            "source_ledger_ids": tuple(f"P0R{number:05d}" for number in range(4921, 4935)),
            "claim_boundary": CLAIM_BOUNDARY,
            "protocol_state": "source_iv_the_neuro_immuno_endocrine_nie_super_system_only_no_experiment",
        },
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "HARDWARE_STATUS",
    "SOURCE_LEDGER_SPAN",
    "IvTheNeuroImmunoEndocrineNieSuperSystemConfig",
    "IvTheNeuroImmunoEndocrineNieSuperSystemFixtureResult",
    "classify_iv_the_neuro_immuno_endocrine_nie_super_system_component",
    "iv_the_neuro_immuno_endocrine_nie_super_system_labels",
    "validate_iv_the_neuro_immuno_endocrine_nie_super_system_fixture",
]
