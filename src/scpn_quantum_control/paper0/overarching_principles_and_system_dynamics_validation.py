# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 Overarching Principles and System Dynamics validation
"""Source-accounting checks for Paper 0 Overarching Principles and System Dynamics records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

CLAIM_BOUNDARY = "source-bounded overarching principles and system dynamics source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"
SOURCE_LEDGER_SPAN = ("P0R05919", "P0R05927")


@dataclass(frozen=True, slots=True)
class OverarchingPrinciplesAndSystemDynamicsConfig:
    """Configuration for this Paper 0 source-accounting fixture."""

    expected_source_record_count: int = 9
    expected_component_count: int = 3
    next_source_boundary: str = "P0R05928"

    def __post_init__(self) -> None:
        if self.expected_source_record_count != 9:
            raise ValueError("expected_source_record_count must equal 9")
        if self.expected_component_count != 3:
            raise ValueError("expected_component_count must equal 3")
        if self.next_source_boundary != "P0R05928":
            raise ValueError("next_source_boundary must equal P0R05928")


@dataclass(frozen=True, slots=True)
class OverarchingPrinciplesAndSystemDynamicsFixtureResult:
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


def classify_overarching_principles_and_system_dynamics_component(component: str) -> str:
    """Classify source-defined components."""
    mapping = {
        "overarching_principles_and_system_dynamics": "overarching_principles_and_system_dynamics_source_boundary",
        "computational_unifier": "computational_unifier_source_boundary",
        "layer_5_strange_loop_as_active_inference_engine_sn_precision_control_oct": "layer_5_strange_loop_as_active_inference_engine_sn_precision_control_oct_source_boundary",
    }
    try:
        return mapping[component]
    except KeyError as exc:
        raise ValueError("unknown overarching_principles_and_system_dynamics component") from exc


def overarching_principles_and_system_dynamics_labels() -> dict[str, str]:
    """Return source-bounded labels for this slice."""
    return {
        "section": "Overarching Principles and System Dynamics",
        "source_span": "P0R05919-P0R05927",
        "component_count": "3",
        "next_boundary": "P0R05928",
        "component_1": "Overarching Principles and System Dynamics",
        "component_2": "Computational Unifier.",
        "component_3": "Layer-5 Strange Loop as Active-Inference Engine; SN precision control; OCT at Meta-Layer 16.",
    }


def validate_overarching_principles_and_system_dynamics_fixture(
    config: OverarchingPrinciplesAndSystemDynamicsConfig | None = None,
) -> OverarchingPrinciplesAndSystemDynamicsFixtureResult:
    """Validate source accounting for this Paper 0 slice."""
    cfg = config or OverarchingPrinciplesAndSystemDynamicsConfig()
    components = (
        "overarching_principles_and_system_dynamics",
        "computational_unifier",
        "layer_5_strange_loop_as_active_inference_engine_sn_precision_control_oct",
    )
    return OverarchingPrinciplesAndSystemDynamicsFixtureResult(
        source_ledger_span=SOURCE_LEDGER_SPAN,
        hardware_status=HARDWARE_STATUS,
        claim_boundary=CLAIM_BOUNDARY,
        components={
            component: classify_overarching_principles_and_system_dynamics_component(component)
            for component in components
        },
        labels=overarching_principles_and_system_dynamics_labels(),
        source_record_count=cfg.expected_source_record_count,
        component_count=cfg.expected_component_count,
        next_source_boundary=cfg.next_source_boundary,
        null_controls={
            "overarching_principles_and_system_dynamics_is_not_empirical_validation_evidence": 1.0,
            "computational_unifier_is_not_empirical_validation_evidence": 1.0,
            "layer_5_strange_loop_as_active_inference_engine_sn_precision_control_oct_is_not_empirical_validation_evidence": 1.0,
        },
        problem_metadata={
            "source_ledger_span": SOURCE_LEDGER_SPAN,
            "source_ledger_ids": tuple(f"P0R{number:05d}" for number in range(5919, 5928)),
            "claim_boundary": CLAIM_BOUNDARY,
            "protocol_state": "source_overarching_principles_and_system_dynamics_only_no_experiment",
        },
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "HARDWARE_STATUS",
    "SOURCE_LEDGER_SPAN",
    "OverarchingPrinciplesAndSystemDynamicsConfig",
    "OverarchingPrinciplesAndSystemDynamicsFixtureResult",
    "classify_overarching_principles_and_system_dynamics_component",
    "overarching_principles_and_system_dynamics_labels",
    "validate_overarching_principles_and_system_dynamics_fixture",
]
