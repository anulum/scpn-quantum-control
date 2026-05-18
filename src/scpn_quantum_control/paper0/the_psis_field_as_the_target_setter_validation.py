# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 The Psis Field as the Target-Setter: validation
"""Source-accounting checks for Paper 0 The Psis Field as the Target-Setter: records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

CLAIM_BOUNDARY = "source-bounded the psis field as the target setter source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"
SOURCE_LEDGER_SPAN = ("P0R02904", "P0R02914")


@dataclass(frozen=True, slots=True)
class ThePsisFieldAsTheTargetSetterConfig:
    """Configuration for this Paper 0 source-accounting fixture."""

    expected_source_record_count: int = 11
    expected_component_count: int = 3
    next_source_boundary: str = "P0R02915"

    def __post_init__(self) -> None:
        if self.expected_source_record_count != 11:
            raise ValueError("expected_source_record_count must equal 11")
        if self.expected_component_count != 3:
            raise ValueError("expected_component_count must equal 3")
        if self.next_source_boundary != "P0R02915":
            raise ValueError("next_source_boundary must equal P0R02915")


@dataclass(frozen=True, slots=True)
class ThePsisFieldAsTheTargetSetterFixtureResult:
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


def classify_the_psis_field_as_the_target_setter_component(component: str) -> str:
    """Classify source-defined components."""
    mapping = {
        "the_psis_field_as_the_target_setter": "the_psis_field_as_the_target_setter_source_boundary",
        "sigma_is_the_lyapunov_function_itself": "sigma_is_the_lyapunov_function_itself_source_boundary",
        "homeostatic_quasicritical_controller": "homeostatic_quasicritical_controller_source_boundary",
    }
    try:
        return mapping[component]
    except KeyError as exc:
        raise ValueError("unknown the_psis_field_as_the_target_setter component") from exc


def the_psis_field_as_the_target_setter_labels() -> dict[str, str]:
    """Return source-bounded labels for this slice."""
    return {
        "section": "The Psis Field as the Target-Setter:",
        "source_span": "P0R02904-P0R02914",
        "component_count": "3",
        "next_boundary": "P0R02915",
        "component_1": "The Psis Field as the Target-Setter:",
        "component_2": "sigma is the Lyapunov Function itself:",
        "component_3": "Homeostatic Quasicritical Controller",
    }


def validate_the_psis_field_as_the_target_setter_fixture(
    config: ThePsisFieldAsTheTargetSetterConfig | None = None,
) -> ThePsisFieldAsTheTargetSetterFixtureResult:
    """Validate source accounting for this Paper 0 slice."""
    cfg = config or ThePsisFieldAsTheTargetSetterConfig()
    components = (
        "the_psis_field_as_the_target_setter",
        "sigma_is_the_lyapunov_function_itself",
        "homeostatic_quasicritical_controller",
    )
    return ThePsisFieldAsTheTargetSetterFixtureResult(
        source_ledger_span=SOURCE_LEDGER_SPAN,
        hardware_status=HARDWARE_STATUS,
        claim_boundary=CLAIM_BOUNDARY,
        components={
            component: classify_the_psis_field_as_the_target_setter_component(component)
            for component in components
        },
        labels=the_psis_field_as_the_target_setter_labels(),
        source_record_count=cfg.expected_source_record_count,
        component_count=cfg.expected_component_count,
        next_source_boundary=cfg.next_source_boundary,
        null_controls={
            "the_psis_field_as_the_target_setter_is_not_empirical_validation_evidence": 1.0,
            "sigma_is_the_lyapunov_function_itself_is_not_empirical_validation_evidence": 1.0,
            "homeostatic_quasicritical_controller_is_not_empirical_validation_evidence": 1.0,
        },
        problem_metadata={
            "source_ledger_span": SOURCE_LEDGER_SPAN,
            "source_ledger_ids": tuple(f"P0R{number:05d}" for number in range(2904, 2915)),
            "claim_boundary": CLAIM_BOUNDARY,
            "protocol_state": "source_the_psis_field_as_the_target_setter_only_no_experiment",
        },
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "HARDWARE_STATUS",
    "SOURCE_LEDGER_SPAN",
    "ThePsisFieldAsTheTargetSetterConfig",
    "ThePsisFieldAsTheTargetSetterFixtureResult",
    "classify_the_psis_field_as_the_target_setter_component",
    "the_psis_field_as_the_target_setter_labels",
    "validate_the_psis_field_as_the_target_setter_fixture",
]
