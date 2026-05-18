# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 The Glial-Neuronal Coupling Mechanism: Slow Control of Neuronal Criticality validation
"""Source-accounting checks for Paper 0 The Glial-Neuronal Coupling Mechanism: Slow Control of Neuronal Criticality records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

CLAIM_BOUNDARY = "source-bounded the glial neuronal coupling mechanism slow control of neuronal criticali source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"
SOURCE_LEDGER_SPAN = ("P0R05366", "P0R05379")


@dataclass(frozen=True, slots=True)
class TheGlialNeuronalCouplingMechanismSlowControlOfNeuronalCriticaliConfig:
    """Configuration for this Paper 0 source-accounting fixture."""

    expected_source_record_count: int = 14
    expected_component_count: int = 1
    next_source_boundary: str = "P0R05380"

    def __post_init__(self) -> None:
        if self.expected_source_record_count != 14:
            raise ValueError("expected_source_record_count must equal 14")
        if self.expected_component_count != 1:
            raise ValueError("expected_component_count must equal 1")
        if self.next_source_boundary != "P0R05380":
            raise ValueError("next_source_boundary must equal P0R05380")


@dataclass(frozen=True, slots=True)
class TheGlialNeuronalCouplingMechanismSlowControlOfNeuronalCriticaliFixtureResult:
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


def classify_the_glial_neuronal_coupling_mechanism_slow_control_of_neuronal_criticali_component(
    component: str,
) -> str:
    """Classify source-defined components."""
    mapping = {
        "the_glial_neuronal_coupling_mechanism_slow_control_of_neuronal_criticali": "the_glial_neuronal_coupling_mechanism_slow_control_of_neuronal_criticali_source_boundary"
    }
    try:
        return mapping[component]
    except KeyError as exc:
        raise ValueError(
            "unknown the_glial_neuronal_coupling_mechanism_slow_control_of_neuronal_criticali component"
        ) from exc


def the_glial_neuronal_coupling_mechanism_slow_control_of_neuronal_criticali_labels() -> dict[
    str, str
]:
    """Return source-bounded labels for this slice."""
    return {
        "section": "The Glial-Neuronal Coupling Mechanism: Slow Control of Neuronal Criticality",
        "source_span": "P0R05366-P0R05379",
        "component_count": "1",
        "next_boundary": "P0R05380",
        "component_1": "The Glial-Neuronal Coupling Mechanism: Slow Control of Neuronal Criticality",
    }


def validate_the_glial_neuronal_coupling_mechanism_slow_control_of_neuronal_criticali_fixture(
    config: TheGlialNeuronalCouplingMechanismSlowControlOfNeuronalCriticaliConfig | None = None,
) -> TheGlialNeuronalCouplingMechanismSlowControlOfNeuronalCriticaliFixtureResult:
    """Validate source accounting for this Paper 0 slice."""
    cfg = config or TheGlialNeuronalCouplingMechanismSlowControlOfNeuronalCriticaliConfig()
    components = ("the_glial_neuronal_coupling_mechanism_slow_control_of_neuronal_criticali",)
    return TheGlialNeuronalCouplingMechanismSlowControlOfNeuronalCriticaliFixtureResult(
        source_ledger_span=SOURCE_LEDGER_SPAN,
        hardware_status=HARDWARE_STATUS,
        claim_boundary=CLAIM_BOUNDARY,
        components={
            component: classify_the_glial_neuronal_coupling_mechanism_slow_control_of_neuronal_criticali_component(
                component
            )
            for component in components
        },
        labels=the_glial_neuronal_coupling_mechanism_slow_control_of_neuronal_criticali_labels(),
        source_record_count=cfg.expected_source_record_count,
        component_count=cfg.expected_component_count,
        next_source_boundary=cfg.next_source_boundary,
        null_controls={
            "the_glial_neuronal_coupling_mechanism_slow_control_of_neuronal_criticali_is_not_empirical_validation_evidence": 1.0
        },
        problem_metadata={
            "source_ledger_span": SOURCE_LEDGER_SPAN,
            "source_ledger_ids": tuple(f"P0R{number:05d}" for number in range(5366, 5380)),
            "claim_boundary": CLAIM_BOUNDARY,
            "protocol_state": "source_the_glial_neuronal_coupling_mechanism_slow_control_of_neuronal_criticali_only_no_experiment",
        },
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "HARDWARE_STATUS",
    "SOURCE_LEDGER_SPAN",
    "TheGlialNeuronalCouplingMechanismSlowControlOfNeuronalCriticaliConfig",
    "TheGlialNeuronalCouplingMechanismSlowControlOfNeuronalCriticaliFixtureResult",
    "classify_the_glial_neuronal_coupling_mechanism_slow_control_of_neuronal_criticali_component",
    "the_glial_neuronal_coupling_mechanism_slow_control_of_neuronal_criticali_labels",
    "validate_the_glial_neuronal_coupling_mechanism_slow_control_of_neuronal_criticali_fixture",
]
