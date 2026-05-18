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

CLAIM_BOUNDARY = "source-bounded the glial neuronal coupling mechanism slow control of neuronal criticali p0r05390 source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"
SOURCE_LEDGER_SPAN = ("P0R05390", "P0R05407")


@dataclass(frozen=True, slots=True)
class TheGlialNeuronalCouplingMechanismSlowControlOfNeuronalCriticaliP0r05390Config:
    """Configuration for this Paper 0 source-accounting fixture."""

    expected_source_record_count: int = 18
    expected_component_count: int = 3
    next_source_boundary: str = "P0R05408"

    def __post_init__(self) -> None:
        if self.expected_source_record_count != 18:
            raise ValueError("expected_source_record_count must equal 18")
        if self.expected_component_count != 3:
            raise ValueError("expected_component_count must equal 3")
        if self.next_source_boundary != "P0R05408":
            raise ValueError("next_source_boundary must equal P0R05408")


@dataclass(frozen=True, slots=True)
class TheGlialNeuronalCouplingMechanismSlowControlOfNeuronalCriticaliP0r05390FixtureResult:
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


def classify_the_glial_neuronal_coupling_mechanism_slow_control_of_neuronal_criticali_p0r05390_component(
    component: str,
) -> str:
    """Classify source-defined components."""
    mapping = {
        "the_glial_neuronal_coupling_mechanism_slow_control_of_neuronal_criticali": "the_glial_neuronal_coupling_mechanism_slow_control_of_neuronal_criticali_source_boundary",
        "the_astrocyte_network_as_the_slow_control_layer": "the_astrocyte_network_as_the_slow_control_layer_source_boundary",
        "formal_model_of_glial_neuronal_coupling": "formal_model_of_glial_neuronal_coupling_source_boundary",
    }
    try:
        return mapping[component]
    except KeyError as exc:
        raise ValueError(
            "unknown the_glial_neuronal_coupling_mechanism_slow_control_of_neuronal_criticali_p0r05390 component"
        ) from exc


def the_glial_neuronal_coupling_mechanism_slow_control_of_neuronal_criticali_p0r05390_labels() -> (
    dict[str, str]
):
    """Return source-bounded labels for this slice."""
    return {
        "section": "The Glial-Neuronal Coupling Mechanism: Slow Control of Neuronal Criticality",
        "source_span": "P0R05390-P0R05407",
        "component_count": "3",
        "next_boundary": "P0R05408",
        "component_1": "The Glial-Neuronal Coupling Mechanism: Slow Control of Neuronal Criticality",
        "component_2": "The Astrocyte Network as the Slow Control Layer",
        "component_3": "Formal Model of Glial-Neuronal Coupling",
    }


def validate_the_glial_neuronal_coupling_mechanism_slow_control_of_neuronal_criticali_p0r05390_fixture(
    config: TheGlialNeuronalCouplingMechanismSlowControlOfNeuronalCriticaliP0r05390Config
    | None = None,
) -> TheGlialNeuronalCouplingMechanismSlowControlOfNeuronalCriticaliP0r05390FixtureResult:
    """Validate source accounting for this Paper 0 slice."""
    cfg = config or TheGlialNeuronalCouplingMechanismSlowControlOfNeuronalCriticaliP0r05390Config()
    components = (
        "the_glial_neuronal_coupling_mechanism_slow_control_of_neuronal_criticali",
        "the_astrocyte_network_as_the_slow_control_layer",
        "formal_model_of_glial_neuronal_coupling",
    )
    return TheGlialNeuronalCouplingMechanismSlowControlOfNeuronalCriticaliP0r05390FixtureResult(
        source_ledger_span=SOURCE_LEDGER_SPAN,
        hardware_status=HARDWARE_STATUS,
        claim_boundary=CLAIM_BOUNDARY,
        components={
            component: classify_the_glial_neuronal_coupling_mechanism_slow_control_of_neuronal_criticali_p0r05390_component(
                component
            )
            for component in components
        },
        labels=the_glial_neuronal_coupling_mechanism_slow_control_of_neuronal_criticali_p0r05390_labels(),
        source_record_count=cfg.expected_source_record_count,
        component_count=cfg.expected_component_count,
        next_source_boundary=cfg.next_source_boundary,
        null_controls={
            "the_glial_neuronal_coupling_mechanism_slow_control_of_neuronal_criticali_is_not_empirical_validation_evidence": 1.0,
            "the_astrocyte_network_as_the_slow_control_layer_is_not_empirical_validation_evidence": 1.0,
            "formal_model_of_glial_neuronal_coupling_is_not_empirical_validation_evidence": 1.0,
        },
        problem_metadata={
            "source_ledger_span": SOURCE_LEDGER_SPAN,
            "source_ledger_ids": tuple(f"P0R{number:05d}" for number in range(5390, 5408)),
            "claim_boundary": CLAIM_BOUNDARY,
            "protocol_state": "source_the_glial_neuronal_coupling_mechanism_slow_control_of_neuronal_criticali_p0r05390_only_no_experiment",
        },
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "HARDWARE_STATUS",
    "SOURCE_LEDGER_SPAN",
    "TheGlialNeuronalCouplingMechanismSlowControlOfNeuronalCriticaliP0r05390Config",
    "TheGlialNeuronalCouplingMechanismSlowControlOfNeuronalCriticaliP0r05390FixtureResult",
    "classify_the_glial_neuronal_coupling_mechanism_slow_control_of_neuronal_criticali_p0r05390_component",
    "the_glial_neuronal_coupling_mechanism_slow_control_of_neuronal_criticali_p0r05390_labels",
    "validate_the_glial_neuronal_coupling_mechanism_slow_control_of_neuronal_criticali_p0r05390_fixture",
]
