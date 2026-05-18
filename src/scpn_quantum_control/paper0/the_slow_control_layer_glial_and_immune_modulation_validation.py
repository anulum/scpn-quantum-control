# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 The Slow Control Layer - Glial and Immune Modulation validation
"""Source-accounting checks for Paper 0 The Slow Control Layer - Glial and Immune Modulation records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

CLAIM_BOUNDARY = "source-bounded the slow control layer glial and immune modulation source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"
SOURCE_LEDGER_SPAN = ("P0R05347", "P0R05357")


@dataclass(frozen=True, slots=True)
class TheSlowControlLayerGlialAndImmuneModulationConfig:
    """Configuration for this Paper 0 source-accounting fixture."""

    expected_source_record_count: int = 11
    expected_component_count: int = 2
    next_source_boundary: str = "P0R05358"

    def __post_init__(self) -> None:
        if self.expected_source_record_count != 11:
            raise ValueError("expected_source_record_count must equal 11")
        if self.expected_component_count != 2:
            raise ValueError("expected_component_count must equal 2")
        if self.next_source_boundary != "P0R05358":
            raise ValueError("next_source_boundary must equal P0R05358")


@dataclass(frozen=True, slots=True)
class TheSlowControlLayerGlialAndImmuneModulationFixtureResult:
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


def classify_the_slow_control_layer_glial_and_immune_modulation_component(component: str) -> str:
    """Classify source-defined components."""
    mapping = {
        "the_slow_control_layer_glial_and_immune_modulation": "the_slow_control_layer_glial_and_immune_modulation_source_boundary",
        "i_the_astrocyte_neuron_lattice_l2_l4_modulation": "i_the_astrocyte_neuron_lattice_l2_l4_modulation_source_boundary",
    }
    try:
        return mapping[component]
    except KeyError as exc:
        raise ValueError(
            "unknown the_slow_control_layer_glial_and_immune_modulation component"
        ) from exc


def the_slow_control_layer_glial_and_immune_modulation_labels() -> dict[str, str]:
    """Return source-bounded labels for this slice."""
    return {
        "section": "The Slow Control Layer - Glial and Immune Modulation",
        "source_span": "P0R05347-P0R05357",
        "component_count": "2",
        "next_boundary": "P0R05358",
        "component_1": "The Slow Control Layer - Glial and Immune Modulation",
        "component_2": "I. The Astrocyte-Neuron Lattice (L2/L4 Modulation)",
    }


def validate_the_slow_control_layer_glial_and_immune_modulation_fixture(
    config: TheSlowControlLayerGlialAndImmuneModulationConfig | None = None,
) -> TheSlowControlLayerGlialAndImmuneModulationFixtureResult:
    """Validate source accounting for this Paper 0 slice."""
    cfg = config or TheSlowControlLayerGlialAndImmuneModulationConfig()
    components = (
        "the_slow_control_layer_glial_and_immune_modulation",
        "i_the_astrocyte_neuron_lattice_l2_l4_modulation",
    )
    return TheSlowControlLayerGlialAndImmuneModulationFixtureResult(
        source_ledger_span=SOURCE_LEDGER_SPAN,
        hardware_status=HARDWARE_STATUS,
        claim_boundary=CLAIM_BOUNDARY,
        components={
            component: classify_the_slow_control_layer_glial_and_immune_modulation_component(
                component
            )
            for component in components
        },
        labels=the_slow_control_layer_glial_and_immune_modulation_labels(),
        source_record_count=cfg.expected_source_record_count,
        component_count=cfg.expected_component_count,
        next_source_boundary=cfg.next_source_boundary,
        null_controls={
            "the_slow_control_layer_glial_and_immune_modulation_is_not_empirical_validation_evidence": 1.0,
            "i_the_astrocyte_neuron_lattice_l2_l4_modulation_is_not_empirical_validation_evidence": 1.0,
        },
        problem_metadata={
            "source_ledger_span": SOURCE_LEDGER_SPAN,
            "source_ledger_ids": tuple(f"P0R{number:05d}" for number in range(5347, 5358)),
            "claim_boundary": CLAIM_BOUNDARY,
            "protocol_state": "source_the_slow_control_layer_glial_and_immune_modulation_only_no_experiment",
        },
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "HARDWARE_STATUS",
    "SOURCE_LEDGER_SPAN",
    "TheSlowControlLayerGlialAndImmuneModulationConfig",
    "TheSlowControlLayerGlialAndImmuneModulationFixtureResult",
    "classify_the_slow_control_layer_glial_and_immune_modulation_component",
    "the_slow_control_layer_glial_and_immune_modulation_labels",
    "validate_the_slow_control_layer_glial_and_immune_modulation_fixture",
]
