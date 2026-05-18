# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 IV. Sub-Synaptic and Axonal Architecture (L1-L3) validation
"""Source-accounting checks for Paper 0 IV. Sub-Synaptic and Axonal Architecture (L1-L3) records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

CLAIM_BOUNDARY = "source-bounded iv sub synaptic and axonal architecture l1 l3 source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"
SOURCE_LEDGER_SPAN = ("P0R04786", "P0R04793")


@dataclass(frozen=True, slots=True)
class IvSubSynapticAndAxonalArchitectureL1L3Config:
    """Configuration for this Paper 0 source-accounting fixture."""

    expected_source_record_count: int = 8
    expected_component_count: int = 4
    next_source_boundary: str = "P0R04794"

    def __post_init__(self) -> None:
        if self.expected_source_record_count != 8:
            raise ValueError("expected_source_record_count must equal 8")
        if self.expected_component_count != 4:
            raise ValueError("expected_component_count must equal 4")
        if self.next_source_boundary != "P0R04794":
            raise ValueError("next_source_boundary must equal P0R04794")


@dataclass(frozen=True, slots=True)
class IvSubSynapticAndAxonalArchitectureL1L3FixtureResult:
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


def classify_iv_sub_synaptic_and_axonal_architecture_l1_l3_component(component: str) -> str:
    """Classify source-defined components."""
    mapping = {
        "iv_sub_synaptic_and_axonal_architecture_l1_l3": "iv_sub_synaptic_and_axonal_architecture_l1_l3_source_boundary",
        "1_the_post_synaptic_density_psd": "1_the_post_synaptic_density_psd_source_boundary",
        "2_axonal_structure_and_transport": "2_axonal_structure_and_transport_source_boundary",
        "v_the_deep_quantum_milieu_l1": "v_the_deep_quantum_milieu_l1_source_boundary",
    }
    try:
        return mapping[component]
    except KeyError as exc:
        raise ValueError(
            "unknown iv_sub_synaptic_and_axonal_architecture_l1_l3 component"
        ) from exc


def iv_sub_synaptic_and_axonal_architecture_l1_l3_labels() -> dict[str, str]:
    """Return source-bounded labels for this slice."""
    return {
        "section": "IV. Sub-Synaptic and Axonal Architecture (L1-L3)",
        "source_span": "P0R04786-P0R04793",
        "component_count": "4",
        "next_boundary": "P0R04794",
        "component_1": "IV. Sub-Synaptic and Axonal Architecture (L1-L3)",
        "component_2": "1. The Post-Synaptic Density (PSD):",
        "component_3": "2. Axonal Structure and Transport:",
        "component_4": "V. The Deep Quantum Milieu (L1)",
    }


def validate_iv_sub_synaptic_and_axonal_architecture_l1_l3_fixture(
    config: IvSubSynapticAndAxonalArchitectureL1L3Config | None = None,
) -> IvSubSynapticAndAxonalArchitectureL1L3FixtureResult:
    """Validate source accounting for this Paper 0 slice."""
    cfg = config or IvSubSynapticAndAxonalArchitectureL1L3Config()
    components = (
        "iv_sub_synaptic_and_axonal_architecture_l1_l3",
        "1_the_post_synaptic_density_psd",
        "2_axonal_structure_and_transport",
        "v_the_deep_quantum_milieu_l1",
    )
    return IvSubSynapticAndAxonalArchitectureL1L3FixtureResult(
        source_ledger_span=SOURCE_LEDGER_SPAN,
        hardware_status=HARDWARE_STATUS,
        claim_boundary=CLAIM_BOUNDARY,
        components={
            component: classify_iv_sub_synaptic_and_axonal_architecture_l1_l3_component(component)
            for component in components
        },
        labels=iv_sub_synaptic_and_axonal_architecture_l1_l3_labels(),
        source_record_count=cfg.expected_source_record_count,
        component_count=cfg.expected_component_count,
        next_source_boundary=cfg.next_source_boundary,
        null_controls={
            "iv_sub_synaptic_and_axonal_architecture_l1_l3_is_not_empirical_validation_evidence": 1.0,
            "1_the_post_synaptic_density_psd_is_not_empirical_validation_evidence": 1.0,
            "2_axonal_structure_and_transport_is_not_empirical_validation_evidence": 1.0,
            "v_the_deep_quantum_milieu_l1_is_not_empirical_validation_evidence": 1.0,
        },
        problem_metadata={
            "source_ledger_span": SOURCE_LEDGER_SPAN,
            "source_ledger_ids": tuple(f"P0R{number:05d}" for number in range(4786, 4794)),
            "claim_boundary": CLAIM_BOUNDARY,
            "protocol_state": "source_iv_sub_synaptic_and_axonal_architecture_l1_l3_only_no_experiment",
        },
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "HARDWARE_STATUS",
    "SOURCE_LEDGER_SPAN",
    "IvSubSynapticAndAxonalArchitectureL1L3Config",
    "IvSubSynapticAndAxonalArchitectureL1L3FixtureResult",
    "classify_iv_sub_synaptic_and_axonal_architecture_l1_l3_component",
    "iv_sub_synaptic_and_axonal_architecture_l1_l3_labels",
    "validate_iv_sub_synaptic_and_axonal_architecture_l1_l3_fixture",
]
