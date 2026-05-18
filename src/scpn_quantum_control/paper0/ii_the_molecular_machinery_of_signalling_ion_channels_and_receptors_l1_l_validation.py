# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 II. The Molecular Machinery of Signalling: Ion Channels and Receptors (L1/L2) validation
"""Source-accounting checks for Paper 0 II. The Molecular Machinery of Signalling: Ion Channels and Receptors (L1/L2) records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

CLAIM_BOUNDARY = "source-bounded ii the molecular machinery of signalling ion channels and receptors l1 l source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"
SOURCE_LEDGER_SPAN = ("P0R04769", "P0R04777")


@dataclass(frozen=True, slots=True)
class IiTheMolecularMachineryOfSignallingIonChannelsAndReceptorsL1LConfig:
    """Configuration for this Paper 0 source-accounting fixture."""

    expected_source_record_count: int = 9
    expected_component_count: int = 4
    next_source_boundary: str = "P0R04778"

    def __post_init__(self) -> None:
        if self.expected_source_record_count != 9:
            raise ValueError("expected_source_record_count must equal 9")
        if self.expected_component_count != 4:
            raise ValueError("expected_component_count must equal 4")
        if self.next_source_boundary != "P0R04778":
            raise ValueError("next_source_boundary must equal P0R04778")


@dataclass(frozen=True, slots=True)
class IiTheMolecularMachineryOfSignallingIonChannelsAndReceptorsL1LFixtureResult:
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


def classify_ii_the_molecular_machinery_of_signalling_ion_channels_and_receptors_l1_l_component(
    component: str,
) -> str:
    """Classify source-defined components."""
    mapping = {
        "ii_the_molecular_machinery_of_signalling_ion_channels_and_receptors_l1_l": "ii_the_molecular_machinery_of_signalling_ion_channels_and_receptors_l1_l_source_boundary",
        "1_the_architecture_of_gating_and_iet": "1_the_architecture_of_gating_and_iet_source_boundary",
        "2_quantum_effects_in_selectivity_and_binding_l1_l2": "2_quantum_effects_in_selectivity_and_binding_l1_l2_source_boundary",
        "3_qze_and_attentional_stabilisation": "3_qze_and_attentional_stabilisation_source_boundary",
    }
    try:
        return mapping[component]
    except KeyError as exc:
        raise ValueError(
            "unknown ii_the_molecular_machinery_of_signalling_ion_channels_and_receptors_l1_l component"
        ) from exc


def ii_the_molecular_machinery_of_signalling_ion_channels_and_receptors_l1_l_labels() -> dict[
    str, str
]:
    """Return source-bounded labels for this slice."""
    return {
        "section": "II. The Molecular Machinery of Signalling: Ion Channels and Receptors (L1/L2)",
        "source_span": "P0R04769-P0R04777",
        "component_count": "4",
        "next_boundary": "P0R04778",
        "component_1": "II. The Molecular Machinery of Signalling: Ion Channels and Receptors (L1/L2)",
        "component_2": "1. The Architecture of Gating and IET",
        "component_3": "2. Quantum Effects in Selectivity and Binding (L1/L2)",
        "component_4": "3. QZE and Attentional Stabilisation:",
    }


def validate_ii_the_molecular_machinery_of_signalling_ion_channels_and_receptors_l1_l_fixture(
    config: IiTheMolecularMachineryOfSignallingIonChannelsAndReceptorsL1LConfig | None = None,
) -> IiTheMolecularMachineryOfSignallingIonChannelsAndReceptorsL1LFixtureResult:
    """Validate source accounting for this Paper 0 slice."""
    cfg = config or IiTheMolecularMachineryOfSignallingIonChannelsAndReceptorsL1LConfig()
    components = (
        "ii_the_molecular_machinery_of_signalling_ion_channels_and_receptors_l1_l",
        "1_the_architecture_of_gating_and_iet",
        "2_quantum_effects_in_selectivity_and_binding_l1_l2",
        "3_qze_and_attentional_stabilisation",
    )
    return IiTheMolecularMachineryOfSignallingIonChannelsAndReceptorsL1LFixtureResult(
        source_ledger_span=SOURCE_LEDGER_SPAN,
        hardware_status=HARDWARE_STATUS,
        claim_boundary=CLAIM_BOUNDARY,
        components={
            component: classify_ii_the_molecular_machinery_of_signalling_ion_channels_and_receptors_l1_l_component(
                component
            )
            for component in components
        },
        labels=ii_the_molecular_machinery_of_signalling_ion_channels_and_receptors_l1_l_labels(),
        source_record_count=cfg.expected_source_record_count,
        component_count=cfg.expected_component_count,
        next_source_boundary=cfg.next_source_boundary,
        null_controls={
            "ii_the_molecular_machinery_of_signalling_ion_channels_and_receptors_l1_l_is_not_empirical_validation_evidence": 1.0,
            "1_the_architecture_of_gating_and_iet_is_not_empirical_validation_evidence": 1.0,
            "2_quantum_effects_in_selectivity_and_binding_l1_l2_is_not_empirical_validation_evidence": 1.0,
            "3_qze_and_attentional_stabilisation_is_not_empirical_validation_evidence": 1.0,
        },
        problem_metadata={
            "source_ledger_span": SOURCE_LEDGER_SPAN,
            "source_ledger_ids": tuple(f"P0R{number:05d}" for number in range(4769, 4778)),
            "claim_boundary": CLAIM_BOUNDARY,
            "protocol_state": "source_ii_the_molecular_machinery_of_signalling_ion_channels_and_receptors_l1_l_only_no_experiment",
        },
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "HARDWARE_STATUS",
    "SOURCE_LEDGER_SPAN",
    "IiTheMolecularMachineryOfSignallingIonChannelsAndReceptorsL1LConfig",
    "IiTheMolecularMachineryOfSignallingIonChannelsAndReceptorsL1LFixtureResult",
    "classify_ii_the_molecular_machinery_of_signalling_ion_channels_and_receptors_l1_l_component",
    "ii_the_molecular_machinery_of_signalling_ion_channels_and_receptors_l1_l_labels",
    "validate_ii_the_molecular_machinery_of_signalling_ion_channels_and_receptors_l1_l_fixture",
]
