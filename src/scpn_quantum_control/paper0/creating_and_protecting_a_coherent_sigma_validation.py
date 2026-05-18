# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 Creating and Protecting a Coherent sigma: validation
"""Source-accounting checks for Paper 0 Creating and Protecting a Coherent sigma: records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

CLAIM_BOUNDARY = "source-bounded creating and protecting a coherent sigma source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"
SOURCE_LEDGER_SPAN = ("P0R03034", "P0R03041")


@dataclass(frozen=True, slots=True)
class CreatingAndProtectingACoherentSigmaConfig:
    """Configuration for this Paper 0 source-accounting fixture."""

    expected_source_record_count: int = 8
    expected_component_count: int = 3
    next_source_boundary: str = "P0R03042"

    def __post_init__(self) -> None:
        if self.expected_source_record_count != 8:
            raise ValueError("expected_source_record_count must equal 8")
        if self.expected_component_count != 3:
            raise ValueError("expected_component_count must equal 3")
        if self.next_source_boundary != "P0R03042":
            raise ValueError("next_source_boundary must equal P0R03042")


@dataclass(frozen=True, slots=True)
class CreatingAndProtectingACoherentSigmaFixtureResult:
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


def classify_creating_and_protecting_a_coherent_sigma_component(component: str) -> str:
    """Classify source-defined components."""
    mapping = {
        "creating_and_protecting_a_coherent_sigma": "creating_and_protecting_a_coherent_sigma_source_boundary",
        "the_hierarchy_of_protection": "the_hierarchy_of_protection_source_boundary",
        "the_coherence_backbone_multi_scale_quantum_error_correction_ms_qec": "the_coherence_backbone_multi_scale_quantum_error_correction_ms_qec_source_boundary",
    }
    try:
        return mapping[component]
    except KeyError as exc:
        raise ValueError("unknown creating_and_protecting_a_coherent_sigma component") from exc


def creating_and_protecting_a_coherent_sigma_labels() -> dict[str, str]:
    """Return source-bounded labels for this slice."""
    return {
        "section": "Creating and Protecting a Coherent sigma:",
        "source_span": "P0R03034-P0R03041",
        "component_count": "3",
        "next_boundary": "P0R03042",
        "component_1": "Creating and Protecting a Coherent sigma:",
        "component_2": "The Hierarchy of Protection:",
        "component_3": "The Coherence Backbone: Multi-Scale Quantum Error Correction (MS-QEC)",
    }


def validate_creating_and_protecting_a_coherent_sigma_fixture(
    config: CreatingAndProtectingACoherentSigmaConfig | None = None,
) -> CreatingAndProtectingACoherentSigmaFixtureResult:
    """Validate source accounting for this Paper 0 slice."""
    cfg = config or CreatingAndProtectingACoherentSigmaConfig()
    components = (
        "creating_and_protecting_a_coherent_sigma",
        "the_hierarchy_of_protection",
        "the_coherence_backbone_multi_scale_quantum_error_correction_ms_qec",
    )
    return CreatingAndProtectingACoherentSigmaFixtureResult(
        source_ledger_span=SOURCE_LEDGER_SPAN,
        hardware_status=HARDWARE_STATUS,
        claim_boundary=CLAIM_BOUNDARY,
        components={
            component: classify_creating_and_protecting_a_coherent_sigma_component(component)
            for component in components
        },
        labels=creating_and_protecting_a_coherent_sigma_labels(),
        source_record_count=cfg.expected_source_record_count,
        component_count=cfg.expected_component_count,
        next_source_boundary=cfg.next_source_boundary,
        null_controls={
            "creating_and_protecting_a_coherent_sigma_is_not_empirical_validation_evidence": 1.0,
            "the_hierarchy_of_protection_is_not_empirical_validation_evidence": 1.0,
            "the_coherence_backbone_multi_scale_quantum_error_correction_ms_qec_is_not_empirical_validation_evidence": 1.0,
        },
        problem_metadata={
            "source_ledger_span": SOURCE_LEDGER_SPAN,
            "source_ledger_ids": tuple(f"P0R{number:05d}" for number in range(3034, 3042)),
            "claim_boundary": CLAIM_BOUNDARY,
            "protocol_state": "source_creating_and_protecting_a_coherent_sigma_only_no_experiment",
        },
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "HARDWARE_STATUS",
    "SOURCE_LEDGER_SPAN",
    "CreatingAndProtectingACoherentSigmaConfig",
    "CreatingAndProtectingACoherentSigmaFixtureResult",
    "classify_creating_and_protecting_a_coherent_sigma_component",
    "creating_and_protecting_a_coherent_sigma_labels",
    "validate_creating_and_protecting_a_coherent_sigma_fixture",
]
