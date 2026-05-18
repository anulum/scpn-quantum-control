# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 IV. The Coherence Backbone (MS-QEC and Symmetry) validation
"""Source-accounting checks for Paper 0 IV. The Coherence Backbone (MS-QEC and Symmetry) records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

CLAIM_BOUNDARY = "source-bounded iv the coherence backbone ms qec and symmetry source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"
SOURCE_LEDGER_SPAN = ("P0R06123", "P0R06131")


@dataclass(frozen=True, slots=True)
class IvTheCoherenceBackboneMsQecAndSymmetryConfig:
    """Configuration for this Paper 0 source-accounting fixture."""

    expected_source_record_count: int = 9
    expected_component_count: int = 4
    next_source_boundary: str = "P0R06132"

    def __post_init__(self) -> None:
        if self.expected_source_record_count != 9:
            raise ValueError("expected_source_record_count must equal 9")
        if self.expected_component_count != 4:
            raise ValueError("expected_component_count must equal 4")
        if self.next_source_boundary != "P0R06132":
            raise ValueError("next_source_boundary must equal P0R06132")


@dataclass(frozen=True, slots=True)
class IvTheCoherenceBackboneMsQecAndSymmetryFixtureResult:
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


def classify_iv_the_coherence_backbone_ms_qec_and_symmetry_component(component: str) -> str:
    """Classify source-defined components."""
    mapping = {
        "iv_the_coherence_backbone_ms_qec_and_symmetry": "iv_the_coherence_backbone_ms_qec_and_symmetry_source_boundary",
        "v_the_architecture_of_time_mmc_tsvf_and_synchronicity": "v_the_architecture_of_time_mmc_tsvf_and_synchronicity_source_boundary",
        "vi_thermodynamics_and_energetics": "vi_thermodynamics_and_energetics_source_boundary",
        "vii_the_scpn_measurement_postulate_intrinsic_measurement": "vii_the_scpn_measurement_postulate_intrinsic_measurement_source_boundary",
    }
    try:
        return mapping[component]
    except KeyError as exc:
        raise ValueError(
            "unknown iv_the_coherence_backbone_ms_qec_and_symmetry component"
        ) from exc


def iv_the_coherence_backbone_ms_qec_and_symmetry_labels() -> dict[str, str]:
    """Return source-bounded labels for this slice."""
    return {
        "section": "IV. The Coherence Backbone (MS-QEC and Symmetry)",
        "source_span": "P0R06123-P0R06131",
        "component_count": "4",
        "next_boundary": "P0R06132",
        "component_1": "IV. The Coherence Backbone (MS-QEC and Symmetry)",
        "component_2": "V. The Architecture of Time (MMC, TSVF, and Synchronicity)",
        "component_3": "VI. Thermodynamics and Energetics",
        "component_4": "VII. The SCPN Measurement Postulate (Intrinsic Measurement)",
    }


def validate_iv_the_coherence_backbone_ms_qec_and_symmetry_fixture(
    config: IvTheCoherenceBackboneMsQecAndSymmetryConfig | None = None,
) -> IvTheCoherenceBackboneMsQecAndSymmetryFixtureResult:
    """Validate source accounting for this Paper 0 slice."""
    cfg = config or IvTheCoherenceBackboneMsQecAndSymmetryConfig()
    components = (
        "iv_the_coherence_backbone_ms_qec_and_symmetry",
        "v_the_architecture_of_time_mmc_tsvf_and_synchronicity",
        "vi_thermodynamics_and_energetics",
        "vii_the_scpn_measurement_postulate_intrinsic_measurement",
    )
    return IvTheCoherenceBackboneMsQecAndSymmetryFixtureResult(
        source_ledger_span=SOURCE_LEDGER_SPAN,
        hardware_status=HARDWARE_STATUS,
        claim_boundary=CLAIM_BOUNDARY,
        components={
            component: classify_iv_the_coherence_backbone_ms_qec_and_symmetry_component(component)
            for component in components
        },
        labels=iv_the_coherence_backbone_ms_qec_and_symmetry_labels(),
        source_record_count=cfg.expected_source_record_count,
        component_count=cfg.expected_component_count,
        next_source_boundary=cfg.next_source_boundary,
        null_controls={
            "iv_the_coherence_backbone_ms_qec_and_symmetry_is_not_empirical_validation_evidence": 1.0,
            "v_the_architecture_of_time_mmc_tsvf_and_synchronicity_is_not_empirical_validation_evidence": 1.0,
            "vi_thermodynamics_and_energetics_is_not_empirical_validation_evidence": 1.0,
            "vii_the_scpn_measurement_postulate_intrinsic_measurement_is_not_empirical_validation_evidence": 1.0,
        },
        problem_metadata={
            "source_ledger_span": SOURCE_LEDGER_SPAN,
            "source_ledger_ids": tuple(f"P0R{number:05d}" for number in range(6123, 6132)),
            "claim_boundary": CLAIM_BOUNDARY,
            "protocol_state": "source_iv_the_coherence_backbone_ms_qec_and_symmetry_only_no_experiment",
        },
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "HARDWARE_STATUS",
    "SOURCE_LEDGER_SPAN",
    "IvTheCoherenceBackboneMsQecAndSymmetryConfig",
    "IvTheCoherenceBackboneMsQecAndSymmetryFixtureResult",
    "classify_iv_the_coherence_backbone_ms_qec_and_symmetry_component",
    "iv_the_coherence_backbone_ms_qec_and_symmetry_labels",
    "validate_iv_the_coherence_backbone_ms_qec_and_symmetry_fixture",
]
