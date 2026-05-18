# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 III. The Coherence Backbone: Multi-Scale Quantum Error Correction (MS-QEC) validation
"""Source-accounting checks for Paper 0 III. The Coherence Backbone: Multi-Scale Quantum Error Correction (MS-QEC) records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

CLAIM_BOUNDARY = "source-bounded iii the coherence backbone multi scale quantum error correction ms qec source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"
SOURCE_LEDGER_SPAN = ("P0R02521", "P0R02531")


@dataclass(frozen=True, slots=True)
class IiiTheCoherenceBackboneMultiScaleQuantumErrorCorrectionMsQecConfig:
    """Configuration for this Paper 0 source-accounting fixture."""

    expected_source_record_count: int = 11
    expected_component_count: int = 1
    next_source_boundary: str = "P0R02532"

    def __post_init__(self) -> None:
        if self.expected_source_record_count != 11:
            raise ValueError("expected_source_record_count must equal 11")
        if self.expected_component_count != 1:
            raise ValueError("expected_component_count must equal 1")
        if self.next_source_boundary != "P0R02532":
            raise ValueError("next_source_boundary must equal P0R02532")


@dataclass(frozen=True, slots=True)
class IiiTheCoherenceBackboneMultiScaleQuantumErrorCorrectionMsQecFixtureResult:
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


def classify_iii_the_coherence_backbone_multi_scale_quantum_error_correction_ms_qec_component(
    component: str,
) -> str:
    """Classify source-defined components."""
    mapping = {
        "iii_the_coherence_backbone_multi_scale_quantum_error_correction_ms_qec": "iii_the_coherence_backbone_multi_scale_quantum_error_correction_ms_qec_source_boundary"
    }
    try:
        return mapping[component]
    except KeyError as exc:
        raise ValueError(
            "unknown iii_the_coherence_backbone_multi_scale_quantum_error_correction_ms_qec component"
        ) from exc


def iii_the_coherence_backbone_multi_scale_quantum_error_correction_ms_qec_labels() -> dict[
    str, str
]:
    """Return source-bounded labels for this slice."""
    return {
        "section": "III. The Coherence Backbone: Multi-Scale Quantum Error Correction (MS-QEC)",
        "source_span": "P0R02521-P0R02531",
        "component_count": "1",
        "next_boundary": "P0R02532",
        "component_1": "III. The Coherence Backbone: Multi-Scale Quantum Error Correction (MS-QEC)",
    }


def validate_iii_the_coherence_backbone_multi_scale_quantum_error_correction_ms_qec_fixture(
    config: IiiTheCoherenceBackboneMultiScaleQuantumErrorCorrectionMsQecConfig | None = None,
) -> IiiTheCoherenceBackboneMultiScaleQuantumErrorCorrectionMsQecFixtureResult:
    """Validate source accounting for this Paper 0 slice."""
    cfg = config or IiiTheCoherenceBackboneMultiScaleQuantumErrorCorrectionMsQecConfig()
    components = ("iii_the_coherence_backbone_multi_scale_quantum_error_correction_ms_qec",)
    return IiiTheCoherenceBackboneMultiScaleQuantumErrorCorrectionMsQecFixtureResult(
        source_ledger_span=SOURCE_LEDGER_SPAN,
        hardware_status=HARDWARE_STATUS,
        claim_boundary=CLAIM_BOUNDARY,
        components={
            component: classify_iii_the_coherence_backbone_multi_scale_quantum_error_correction_ms_qec_component(
                component
            )
            for component in components
        },
        labels=iii_the_coherence_backbone_multi_scale_quantum_error_correction_ms_qec_labels(),
        source_record_count=cfg.expected_source_record_count,
        component_count=cfg.expected_component_count,
        next_source_boundary=cfg.next_source_boundary,
        null_controls={
            "iii_the_coherence_backbone_multi_scale_quantum_error_correction_ms_qec_is_not_empirical_validation_evidence": 1.0
        },
        problem_metadata={
            "source_ledger_span": SOURCE_LEDGER_SPAN,
            "source_ledger_ids": tuple(f"P0R{number:05d}" for number in range(2521, 2532)),
            "claim_boundary": CLAIM_BOUNDARY,
            "protocol_state": "source_iii_the_coherence_backbone_multi_scale_quantum_error_correction_ms_qec_only_no_experiment",
        },
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "HARDWARE_STATUS",
    "SOURCE_LEDGER_SPAN",
    "IiiTheCoherenceBackboneMultiScaleQuantumErrorCorrectionMsQecConfig",
    "IiiTheCoherenceBackboneMultiScaleQuantumErrorCorrectionMsQecFixtureResult",
    "classify_iii_the_coherence_backbone_multi_scale_quantum_error_correction_ms_qec_component",
    "iii_the_coherence_backbone_multi_scale_quantum_error_correction_ms_qec_labels",
    "validate_iii_the_coherence_backbone_multi_scale_quantum_error_correction_ms_qec_fixture",
]
