# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 V. The Integrated Body Matrix (Fascia and Tensegrity) validation
"""Source-accounting checks for Paper 0 V. The Integrated Body Matrix (Fascia and Tensegrity) records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

CLAIM_BOUNDARY = "source-bounded v the integrated body matrix fascia and tensegrity source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"
SOURCE_LEDGER_SPAN = ("P0R04943", "P0R04955")


@dataclass(frozen=True, slots=True)
class VTheIntegratedBodyMatrixFasciaAndTensegrityConfig:
    """Configuration for this Paper 0 source-accounting fixture."""

    expected_source_record_count: int = 13
    expected_component_count: int = 2
    next_source_boundary: str = "P0R04956"

    def __post_init__(self) -> None:
        if self.expected_source_record_count != 13:
            raise ValueError("expected_source_record_count must equal 13")
        if self.expected_component_count != 2:
            raise ValueError("expected_component_count must equal 2")
        if self.next_source_boundary != "P0R04956":
            raise ValueError("next_source_boundary must equal P0R04956")


@dataclass(frozen=True, slots=True)
class VTheIntegratedBodyMatrixFasciaAndTensegrityFixtureResult:
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


def classify_v_the_integrated_body_matrix_fascia_and_tensegrity_component(component: str) -> str:
    """Classify source-defined components."""
    mapping = {
        "v_the_integrated_body_matrix_fascia_and_tensegrity": "v_the_integrated_body_matrix_fascia_and_tensegrity_source_boundary",
        "vi_synthesis_the_holistic_pathology_of_the_scpn": "vi_synthesis_the_holistic_pathology_of_the_scpn_source_boundary",
    }
    try:
        return mapping[component]
    except KeyError as exc:
        raise ValueError(
            "unknown v_the_integrated_body_matrix_fascia_and_tensegrity component"
        ) from exc


def v_the_integrated_body_matrix_fascia_and_tensegrity_labels() -> dict[str, str]:
    """Return source-bounded labels for this slice."""
    return {
        "section": "V. The Integrated Body Matrix (Fascia and Tensegrity)",
        "source_span": "P0R04943-P0R04955",
        "component_count": "2",
        "next_boundary": "P0R04956",
        "component_1": "V. The Integrated Body Matrix (Fascia and Tensegrity)",
        "component_2": "VI. Synthesis: The Holistic Pathology of the SCPN",
    }


def validate_v_the_integrated_body_matrix_fascia_and_tensegrity_fixture(
    config: VTheIntegratedBodyMatrixFasciaAndTensegrityConfig | None = None,
) -> VTheIntegratedBodyMatrixFasciaAndTensegrityFixtureResult:
    """Validate source accounting for this Paper 0 slice."""
    cfg = config or VTheIntegratedBodyMatrixFasciaAndTensegrityConfig()
    components = (
        "v_the_integrated_body_matrix_fascia_and_tensegrity",
        "vi_synthesis_the_holistic_pathology_of_the_scpn",
    )
    return VTheIntegratedBodyMatrixFasciaAndTensegrityFixtureResult(
        source_ledger_span=SOURCE_LEDGER_SPAN,
        hardware_status=HARDWARE_STATUS,
        claim_boundary=CLAIM_BOUNDARY,
        components={
            component: classify_v_the_integrated_body_matrix_fascia_and_tensegrity_component(
                component
            )
            for component in components
        },
        labels=v_the_integrated_body_matrix_fascia_and_tensegrity_labels(),
        source_record_count=cfg.expected_source_record_count,
        component_count=cfg.expected_component_count,
        next_source_boundary=cfg.next_source_boundary,
        null_controls={
            "v_the_integrated_body_matrix_fascia_and_tensegrity_is_not_empirical_validation_evidence": 1.0,
            "vi_synthesis_the_holistic_pathology_of_the_scpn_is_not_empirical_validation_evidence": 1.0,
        },
        problem_metadata={
            "source_ledger_span": SOURCE_LEDGER_SPAN,
            "source_ledger_ids": tuple(f"P0R{number:05d}" for number in range(4943, 4956)),
            "claim_boundary": CLAIM_BOUNDARY,
            "protocol_state": "source_v_the_integrated_body_matrix_fascia_and_tensegrity_only_no_experiment",
        },
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "HARDWARE_STATUS",
    "SOURCE_LEDGER_SPAN",
    "VTheIntegratedBodyMatrixFasciaAndTensegrityConfig",
    "VTheIntegratedBodyMatrixFasciaAndTensegrityFixtureResult",
    "classify_v_the_integrated_body_matrix_fascia_and_tensegrity_component",
    "v_the_integrated_body_matrix_fascia_and_tensegrity_labels",
    "validate_v_the_integrated_body_matrix_fascia_and_tensegrity_fixture",
]
