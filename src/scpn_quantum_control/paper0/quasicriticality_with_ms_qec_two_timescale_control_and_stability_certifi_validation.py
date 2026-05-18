# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 Quasicriticality with MS-QEC: Two-Timescale Control and Stability Certificates (revision 11.07) validation
"""Source-accounting checks for Paper 0 Quasicriticality with MS-QEC: Two-Timescale Control and Stability Certificates (revision 11.07) records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

CLAIM_BOUNDARY = "source-bounded quasicriticality with ms qec two timescale control and stability certifi source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"
SOURCE_LEDGER_SPAN = ("P0R02983", "P0R02990")


@dataclass(frozen=True, slots=True)
class QuasicriticalityWithMsQecTwoTimescaleControlAndStabilityCertifiConfig:
    """Configuration for this Paper 0 source-accounting fixture."""

    expected_source_record_count: int = 8
    expected_component_count: int = 3
    next_source_boundary: str = "P0R02991"

    def __post_init__(self) -> None:
        if self.expected_source_record_count != 8:
            raise ValueError("expected_source_record_count must equal 8")
        if self.expected_component_count != 3:
            raise ValueError("expected_component_count must equal 3")
        if self.next_source_boundary != "P0R02991":
            raise ValueError("next_source_boundary must equal P0R02991")


@dataclass(frozen=True, slots=True)
class QuasicriticalityWithMsQecTwoTimescaleControlAndStabilityCertifiFixtureResult:
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


def classify_quasicriticality_with_ms_qec_two_timescale_control_and_stability_certifi_component(
    component: str,
) -> str:
    """Classify source-defined components."""
    mapping = {
        "quasicriticality_with_ms_qec_two_timescale_control_and_stability_certifi": "quasicriticality_with_ms_qec_two_timescale_control_and_stability_certifi_source_boundary",
        "two_timescale_structure": "two_timescale_structure_source_boundary",
        "gain_scheduling_via_affective_field_sensitivity": "gain_scheduling_via_affective_field_sensitivity_source_boundary",
    }
    try:
        return mapping[component]
    except KeyError as exc:
        raise ValueError(
            "unknown quasicriticality_with_ms_qec_two_timescale_control_and_stability_certifi component"
        ) from exc


def quasicriticality_with_ms_qec_two_timescale_control_and_stability_certifi_labels() -> dict[
    str, str
]:
    """Return source-bounded labels for this slice."""
    return {
        "section": "Quasicriticality with MS-QEC: Two-Timescale Control and Stability Certificates (revision 11.07)",
        "source_span": "P0R02983-P0R02990",
        "component_count": "3",
        "next_boundary": "P0R02991",
        "component_1": "Quasicriticality with MS-QEC: Two-Timescale Control and Stability Certificates (revision 11.07)",
        "component_2": "Two-Timescale Structure:",
        "component_3": "Gain Scheduling via Affective Field Sensitivity:",
    }


def validate_quasicriticality_with_ms_qec_two_timescale_control_and_stability_certifi_fixture(
    config: QuasicriticalityWithMsQecTwoTimescaleControlAndStabilityCertifiConfig | None = None,
) -> QuasicriticalityWithMsQecTwoTimescaleControlAndStabilityCertifiFixtureResult:
    """Validate source accounting for this Paper 0 slice."""
    cfg = config or QuasicriticalityWithMsQecTwoTimescaleControlAndStabilityCertifiConfig()
    components = (
        "quasicriticality_with_ms_qec_two_timescale_control_and_stability_certifi",
        "two_timescale_structure",
        "gain_scheduling_via_affective_field_sensitivity",
    )
    return QuasicriticalityWithMsQecTwoTimescaleControlAndStabilityCertifiFixtureResult(
        source_ledger_span=SOURCE_LEDGER_SPAN,
        hardware_status=HARDWARE_STATUS,
        claim_boundary=CLAIM_BOUNDARY,
        components={
            component: classify_quasicriticality_with_ms_qec_two_timescale_control_and_stability_certifi_component(
                component
            )
            for component in components
        },
        labels=quasicriticality_with_ms_qec_two_timescale_control_and_stability_certifi_labels(),
        source_record_count=cfg.expected_source_record_count,
        component_count=cfg.expected_component_count,
        next_source_boundary=cfg.next_source_boundary,
        null_controls={
            "quasicriticality_with_ms_qec_two_timescale_control_and_stability_certifi_is_not_empirical_validation_evidence": 1.0,
            "two_timescale_structure_is_not_empirical_validation_evidence": 1.0,
            "gain_scheduling_via_affective_field_sensitivity_is_not_empirical_validation_evidence": 1.0,
        },
        problem_metadata={
            "source_ledger_span": SOURCE_LEDGER_SPAN,
            "source_ledger_ids": tuple(f"P0R{number:05d}" for number in range(2983, 2991)),
            "claim_boundary": CLAIM_BOUNDARY,
            "protocol_state": "source_quasicriticality_with_ms_qec_two_timescale_control_and_stability_certifi_only_no_experiment",
        },
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "HARDWARE_STATUS",
    "SOURCE_LEDGER_SPAN",
    "QuasicriticalityWithMsQecTwoTimescaleControlAndStabilityCertifiConfig",
    "QuasicriticalityWithMsQecTwoTimescaleControlAndStabilityCertifiFixtureResult",
    "classify_quasicriticality_with_ms_qec_two_timescale_control_and_stability_certifi_component",
    "quasicriticality_with_ms_qec_two_timescale_control_and_stability_certifi_labels",
    "validate_quasicriticality_with_ms_qec_two_timescale_control_and_stability_certifi_fixture",
]
