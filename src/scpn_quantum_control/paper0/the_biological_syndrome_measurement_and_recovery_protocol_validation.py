# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 The Biological Syndrome Measurement and Recovery Protocol validation
"""Source-accounting checks for Paper 0 The Biological Syndrome Measurement and Recovery Protocol records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

CLAIM_BOUNDARY = "source-bounded the biological syndrome measurement and recovery protocol source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"
SOURCE_LEDGER_SPAN = ("P0R03076", "P0R03098")


@dataclass(frozen=True, slots=True)
class TheBiologicalSyndromeMeasurementAndRecoveryProtocolConfig:
    """Configuration for this Paper 0 source-accounting fixture."""

    expected_source_record_count: int = 23
    expected_component_count: int = 1
    next_source_boundary: str = "P0R03099"

    def __post_init__(self) -> None:
        if self.expected_source_record_count != 23:
            raise ValueError("expected_source_record_count must equal 23")
        if self.expected_component_count != 1:
            raise ValueError("expected_component_count must equal 1")
        if self.next_source_boundary != "P0R03099":
            raise ValueError("next_source_boundary must equal P0R03099")


@dataclass(frozen=True, slots=True)
class TheBiologicalSyndromeMeasurementAndRecoveryProtocolFixtureResult:
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


def classify_the_biological_syndrome_measurement_and_recovery_protocol_component(
    component: str,
) -> str:
    """Classify source-defined components."""
    mapping = {
        "the_biological_syndrome_measurement_and_recovery_protocol": "the_biological_syndrome_measurement_and_recovery_protocol_source_boundary"
    }
    try:
        return mapping[component]
    except KeyError as exc:
        raise ValueError(
            "unknown the_biological_syndrome_measurement_and_recovery_protocol component"
        ) from exc


def the_biological_syndrome_measurement_and_recovery_protocol_labels() -> dict[str, str]:
    """Return source-bounded labels for this slice."""
    return {
        "section": "The Biological Syndrome Measurement and Recovery Protocol",
        "source_span": "P0R03076-P0R03098",
        "component_count": "1",
        "next_boundary": "P0R03099",
        "component_1": "The Biological Syndrome Measurement and Recovery Protocol",
    }


def validate_the_biological_syndrome_measurement_and_recovery_protocol_fixture(
    config: TheBiologicalSyndromeMeasurementAndRecoveryProtocolConfig | None = None,
) -> TheBiologicalSyndromeMeasurementAndRecoveryProtocolFixtureResult:
    """Validate source accounting for this Paper 0 slice."""
    cfg = config or TheBiologicalSyndromeMeasurementAndRecoveryProtocolConfig()
    components = ("the_biological_syndrome_measurement_and_recovery_protocol",)
    return TheBiologicalSyndromeMeasurementAndRecoveryProtocolFixtureResult(
        source_ledger_span=SOURCE_LEDGER_SPAN,
        hardware_status=HARDWARE_STATUS,
        claim_boundary=CLAIM_BOUNDARY,
        components={
            component: classify_the_biological_syndrome_measurement_and_recovery_protocol_component(
                component
            )
            for component in components
        },
        labels=the_biological_syndrome_measurement_and_recovery_protocol_labels(),
        source_record_count=cfg.expected_source_record_count,
        component_count=cfg.expected_component_count,
        next_source_boundary=cfg.next_source_boundary,
        null_controls={
            "the_biological_syndrome_measurement_and_recovery_protocol_is_not_empirical_validation_evidence": 1.0
        },
        problem_metadata={
            "source_ledger_span": SOURCE_LEDGER_SPAN,
            "source_ledger_ids": tuple(f"P0R{number:05d}" for number in range(3076, 3099)),
            "claim_boundary": CLAIM_BOUNDARY,
            "protocol_state": "source_the_biological_syndrome_measurement_and_recovery_protocol_only_no_experiment",
        },
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "HARDWARE_STATUS",
    "SOURCE_LEDGER_SPAN",
    "TheBiologicalSyndromeMeasurementAndRecoveryProtocolConfig",
    "TheBiologicalSyndromeMeasurementAndRecoveryProtocolFixtureResult",
    "classify_the_biological_syndrome_measurement_and_recovery_protocol_component",
    "the_biological_syndrome_measurement_and_recovery_protocol_labels",
    "validate_the_biological_syndrome_measurement_and_recovery_protocol_fixture",
]
