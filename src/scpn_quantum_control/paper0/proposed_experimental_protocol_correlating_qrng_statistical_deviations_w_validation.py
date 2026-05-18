# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 Proposed Experimental Protocol: Correlating QRNG Statistical Deviations with Collective Coherence Metrics validation
"""Source-accounting checks for Paper 0 Proposed Experimental Protocol: Correlating QRNG Statistical Deviations with Collective Coherence Metrics records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

CLAIM_BOUNDARY = "source-bounded proposed experimental protocol correlating qrng statistical deviations w source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"
SOURCE_LEDGER_SPAN = ("P0R05217", "P0R05227")


@dataclass(frozen=True, slots=True)
class ProposedExperimentalProtocolCorrelatingQrngStatisticalDeviationsWConfig:
    """Configuration for this Paper 0 source-accounting fixture."""

    expected_source_record_count: int = 11
    expected_component_count: int = 3
    next_source_boundary: str = "P0R05228"

    def __post_init__(self) -> None:
        if self.expected_source_record_count != 11:
            raise ValueError("expected_source_record_count must equal 11")
        if self.expected_component_count != 3:
            raise ValueError("expected_component_count must equal 3")
        if self.next_source_boundary != "P0R05228":
            raise ValueError("next_source_boundary must equal P0R05228")


@dataclass(frozen=True, slots=True)
class ProposedExperimentalProtocolCorrelatingQrngStatisticalDeviationsWFixtureResult:
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


def classify_proposed_experimental_protocol_correlating_qrng_statistical_deviations_w_component(
    component: str,
) -> str:
    """Classify source-defined components."""
    mapping = {
        "proposed_experimental_protocol_correlating_qrng_statistical_deviations_w": "proposed_experimental_protocol_correlating_qrng_statistical_deviations_w_source_boundary",
        "apparatus": "apparatus_source_boundary",
        "protocol": "protocol_source_boundary",
    }
    try:
        return mapping[component]
    except KeyError as exc:
        raise ValueError(
            "unknown proposed_experimental_protocol_correlating_qrng_statistical_deviations_w component"
        ) from exc


def proposed_experimental_protocol_correlating_qrng_statistical_deviations_w_labels() -> dict[
    str, str
]:
    """Return source-bounded labels for this slice."""
    return {
        "section": "Proposed Experimental Protocol: Correlating QRNG Statistical Deviations with Collective Coherence Metrics",
        "source_span": "P0R05217-P0R05227",
        "component_count": "3",
        "next_boundary": "P0R05228",
        "component_1": "Proposed Experimental Protocol: Correlating QRNG Statistical Deviations with Collective Coherence Metrics",
        "component_2": "Apparatus",
        "component_3": "Protocol",
    }


def validate_proposed_experimental_protocol_correlating_qrng_statistical_deviations_w_fixture(
    config: ProposedExperimentalProtocolCorrelatingQrngStatisticalDeviationsWConfig | None = None,
) -> ProposedExperimentalProtocolCorrelatingQrngStatisticalDeviationsWFixtureResult:
    """Validate source accounting for this Paper 0 slice."""
    cfg = config or ProposedExperimentalProtocolCorrelatingQrngStatisticalDeviationsWConfig()
    components = (
        "proposed_experimental_protocol_correlating_qrng_statistical_deviations_w",
        "apparatus",
        "protocol",
    )
    return ProposedExperimentalProtocolCorrelatingQrngStatisticalDeviationsWFixtureResult(
        source_ledger_span=SOURCE_LEDGER_SPAN,
        hardware_status=HARDWARE_STATUS,
        claim_boundary=CLAIM_BOUNDARY,
        components={
            component: classify_proposed_experimental_protocol_correlating_qrng_statistical_deviations_w_component(
                component
            )
            for component in components
        },
        labels=proposed_experimental_protocol_correlating_qrng_statistical_deviations_w_labels(),
        source_record_count=cfg.expected_source_record_count,
        component_count=cfg.expected_component_count,
        next_source_boundary=cfg.next_source_boundary,
        null_controls={
            "proposed_experimental_protocol_correlating_qrng_statistical_deviations_w_is_not_empirical_validation_evidence": 1.0,
            "apparatus_is_not_empirical_validation_evidence": 1.0,
            "protocol_is_not_empirical_validation_evidence": 1.0,
        },
        problem_metadata={
            "source_ledger_span": SOURCE_LEDGER_SPAN,
            "source_ledger_ids": tuple(f"P0R{number:05d}" for number in range(5217, 5228)),
            "claim_boundary": CLAIM_BOUNDARY,
            "protocol_state": "source_proposed_experimental_protocol_correlating_qrng_statistical_deviations_w_only_no_experiment",
        },
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "HARDWARE_STATUS",
    "SOURCE_LEDGER_SPAN",
    "ProposedExperimentalProtocolCorrelatingQrngStatisticalDeviationsWConfig",
    "ProposedExperimentalProtocolCorrelatingQrngStatisticalDeviationsWFixtureResult",
    "classify_proposed_experimental_protocol_correlating_qrng_statistical_deviations_w_component",
    "proposed_experimental_protocol_correlating_qrng_statistical_deviations_w_labels",
    "validate_proposed_experimental_protocol_correlating_qrng_statistical_deviations_w_fixture",
]
