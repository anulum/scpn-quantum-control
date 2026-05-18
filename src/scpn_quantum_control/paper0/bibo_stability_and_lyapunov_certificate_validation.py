# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 BIBO Stability and Lyapunov Certificate: validation
"""Source-accounting checks for Paper 0 BIBO Stability and Lyapunov Certificate: records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

CLAIM_BOUNDARY = "source-bounded bibo stability and lyapunov certificate source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"
SOURCE_LEDGER_SPAN = ("P0R02991", "P0R03009")


@dataclass(frozen=True, slots=True)
class BiboStabilityAndLyapunovCertificateConfig:
    """Configuration for this Paper 0 source-accounting fixture."""

    expected_source_record_count: int = 19
    expected_component_count: int = 1
    next_source_boundary: str = "P0R03010"

    def __post_init__(self) -> None:
        if self.expected_source_record_count != 19:
            raise ValueError("expected_source_record_count must equal 19")
        if self.expected_component_count != 1:
            raise ValueError("expected_component_count must equal 1")
        if self.next_source_boundary != "P0R03010":
            raise ValueError("next_source_boundary must equal P0R03010")


@dataclass(frozen=True, slots=True)
class BiboStabilityAndLyapunovCertificateFixtureResult:
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


def classify_bibo_stability_and_lyapunov_certificate_component(component: str) -> str:
    """Classify source-defined components."""
    mapping = {
        "bibo_stability_and_lyapunov_certificate": "bibo_stability_and_lyapunov_certificate_source_boundary"
    }
    try:
        return mapping[component]
    except KeyError as exc:
        raise ValueError("unknown bibo_stability_and_lyapunov_certificate component") from exc


def bibo_stability_and_lyapunov_certificate_labels() -> dict[str, str]:
    """Return source-bounded labels for this slice."""
    return {
        "section": "BIBO Stability and Lyapunov Certificate:",
        "source_span": "P0R02991-P0R03009",
        "component_count": "1",
        "next_boundary": "P0R03010",
        "component_1": "BIBO Stability and Lyapunov Certificate:",
    }


def validate_bibo_stability_and_lyapunov_certificate_fixture(
    config: BiboStabilityAndLyapunovCertificateConfig | None = None,
) -> BiboStabilityAndLyapunovCertificateFixtureResult:
    """Validate source accounting for this Paper 0 slice."""
    cfg = config or BiboStabilityAndLyapunovCertificateConfig()
    components = ("bibo_stability_and_lyapunov_certificate",)
    return BiboStabilityAndLyapunovCertificateFixtureResult(
        source_ledger_span=SOURCE_LEDGER_SPAN,
        hardware_status=HARDWARE_STATUS,
        claim_boundary=CLAIM_BOUNDARY,
        components={
            component: classify_bibo_stability_and_lyapunov_certificate_component(component)
            for component in components
        },
        labels=bibo_stability_and_lyapunov_certificate_labels(),
        source_record_count=cfg.expected_source_record_count,
        component_count=cfg.expected_component_count,
        next_source_boundary=cfg.next_source_boundary,
        null_controls={
            "bibo_stability_and_lyapunov_certificate_is_not_empirical_validation_evidence": 1.0
        },
        problem_metadata={
            "source_ledger_span": SOURCE_LEDGER_SPAN,
            "source_ledger_ids": tuple(f"P0R{number:05d}" for number in range(2991, 3010)),
            "claim_boundary": CLAIM_BOUNDARY,
            "protocol_state": "source_bibo_stability_and_lyapunov_certificate_only_no_experiment",
        },
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "HARDWARE_STATUS",
    "SOURCE_LEDGER_SPAN",
    "BiboStabilityAndLyapunovCertificateConfig",
    "BiboStabilityAndLyapunovCertificateFixtureResult",
    "classify_bibo_stability_and_lyapunov_certificate_component",
    "bibo_stability_and_lyapunov_certificate_labels",
    "validate_bibo_stability_and_lyapunov_certificate_fixture",
]
