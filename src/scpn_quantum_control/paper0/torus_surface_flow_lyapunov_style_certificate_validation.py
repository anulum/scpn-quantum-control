# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 Torus surface flow: Lyapunov-style certificate. validation
"""Source-accounting checks for Paper 0 Torus surface flow: Lyapunov-style certificate. records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

CLAIM_BOUNDARY = "source-bounded torus surface flow lyapunov style certificate source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"
SOURCE_LEDGER_SPAN = ("P0R02975", "P0R02982")


@dataclass(frozen=True, slots=True)
class TorusSurfaceFlowLyapunovStyleCertificateConfig:
    """Configuration for this Paper 0 source-accounting fixture."""

    expected_source_record_count: int = 8
    expected_component_count: int = 3
    next_source_boundary: str = "P0R02983"

    def __post_init__(self) -> None:
        if self.expected_source_record_count != 8:
            raise ValueError("expected_source_record_count must equal 8")
        if self.expected_component_count != 3:
            raise ValueError("expected_component_count must equal 3")
        if self.next_source_boundary != "P0R02983":
            raise ValueError("next_source_boundary must equal P0R02983")


@dataclass(frozen=True, slots=True)
class TorusSurfaceFlowLyapunovStyleCertificateFixtureResult:
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


def classify_torus_surface_flow_lyapunov_style_certificate_component(component: str) -> str:
    """Classify source-defined components."""
    mapping = {
        "torus_surface_flow_lyapunov_style_certificate": "torus_surface_flow_lyapunov_style_certificate_source_boundary",
        "ms_qec_integration_fast_channel_realiser": "ms_qec_integration_fast_channel_realiser_source_boundary",
        "implementation_notes": "implementation_notes_source_boundary",
    }
    try:
        return mapping[component]
    except KeyError as exc:
        raise ValueError(
            "unknown torus_surface_flow_lyapunov_style_certificate component"
        ) from exc


def torus_surface_flow_lyapunov_style_certificate_labels() -> dict[str, str]:
    """Return source-bounded labels for this slice."""
    return {
        "section": "Torus surface flow: Lyapunov-style certificate.",
        "source_span": "P0R02975-P0R02982",
        "component_count": "3",
        "next_boundary": "P0R02983",
        "component_1": "Torus surface flow: Lyapunov-style certificate.",
        "component_2": "MS-QEC integration (fast channel realiser).",
        "component_3": "Implementation notes.",
    }


def validate_torus_surface_flow_lyapunov_style_certificate_fixture(
    config: TorusSurfaceFlowLyapunovStyleCertificateConfig | None = None,
) -> TorusSurfaceFlowLyapunovStyleCertificateFixtureResult:
    """Validate source accounting for this Paper 0 slice."""
    cfg = config or TorusSurfaceFlowLyapunovStyleCertificateConfig()
    components = (
        "torus_surface_flow_lyapunov_style_certificate",
        "ms_qec_integration_fast_channel_realiser",
        "implementation_notes",
    )
    return TorusSurfaceFlowLyapunovStyleCertificateFixtureResult(
        source_ledger_span=SOURCE_LEDGER_SPAN,
        hardware_status=HARDWARE_STATUS,
        claim_boundary=CLAIM_BOUNDARY,
        components={
            component: classify_torus_surface_flow_lyapunov_style_certificate_component(component)
            for component in components
        },
        labels=torus_surface_flow_lyapunov_style_certificate_labels(),
        source_record_count=cfg.expected_source_record_count,
        component_count=cfg.expected_component_count,
        next_source_boundary=cfg.next_source_boundary,
        null_controls={
            "torus_surface_flow_lyapunov_style_certificate_is_not_empirical_validation_evidence": 1.0,
            "ms_qec_integration_fast_channel_realiser_is_not_empirical_validation_evidence": 1.0,
            "implementation_notes_is_not_empirical_validation_evidence": 1.0,
        },
        problem_metadata={
            "source_ledger_span": SOURCE_LEDGER_SPAN,
            "source_ledger_ids": tuple(f"P0R{number:05d}" for number in range(2975, 2983)),
            "claim_boundary": CLAIM_BOUNDARY,
            "protocol_state": "source_torus_surface_flow_lyapunov_style_certificate_only_no_experiment",
        },
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "HARDWARE_STATUS",
    "SOURCE_LEDGER_SPAN",
    "TorusSurfaceFlowLyapunovStyleCertificateConfig",
    "TorusSurfaceFlowLyapunovStyleCertificateFixtureResult",
    "classify_torus_surface_flow_lyapunov_style_certificate_component",
    "torus_surface_flow_lyapunov_style_certificate_labels",
    "validate_torus_surface_flow_lyapunov_style_certificate_fixture",
]
