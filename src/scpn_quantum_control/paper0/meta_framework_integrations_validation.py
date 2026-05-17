# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 meta-framework integrations validation
"""Source-accounting checks for Paper 0 meta-framework integrations records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

CLAIM_BOUNDARY = "source-bounded meta-framework integrations bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"
SOURCE_LEDGER_SPAN = ("P0R01714", "P0R01726")


@dataclass(frozen=True, slots=True)
class MetaFrameworkIntegrationsConfig:
    """Configuration for the meta-framework integrations fixture."""

    expected_source_record_count: int = 13
    expected_component_count: int = 3
    next_source_boundary: str = "P0R01727"

    def __post_init__(self) -> None:
        if self.expected_source_record_count != 13:
            raise ValueError("expected_source_record_count must equal 13")
        if self.expected_component_count != 3:
            raise ValueError("expected_component_count must equal 3")
        if self.next_source_boundary != "P0R01727":
            raise ValueError("next_source_boundary must equal P0R01727")


@dataclass(frozen=True, slots=True)
class MetaFrameworkIntegrationsFixtureResult:
    """Result for the Paper 0 meta-framework integrations fixture."""

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


def classify_meta_framework_integrations_component(component: str) -> str:
    """Classify source-defined meta-framework integration components."""
    mapping = {
        "predictive_coding_flat_prior": "predictive_coding_flat_prior_hierarchy_boundary",
        "psi_s_field_coupling": "psi_s_h_int_coupling_source_boundary",
        "differentiated_sigma_interface": "differentiated_sigma_interface_source_boundary",
    }
    try:
        return mapping[component]
    except KeyError as exc:
        raise ValueError("unknown meta-framework integrations component") from exc


def meta_framework_integrations_labels() -> dict[str, str]:
    """Return source-bounded labels for the meta-framework integrations slice."""
    return {
        "section": "Meta-Framework Integrations",
        "predictive_coding": "flat prior to deep hierarchy",
        "coupling": "H_int = -lambda * Psi_s * sigma",
        "interface": "layered sigma variables",
        "next_boundary": "The Genesis of the Hierarchy: Sequential Symmetry Breaking (SSB)",
    }


def validate_meta_framework_integrations_fixture(
    config: MetaFrameworkIntegrationsConfig | None = None,
) -> MetaFrameworkIntegrationsFixtureResult:
    """Validate source accounting for the meta-framework integrations slice."""
    cfg = config or MetaFrameworkIntegrationsConfig()
    components = (
        "predictive_coding_flat_prior",
        "psi_s_field_coupling",
        "differentiated_sigma_interface",
    )

    return MetaFrameworkIntegrationsFixtureResult(
        source_ledger_span=SOURCE_LEDGER_SPAN,
        hardware_status=HARDWARE_STATUS,
        claim_boundary=CLAIM_BOUNDARY,
        components={
            component: classify_meta_framework_integrations_component(component)
            for component in components
        },
        labels=meta_framework_integrations_labels(),
        source_record_count=cfg.expected_source_record_count,
        component_count=cfg.expected_component_count,
        next_source_boundary=cfg.next_source_boundary,
        null_controls={
            "predictive_coding_mapping_is_not_measured_cosmic_inference": 1.0,
            "psi_s_coupling_claim_is_not_measured_interaction": 1.0,
            "differentiated_sigma_interface_is_not_empirical_layer_measurement": 1.0,
        },
        problem_metadata={
            "source_ledger_span": SOURCE_LEDGER_SPAN,
            "source_ledger_ids": tuple(f"P0R{number:05d}" for number in range(1714, 1727)),
            "claim_boundary": CLAIM_BOUNDARY,
            "protocol_state": "source_meta_framework_integrations_only_no_experiment",
        },
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "HARDWARE_STATUS",
    "SOURCE_LEDGER_SPAN",
    "MetaFrameworkIntegrationsConfig",
    "MetaFrameworkIntegrationsFixtureResult",
    "classify_meta_framework_integrations_component",
    "meta_framework_integrations_labels",
    "validate_meta_framework_integrations_fixture",
]
