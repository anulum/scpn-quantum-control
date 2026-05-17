# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 operational pullback protocol validation
"""Source-accounting checks for Paper 0 operational-pullback protocol records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

CLAIM_BOUNDARY = "source-bounded operational pullback protocol; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"
SOURCE_LEDGER_SPAN = ("P0R01242", "P0R01271")


@dataclass(frozen=True, slots=True)
class OperationalPullbackProtocolConfig:
    """Configuration for the operational-pullback protocol fixture."""

    expected_source_record_count: int = 30
    expected_component_count: int = 6
    next_source_boundary: str = "P0R01272"

    def __post_init__(self) -> None:
        if self.expected_source_record_count != 30:
            raise ValueError("expected_source_record_count must equal 30")
        if self.expected_component_count != 6:
            raise ValueError("expected_component_count must equal 6")
        if self.next_source_boundary != "P0R01272":
            raise ValueError("next_source_boundary must equal P0R01272")


@dataclass(frozen=True, slots=True)
class OperationalPullbackProtocolFixtureResult:
    """Result for the Paper 0 operational-pullback protocol fixture."""

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


def classify_operational_pullback_protocol_component(component: str) -> str:
    """Classify source-defined operational-pullback protocol components."""
    mapping = {
        "section_and_protocol_boundary": "ssb_section_operational_pullback_protocol_boundary",
        "statistical_bundle_and_fim": "statistical_bundle_section_and_fim_source_definition",
        "spacetime_pullback_and_normalisation": "fim_spacetime_pullback_and_lambda_i_normalisation",
        "observable_sections_and_l4_l5_case": "observable_sections_l4_l5_case_and_nv_prediction_boundary",
        "full_covariance_fim_strategy": "full_covariance_fim_computation_requirement",
        "eft_lorentz_locality_constraints": "eft_lorentz_locality_constraint_boundary",
    }
    try:
        return mapping[component]
    except KeyError as exc:
        raise ValueError("unknown operational-pullback protocol component") from exc


def operational_pullback_protocol_labels() -> dict[str, str]:
    """Return source-bounded labels for the operational-pullback protocol slice."""
    return {
        "section": "2.3 The Physics of Form: Spontaneous Symmetry Breaking",
        "protocol": "Operational Pullback Protocol Revision 11.00",
        "fim": "I_ij(theta) = E[partial_i log p partial_j log p]",
        "pullback": "g_F_mu_nu = partial_mu theta I partial_nu theta",
        "next_boundary": "The Physics of Form: Spontaneous Symmetry Breaking and the Psi-Field",
    }


def validate_operational_pullback_protocol_fixture(
    config: OperationalPullbackProtocolConfig | None = None,
) -> OperationalPullbackProtocolFixtureResult:
    """Validate source accounting for the operational-pullback protocol slice."""
    cfg = config or OperationalPullbackProtocolConfig()
    components = (
        "section_and_protocol_boundary",
        "statistical_bundle_and_fim",
        "spacetime_pullback_and_normalisation",
        "observable_sections_and_l4_l5_case",
        "full_covariance_fim_strategy",
        "eft_lorentz_locality_constraints",
    )

    return OperationalPullbackProtocolFixtureResult(
        source_ledger_span=SOURCE_LEDGER_SPAN,
        hardware_status=HARDWARE_STATUS,
        claim_boundary=CLAIM_BOUNDARY,
        components={
            component: classify_operational_pullback_protocol_component(component)
            for component in components
        },
        labels=operational_pullback_protocol_labels(),
        source_record_count=cfg.expected_source_record_count,
        component_count=cfg.expected_component_count,
        next_source_boundary=cfg.next_source_boundary,
        null_controls={
            "operational_pullback_protocol_is_source_protocol_not_measurement": 1.0,
            "nv_centre_prediction_is_not_experimental_evidence": 1.0,
            "diagonal_or_mean_only_fim_shortcut_rejected_for_full_covariance_protocol": 1.0,
        },
        problem_metadata={
            "source_ledger_span": SOURCE_LEDGER_SPAN,
            "source_ledger_ids": tuple(f"P0R{number:05d}" for number in range(1242, 1272)),
            "claim_boundary": CLAIM_BOUNDARY,
            "protocol_state": "source_operational_pullback_protocol_only_no_experiment",
        },
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "HARDWARE_STATUS",
    "SOURCE_LEDGER_SPAN",
    "OperationalPullbackProtocolConfig",
    "OperationalPullbackProtocolFixtureResult",
    "classify_operational_pullback_protocol_component",
    "operational_pullback_protocol_labels",
    "validate_operational_pullback_protocol_fixture",
]
