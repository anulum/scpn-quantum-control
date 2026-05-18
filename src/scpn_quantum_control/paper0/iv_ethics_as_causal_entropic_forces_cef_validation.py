# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 IV. Ethics as Causal Entropic Forces (CEF): validation
"""Source-accounting checks for Paper 0 IV. Ethics as Causal Entropic Forces (CEF): records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

CLAIM_BOUNDARY = "source-bounded iv ethics as causal entropic forces cef source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"
SOURCE_LEDGER_SPAN = ("P0R06107", "P0R06114")


@dataclass(frozen=True, slots=True)
class IvEthicsAsCausalEntropicForcesCefConfig:
    """Configuration for this Paper 0 source-accounting fixture."""

    expected_source_record_count: int = 8
    expected_component_count: int = 2
    next_source_boundary: str = "P0R06115"

    def __post_init__(self) -> None:
        if self.expected_source_record_count != 8:
            raise ValueError("expected_source_record_count must equal 8")
        if self.expected_component_count != 2:
            raise ValueError("expected_component_count must equal 2")
        if self.next_source_boundary != "P0R06115":
            raise ValueError("next_source_boundary must equal P0R06115")


@dataclass(frozen=True, slots=True)
class IvEthicsAsCausalEntropicForcesCefFixtureResult:
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


def classify_iv_ethics_as_causal_entropic_forces_cef_component(component: str) -> str:
    """Classify source-defined components."""
    mapping = {
        "iv_ethics_as_causal_entropic_forces_cef": "iv_ethics_as_causal_entropic_forces_cef_source_boundary",
        "overarching_principles_and_system_dynamics_in_short": "overarching_principles_and_system_dynamics_in_short_source_boundary",
    }
    try:
        return mapping[component]
    except KeyError as exc:
        raise ValueError("unknown iv_ethics_as_causal_entropic_forces_cef component") from exc


def iv_ethics_as_causal_entropic_forces_cef_labels() -> dict[str, str]:
    """Return source-bounded labels for this slice."""
    return {
        "section": "IV. Ethics as Causal Entropic Forces (CEF):",
        "source_span": "P0R06107-P0R06114",
        "component_count": "2",
        "next_boundary": "P0R06115",
        "component_1": "IV. Ethics as Causal Entropic Forces (CEF):",
        "component_2": "Overarching Principles and System Dynamics In short",
    }


def validate_iv_ethics_as_causal_entropic_forces_cef_fixture(
    config: IvEthicsAsCausalEntropicForcesCefConfig | None = None,
) -> IvEthicsAsCausalEntropicForcesCefFixtureResult:
    """Validate source accounting for this Paper 0 slice."""
    cfg = config or IvEthicsAsCausalEntropicForcesCefConfig()
    components = (
        "iv_ethics_as_causal_entropic_forces_cef",
        "overarching_principles_and_system_dynamics_in_short",
    )
    return IvEthicsAsCausalEntropicForcesCefFixtureResult(
        source_ledger_span=SOURCE_LEDGER_SPAN,
        hardware_status=HARDWARE_STATUS,
        claim_boundary=CLAIM_BOUNDARY,
        components={
            component: classify_iv_ethics_as_causal_entropic_forces_cef_component(component)
            for component in components
        },
        labels=iv_ethics_as_causal_entropic_forces_cef_labels(),
        source_record_count=cfg.expected_source_record_count,
        component_count=cfg.expected_component_count,
        next_source_boundary=cfg.next_source_boundary,
        null_controls={
            "iv_ethics_as_causal_entropic_forces_cef_is_not_empirical_validation_evidence": 1.0,
            "overarching_principles_and_system_dynamics_in_short_is_not_empirical_validation_evidence": 1.0,
        },
        problem_metadata={
            "source_ledger_span": SOURCE_LEDGER_SPAN,
            "source_ledger_ids": tuple(f"P0R{number:05d}" for number in range(6107, 6115)),
            "claim_boundary": CLAIM_BOUNDARY,
            "protocol_state": "source_iv_ethics_as_causal_entropic_forces_cef_only_no_experiment",
        },
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "HARDWARE_STATUS",
    "SOURCE_LEDGER_SPAN",
    "IvEthicsAsCausalEntropicForcesCefConfig",
    "IvEthicsAsCausalEntropicForcesCefFixtureResult",
    "classify_iv_ethics_as_causal_entropic_forces_cef_component",
    "iv_ethics_as_causal_entropic_forces_cef_labels",
    "validate_iv_ethics_as_causal_entropic_forces_cef_fixture",
]
