# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 I. The Unified Mathematical Description (The Dynamic Fiber Bundle) validation
"""Source-accounting checks for Paper 0 I. The Unified Mathematical Description (The Dynamic Fiber Bundle) records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

CLAIM_BOUNDARY = "source-bounded i the unified mathematical description the dynamic fiber bundle source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"
SOURCE_LEDGER_SPAN = ("P0R06115", "P0R06122")


@dataclass(frozen=True, slots=True)
class ITheUnifiedMathematicalDescriptionTheDynamicFiberBundleConfig:
    """Configuration for this Paper 0 source-accounting fixture."""

    expected_source_record_count: int = 8
    expected_component_count: int = 3
    next_source_boundary: str = "P0R06123"

    def __post_init__(self) -> None:
        if self.expected_source_record_count != 8:
            raise ValueError("expected_source_record_count must equal 8")
        if self.expected_component_count != 3:
            raise ValueError("expected_component_count must equal 3")
        if self.next_source_boundary != "P0R06123":
            raise ValueError("next_source_boundary must equal P0R06123")


@dataclass(frozen=True, slots=True)
class ITheUnifiedMathematicalDescriptionTheDynamicFiberBundleFixtureResult:
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


def classify_i_the_unified_mathematical_description_the_dynamic_fiber_bundle_component(
    component: str,
) -> str:
    """Classify source-defined components."""
    mapping = {
        "i_the_unified_mathematical_description_the_dynamic_fiber_bundle": "i_the_unified_mathematical_description_the_dynamic_fiber_bundle_source_boundary",
        "ii_the_unified_phase_dynamics_equation_upde_the_spine": "ii_the_unified_phase_dynamics_equation_upde_the_spine_source_boundary",
        "iii_the_universal_dynamic_regime_quasicriticality": "iii_the_universal_dynamic_regime_quasicriticality_source_boundary",
    }
    try:
        return mapping[component]
    except KeyError as exc:
        raise ValueError(
            "unknown i_the_unified_mathematical_description_the_dynamic_fiber_bundle component"
        ) from exc


def i_the_unified_mathematical_description_the_dynamic_fiber_bundle_labels() -> dict[str, str]:
    """Return source-bounded labels for this slice."""
    return {
        "section": "I. The Unified Mathematical Description (The Dynamic Fiber Bundle)",
        "source_span": "P0R06115-P0R06122",
        "component_count": "3",
        "next_boundary": "P0R06123",
        "component_1": "I. The Unified Mathematical Description (The Dynamic Fiber Bundle)",
        "component_2": "II. The Unified Phase Dynamics Equation (UPDE) - The Spine",
        "component_3": "III. The Universal Dynamic Regime (Quasicriticality)",
    }


def validate_i_the_unified_mathematical_description_the_dynamic_fiber_bundle_fixture(
    config: ITheUnifiedMathematicalDescriptionTheDynamicFiberBundleConfig | None = None,
) -> ITheUnifiedMathematicalDescriptionTheDynamicFiberBundleFixtureResult:
    """Validate source accounting for this Paper 0 slice."""
    cfg = config or ITheUnifiedMathematicalDescriptionTheDynamicFiberBundleConfig()
    components = (
        "i_the_unified_mathematical_description_the_dynamic_fiber_bundle",
        "ii_the_unified_phase_dynamics_equation_upde_the_spine",
        "iii_the_universal_dynamic_regime_quasicriticality",
    )
    return ITheUnifiedMathematicalDescriptionTheDynamicFiberBundleFixtureResult(
        source_ledger_span=SOURCE_LEDGER_SPAN,
        hardware_status=HARDWARE_STATUS,
        claim_boundary=CLAIM_BOUNDARY,
        components={
            component: classify_i_the_unified_mathematical_description_the_dynamic_fiber_bundle_component(
                component
            )
            for component in components
        },
        labels=i_the_unified_mathematical_description_the_dynamic_fiber_bundle_labels(),
        source_record_count=cfg.expected_source_record_count,
        component_count=cfg.expected_component_count,
        next_source_boundary=cfg.next_source_boundary,
        null_controls={
            "i_the_unified_mathematical_description_the_dynamic_fiber_bundle_is_not_empirical_validation_evidence": 1.0,
            "ii_the_unified_phase_dynamics_equation_upde_the_spine_is_not_empirical_validation_evidence": 1.0,
            "iii_the_universal_dynamic_regime_quasicriticality_is_not_empirical_validation_evidence": 1.0,
        },
        problem_metadata={
            "source_ledger_span": SOURCE_LEDGER_SPAN,
            "source_ledger_ids": tuple(f"P0R{number:05d}" for number in range(6115, 6123)),
            "claim_boundary": CLAIM_BOUNDARY,
            "protocol_state": "source_i_the_unified_mathematical_description_the_dynamic_fiber_bundle_only_no_experiment",
        },
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "HARDWARE_STATUS",
    "SOURCE_LEDGER_SPAN",
    "ITheUnifiedMathematicalDescriptionTheDynamicFiberBundleConfig",
    "ITheUnifiedMathematicalDescriptionTheDynamicFiberBundleFixtureResult",
    "classify_i_the_unified_mathematical_description_the_dynamic_fiber_bundle_component",
    "i_the_unified_mathematical_description_the_dynamic_fiber_bundle_labels",
    "validate_i_the_unified_mathematical_description_the_dynamic_fiber_bundle_fixture",
]
