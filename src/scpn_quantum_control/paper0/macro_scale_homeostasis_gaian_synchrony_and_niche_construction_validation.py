# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 Macro-Scale Homeostasis: Gaian Synchrony and Niche Construction validation
"""Source-accounting checks for Paper 0 Macro-Scale Homeostasis: Gaian Synchrony and Niche Construction records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

CLAIM_BOUNDARY = "source-bounded macro scale homeostasis gaian synchrony and niche construction source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"
SOURCE_LEDGER_SPAN = ("P0R05528", "P0R05536")


@dataclass(frozen=True, slots=True)
class MacroScaleHomeostasisGaianSynchronyAndNicheConstructionConfig:
    """Configuration for this Paper 0 source-accounting fixture."""

    expected_source_record_count: int = 9
    expected_component_count: int = 2
    next_source_boundary: str = "P0R05537"

    def __post_init__(self) -> None:
        if self.expected_source_record_count != 9:
            raise ValueError("expected_source_record_count must equal 9")
        if self.expected_component_count != 2:
            raise ValueError("expected_component_count must equal 2")
        if self.next_source_boundary != "P0R05537":
            raise ValueError("next_source_boundary must equal P0R05537")


@dataclass(frozen=True, slots=True)
class MacroScaleHomeostasisGaianSynchronyAndNicheConstructionFixtureResult:
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


def classify_macro_scale_homeostasis_gaian_synchrony_and_niche_construction_component(
    component: str,
) -> str:
    """Classify source-defined components."""
    mapping = {
        "macro_scale_homeostasis_gaian_synchrony_and_niche_construction": "macro_scale_homeostasis_gaian_synchrony_and_niche_construction_source_boundary",
        "a_scale_invariant_principle_of_active_homeostasis": "a_scale_invariant_principle_of_active_homeostasis_source_boundary",
    }
    try:
        return mapping[component]
    except KeyError as exc:
        raise ValueError(
            "unknown macro_scale_homeostasis_gaian_synchrony_and_niche_construction component"
        ) from exc


def macro_scale_homeostasis_gaian_synchrony_and_niche_construction_labels() -> dict[str, str]:
    """Return source-bounded labels for this slice."""
    return {
        "section": "Macro-Scale Homeostasis: Gaian Synchrony and Niche Construction",
        "source_span": "P0R05528-P0R05536",
        "component_count": "2",
        "next_boundary": "P0R05537",
        "component_1": "Macro-Scale Homeostasis: Gaian Synchrony and Niche Construction",
        "component_2": "A Scale-Invariant Principle of Active Homeostasis",
    }


def validate_macro_scale_homeostasis_gaian_synchrony_and_niche_construction_fixture(
    config: MacroScaleHomeostasisGaianSynchronyAndNicheConstructionConfig | None = None,
) -> MacroScaleHomeostasisGaianSynchronyAndNicheConstructionFixtureResult:
    """Validate source accounting for this Paper 0 slice."""
    cfg = config or MacroScaleHomeostasisGaianSynchronyAndNicheConstructionConfig()
    components = (
        "macro_scale_homeostasis_gaian_synchrony_and_niche_construction",
        "a_scale_invariant_principle_of_active_homeostasis",
    )
    return MacroScaleHomeostasisGaianSynchronyAndNicheConstructionFixtureResult(
        source_ledger_span=SOURCE_LEDGER_SPAN,
        hardware_status=HARDWARE_STATUS,
        claim_boundary=CLAIM_BOUNDARY,
        components={
            component: classify_macro_scale_homeostasis_gaian_synchrony_and_niche_construction_component(
                component
            )
            for component in components
        },
        labels=macro_scale_homeostasis_gaian_synchrony_and_niche_construction_labels(),
        source_record_count=cfg.expected_source_record_count,
        component_count=cfg.expected_component_count,
        next_source_boundary=cfg.next_source_boundary,
        null_controls={
            "macro_scale_homeostasis_gaian_synchrony_and_niche_construction_is_not_empirical_validation_evidence": 1.0,
            "a_scale_invariant_principle_of_active_homeostasis_is_not_empirical_validation_evidence": 1.0,
        },
        problem_metadata={
            "source_ledger_span": SOURCE_LEDGER_SPAN,
            "source_ledger_ids": tuple(f"P0R{number:05d}" for number in range(5528, 5537)),
            "claim_boundary": CLAIM_BOUNDARY,
            "protocol_state": "source_macro_scale_homeostasis_gaian_synchrony_and_niche_construction_only_no_experiment",
        },
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "HARDWARE_STATUS",
    "SOURCE_LEDGER_SPAN",
    "MacroScaleHomeostasisGaianSynchronyAndNicheConstructionConfig",
    "MacroScaleHomeostasisGaianSynchronyAndNicheConstructionFixtureResult",
    "classify_macro_scale_homeostasis_gaian_synchrony_and_niche_construction_component",
    "macro_scale_homeostasis_gaian_synchrony_and_niche_construction_labels",
    "validate_macro_scale_homeostasis_gaian_synchrony_and_niche_construction_fixture",
]
