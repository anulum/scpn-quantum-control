# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 II. The Universal Dynamic Regime: Quasicriticality validation
"""Source-accounting checks for Paper 0 II. The Universal Dynamic Regime: Quasicriticality records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

CLAIM_BOUNDARY = "source-bounded ii the universal dynamic regime quasicriticality source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"
SOURCE_LEDGER_SPAN = ("P0R02513", "P0R02520")


@dataclass(frozen=True, slots=True)
class IiTheUniversalDynamicRegimeQuasicriticalityConfig:
    """Configuration for this Paper 0 source-accounting fixture."""

    expected_source_record_count: int = 8
    expected_component_count: int = 1
    next_source_boundary: str = "P0R02521"

    def __post_init__(self) -> None:
        if self.expected_source_record_count != 8:
            raise ValueError("expected_source_record_count must equal 8")
        if self.expected_component_count != 1:
            raise ValueError("expected_component_count must equal 1")
        if self.next_source_boundary != "P0R02521":
            raise ValueError("next_source_boundary must equal P0R02521")


@dataclass(frozen=True, slots=True)
class IiTheUniversalDynamicRegimeQuasicriticalityFixtureResult:
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


def classify_ii_the_universal_dynamic_regime_quasicriticality_component(component: str) -> str:
    """Classify source-defined components."""
    mapping = {
        "ii_the_universal_dynamic_regime_quasicriticality": "ii_the_universal_dynamic_regime_quasicriticality_source_boundary"
    }
    try:
        return mapping[component]
    except KeyError as exc:
        raise ValueError(
            "unknown ii_the_universal_dynamic_regime_quasicriticality component"
        ) from exc


def ii_the_universal_dynamic_regime_quasicriticality_labels() -> dict[str, str]:
    """Return source-bounded labels for this slice."""
    return {
        "section": "II. The Universal Dynamic Regime: Quasicriticality",
        "source_span": "P0R02513-P0R02520",
        "component_count": "1",
        "next_boundary": "P0R02521",
        "component_1": "II. The Universal Dynamic Regime: Quasicriticality",
    }


def validate_ii_the_universal_dynamic_regime_quasicriticality_fixture(
    config: IiTheUniversalDynamicRegimeQuasicriticalityConfig | None = None,
) -> IiTheUniversalDynamicRegimeQuasicriticalityFixtureResult:
    """Validate source accounting for this Paper 0 slice."""
    cfg = config or IiTheUniversalDynamicRegimeQuasicriticalityConfig()
    components = ("ii_the_universal_dynamic_regime_quasicriticality",)
    return IiTheUniversalDynamicRegimeQuasicriticalityFixtureResult(
        source_ledger_span=SOURCE_LEDGER_SPAN,
        hardware_status=HARDWARE_STATUS,
        claim_boundary=CLAIM_BOUNDARY,
        components={
            component: classify_ii_the_universal_dynamic_regime_quasicriticality_component(
                component
            )
            for component in components
        },
        labels=ii_the_universal_dynamic_regime_quasicriticality_labels(),
        source_record_count=cfg.expected_source_record_count,
        component_count=cfg.expected_component_count,
        next_source_boundary=cfg.next_source_boundary,
        null_controls={
            "ii_the_universal_dynamic_regime_quasicriticality_is_not_empirical_validation_evidence": 1.0
        },
        problem_metadata={
            "source_ledger_span": SOURCE_LEDGER_SPAN,
            "source_ledger_ids": tuple(f"P0R{number:05d}" for number in range(2513, 2521)),
            "claim_boundary": CLAIM_BOUNDARY,
            "protocol_state": "source_ii_the_universal_dynamic_regime_quasicriticality_only_no_experiment",
        },
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "HARDWARE_STATUS",
    "SOURCE_LEDGER_SPAN",
    "IiTheUniversalDynamicRegimeQuasicriticalityConfig",
    "IiTheUniversalDynamicRegimeQuasicriticalityFixtureResult",
    "classify_ii_the_universal_dynamic_regime_quasicriticality_component",
    "ii_the_universal_dynamic_regime_quasicriticality_labels",
    "validate_ii_the_universal_dynamic_regime_quasicriticality_fixture",
]
