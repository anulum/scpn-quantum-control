# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 Timing the Engine: UPDE Phase-Lags ($\tau_{ij}$) and Physiological Delays validation
"""Source-accounting checks for Paper 0 Timing the Engine: UPDE Phase-Lags ($\tau_{ij}$) and Physiological Delays records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

CLAIM_BOUNDARY = "source-bounded timing the engine upde phase lags tau ij and physiological delays source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"
SOURCE_LEDGER_SPAN = ("P0R02223", "P0R02236")


@dataclass(frozen=True, slots=True)
class TimingTheEngineUpdePhaseLagsTauIjAndPhysiologicalDelaysConfig:
    """Configuration for this Paper 0 source-accounting fixture."""

    expected_source_record_count: int = 14
    expected_component_count: int = 1
    next_source_boundary: str = "P0R02237"

    def __post_init__(self) -> None:
        if self.expected_source_record_count != 14:
            raise ValueError("expected_source_record_count must equal 14")
        if self.expected_component_count != 1:
            raise ValueError("expected_component_count must equal 1")
        if self.next_source_boundary != "P0R02237":
            raise ValueError("next_source_boundary must equal P0R02237")


@dataclass(frozen=True, slots=True)
class TimingTheEngineUpdePhaseLagsTauIjAndPhysiologicalDelaysFixtureResult:
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


def classify_timing_the_engine_upde_phase_lags_tau_ij_and_physiological_delays_component(
    component: str,
) -> str:
    """Classify source-defined components."""
    mapping = {
        "timing_the_engine_upde_phase_lags_tau_ij_and_physiological_delays": "timing_the_engine_upde_phase_lags_tau_ij_and_physiological_delays_source_boundary"
    }
    try:
        return mapping[component]
    except KeyError as exc:
        raise ValueError(
            "unknown timing_the_engine_upde_phase_lags_tau_ij_and_physiological_delays component"
        ) from exc


def timing_the_engine_upde_phase_lags_tau_ij_and_physiological_delays_labels() -> dict[str, str]:
    """Return source-bounded labels for this slice."""
    return {
        "section": "Timing the Engine: UPDE Phase-Lags ($\\tau_{ij}$) and Physiological Delays",
        "source_span": "P0R02223-P0R02236",
        "component_count": "1",
        "next_boundary": "P0R02237",
        "component_1": "Timing the Engine: UPDE Phase-Lags ($\\tau_{ij}$) and Physiological Delays",
    }


def validate_timing_the_engine_upde_phase_lags_tau_ij_and_physiological_delays_fixture(
    config: TimingTheEngineUpdePhaseLagsTauIjAndPhysiologicalDelaysConfig | None = None,
) -> TimingTheEngineUpdePhaseLagsTauIjAndPhysiologicalDelaysFixtureResult:
    """Validate source accounting for this Paper 0 slice."""
    cfg = config or TimingTheEngineUpdePhaseLagsTauIjAndPhysiologicalDelaysConfig()
    components = ("timing_the_engine_upde_phase_lags_tau_ij_and_physiological_delays",)
    return TimingTheEngineUpdePhaseLagsTauIjAndPhysiologicalDelaysFixtureResult(
        source_ledger_span=SOURCE_LEDGER_SPAN,
        hardware_status=HARDWARE_STATUS,
        claim_boundary=CLAIM_BOUNDARY,
        components={
            component: classify_timing_the_engine_upde_phase_lags_tau_ij_and_physiological_delays_component(
                component
            )
            for component in components
        },
        labels=timing_the_engine_upde_phase_lags_tau_ij_and_physiological_delays_labels(),
        source_record_count=cfg.expected_source_record_count,
        component_count=cfg.expected_component_count,
        next_source_boundary=cfg.next_source_boundary,
        null_controls={
            "timing_the_engine_upde_phase_lags_tau_ij_and_physiological_delays_is_not_empirical_validation_evidence": 1.0
        },
        problem_metadata={
            "source_ledger_span": SOURCE_LEDGER_SPAN,
            "source_ledger_ids": tuple(f"P0R{number:05d}" for number in range(2223, 2237)),
            "claim_boundary": CLAIM_BOUNDARY,
            "protocol_state": "source_timing_the_engine_upde_phase_lags_tau_ij_and_physiological_delays_only_no_experiment",
        },
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "HARDWARE_STATUS",
    "SOURCE_LEDGER_SPAN",
    "TimingTheEngineUpdePhaseLagsTauIjAndPhysiologicalDelaysConfig",
    "TimingTheEngineUpdePhaseLagsTauIjAndPhysiologicalDelaysFixtureResult",
    "classify_timing_the_engine_upde_phase_lags_tau_ij_and_physiological_delays_component",
    "timing_the_engine_upde_phase_lags_tau_ij_and_physiological_delays_labels",
    "validate_timing_the_engine_upde_phase_lags_tau_ij_and_physiological_delays_fixture",
]
