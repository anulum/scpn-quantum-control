# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 advanced mechanisms validation fixtures
"""Simulator-only advanced mechanisms fixtures."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np

from .spec_loader import load_advanced_mechanisms_validation_spec

CLAIM_BOUNDARY = "source-bounded advanced-mechanisms simulator contract; not empirical evidence"
SOURCE_LEDGER_SPAN = ("P0R06382", "P0R06401")


@dataclass(frozen=True, slots=True)
class AdvancedMechanismsConfig:
    """Finite simulator settings for advanced mechanisms fixtures."""

    mechanism_threshold: float = 0.72
    consilium_threshold: float = 0.70
    spec_bundle_path: Path | None = None

    def __post_init__(self) -> None:
        _require_positive("mechanism_threshold", self.mechanism_threshold)
        _require_positive("consilium_threshold", self.consilium_threshold)


@dataclass(frozen=True, slots=True)
class AdvancedMechanismsFixtureResult:
    """Combined advanced mechanisms fixture result."""

    spec_keys: tuple[str, str, str, str]
    hardware_status: str
    gauge_transduction: float
    holographic_encoding: float
    holographic_retrieval: float
    consilium_pareto_support: float
    null_controls: MappingProxyType[str, float]
    config_thresholds: MappingProxyType[str, float]
    source_ledger_span: tuple[str, str]
    claim_boundary: str
    problem_metadata: MappingProxyType[str, Any]


def gauge_transduction_score(
    *,
    symbol_operator_strength: float,
    psi_resonance: float,
    local_gauge_transformation: float,
    field_connection_shift: float,
) -> float:
    """Score the source-bounded L7/L8 gauge-transduction channel."""
    values = _unit_interval_values(
        symbol_operator_strength,
        psi_resonance,
        local_gauge_transformation,
        field_connection_shift,
    )
    return float(np.prod(values) ** 0.25)


def holographic_encoding_score(
    *,
    l4_synchronisation: float,
    l1_quantum_bias: float,
    mera_boundary_mapping: float,
    bulk_entanglement_storage: float,
) -> float:
    """Score the source-bounded L4-to-L1-to-MERA encoding path."""
    values = _unit_interval_values(
        l4_synchronisation,
        l1_quantum_bias,
        mera_boundary_mapping,
        bulk_entanglement_storage,
    )
    return float(np.prod(values) ** 0.25)


def holographic_retrieval_score(
    *,
    cue_syndrome_match: float,
    qec_recovery_operator: float,
    geodesic_flow_trace: float,
    l1_l4_reconstruction_bias: float,
) -> float:
    """Score the source-bounded QEC/geodesic retrieval path."""
    values = _unit_interval_values(
        cue_syndrome_match,
        qec_recovery_operator,
        geodesic_flow_trace,
        l1_l4_reconstruction_bias,
    )
    return float(np.prod(values) ** 0.25)


def consilium_pareto_support_score(
    *,
    coherence: float,
    complexity: float,
    qualia: float,
    pareto_feasibility: float,
    dynamic_weighting: float,
    geodesic_dissonance_reduction: float,
) -> float:
    """Score the source-bounded C/K/Q Pareto and geodesic optimisation path."""
    values = _unit_interval_values(
        coherence,
        complexity,
        qualia,
        pareto_feasibility,
        dynamic_weighting,
        geodesic_dissonance_reduction,
    )
    objective_balance = float(np.mean(values[:3]))
    optimisation_support = float(np.prod(values[3:]) ** (1.0 / 3.0))
    return 0.5 * objective_balance + 0.5 * optimisation_support


def validate_advanced_mechanisms_fixture(
    config: AdvancedMechanismsConfig | None = None,
) -> AdvancedMechanismsFixtureResult:
    """Run the combined advanced mechanisms fixture."""
    cfg = config or AdvancedMechanismsConfig()
    keys = (
        "advanced_mechanisms.geometric_physical_transduction",
        "advanced_mechanisms.holographic_memory_encoding",
        "advanced_mechanisms.holographic_memory_retrieval",
        "advanced_mechanisms.consilium_multiobjective_optimisation",
    )
    specs = tuple(
        load_advanced_mechanisms_validation_spec(key, spec_bundle_path=cfg.spec_bundle_path)
        for key in keys
    )
    gauge = gauge_transduction_score(
        symbol_operator_strength=0.84,
        psi_resonance=0.82,
        local_gauge_transformation=0.88,
        field_connection_shift=0.79,
    )
    encoding = holographic_encoding_score(
        l4_synchronisation=0.83,
        l1_quantum_bias=0.78,
        mera_boundary_mapping=0.81,
        bulk_entanglement_storage=0.86,
    )
    retrieval = holographic_retrieval_score(
        cue_syndrome_match=0.8,
        qec_recovery_operator=0.84,
        geodesic_flow_trace=0.79,
        l1_l4_reconstruction_bias=0.82,
    )
    consilium = consilium_pareto_support_score(
        coherence=0.82,
        complexity=0.77,
        qualia=0.74,
        pareto_feasibility=0.86,
        dynamic_weighting=0.81,
        geodesic_dissonance_reduction=0.79,
    )
    controls = {
        "missing_connection_rejection_label": float(
            gauge_transduction_score(
                symbol_operator_strength=0.84,
                psi_resonance=0.82,
                local_gauge_transformation=0.88,
                field_connection_shift=0.0,
            )
            < cfg.mechanism_threshold
        ),
        "incomplete_memory_path_rejection_label": float(
            holographic_encoding_score(
                l4_synchronisation=0.83,
                l1_quantum_bias=0.0,
                mera_boundary_mapping=0.81,
                bulk_entanglement_storage=0.86,
            )
            < cfg.mechanism_threshold
        ),
        "unsupported_empirical_evidence_rejection_label": 1.0,
    }
    return AdvancedMechanismsFixtureResult(
        spec_keys=keys,
        hardware_status=str(specs[0]["hardware_status"]),
        gauge_transduction=gauge,
        holographic_encoding=encoding,
        holographic_retrieval=retrieval,
        consilium_pareto_support=consilium,
        null_controls=MappingProxyType(controls),
        config_thresholds=MappingProxyType(
            {
                "mechanism_threshold": cfg.mechanism_threshold,
                "consilium_threshold": cfg.consilium_threshold,
            }
        ),
        source_ledger_span=SOURCE_LEDGER_SPAN,
        claim_boundary=CLAIM_BOUNDARY,
        problem_metadata=MappingProxyType(
            {
                "source_ledger_ids": tuple(str(item) for item in specs[0]["source_ledger_ids"]),
                "source_ledger_span": SOURCE_LEDGER_SPAN,
                "claim_boundary": CLAIM_BOUNDARY,
            }
        ),
    )


def _unit_interval_values(*values: float) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(array)) or np.any(array < 0.0) or np.any(array > 1.0):
        raise ValueError("mechanism inputs must be in [0, 1]")
    return array


def _require_positive(name: str, value: float) -> None:
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")


__all__ = [
    "AdvancedMechanismsConfig",
    "AdvancedMechanismsFixtureResult",
    "CLAIM_BOUNDARY",
    "consilium_pareto_support_score",
    "gauge_transduction_score",
    "holographic_encoding_score",
    "holographic_retrieval_score",
    "validate_advanced_mechanisms_fixture",
]
