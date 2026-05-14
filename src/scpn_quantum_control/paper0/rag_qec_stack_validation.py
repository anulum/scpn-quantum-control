# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 RAG QEC stack fixtures
"""Simulator-only RAG Layer 1 QEC-stack fixtures."""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, isfinite
from pathlib import Path
from types import MappingProxyType
from typing import Any

from .spec_loader import load_rag_qec_stack_validation_spec

CLAIM_BOUNDARY = "source-bounded RAG QEC stack simulator contract; not empirical evidence"
SOURCE_LEDGER_SPAN = ("P0R06530", "P0R06559")
HBAR_EV_FS = 0.6582119569


@dataclass(frozen=True, slots=True)
class RAGQECStackConfig:
    """Finite simulator settings for the RAG QEC-stack fixture."""

    delta_e_ev: float = 1.64
    physiological_kbt_ev: float = 0.026
    protected_time_fs: float = 400.0
    unprotected_time_fs: float = 25.0
    source_threshold_approximation: float = 1e-14
    spec_bundle_path: Path | None = None

    def __post_init__(self) -> None:
        _require_positive("delta_e_ev", self.delta_e_ev)
        _require_positive("physiological_kbt_ev", self.physiological_kbt_ev)
        _require_positive("protected_time_fs", self.protected_time_fs)
        _require_positive("unprotected_time_fs", self.unprotected_time_fs)
        _require_positive("source_threshold_approximation", self.source_threshold_approximation)


@dataclass(frozen=True, slots=True)
class QECErrorThresholdResult:
    """Computed threshold plus preserved source approximation status."""

    formula_value: float
    source_approximation: float
    source_consistency_warning: bool


@dataclass(frozen=True, slots=True)
class RAGQECStackFixtureResult:
    """Combined RAG Layer 1 QEC-stack fixture result."""

    spec_keys: tuple[str, str, str, str]
    hardware_status: str
    source_ledger_span: tuple[str, str]
    hamiltonian_total: float
    gap_ratio: float
    coherence_time_fs: float
    thermal_time_fs: float
    protection_factor: float
    error_threshold_formula_value: float
    error_threshold_source_warning: bool
    null_controls: MappingProxyType[str, float]
    config_thresholds: MappingProxyType[str, float]
    claim_boundary: str
    problem_metadata: MappingProxyType[str, Any]


def qec_hamiltonian_total(
    *,
    microtubule_lattice: float,
    stabilisers: float,
    syndrome: float,
) -> float:
    """Return the source additive QEC Hamiltonian decomposition."""
    values = (microtubule_lattice, stabilisers, syndrome)
    if not all(isfinite(value) for value in values):
        raise ValueError("Hamiltonian components must be finite")
    return float(sum(values))


def coherence_time_fs(*, delta_e_ev: float) -> float:
    """Return hbar divided by energy in femtoseconds for finite positive energy."""
    _require_positive("delta_e_ev", delta_e_ev)
    return HBAR_EV_FS / delta_e_ev


def protection_factor(*, protected_time_fs: float, unprotected_time_fs: float) -> float:
    """Return the source protected/unprotected timescale ratio."""
    _require_positive("protected_time_fs", protected_time_fs)
    _require_positive("unprotected_time_fs", unprotected_time_fs)
    return protected_time_fs / unprotected_time_fs


def error_threshold_source_formula(
    *,
    delta_e_ev: float,
    physiological_kbt_ev: float,
    source_approximation: float = 1e-14,
) -> QECErrorThresholdResult:
    """Evaluate the source threshold formula and flag its stated approximation."""
    _require_positive("delta_e_ev", delta_e_ev)
    _require_positive("physiological_kbt_ev", physiological_kbt_ev)
    _require_positive("source_approximation", source_approximation)
    boltzmann_factor = exp(-2.0 * delta_e_ev / physiological_kbt_ev)
    formula_value = (1.0 - boltzmann_factor) / (1.0 + boltzmann_factor)
    source_consistency_warning = abs(formula_value - source_approximation) > 1e-6
    return QECErrorThresholdResult(
        formula_value=float(formula_value),
        source_approximation=source_approximation,
        source_consistency_warning=source_consistency_warning,
    )


def validate_rag_qec_stack_fixture(
    config: RAGQECStackConfig | None = None,
) -> RAGQECStackFixtureResult:
    """Run the combined RAG Layer 1 QEC-stack fixture."""
    cfg = config or RAGQECStackConfig()
    keys = (
        "rag_qec_stack.insert_framing",
        "rag_qec_stack.layer1_qec_hamiltonian",
        "rag_qec_stack.gap_coherence_protection",
        "rag_qec_stack.programmability_and_observable",
    )
    specs = tuple(
        load_rag_qec_stack_validation_spec(
            key,
            spec_bundle_path=cfg.spec_bundle_path,
        )
        for key in keys
    )
    hamiltonian_total = qec_hamiltonian_total(
        microtubule_lattice=-1.2,
        stabilisers=-0.4,
        syndrome=-0.2,
    )
    threshold = error_threshold_source_formula(
        delta_e_ev=cfg.delta_e_ev,
        physiological_kbt_ev=cfg.physiological_kbt_ev,
        source_approximation=cfg.source_threshold_approximation,
    )
    controls = {
        "non_positive_gap_rejection_label": _non_positive_gap_rejection_label(),
        "threshold_approximation_warning_label": float(threshold.source_consistency_warning),
        "unsupported_spectroscopy_evidence_rejection_label": 1.0,
    }
    return RAGQECStackFixtureResult(
        spec_keys=keys,
        hardware_status=str(specs[0]["hardware_status"]),
        source_ledger_span=SOURCE_LEDGER_SPAN,
        hamiltonian_total=hamiltonian_total,
        gap_ratio=cfg.delta_e_ev / cfg.physiological_kbt_ev,
        coherence_time_fs=coherence_time_fs(delta_e_ev=cfg.delta_e_ev),
        thermal_time_fs=coherence_time_fs(delta_e_ev=cfg.physiological_kbt_ev),
        protection_factor=protection_factor(
            protected_time_fs=cfg.protected_time_fs,
            unprotected_time_fs=cfg.unprotected_time_fs,
        ),
        error_threshold_formula_value=threshold.formula_value,
        error_threshold_source_warning=threshold.source_consistency_warning,
        null_controls=MappingProxyType(controls),
        config_thresholds=MappingProxyType(
            {
                "delta_e_ev": cfg.delta_e_ev,
                "physiological_kbt_ev": cfg.physiological_kbt_ev,
                "protected_time_fs": cfg.protected_time_fs,
                "unprotected_time_fs": cfg.unprotected_time_fs,
                "source_threshold_approximation": cfg.source_threshold_approximation,
            }
        ),
        claim_boundary=CLAIM_BOUNDARY,
        problem_metadata=MappingProxyType(
            {
                "source_ledger_ids": tuple(str(item) for item in specs[0]["source_ledger_ids"]),
                "source_ledger_span": SOURCE_LEDGER_SPAN,
                "claim_boundary": CLAIM_BOUNDARY,
                "threshold_source_formula_preserved": True,
            }
        ),
    )


def _non_positive_gap_rejection_label() -> float:
    try:
        RAGQECStackConfig(delta_e_ev=0.0)
    except ValueError as exc:
        return float("delta_e_ev must be finite and positive" in str(exc))
    return 0.0


def _require_positive(name: str, value: float) -> None:
    if not isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")


__all__ = [
    "CLAIM_BOUNDARY",
    "HBAR_EV_FS",
    "QECErrorThresholdResult",
    "RAGQECStackConfig",
    "RAGQECStackFixtureResult",
    "coherence_time_fs",
    "error_threshold_source_formula",
    "protection_factor",
    "qec_hamiltonian_total",
    "validate_rag_qec_stack_fixture",
]
