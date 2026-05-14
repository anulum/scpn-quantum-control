# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 free-energy validation fixtures
"""Executable simulator fixtures for Paper 0 EQ0130-EQ0131 anchors."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any, cast

import numpy as np

from .spec_loader import load_free_energy_validation_spec


@dataclass(frozen=True, slots=True)
class FreeEnergyConfig:
    """Finite probability-simplex settings for the EQ0130-EQ0131 fixture."""

    q_theta_mu: np.ndarray | None = None
    p_theta_y: np.ndarray | None = None
    p_y_theta: np.ndarray | None = None
    evidence: float = 0.7
    spec_bundle_path: Path | None = None

    def __post_init__(self) -> None:
        q = _validate_probability_vector(
            "q_theta_mu",
            self.q_theta_mu
            if self.q_theta_mu is not None
            else np.array([0.2, 0.5, 0.3], dtype=np.float64),
        )
        p = _validate_probability_vector(
            "p_theta_y",
            self.p_theta_y
            if self.p_theta_y is not None
            else np.array([0.25, 0.45, 0.3], dtype=np.float64),
        )
        likelihood = _validate_positive_vector(
            "p_y_theta",
            self.p_y_theta
            if self.p_y_theta is not None
            else np.array([0.7, 0.4, 0.9], dtype=np.float64),
        )
        if q.shape != p.shape or q.shape != likelihood.shape:
            raise ValueError("q_theta_mu, p_theta_y, and p_y_theta must have the same shape")
        if not np.isfinite(self.evidence) or self.evidence <= 0.0:
            raise ValueError("evidence must be finite and positive")
        object.__setattr__(self, "q_theta_mu", q)
        object.__setattr__(self, "p_theta_y", p)
        object.__setattr__(self, "p_y_theta", likelihood)


@dataclass(frozen=True, slots=True)
class FreeEnergyTerms:
    """Computed finite free-energy decomposition."""

    complexity_kl: float
    accuracy_loss: float
    total_free_energy: float
    surprise: float


@dataclass(frozen=True, slots=True)
class FreeEnergyValidationResult:
    """Source-anchored variational-free-energy validation result."""

    spec_key: str
    validation_protocol: str
    hardware_status: str
    source_equation_ids: tuple[str, ...]
    source_ledger_ids: tuple[str, ...]
    complexity_kl: float
    accuracy_loss: float
    total_free_energy: float
    surprise: float
    decomposition_residual: float
    surprise_upper_bound_margin: float
    null_controls: MappingProxyType[str, float]
    problem_metadata: MappingProxyType[str, Any]


def free_energy_terms(config: FreeEnergyConfig) -> FreeEnergyTerms:
    """Compute finite KL complexity, expected accuracy loss, and total F."""
    q = cast(np.ndarray, config.q_theta_mu)
    posterior = cast(np.ndarray, config.p_theta_y)
    likelihood = cast(np.ndarray, config.p_y_theta)
    complexity = float(np.sum(q * np.log(q / posterior)))
    accuracy = float(np.sum(q * (-np.log(likelihood))))
    total = complexity + accuracy
    surprise = float(-np.log(config.evidence))
    return FreeEnergyTerms(
        complexity_kl=complexity,
        accuracy_loss=accuracy,
        total_free_energy=total,
        surprise=surprise,
    )


def validate_variational_free_energy_fixture(
    config: FreeEnergyConfig | None = None,
) -> FreeEnergyValidationResult:
    """Run the source-anchored EQ0130-EQ0131 free-energy fixture."""
    cfg = config or FreeEnergyConfig()
    spec = load_free_energy_validation_spec(
        "computational.variational_free_energy",
        spec_bundle_path=cfg.spec_bundle_path,
    )
    terms = free_energy_terms(cfg)
    identical = free_energy_terms(replace(cfg, p_theta_y=cast(np.ndarray, cfg.q_theta_mu)))
    controls = {
        "identical_density_kl_abs": abs(identical.complexity_kl),
        "support_mismatch_rejection_label": _support_mismatch_rejection_label(),
        "zero_likelihood_rejection_label": _zero_likelihood_rejection_label(),
    }
    metadata = {
        "paper0_spec_key": "computational.variational_free_energy",
        "paper0_validation_protocol": str(spec["validation_protocol"]),
        "hardware_status": str(spec["hardware_status"]),
        "state_count": int(cast(np.ndarray, cfg.q_theta_mu).size),
        "evidence": float(cfg.evidence),
        "simulator_only_mechanism_evidence": True,
        "claim_boundary": "finite_probability_decomposition_not_neurobiological_confirmation",
    }
    return FreeEnergyValidationResult(
        spec_key="computational.variational_free_energy",
        validation_protocol=str(spec["validation_protocol"]),
        hardware_status=str(spec["hardware_status"]),
        source_equation_ids=tuple(str(item) for item in spec["source_equation_ids"]),
        source_ledger_ids=tuple(str(item) for item in spec["source_ledger_ids"]),
        complexity_kl=terms.complexity_kl,
        accuracy_loss=terms.accuracy_loss,
        total_free_energy=terms.total_free_energy,
        surprise=terms.surprise,
        decomposition_residual=abs(
            terms.total_free_energy - (terms.complexity_kl + terms.accuracy_loss)
        ),
        surprise_upper_bound_margin=terms.total_free_energy - terms.surprise,
        null_controls=MappingProxyType(controls),
        problem_metadata=MappingProxyType(metadata),
    )


def _support_mismatch_rejection_label() -> float:
    try:
        FreeEnergyConfig(q_theta_mu=np.array([0.5, 0.5]), p_theta_y=np.array([0.5, 0.0]))
    except ValueError as exc:
        return float("strictly positive" in str(exc))
    return 0.0


def _zero_likelihood_rejection_label() -> float:
    try:
        FreeEnergyConfig(p_y_theta=np.array([0.8, 0.0, 0.3]))
    except ValueError as exc:
        return float("strictly positive" in str(exc))
    return 0.0


def _validate_probability_vector(name: str, values: np.ndarray) -> np.ndarray:
    arr = _validate_positive_vector(name, values)
    total = float(np.sum(arr))
    if total <= np.finfo(np.float64).eps:
        raise ValueError(f"{name} must have positive mass")
    return cast(np.ndarray, arr / total)


def _validate_positive_vector(name: str, values: np.ndarray) -> np.ndarray:
    arr = np.array(values, dtype=np.float64, copy=True)
    if arr.ndim != 1 or arr.size < 2:
        raise ValueError(f"{name} must be a one-dimensional vector with at least two entries")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    if np.any(arr <= 0.0):
        raise ValueError(f"{name} must be strictly positive")
    return arr


__all__ = [
    "FreeEnergyConfig",
    "FreeEnergyTerms",
    "FreeEnergyValidationResult",
    "free_energy_terms",
    "validate_variational_free_energy_fixture",
]
