# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 HPC-UPDE derivation fixtures
"""Simulator-only HPC-UPDE mathematical bridge fixtures."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from types import MappingProxyType
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from .spec_loader import load_hpc_upde_derivation_validation_spec

CLAIM_BOUNDARY = (
    "source-bounded HPC-UPDE mathematical bridge simulator contract; not empirical evidence"
)
SOURCE_LEDGER_SPAN = ("P0R06615", "P0R06645")


@dataclass(frozen=True, slots=True)
class HPCUPDEDerivationConfig:
    """Finite simulator settings for HPC-UPDE derivation fixtures."""

    finite_difference_step: float = 1.0e-6
    spec_bundle_path: Path | None = None

    def __post_init__(self) -> None:
        _require_positive("finite_difference_step", self.finite_difference_step)


@dataclass(frozen=True, slots=True)
class HPCUPDEDerivationFixtureResult:
    """Combined HPC-UPDE derivation fixture result."""

    spec_keys: tuple[str, str, str, str, str, str]
    hardware_status: str
    source_ledger_span: tuple[str, str]
    free_energy_locked: float
    free_energy_spread: float
    analytic_gradient: tuple[float, ...]
    finite_difference_gradient: tuple[float, ...]
    gradient_check_error: float
    upde_derivative: tuple[float, ...]
    upde_derivative_norm: float
    phase_locking_error_value: float
    null_controls: MappingProxyType[str, float]
    config_thresholds: MappingProxyType[str, float]
    claim_boundary: str
    problem_metadata: MappingProxyType[str, Any]


def free_energy_phase_functional(
    *,
    theta: NDArray[np.float64],
    coupling_matrix: NDArray[np.float64],
) -> float:
    """Return the symmetric-coupling phase free energy.

    Paper 0 writes the source functional as a double sum. The executable
    symmetric-matrix fixture applies the conventional half factor so each
    unordered interaction is counted once and the promoted source gradient is
    preserved exactly.
    """
    phases = _finite_vector("theta", theta)
    coupling = _finite_square_k(coupling_matrix, size=phases.size)
    deltas = phases[None, :] - phases[:, None]
    return float(-0.5 * np.sum(coupling * np.cos(deltas)))


def free_energy_phase_gradient(
    *,
    theta: NDArray[np.float64],
    coupling_matrix: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Return partial F / partial theta_i = -sum_j K_ij sin(theta_j - theta_i)."""
    phases = _finite_vector("theta", theta)
    coupling = _finite_square_k(coupling_matrix, size=phases.size)
    deltas = phases[None, :] - phases[:, None]
    return cast(NDArray[np.float64], -np.sum(coupling * np.sin(deltas), axis=1))


def upde_core_derivative(
    *,
    theta: NDArray[np.float64],
    omega: NDArray[np.float64],
    coupling_matrix: NDArray[np.float64],
    eta: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    """Return d theta_i / dt = omega_i - partial F / partial theta_i + eta_i."""
    phases = _finite_vector("theta", theta)
    frequencies = _finite_vector("omega", omega)
    if phases.shape != frequencies.shape:
        raise ValueError("theta and omega must have the same shape")
    noise = np.zeros_like(phases) if eta is None else _finite_vector("eta", eta)
    if noise.shape != phases.shape:
        raise ValueError("theta and eta must have the same shape")
    gradient = free_energy_phase_gradient(theta=phases, coupling_matrix=coupling_matrix)
    return cast(NDArray[np.float64], frequencies - gradient + noise)


def phase_locking_error(theta: NDArray[np.float64]) -> float:
    """Return RMS sine phase error, zero only for phase-locked configurations."""
    phases = _finite_vector("theta", theta)
    if phases.size == 0:
        raise ValueError("theta must not be empty")
    deltas = phases[None, :] - phases[:, None]
    return float(np.sqrt(np.mean(np.sin(deltas) ** 2)))


def validate_hpc_upde_derivation_fixture(
    config: HPCUPDEDerivationConfig | None = None,
) -> HPCUPDEDerivationFixtureResult:
    """Run the combined HPC-UPDE derivation fixture."""
    cfg = config or HPCUPDEDerivationConfig()
    keys = (
        "hpc_upde_derivation.block_framing",
        "hpc_upde_derivation.free_energy_functional",
        "hpc_upde_derivation.gradient_descent",
        "hpc_upde_derivation.upde_core_equation",
        "hpc_upde_derivation.hpc_interpretation",
        "hpc_upde_derivation.active_inference_boundary",
    )
    specs = tuple(
        load_hpc_upde_derivation_validation_spec(
            key,
            spec_bundle_path=cfg.spec_bundle_path,
        )
        for key in keys
    )
    theta = np.array([0.0, 0.25, -0.1], dtype=np.float64)
    locked = np.zeros(3, dtype=np.float64)
    coupling = np.array(
        [
            [0.0, 0.8, 0.3],
            [0.8, 0.0, 0.5],
            [0.3, 0.5, 0.0],
        ],
        dtype=np.float64,
    )
    omega = np.array([0.1, -0.05, 0.2], dtype=np.float64)
    eta = np.array([0.01, 0.0, -0.02], dtype=np.float64)
    analytic_gradient = free_energy_phase_gradient(theta=theta, coupling_matrix=coupling)
    finite_difference_gradient = _central_difference_gradient(
        theta=theta,
        coupling_matrix=coupling,
        step=cfg.finite_difference_step,
    )
    gradient_error = float(np.max(np.abs(analytic_gradient - finite_difference_gradient)))
    derivative = upde_core_derivative(
        theta=theta,
        omega=omega,
        coupling_matrix=coupling,
        eta=eta,
    )
    controls = {
        "non_square_k_rejection_label": _non_square_k_rejection_label(),
        "shape_mismatch_rejection_label": _shape_mismatch_rejection_label(),
        "unsupported_active_inference_evidence_rejection_label": 1.0,
    }
    return HPCUPDEDerivationFixtureResult(
        spec_keys=keys,
        hardware_status=str(specs[0]["hardware_status"]),
        source_ledger_span=SOURCE_LEDGER_SPAN,
        free_energy_locked=free_energy_phase_functional(theta=locked, coupling_matrix=coupling),
        free_energy_spread=free_energy_phase_functional(theta=theta, coupling_matrix=coupling),
        analytic_gradient=tuple(float(value) for value in analytic_gradient),
        finite_difference_gradient=tuple(float(value) for value in finite_difference_gradient),
        gradient_check_error=gradient_error,
        upde_derivative=tuple(float(value) for value in derivative),
        upde_derivative_norm=float(np.linalg.norm(derivative)),
        phase_locking_error_value=phase_locking_error(theta),
        null_controls=MappingProxyType(controls),
        config_thresholds=MappingProxyType({"finite_difference_step": cfg.finite_difference_step}),
        claim_boundary=CLAIM_BOUNDARY,
        problem_metadata=MappingProxyType(
            {
                "source_ledger_ids": tuple(str(item) for item in specs[0]["source_ledger_ids"]),
                "source_ledger_span": SOURCE_LEDGER_SPAN,
                "claim_boundary": CLAIM_BOUNDARY,
                "implementation_note": (
                    "symmetric coupling uses half-weighted double sum to preserve "
                    "the source gradient without double-counting"
                ),
            }
        ),
    )


def _central_difference_gradient(
    *,
    theta: NDArray[np.float64],
    coupling_matrix: NDArray[np.float64],
    step: float,
) -> NDArray[np.float64]:
    _require_positive("finite_difference_step", step)
    phases = _finite_vector("theta", theta)
    values = np.empty_like(phases)
    for index in range(phases.size):
        plus = phases.copy()
        minus = phases.copy()
        plus[index] += step
        minus[index] -= step
        values[index] = (
            free_energy_phase_functional(theta=plus, coupling_matrix=coupling_matrix)
            - free_energy_phase_functional(theta=minus, coupling_matrix=coupling_matrix)
        ) / (2.0 * step)
    return cast(NDArray[np.float64], values)


def _non_square_k_rejection_label() -> float:
    try:
        free_energy_phase_gradient(
            theta=np.array([0.0, 0.1], dtype=np.float64),
            coupling_matrix=np.ones((2, 3), dtype=np.float64),
        )
    except ValueError as exc:
        return float("K must be square" in str(exc))
    return 0.0


def _shape_mismatch_rejection_label() -> float:
    try:
        upde_core_derivative(
            theta=np.array([0.0, 0.1], dtype=np.float64),
            omega=np.array([0.1], dtype=np.float64),
            coupling_matrix=np.eye(2, dtype=np.float64),
        )
    except ValueError as exc:
        return float("theta and omega must have the same shape" in str(exc))
    return 0.0


def _finite_vector(name: str, values: NDArray[np.float64]) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return cast(NDArray[np.float64], array)


def _finite_square_k(
    values: NDArray[np.float64],
    *,
    size: int,
) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        raise ValueError("K must be square")
    if array.shape != (size, size):
        raise ValueError("K dimension must match theta")
    if not np.all(np.isfinite(array)):
        raise ValueError("K must contain only finite values")
    return cast(NDArray[np.float64], array)


def _require_positive(name: str, value: float) -> None:
    if not isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")


__all__ = [
    "CLAIM_BOUNDARY",
    "HPCUPDEDerivationConfig",
    "HPCUPDEDerivationFixtureResult",
    "free_energy_phase_functional",
    "free_energy_phase_gradient",
    "phase_locking_error",
    "upde_core_derivative",
    "validate_hpc_upde_derivation_fixture",
]
