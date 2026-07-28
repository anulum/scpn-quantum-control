# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — ENAQT transport scan
"""Bounded environment-assisted quantum transport (ENAQT) simulation.

This module implements the excitation-transport setting studied by Plenio and
Huelga (2008): one excitation hops on a finite site network, local Lindblad
dephasing suppresses site coherences, an irreversible target sink records
successful transfer, and a competing loss state records recombination.

The scanned observable is finite-horizon sink population, not a Kuramoto order
parameter. An intermediate optimum is scenario-specific evidence only; it is
not a universal optimum, biological calibration, synchronisation result, BKT
correspondence, consciousness proxy, hardware result, or noise-setpoint policy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray
from scipy.sparse.linalg import LinearOperator, expm_multiply

from ..dense_budget import GIB, DenseAllocationError, dense_budget_bytes

DEFAULT_GAMMA_GRID: Final[tuple[float, ...]] = (
    0.0,
    0.01,
    0.03,
    0.1,
    0.3,
    1.0,
    3.0,
    10.0,
    30.0,
)
_BOUND_TOLERANCE: Final[float] = 1e-10
_WORKSPACE_OBJECTS: Final[int] = 12


@dataclass(frozen=True, slots=True)
class ENAQTResult:
    """Result of one finite-horizon local-dephasing transport scan."""

    optimal_gamma: float
    optimal_efficiency: float
    gamma_values: NDArray[np.float64]
    efficiency_values: NDArray[np.float64]
    coherent_efficiency: float
    high_noise_efficiency: float
    enhancement: float
    has_intermediate_optimum: bool
    source_site: int
    target_site: int
    t_evolve: float
    sink_rate: float
    loss_rate: float

    def __post_init__(self) -> None:
        """Freeze arrays and validate the public evidence contract."""
        gamma_values = _readonly_float_vector("gamma_values", self.gamma_values)
        efficiency_values = _readonly_float_vector("efficiency_values", self.efficiency_values)
        if gamma_values.shape != efficiency_values.shape:
            raise ValueError("gamma_values and efficiency_values must have equal shape")
        if np.any(gamma_values < 0.0):
            raise ValueError("gamma_values must be non-negative")
        if np.any(efficiency_values < -_BOUND_TOLERANCE) or np.any(
            efficiency_values > 1.0 + _BOUND_TOLERANCE
        ):
            raise ValueError("efficiency_values must lie in [0, 1]")
        for name, value in (
            ("optimal_gamma", self.optimal_gamma),
            ("optimal_efficiency", self.optimal_efficiency),
            ("coherent_efficiency", self.coherent_efficiency),
            ("high_noise_efficiency", self.high_noise_efficiency),
            ("enhancement", self.enhancement),
            ("t_evolve", self.t_evolve),
            ("sink_rate", self.sink_rate),
            ("loss_rate", self.loss_rate),
        ):
            _require_finite(name, value)
        if self.optimal_gamma < 0.0:
            raise ValueError("optimal_gamma must be non-negative")
        for name, value in (
            ("optimal_efficiency", self.optimal_efficiency),
            ("coherent_efficiency", self.coherent_efficiency),
            ("high_noise_efficiency", self.high_noise_efficiency),
        ):
            if not -_BOUND_TOLERANCE <= value <= 1.0 + _BOUND_TOLERANCE:
                raise ValueError(f"{name} must lie in [0, 1]")
        if self.enhancement < 0.0:
            raise ValueError("enhancement must be non-negative")
        if not isinstance(self.has_intermediate_optimum, bool):
            raise ValueError("has_intermediate_optimum must be boolean")
        if any(
            isinstance(site, bool) or not isinstance(site, int) or site < 0
            for site in (self.source_site, self.target_site)
        ):
            raise ValueError("source_site and target_site must be non-negative integers")
        if self.source_site == self.target_site:
            raise ValueError("source_site and target_site must differ")
        if self.t_evolve <= 0.0 or self.sink_rate <= 0.0 or self.loss_rate < 0.0:
            raise ValueError("time and sink rate must be positive; loss rate non-negative")
        matching_gamma = np.flatnonzero(
            np.isclose(gamma_values, self.optimal_gamma, rtol=1e-12, atol=1e-12)
        )
        if matching_gamma.size == 0:
            raise ValueError("optimal_gamma must be present in gamma_values")
        if not math.isclose(
            self.optimal_efficiency,
            float(np.max(efficiency_values)),
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError("optimal_efficiency must equal the maximum scanned efficiency")
        zero_gamma = np.flatnonzero(np.isclose(gamma_values, 0.0, rtol=0.0, atol=1e-15))
        if zero_gamma.size and not np.allclose(
            efficiency_values[zero_gamma],
            self.coherent_efficiency,
            rtol=1e-12,
            atol=1e-12,
        ):
            raise ValueError("coherent_efficiency disagrees with the gamma-zero row")
        high_noise_index = int(np.argmax(gamma_values))
        if not math.isclose(
            self.high_noise_efficiency,
            float(efficiency_values[high_noise_index]),
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError("high_noise_efficiency disagrees with the largest gamma row")
        object.__setattr__(self, "gamma_values", gamma_values)
        object.__setattr__(self, "efficiency_values", efficiency_values)

    @property
    def optimal_r(self) -> float:
        """Return the optimal efficiency under the legacy field name."""
        return self.optimal_efficiency

    @property
    def r_values(self) -> NDArray[np.float64]:
        """Return efficiency values under the legacy field name."""
        return self.efficiency_values

    @property
    def coherent_r(self) -> float:
        """Return zero-dephasing efficiency under the legacy field name."""
        return self.coherent_efficiency

    @property
    def classical_r(self) -> float:
        """Return the largest-scanned-gamma endpoint under the legacy name.

        Large but finite dephasing is not asserted to be a classical limit.
        """
        return self.high_noise_efficiency

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready result without misleading legacy aliases."""
        return {
            "optimal_gamma": self.optimal_gamma,
            "optimal_efficiency": self.optimal_efficiency,
            "gamma_values": self.gamma_values.tolist(),
            "efficiency_values": self.efficiency_values.tolist(),
            "coherent_efficiency": self.coherent_efficiency,
            "high_noise_efficiency": self.high_noise_efficiency,
            "enhancement": self.enhancement,
            "has_intermediate_optimum": self.has_intermediate_optimum,
            "source_site": self.source_site,
            "target_site": self.target_site,
            "t_evolve": self.t_evolve,
            "sink_rate": self.sink_rate,
            "loss_rate": self.loss_rate,
        }


def _readonly_float_vector(name: str, value: object) -> NDArray[np.float64]:
    """Return one finite, non-empty, immutable float vector."""
    array = np.array(value, dtype=np.float64, copy=True)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    array.setflags(write=False)
    return array


def _require_finite(name: str, value: float) -> None:
    """Reject booleans and non-finite scalar parameters."""
    if isinstance(value, bool) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite")


def _validate_transport_inputs(
    coupling: NDArray[np.float64],
    omega: NDArray[np.float64],
    gamma_values: NDArray[np.float64],
    source_site: int,
    target_site: int | None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], int]:
    """Validate and copy the finite site-network inputs."""
    if np.iscomplexobj(coupling) or np.iscomplexobj(omega):
        raise ValueError("K and omega must be real-valued")
    matrix = np.asarray(coupling, dtype=np.float64)
    frequencies = np.asarray(omega, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("K must be a square matrix")
    sites = matrix.shape[0]
    if sites < 2:
        raise ValueError("ENAQT transport requires at least two sites")
    if frequencies.shape != (sites,):
        raise ValueError("omega must have shape (K.shape[0],)")
    if not np.all(np.isfinite(matrix)) or not np.all(np.isfinite(frequencies)):
        raise ValueError("K and omega must contain only finite values")
    if not np.allclose(matrix, matrix.T, rtol=1e-12, atol=1e-12):
        raise ValueError("K must be symmetric for a Hermitian hopping Hamiltonian")
    if np.any(gamma_values < 0.0):
        raise ValueError("gamma_range must be non-negative")
    selected_target = sites - 1 if target_site is None else target_site
    for name, site in (("source_site", source_site), ("target_site", selected_target)):
        if isinstance(site, bool) or not isinstance(site, int) or not 0 <= site < sites:
            raise ValueError(f"{name} must be an integer site index")
    if source_site == selected_target:
        raise ValueError("source_site and target_site must differ")
    return matrix.copy(), frequencies.copy(), selected_target


def _require_site_workspace(sites: int, max_dense_gib: float | None) -> None:
    """Reject a site-basis density workspace that exceeds the active budget."""
    dimension = sites + 2
    required = _WORKSPACE_OBJECTS * dimension * dimension * np.dtype(np.complex128).itemsize
    budget = dense_budget_bytes(max_dense_gib)
    if required > budget:
        raise DenseAllocationError(
            "ENAQT site-basis density workspace for "
            f"{sites} sites requires {required / GIB:.6f} GiB, above the active "
            f"dense budget {budget / GIB:.6f} GiB"
        )


def _site_hamiltonian(
    coupling: NDArray[np.float64], omega: NDArray[np.float64]
) -> NDArray[np.complex128]:
    """Return ``diag(omega) + K`` in the single-excitation site basis."""
    result = np.asarray(coupling, dtype=np.complex128).copy()
    diagonal = np.diag_indices_from(result)
    result[diagonal] += omega
    return result


def _transport_generator(
    hamiltonian: NDArray[np.complex128],
    gamma: float,
    target_site: int,
    sink_rate: float,
    loss_rate: float,
) -> tuple[LinearOperator, float]:
    """Build the trace-preserving Lindblad generator and its exact trace."""
    sites = hamiltonian.shape[0]
    dimension = sites + 2
    sink_index = sites
    loss_index = sites + 1
    extended_h = np.zeros((dimension, dimension), dtype=np.complex128)
    extended_h[:sites, :sites] = hamiltonian
    flat_size = dimension * dimension

    def action(flat_state: NDArray[np.complex128]) -> NDArray[np.complex128]:
        rho = np.asarray(flat_state, dtype=np.complex128).reshape(dimension, dimension)
        output = -1j * (extended_h @ rho - rho @ extended_h)
        output[:sites, :] -= 0.5 * gamma * rho[:sites, :]
        output[:, :sites] -= 0.5 * gamma * rho[:, :sites]
        site_indices = np.arange(sites)
        output[site_indices, site_indices] += gamma * rho[site_indices, site_indices]
        output[target_site, :] -= 0.5 * sink_rate * rho[target_site, :]
        output[:, target_site] -= 0.5 * sink_rate * rho[:, target_site]
        output[sink_index, sink_index] += sink_rate * rho[target_site, target_site]
        output[:sites, :] -= 0.5 * loss_rate * rho[:sites, :]
        output[:, :sites] -= 0.5 * loss_rate * rho[:, :sites]
        output[loss_index, loss_index] += loss_rate * np.trace(rho[:sites, :sites])
        return np.asarray(output.reshape(-1), dtype=np.complex128)

    def adjoint_action(flat_state: NDArray[np.complex128]) -> NDArray[np.complex128]:
        observable = np.asarray(flat_state, dtype=np.complex128).reshape(dimension, dimension)
        output = 1j * (extended_h @ observable - observable @ extended_h)
        output[:sites, :] -= 0.5 * gamma * observable[:sites, :]
        output[:, :sites] -= 0.5 * gamma * observable[:, :sites]
        site_indices = np.arange(sites)
        output[site_indices, site_indices] += gamma * observable[site_indices, site_indices]
        output[target_site, :] -= 0.5 * sink_rate * observable[target_site, :]
        output[:, target_site] -= 0.5 * sink_rate * observable[:, target_site]
        output[target_site, target_site] += sink_rate * observable[sink_index, sink_index]
        output[:sites, :] -= 0.5 * loss_rate * observable[:sites, :]
        output[:, :sites] -= 0.5 * loss_rate * observable[:, :sites]
        output[site_indices, site_indices] += loss_rate * observable[loss_index, loss_index]
        return np.asarray(output.reshape(-1), dtype=np.complex128)

    generator = LinearOperator(
        (flat_size, flat_size),
        matvec=action,
        rmatvec=adjoint_action,
        dtype=np.complex128,
    )
    trace = (
        -gamma * sites * (dimension - 1) - dimension * sink_rate - dimension * sites * loss_rate
    )
    return generator, trace


def _transport_efficiency(
    hamiltonian: NDArray[np.complex128],
    gamma: float,
    source_site: int,
    target_site: int,
    t_evolve: float,
    n_steps: int,
    sink_rate: float,
    loss_rate: float,
) -> float:
    """Evolve one source excitation and return final target-sink population."""
    sites = hamiltonian.shape[0]
    dimension = sites + 2
    sink_index = sites
    rho = np.zeros((dimension, dimension), dtype=np.complex128)
    rho[source_site, source_site] = 1.0
    generator, generator_trace = _transport_generator(
        hamiltonian, gamma, target_site, sink_rate, loss_rate
    )
    dt = t_evolve / n_steps
    for _ in range(n_steps):
        evolved = expm_multiply(
            dt * generator,
            rho.reshape(-1),
            traceA=dt * generator_trace,
        )
        rho = np.asarray(evolved, dtype=np.complex128).reshape(dimension, dimension)
    efficiency = float(np.real(rho[sink_index, sink_index]))
    if not -_BOUND_TOLERANCE <= efficiency <= 1.0 + _BOUND_TOLERANCE:
        raise RuntimeError("Lindblad transport efficiency escaped [0, 1]")
    return float(np.clip(efficiency, 0.0, 1.0))


def enaqt_scan(
    K: NDArray[np.float64],
    omega: NDArray[np.float64],
    gamma_range: NDArray[np.float64] | None = None,
    t_evolve: float = 1.0,
    n_steps: int = 1,
    *,
    source_site: int = 0,
    target_site: int | None = None,
    sink_rate: float = 1.0,
    loss_rate: float = 0.05,
    minimum_improvement: float = 1e-6,
    max_dense_gib: float | None = None,
) -> ENAQTResult:
    """Scan local dephasing and maximise finite-horizon sink efficiency.

    ``K`` is the real symmetric site-to-site hopping matrix and ``omega`` gives
    site energies. ``n_steps`` only segments the same exponential propagator;
    it is retained for API compatibility and deterministic convergence checks.
    The coherent endpoint is always evaluated at gamma zero, even when zero is
    absent from ``gamma_range``. The high-noise endpoint is the value at the
    largest scanned gamma and is not labelled a classical limit.
    """
    selected_gammas = _readonly_float_vector(
        "gamma_range",
        DEFAULT_GAMMA_GRID if gamma_range is None else gamma_range,
    )
    matrix, frequencies, selected_target = _validate_transport_inputs(
        K, omega, selected_gammas, source_site, target_site
    )
    for name, value, allow_zero in (
        ("t_evolve", t_evolve, False),
        ("sink_rate", sink_rate, False),
        ("loss_rate", loss_rate, True),
        ("minimum_improvement", minimum_improvement, True),
    ):
        _require_finite(name, value)
        if value < 0.0 or (not allow_zero and value == 0.0):
            qualifier = "non-negative" if allow_zero else "positive"
            raise ValueError(f"{name} must be {qualifier}")
    if isinstance(n_steps, bool) or not isinstance(n_steps, int) or n_steps < 1:
        raise ValueError("n_steps must be a positive integer")
    _require_site_workspace(matrix.shape[0], max_dense_gib)
    hamiltonian = _site_hamiltonian(matrix, frequencies)

    cache: dict[float, float] = {}

    def evaluate(gamma: float) -> float:
        key = float(gamma)
        if key not in cache:
            cache[key] = _transport_efficiency(
                hamiltonian,
                key,
                source_site,
                selected_target,
                t_evolve,
                n_steps,
                sink_rate,
                loss_rate,
            )
        return cache[key]

    efficiencies = np.asarray([evaluate(gamma) for gamma in selected_gammas], dtype=np.float64)
    best_index = int(np.argmax(efficiencies))
    coherent_efficiency = evaluate(0.0)
    high_noise_index = int(np.argmax(selected_gammas))
    high_noise_efficiency = float(efficiencies[high_noise_index])
    optimal_gamma = float(selected_gammas[best_index])
    optimal_efficiency = float(efficiencies[best_index])
    maximum_gamma = float(selected_gammas[high_noise_index])
    is_interior = 0.0 < optimal_gamma < maximum_gamma
    endpoint_best = max(coherent_efficiency, high_noise_efficiency)
    has_intermediate_optimum = (
        is_interior and optimal_efficiency > endpoint_best + minimum_improvement
    )
    enhancement = optimal_efficiency / max(coherent_efficiency, np.finfo(float).tiny)
    return ENAQTResult(
        optimal_gamma=optimal_gamma,
        optimal_efficiency=optimal_efficiency,
        gamma_values=selected_gammas,
        efficiency_values=efficiencies,
        coherent_efficiency=coherent_efficiency,
        high_noise_efficiency=high_noise_efficiency,
        enhancement=float(enhancement),
        has_intermediate_optimum=has_intermediate_optimum,
        source_site=source_site,
        target_site=selected_target,
        t_evolve=t_evolve,
        sink_rate=sink_rate,
        loss_rate=loss_rate,
    )


__all__ = ["DEFAULT_GAMMA_GRID", "ENAQTResult", "enaqt_scan"]
