# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — analog-mapping bounded analog/digital comparison
"""Bounded mathematical-model comparison against a Lie–Trotter reference."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import expm

from ..bridge.knm_hamiltonian import knm_to_dense_matrix
from ..hardware.analog_kuramoto import compile_analog_kuramoto
from .contracts import MappingRequest
from .feasibility import reconstruct_compiled_couplings

FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]
ANALOG_DIGITAL_COMPARISON_SCHEMA = "analog_mapping_model_comparison.v1"
COMPARISON_BOUNDARY = (
    "Dense N<=6 mathematical XY-model comparison only: compiler parameter reconstruction "
    "and ideal Lie–Trotter state fidelity; not physical analog dynamics, calibration, noise, "
    "measurement, hardware equivalence, or performance evidence"
)


@dataclass(frozen=True, slots=True)
class AnalogDigitalComparison:
    """Bounded exact-model and Lie–Trotter comparison metrics."""

    schema: str
    n_nodes: int
    trotter_steps: int
    duration: float
    parameter_rmse: float
    compiler_model_state_fidelity: float
    digital_trotter_state_fidelity: float
    digital_trotter_infidelity: float
    within_declared_tolerance: bool
    comparison_tolerance: float
    comparison_boundary: str = COMPARISON_BOUNDARY
    hardware_equivalence_claim_allowed: bool = False
    analog_advantage_claim_allowed: bool = False

    def __post_init__(self) -> None:
        """Validate bounded comparison results and blocked claims."""
        if not 2 <= self.n_nodes <= 6:
            raise ValueError("analog/digital comparison is bounded to 2 <= N <= 6")
        if self.trotter_steps < 1:
            raise ValueError("trotter_steps must be positive")
        metrics = (
            self.parameter_rmse,
            self.compiler_model_state_fidelity,
            self.digital_trotter_state_fidelity,
            self.digital_trotter_infidelity,
            self.comparison_tolerance,
        )
        if not all(math.isfinite(value) and value >= 0.0 for value in metrics):
            raise ValueError("comparison metrics must be finite and non-negative")
        if self.compiler_model_state_fidelity > 1.0 + 1e-9:
            raise ValueError("compiler_model_state_fidelity cannot exceed one")
        if self.digital_trotter_state_fidelity > 1.0 + 1e-9:
            raise ValueError("digital_trotter_state_fidelity cannot exceed one")
        if self.hardware_equivalence_claim_allowed or self.analog_advantage_claim_allowed:
            raise ValueError("bounded model comparison cannot promote hardware claims")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready comparison record."""
        return asdict(self)


def compare_analog_model_to_trotter(
    request: MappingRequest,
    *,
    compiler_platform: str = "circuit_qed",
    trotter_steps: int = 32,
) -> AnalogDigitalComparison:
    """Compare reconstructed compiler-model dynamics to a digital product formula.

    The fixed single-excitation initial state avoids the trivially stationary all-zero
    state while keeping the comparison deterministic.
    """
    if not 2 <= request.n_nodes <= 6:
        raise ValueError("analog/digital comparison is bounded to 2 <= N <= 6")
    if isinstance(trotter_steps, bool) or trotter_steps < 1:
        raise ValueError("trotter_steps must be a positive integer")

    target_couplings = request.coupling_scale * request.coupling_matrix
    program = compile_analog_kuramoto(
        request.coupling_matrix,
        request.detuning_array,
        platform=compiler_platform,
        duration=request.duration,
        coupling_scale=request.coupling_scale,
    )
    reconstructed = reconstruct_compiled_couplings(program.to_dict(), request.n_nodes)
    parameter_rmse = float(np.sqrt(np.mean((reconstructed - target_couplings) ** 2)))

    target_hamiltonian = _dense_hamiltonian(target_couplings, request.detuning_array)
    compiler_hamiltonian = _dense_hamiltonian(reconstructed, request.detuning_array)
    initial = _single_excitation_state(request.n_nodes)
    exact_target = expm(-1j * target_hamiltonian * request.duration) @ initial
    exact_compiler = expm(-1j * compiler_hamiltonian * request.duration) @ initial
    trotter = _lie_trotter_state(
        initial,
        target_couplings,
        request.detuning_array,
        duration=request.duration,
        trotter_steps=trotter_steps,
    )
    model_fidelity = _state_fidelity(exact_target, exact_compiler)
    trotter_fidelity = _state_fidelity(exact_target, trotter)
    trotter_infidelity = max(0.0, 1.0 - trotter_fidelity)
    within_tolerance = (
        max(
            parameter_rmse,
            max(0.0, 1.0 - model_fidelity),
            trotter_infidelity,
        )
        <= request.comparison_tolerance
    )
    return AnalogDigitalComparison(
        schema=ANALOG_DIGITAL_COMPARISON_SCHEMA,
        n_nodes=request.n_nodes,
        trotter_steps=trotter_steps,
        duration=request.duration,
        parameter_rmse=parameter_rmse,
        compiler_model_state_fidelity=model_fidelity,
        digital_trotter_state_fidelity=trotter_fidelity,
        digital_trotter_infidelity=trotter_infidelity,
        within_declared_tolerance=within_tolerance,
        comparison_tolerance=request.comparison_tolerance,
    )


def _dense_hamiltonian(couplings: FloatArray, detunings: FloatArray) -> ComplexArray:
    matrix = knm_to_dense_matrix(couplings, detunings)
    return np.asarray(matrix, dtype=np.complex128)


def _single_excitation_state(n_nodes: int) -> ComplexArray:
    state = np.zeros(1 << n_nodes, dtype=np.complex128)
    state[1] = 1.0
    return state


def _lie_trotter_state(
    initial: ComplexArray,
    couplings: FloatArray,
    detunings: FloatArray,
    *,
    duration: float,
    trotter_steps: int,
) -> ComplexArray:
    n_nodes = detunings.size
    zeros = np.zeros_like(couplings)
    dt = duration / float(trotter_steps)
    factors = [expm(-1j * _dense_hamiltonian(zeros, detunings) * dt)]
    for source in range(n_nodes):
        for target in range(source + 1, n_nodes):
            strength = float(couplings[source, target])
            if abs(strength) <= 1e-12:
                continue
            edge = np.zeros_like(couplings)
            edge[source, target] = strength
            edge[target, source] = strength
            factors.append(expm(-1j * _dense_hamiltonian(edge, np.zeros_like(detunings)) * dt))
    state = initial.copy()
    for _ in range(trotter_steps):
        for factor in factors:
            state = factor @ state
    return state


def _state_fidelity(left: ComplexArray, right: ComplexArray) -> float:
    fidelity = float(abs(np.vdot(left, right)) ** 2)
    return min(1.0, max(0.0, fidelity))


__all__ = [
    "ANALOG_DIGITAL_COMPARISON_SCHEMA",
    "COMPARISON_BOUNDARY",
    "AnalogDigitalComparison",
    "compare_analog_model_to_trotter",
]
