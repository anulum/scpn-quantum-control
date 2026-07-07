# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Kuramoto core facade
"""Small public facade for Kuramoto-XY problems."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Statevector

from .bridge.knm_hamiltonian import knm_to_dense_matrix, knm_to_hamiltonian
from .phase.xy_kuramoto import QuantumKuramotoSolver

if TYPE_CHECKING:
    from .hardware.analog_kuramoto import AnalogKuramotoPlatform, AnalogKuramotoProgram
    from .hardware.hybrid_digital_analog import HybridDigitalAnalogProgram
    from .phase.kuramoto_variants import KuramotoVariantResult

JsonScalar = str | int | float | bool | None


def _as_real_numeric_array(name: str, values: Any) -> NDArray[np.float64]:
    """Return a real numeric array without implicit string/bool/object coercion."""
    try:
        raw = np.asarray(values)
    except ValueError as exc:
        raise ValueError(f"{name} must be a rectangular numeric array") from exc

    if raw.dtype.kind in {"b", "O", "S", "U"}:
        raise ValueError(f"{name} must contain real numeric scalars")
    if raw.dtype.kind == "c":
        raise ValueError(f"{name} must contain real numeric scalars")
    try:
        return np.array(raw, dtype=np.float64, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain real numeric scalars") from exc


@dataclass(frozen=True)
class KuramotoProblem:
    """Validated coupling matrix, frequencies, and serialisable metadata."""

    K_nm: NDArray[np.float64]
    omega: NDArray[np.float64]
    metadata: Mapping[str, JsonScalar] = field(default_factory=dict)

    def __post_init__(self) -> None:
        K_nm, omega = validate_kuramoto_inputs(self.K_nm, self.omega)
        metadata = dict(self.metadata)
        try:
            json.dumps(metadata, sort_keys=True)
        except TypeError as exc:
            raise TypeError("metadata must be JSON-serialisable") from exc

        K_nm.setflags(write=False)
        omega.setflags(write=False)
        object.__setattr__(self, "K_nm", K_nm)
        object.__setattr__(self, "omega", omega)
        object.__setattr__(self, "metadata", MappingProxyType(metadata))

    @property
    def n_oscillators(self) -> int:
        """Number of oscillators/qubits represented by the problem."""
        return int(self.omega.shape[0])

    @property
    def K(self) -> NDArray[np.float64]:
        """Alias for the validated coupling matrix."""
        return self.K_nm

    def validate(self) -> None:
        """Re-run the public validation contract for this problem."""
        validate_kuramoto_inputs(self.K_nm, self.omega)

    def to_metadata(self) -> dict[str, Any]:
        """Return serialisable metadata for result artifacts."""
        return {
            "n_oscillators": self.n_oscillators,
            "metadata": dict(self.metadata),
            "K_nm_shape": list(self.K_nm.shape),
            "omega_shape": list(self.omega.shape),
        }


def validate_kuramoto_inputs(
    K_nm: NDArray[np.float64], omega: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Validate and copy a symmetric Kuramoto coupling problem."""
    K_arr = _as_real_numeric_array("K_nm", K_nm)
    omega_arr = _as_real_numeric_array("omega", omega)

    if K_arr.ndim != 2 or K_arr.shape[0] != K_arr.shape[1]:
        raise ValueError(f"K_nm must be a square matrix, got shape {K_arr.shape}")
    n_oscillators = K_arr.shape[0]
    if n_oscillators == 0:
        raise ValueError("K_nm must contain at least one oscillator")
    if omega_arr.shape != (n_oscillators,):
        raise ValueError(f"omega must have shape ({n_oscillators},), got {omega_arr.shape}")
    if not np.all(np.isfinite(K_arr)):
        raise ValueError("K_nm must contain only finite values")
    if not np.all(np.isfinite(omega_arr)):
        raise ValueError("omega must contain only finite values")
    if not np.allclose(K_arr, K_arr.T, atol=1e-12, rtol=1e-12):
        raise ValueError("K_nm must be symmetric for the gate-model XY mapping")

    np.fill_diagonal(K_arr, 0.0)
    return K_arr, omega_arr


def build_kuramoto_problem(
    K_nm: NDArray[np.float64],
    omega: NDArray[np.float64],
    metadata: Mapping[str, JsonScalar] | None = None,
) -> KuramotoProblem:
    """Create a validated Kuramoto-XY problem from arbitrary arrays."""
    return KuramotoProblem(K_nm=K_nm, omega=omega, metadata=metadata or {})


def compile_hamiltonian(problem: KuramotoProblem) -> SparsePauliOp:
    """Compile a Kuramoto problem into the XY SparsePauliOp Hamiltonian."""
    return knm_to_hamiltonian(problem.K_nm, problem.omega)


def compile_dense_hamiltonian(
    problem: KuramotoProblem,
    *,
    max_dense_gib: float | None = None,
) -> NDArray[np.complex128]:
    """Compile a dense Hamiltonian, using the Rust engine when installed."""
    return knm_to_dense_matrix(problem.K_nm, problem.omega, max_dense_gib=max_dense_gib)


def compile_trotter_circuit(
    problem: KuramotoProblem,
    time: float,
    trotter_steps: int = 10,
    trotter_order: int = 1,
) -> QuantumCircuit:
    """Compile a Trotterised gate-model evolution circuit."""
    solver = QuantumKuramotoSolver(
        problem.n_oscillators,
        problem.K_nm,
        problem.omega,
        trotter_order=trotter_order,
    )
    return solver.evolve(time=time, trotter_steps=trotter_steps)


def compile_analog_program(
    problem: KuramotoProblem,
    *,
    platform: AnalogKuramotoPlatform | str,
    duration: float,
    coupling_scale: float = 1.0,
) -> AnalogKuramotoProgram:
    """Compile a Kuramoto problem into a native analog hardware programme."""
    from .hardware.analog_kuramoto import AnalogKuramotoBackend

    backend = AnalogKuramotoBackend(platform)
    return backend.compile(problem, duration=duration, coupling_scale=coupling_scale)


def compile_hybrid_program(
    problem: KuramotoProblem,
    *,
    platform: AnalogKuramotoPlatform | str,
    duration: float,
    digital_time: float | None = None,
    max_analog_couplers: int | None = None,
    analog_threshold: float = 0.0,
    trotter_steps: int = 8,
    trotter_order: int = 1,
) -> HybridDigitalAnalogProgram:
    """Compile a split analog-native plus digital-residual programme."""
    from .hardware.hybrid_digital_analog import HybridDigitalAnalogBackend

    backend = HybridDigitalAnalogBackend(platform)
    return backend.compile(
        problem,
        duration=duration,
        digital_time=digital_time,
        max_analog_couplers=max_analog_couplers,
        analog_threshold=analog_threshold,
        trotter_steps=trotter_steps,
        trotter_order=trotter_order,
    )


def measure_order_parameter(
    problem: KuramotoProblem, statevector: Statevector
) -> tuple[float, float]:
    """Measure the Kuramoto order parameter from a statevector."""
    solver = QuantumKuramotoSolver(problem.n_oscillators, problem.K_nm, problem.omega)
    return solver.measure_order_parameter(statevector)


def simulate_variant_trajectory(
    problem: KuramotoProblem,
    variant: str,
    *,
    dt: float,
    n_steps: int,
    theta0: NDArray[np.float64] | None = None,
    hyperedges: NDArray[np.int64] | None = None,
    hyper_weights: NDArray[np.float64] | None = None,
    target_r: float = 0.75,
    monitor_gain: float = 0.8,
    measurement_strength: float = 0.2,
    gain_loss: NDArray[np.float64] | None = None,
    prefer_rust: bool = True,
) -> KuramotoVariantResult:
    """Run a higher-order, monitored, or PT-symmetric Kuramoto variant."""
    from .phase.kuramoto_variants import (
        HigherOrderKuramotoSpec,
        MonitoredKuramotoSpec,
        PTSymmetricKuramotoSpec,
        simulate_higher_order_kuramoto,
        simulate_monitored_kuramoto,
        simulate_pt_symmetric_kuramoto,
    )

    if variant == "higher_order":
        if hyperedges is None or hyper_weights is None:
            raise ValueError("higher_order variant requires hyperedges and hyper_weights")
        return simulate_higher_order_kuramoto(
            HigherOrderKuramotoSpec(
                problem.K_nm,
                problem.omega,
                hyperedges,
                hyper_weights,
                theta0=theta0,
                metadata=problem.metadata,
            ),
            dt=dt,
            n_steps=n_steps,
            prefer_rust=prefer_rust,
        )
    if variant == "monitored":
        return simulate_monitored_kuramoto(
            MonitoredKuramotoSpec(
                problem.K_nm,
                problem.omega,
                target_r=target_r,
                monitor_gain=monitor_gain,
                measurement_strength=measurement_strength,
                theta0=theta0,
                metadata=problem.metadata,
            ),
            dt=dt,
            n_steps=n_steps,
            prefer_rust=prefer_rust,
        )
    if variant == "pt_symmetric":
        if gain_loss is None:
            raise ValueError("pt_symmetric variant requires gain_loss")
        return simulate_pt_symmetric_kuramoto(
            PTSymmetricKuramotoSpec(
                problem.K_nm,
                problem.omega,
                gain_loss,
                theta0=theta0,
                metadata=problem.metadata,
            ),
            dt=dt,
            n_steps=n_steps,
            prefer_rust=prefer_rust,
        )
    raise ValueError("variant must be one of 'higher_order', 'monitored', or 'pt_symmetric'")
