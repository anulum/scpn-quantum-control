# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Xy Kuramoto
"""Quantum Kuramoto solver via XY spin Hamiltonian + Trotter evolution.

The Kuramoto model d(theta_i)/dt = omega_i + K*sum_j sin(theta_j - theta_i)
is isomorphic to the XY spin Hamiltonian:
    H = -sum_{i<j} K_ij (X_iX_j + Y_iY_j) - sum_i omega_i Z_i

Quantum hardware simulates this natively via Trotterized time evolution.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray
from qiskit import QuantumCircuit
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit.synthesis import LieTrotter, SuzukiTrotter

from .._rust_accel import optional_rust_engine
from ..bridge.knm_hamiltonian import knm_to_hamiltonian
from ..dense_budget import require_dense_allocation
from .results import TrajectoryResult

FloatArray: TypeAlias = NDArray[np.float64]


def _as_real_numeric_array(name: str, values: object) -> FloatArray:
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
class TrotterEvolutionConfig:
    """Typed defaults for Kuramoto-XY Trotter evolution.

    `order` selects Lie-Trotter (`1`) or second-order Suzuki-Trotter (`2`).
    `evolve_steps` is used when `evolve()` is called without an explicit
    `trotter_steps` value. `run_steps_per_step` is used when `run()` is called
    without an explicit `trotter_per_step` value.
    """

    order: int = 1
    evolve_steps: int = 10
    run_steps_per_step: int = 5

    def __post_init__(self) -> None:
        if self.order not in (1, 2):
            raise ValueError(f"order must be 1 or 2, got {self.order}")
        if not isinstance(self.evolve_steps, int) or self.evolve_steps < 1:
            raise ValueError(f"evolve_steps must be a positive integer, got {self.evolve_steps}")
        if not isinstance(self.run_steps_per_step, int) or self.run_steps_per_step < 1:
            raise ValueError(
                f"run_steps_per_step must be a positive integer, got {self.run_steps_per_step}"
            )


class QuantumKuramotoSolver:
    """Trotterized quantum simulation of Kuramoto oscillators.

    Each oscillator maps to one qubit. The XY coupling simulates
    the sin(theta_j - theta_i) interaction natively.
    """

    def __init__(
        self,
        n_oscillators: int,
        K_coupling: FloatArray,
        omega_natural: FloatArray,
        trotter_order: int | None = None,
        evolution_config: TrotterEvolutionConfig | None = None,
    ):
        """K_coupling: (n,n) coupling matrix, omega_natural: (n,) frequencies."""
        self.n = self._validate_n_oscillators(n_oscillators)
        self.K = _as_real_numeric_array("K_coupling", K_coupling)
        self.omega = _as_real_numeric_array("omega_natural", omega_natural)
        self._validate_coupling_inputs(self.n, self.K, self.omega)
        np.fill_diagonal(self.K, 0.0)
        config = evolution_config or TrotterEvolutionConfig()
        order = config.order if trotter_order is None else trotter_order
        if order not in (1, 2):
            raise ValueError(f"trotter_order must be 1 or 2, got {order}")
        self.evolution_config = replace(config, order=order)
        self.trotter_order = self.evolution_config.order
        self._hamiltonian: SparsePauliOp | None = None

    @staticmethod
    def _validate_n_oscillators(n_oscillators: int) -> int:
        if not isinstance(n_oscillators, int) or n_oscillators < 1:
            raise ValueError(f"n_oscillators must be a positive integer, got {n_oscillators}")
        return n_oscillators

    @staticmethod
    def _validate_coupling_inputs(n: int, K: FloatArray, omega: FloatArray) -> None:
        if K.shape != (n, n):
            raise ValueError(f"K_coupling shape must be ({n}, {n}), got {K.shape}")
        if omega.shape != (n,):
            raise ValueError(f"omega_natural shape must be ({n},), got {omega.shape}")
        if not np.all(np.isfinite(K)):
            raise ValueError("K_coupling must contain only finite values")
        if not np.all(np.isfinite(omega)):
            raise ValueError("omega_natural must contain only finite values")
        if not np.allclose(K, K.T, atol=1e-12, rtol=1e-12):
            raise ValueError("K_coupling must be symmetric for the XY Kuramoto solver")

    def build_hamiltonian(self) -> SparsePauliOp:
        """Compile K + omega into SparsePauliOp. Called automatically by evolve()."""
        self._hamiltonian = knm_to_hamiltonian(self.K, self.omega)
        return self._hamiltonian

    def evolve(self, time: float, trotter_steps: int | None = None) -> QuantumCircuit:
        """Build Trotterized evolution circuit U(t) = exp(-iHt).

        Uses LieTrotter (order=1, O(t²/reps)) or SuzukiTrotter (order=2,
        O(t³/reps²)) depending on self.trotter_order.
        """
        if not np.isfinite(time) or time < 0.0:
            raise ValueError(f"time must be finite and non-negative, got {time}")
        trotter_steps = (
            self.evolution_config.evolve_steps if trotter_steps is None else trotter_steps
        )
        if not isinstance(trotter_steps, int) or trotter_steps < 1:
            raise ValueError(f"trotter_steps must be a positive integer, got {trotter_steps}")
        if self._hamiltonian is None:
            self.build_hamiltonian()
        if self.trotter_order == 2:
            synth = SuzukiTrotter(order=2, reps=trotter_steps)
        else:
            synth = LieTrotter(reps=trotter_steps)
        evo_gate = PauliEvolutionGate(self._hamiltonian, time=time, synthesis=synth)
        qc = QuantumCircuit(self.n)
        qc.append(evo_gate, range(self.n))
        return qc

    def measure_order_parameter(self, sv: Statevector) -> tuple[float, float]:
        """Compute Kuramoto R from qubit X,Y expectations.

        R*exp(i*psi) = (1/N) sum_j (<X_j> + i<Y_j>)
        """
        z_complex: complex
        try:
            _engine = optional_rust_engine()
            if _engine is None:
                raise AttributeError("scpn_quantum_engine absent")
            # Fast Rust path: compute all X,Y expectations in one parallel pass
            sv_arr = np.asarray(sv.data)
            exp_x, exp_y = _engine.all_xy_expectations(
                sv_arr.real.astype(np.float64),
                sv_arr.imag.astype(np.float64),
                self.n,
            )
            z_complex = complex(np.sum(exp_x) + 1j * np.sum(exp_y)) / self.n
        except AttributeError:
            # Fallback to slow Qiskit path
            z_complex = 0.0 + 0.0j
            for j in range(self.n):
                exp_x_j = float(sv.expectation_value(self._pauli_op("X", j)).real)
                exp_y_j = float(sv.expectation_value(self._pauli_op("Y", j)).real)
                z_complex += exp_x_j + 1j * exp_y_j
            z_complex /= self.n

        R = float(abs(z_complex))
        psi = float(np.angle(z_complex))
        return R, psi

    def run(
        self,
        t_max: float,
        dt: float,
        trotter_per_step: int | None = None,
        *,
        max_statevector_gib: float | None = None,
    ) -> TrajectoryResult:
        """Time-stepped evolution returning R(t) and per-qubit expectations.

        This local simulator path stores an exact dense statevector. Use
        ``max_statevector_gib`` to fail closed before Qiskit allocates that
        vector; hardware or tensor-network paths should be used for larger
        systems.
        """
        if not np.isfinite(t_max) or t_max < 0.0:
            raise ValueError(f"t_max must be finite and non-negative, got {t_max}")
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError(f"dt must be finite and positive, got {dt}")
        if trotter_per_step is None:
            trotter_per_step = self.evolution_config.run_steps_per_step
        if not isinstance(trotter_per_step, int) or trotter_per_step < 1:
            raise ValueError(
                f"trotter_per_step must be a positive integer, got {trotter_per_step}"
            )
        require_dense_allocation(
            self.n,
            rank=1,
            max_gib=max_statevector_gib,
            label="Kuramoto statevector trajectory",
        )
        if self._hamiltonian is None:
            self.build_hamiltonian()

        times_list = [0.0]
        current_time = 0.0
        tolerance = max(np.finfo(float).eps * max(1.0, abs(t_max), abs(dt)) * 16.0, 1e-15)
        while current_time + dt < t_max - tolerance:
            current_time += dt
            times_list.append(current_time)
        if t_max > times_list[-1] + tolerance:
            times_list.append(float(t_max))
        else:
            times_list[-1] = float(t_max)

        times = np.asarray(times_list, dtype=float)
        step_sizes = np.diff(times)
        R_history = np.zeros(times.shape[0])

        # Initial state: each qubit at angle ~ omega_i (Ry rotation)
        init_qc = QuantumCircuit(self.n)
        for i in range(self.n):
            angle = float(self.omega[i]) % (2 * np.pi)
            init_qc.ry(angle, i)

        sv = Statevector.from_instruction(init_qc)
        R_history[0], _ = self.measure_order_parameter(sv)

        for step, step_dt in enumerate(step_sizes, start=1):
            evo_qc = self.evolve(float(step_dt), trotter_per_step)
            sv = sv.evolve(evo_qc)
            R_history[step], _ = self.measure_order_parameter(sv)

        return TrajectoryResult(
            times=times,
            R=R_history,
            metadata={
                "backend": "statevector",
                "trotter_per_step": trotter_per_step,
                "trotter_order": self.trotter_order,
            },
        )

    def energy_expectation(self, sv: Statevector) -> float:
        """Compute <H> for a given statevector."""
        if self._hamiltonian is None:
            self.build_hamiltonian()
        return float(sv.expectation_value(self._hamiltonian).real)

    def _pauli_op(self, pauli: str, qubit: int) -> SparsePauliOp:
        label = ["I"] * self.n
        label[qubit] = pauli
        return SparsePauliOp("".join(reversed(label)))
