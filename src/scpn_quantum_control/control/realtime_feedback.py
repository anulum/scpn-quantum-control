# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Real-Time Synchronisation Feedback
"""Closed-loop Kuramoto-XY synchronisation feedback.

The controller keeps a statevector simulation and applies the same state
updates that a low-latency runtime would apply between live shots: evolve,
sample finite-shot observables, choose a coupling update, and apply a small
phase-alignment correction. A separate circuit builder emits the monitored
dynamic-circuit template with mid-circuit measurement and conditional reset.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.quantum_info import Statevector

from ..phase.xy_kuramoto import QuantumKuramotoSolver

FloatArray: TypeAlias = NDArray[np.float64]

_ACTION_LABELS = {-1: "release", 0: "hold", 1: "synchronise"}


@dataclass(frozen=True)
class RealtimeFeedbackConfig:
    """Configuration for live-shot synchronisation feedback."""

    target_r: float = 0.75
    deadband: float = 0.04
    base_dt: float = 0.08
    trotter_steps: int = 3
    measurement_shots: int = 128
    base_gain: float = 0.8
    max_gain: float = 2.0
    monitor_strength: float = 0.18
    correction_angle: float = 0.12

    def __post_init__(self) -> None:
        _require_range(self.target_r, 0.0, 1.0, "target_r")
        _require_range(self.deadband, 0.0, 1.0, "deadband")
        _require_positive(self.base_dt, "base_dt")
        _require_positive(self.base_gain, "base_gain", allow_zero=True)
        _require_range(self.monitor_strength, 0.0, np.pi, "monitor_strength")
        _require_range(self.correction_angle, 0.0, np.pi, "correction_angle")
        if not isinstance(self.trotter_steps, int) or self.trotter_steps < 1:
            raise ValueError("trotter_steps must be a positive integer")
        if not isinstance(self.measurement_shots, int) or self.measurement_shots < 1:
            raise ValueError("measurement_shots must be a positive integer")
        if not np.isfinite(self.max_gain) or self.max_gain < 1.0:
            raise ValueError("max_gain must be finite and at least 1.0")


@dataclass(frozen=True)
class FeedbackStep:
    """One closed-loop feedback update."""

    index: int
    action: str
    r_live: float
    r_statevector: float
    psi_statevector: float
    error: float
    applied_coupling_scale: float
    next_coupling_scale: float
    correction_angle: float
    readout_counts: Mapping[str, int] = field(default_factory=dict)


class RealtimeSyncFeedbackController:
    """Stateful live-shot feedback controller for Kuramoto-XY circuits."""

    def __init__(
        self,
        K_coupling: FloatArray,
        omega_natural: FloatArray,
        config: RealtimeFeedbackConfig | None = None,
        trotter_order: int = 1,
    ) -> None:
        self.config = config or RealtimeFeedbackConfig()
        self.K = np.asarray(K_coupling, dtype=np.float64)
        self.omega = np.asarray(omega_natural, dtype=np.float64)
        self.n = int(self.omega.shape[0])
        self.trotter_order = trotter_order
        self._base_solver = QuantumKuramotoSolver(
            self.n, self.K, self.omega, trotter_order=trotter_order
        )
        self._state = self._initial_statevector()
        self._coupling_scale = 1.0
        self.history: list[FeedbackStep] = []

    @property
    def statevector(self) -> Statevector:
        """Current controller state."""
        return self._state

    @property
    def coupling_scale(self) -> float:
        """Coupling multiplier that will be applied on the next step."""
        return self._coupling_scale

    def set_coupling_scale(self, scale: float) -> None:
        """Set the next cross-shot coupling multiplier within policy bounds."""
        _require_range(scale, 1.0 / self.config.max_gain, self.config.max_gain, "scale")
        self._coupling_scale = float(scale)

    def reset(self) -> None:
        """Reset the controller to the prepared Kuramoto initial state."""
        self._state = self._initial_statevector()
        self._coupling_scale = 1.0
        self.history.clear()

    def step(self, seed: int | None = None) -> FeedbackStep:
        """Advance the simulator by one live-shot feedback update."""
        rng = np.random.default_rng(seed)
        applied_scale = self._coupling_scale
        scaled_solver = QuantumKuramotoSolver(
            self.n,
            self.K * applied_scale,
            self.omega,
            trotter_order=self.trotter_order,
        )
        evolution = scaled_solver.evolve(self.config.base_dt, self.config.trotter_steps)
        self._state = self._state.evolve(evolution)

        r_exact, psi_exact = self._base_solver.measure_order_parameter(self._state)
        r_live = self._sample_live_order_parameter(self._state, rng)
        action_code, next_scale, error = _feedback_policy(
            np.array([r_live], dtype=np.float64),
            self.config.target_r,
            self.config.deadband,
            self.config.base_gain,
            self.config.max_gain,
        )
        action = _ACTION_LABELS[int(action_code[0])]
        correction = self._apply_feedback_correction(float(error[0]), action)
        counts = _sample_readout_counts(self._state, self.n, self.config.measurement_shots, rng)
        self._coupling_scale = float(next_scale[0])

        step = FeedbackStep(
            index=len(self.history),
            action=action,
            r_live=float(r_live),
            r_statevector=float(r_exact),
            psi_statevector=float(psi_exact),
            error=float(error[0]),
            applied_coupling_scale=float(applied_scale),
            next_coupling_scale=self._coupling_scale,
            correction_angle=correction,
            readout_counts=counts,
        )
        self.history.append(step)
        return step

    def run(self, n_steps: int, seed: int | None = None) -> list[FeedbackStep]:
        """Run ``n_steps`` feedback updates with deterministic child seeds."""
        if not isinstance(n_steps, int) or n_steps < 1:
            raise ValueError("n_steps must be a positive integer")
        rng = np.random.default_rng(seed)
        return [self.step(seed=int(rng.integers(0, 2**32 - 1))) for _ in range(n_steps)]

    def build_monitored_circuit(self, n_rounds: int) -> QuantumCircuit:
        """Build the dynamic monitored circuit matching this controller."""
        return build_monitored_feedback_circuit(
            self.K,
            self.omega,
            config=self.config,
            n_rounds=n_rounds,
            trotter_order=self.trotter_order,
        )

    def _initial_statevector(self) -> Statevector:
        qc = QuantumCircuit(self.n)
        for qubit, omega_i in enumerate(self.omega):
            qc.ry(float(omega_i) % (2 * np.pi), qubit)
        return Statevector.from_instruction(qc)

    def _sample_live_order_parameter(self, state: Statevector, rng: np.random.Generator) -> float:
        exp_x, exp_y = _xy_expectations(state, self.n, self._base_solver)
        shots = self.config.measurement_shots
        x_samples = _sample_axis(exp_x, shots, rng)
        y_samples = _sample_axis(exp_y, shots, rng)
        z_complex = complex(float(np.sum(x_samples)), float(np.sum(y_samples))) / self.n
        return float(abs(z_complex))

    def _apply_feedback_correction(self, error: float, action: str) -> float:
        if action == "hold":
            return 0.0
        correction = float(
            np.clip(
                error * self.config.correction_angle,
                -self.config.correction_angle,
                self.config.correction_angle,
            )
        )
        if abs(correction) < 1e-15:
            return 0.0
        qc = QuantumCircuit(self.n)
        for qubit, omega_i in enumerate(self.omega):
            direction = 1.0 if np.cos(float(omega_i)) >= 0.0 else -1.0
            qc.ry(correction * direction, qubit)
        self._state = self._state.evolve(qc)
        return correction


def build_monitored_feedback_circuit(
    K_coupling: FloatArray,
    omega_natural: FloatArray,
    config: RealtimeFeedbackConfig | None = None,
    n_rounds: int = 3,
    trotter_order: int = 1,
) -> QuantumCircuit:
    """Build a monitored dynamic circuit with conditional reset/correction."""
    cfg = config or RealtimeFeedbackConfig()
    if not isinstance(n_rounds, int) or n_rounds < 1:
        raise ValueError("n_rounds must be a positive integer")
    K = np.asarray(K_coupling, dtype=np.float64)
    omega = np.asarray(omega_natural, dtype=np.float64)
    n = int(omega.shape[0])
    QuantumKuramotoSolver(n, K, omega, trotter_order=trotter_order)

    sys_reg = QuantumRegister(n, "sys")
    monitor_reg = QuantumRegister(1, "monitor")
    monitor_bits = ClassicalRegister(n_rounds, "monitor_bit")
    readout_bits = ClassicalRegister(n, "readout")
    qc = QuantumCircuit(sys_reg, monitor_reg, monitor_bits, readout_bits)

    for qubit, omega_i in enumerate(omega):
        qc.ry(float(omega_i) % (2 * np.pi), sys_reg[qubit])

    for round_index in range(n_rounds):
        scale = 1.0 + min(round_index, 2) * cfg.base_gain * cfg.deadband
        solver = QuantumKuramotoSolver(n, K * scale, omega, trotter_order=trotter_order)
        qc.compose(
            solver.evolve(cfg.base_dt, cfg.trotter_steps),
            qubits=list(sys_reg),
            inplace=True,
        )
        _append_monitor_interaction(qc, sys_reg, monitor_reg, cfg.monitor_strength)
        qc.measure(monitor_reg[0], monitor_bits[round_index])
        with qc.if_test((monitor_bits[round_index], 1)):
            qc.reset(monitor_reg[0])
        with qc.if_test((monitor_bits[round_index], 1)):
            for qubit in range(n):
                qc.ry(-cfg.correction_angle / max(n, 1), sys_reg[qubit])

    qc.measure(sys_reg, readout_bits)
    return qc


def feedback_policy_numpy(
    r_values: FloatArray,
    target_r: float,
    deadband: float,
    base_gain: float,
    max_gain: float,
) -> tuple[NDArray[np.int32], FloatArray, FloatArray]:
    """NumPy fallback for the Rust feedback-policy kernel."""
    r = np.asarray(r_values, dtype=np.float64)
    if not np.all(np.isfinite(r)):
        raise ValueError("r_values must contain only finite values")
    _require_range(target_r, 0.0, 1.0, "target_r")
    _require_range(deadband, 0.0, 1.0, "deadband")
    _require_positive(base_gain, "base_gain", allow_zero=True)
    if not np.isfinite(max_gain) or max_gain < 1.0:
        raise ValueError("max_gain must be finite and at least 1.0")
    errors = target_r - r
    actions = np.zeros(r.shape, dtype=np.int32)
    gains = np.ones(r.shape, dtype=np.float64)

    low = errors > deadband
    high = errors < -deadband
    actions[low] = 1
    actions[high] = -1
    gains[low] = np.minimum(1.0 + base_gain * errors[low], max_gain)
    gains[high] = np.clip(1.0 + base_gain * errors[high], 1.0 / max_gain, 1.0)
    return actions, gains, errors.astype(np.float64)


def _feedback_policy(
    r_values: FloatArray,
    target_r: float,
    deadband: float,
    base_gain: float,
    max_gain: float,
) -> tuple[NDArray[np.int32], FloatArray, FloatArray]:
    try:
        import scpn_quantum_engine as _engine

        if hasattr(_engine, "feedback_policy_batch"):
            actions, gains, errors = _engine.feedback_policy_batch(
                r_values,
                target_r,
                deadband,
                base_gain,
                max_gain,
            )
            return (
                np.asarray(actions, dtype=np.int32),
                np.asarray(gains, dtype=np.float64),
                np.asarray(errors, dtype=np.float64),
            )
    except (ImportError, AttributeError, ValueError):
        pass
    return feedback_policy_numpy(r_values, target_r, deadband, base_gain, max_gain)


def _append_monitor_interaction(
    qc: QuantumCircuit,
    sys_reg: QuantumRegister,
    monitor_reg: QuantumRegister,
    strength: float,
) -> None:
    angle = strength / max(len(sys_reg), 1)
    for qubit in sys_reg:
        qc.cx(qubit, monitor_reg[0])
        qc.ry(angle, monitor_reg[0])
        qc.cx(qubit, monitor_reg[0])


def _xy_expectations(
    state: Statevector,
    n: int,
    solver: QuantumKuramotoSolver,
) -> tuple[FloatArray, FloatArray]:
    try:
        import scpn_quantum_engine as _engine

        state_data = np.asarray(state.data)
        exp_x, exp_y = _engine.all_xy_expectations(
            state_data.real.astype(np.float64),
            state_data.imag.astype(np.float64),
            n,
        )
        return np.asarray(exp_x, dtype=np.float64), np.asarray(exp_y, dtype=np.float64)
    except (ImportError, AttributeError, ValueError):
        exp_x = np.zeros(n, dtype=np.float64)
        exp_y = np.zeros(n, dtype=np.float64)
        for qubit in range(n):
            exp_x[qubit] = float(state.expectation_value(solver._pauli_op("X", qubit)).real)
            exp_y[qubit] = float(state.expectation_value(solver._pauli_op("Y", qubit)).real)
        return exp_x, exp_y


def _sample_axis(
    expectations: FloatArray,
    shots: int,
    rng: np.random.Generator,
) -> FloatArray:
    probabilities = np.clip((1.0 + expectations) / 2.0, 0.0, 1.0)
    plus_counts = rng.binomial(shots, probabilities)
    return (2.0 * plus_counts.astype(np.float64) / shots) - 1.0


def _sample_readout_counts(
    state: Statevector,
    n: int,
    shots: int,
    rng: np.random.Generator,
) -> dict[str, int]:
    probabilities = np.asarray(state.probabilities(), dtype=np.float64)
    draws = rng.multinomial(shots, probabilities / probabilities.sum())
    return {
        format(index, f"0{n}b"): int(count) for index, count in enumerate(draws) if int(count) > 0
    }


def _require_positive(value: float, name: str, allow_zero: bool = False) -> None:
    if not np.isfinite(value) or value < 0.0 or (value == 0.0 and not allow_zero):
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be finite and {qualifier}")


def _require_range(value: float, lower: float, upper: float, name: str) -> None:
    if not np.isfinite(value) or value < lower or value > upper:
        raise ValueError(f"{name} must be finite and in [{lower}, {upper}]")
