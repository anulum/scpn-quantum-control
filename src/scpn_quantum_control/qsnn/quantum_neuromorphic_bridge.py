# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Quantum neuromorphic bridge for QSNN experiments.

The bridge composes three pieces into one time-stepped engine:

* vectorised quantum LIF dynamics, where membrane voltage maps to Ry angles;
* trace-based STDP for input and recurrent synapses;
* dynamic recurrent coupling driven by spike coactivity and quantum
  spike-probability coherence.

This is a simulator and framework primitive. It is not empirical evidence for
quantum biology by itself; experiments must bind it to measured spike trains,
calibration metadata, and preregistered reducers before making biological
claims.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias, cast

import numpy as np
from numpy.typing import NDArray
from qiskit import QuantumCircuit

CLAIM_BOUNDARY = (
    "simulator QSNN neuromorphic bridge; not empirical quantum-biology evidence "
    "without measured biological data and preregistered validation"
)

FloatArray: TypeAlias = NDArray[np.float64]
IntArray: TypeAlias = NDArray[np.int64]


def _as_finite_vector(
    name: str, values: np.ndarray, expected_shape: tuple[int, ...]
) -> FloatArray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.shape != expected_shape:
        raise ValueError(f"{name} shape must be {expected_shape}, got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    return cast(FloatArray, arr)


def _as_finite_matrix(
    name: str, values: np.ndarray, expected_shape: tuple[int, int]
) -> FloatArray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.shape != expected_shape:
        raise ValueError(f"{name} shape must be {expected_shape}, got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    return cast(FloatArray, arr)


@dataclass(frozen=True)
class QuantumLIFConfig:
    """Vectorised leaky-integrate-and-fire settings."""

    v_rest: float = 0.0
    v_reset: float = 0.0
    v_threshold: float = 1.0
    tau_mem: float = 20.0
    dt: float = 1.0
    resistance: float = 1.0
    n_shots: int = 0
    refractory_steps: int = 0

    def __post_init__(self) -> None:
        if self.v_threshold <= self.v_rest:
            raise ValueError("v_threshold must exceed v_rest")
        if self.tau_mem <= 0.0:
            raise ValueError("tau_mem must be positive")
        if self.dt <= 0.0:
            raise ValueError("dt must be positive")
        if self.n_shots < 0:
            raise ValueError("n_shots must be >= 0")
        if self.refractory_steps < 0:
            raise ValueError("refractory_steps must be >= 0")


@dataclass(frozen=True)
class TraceSTDPConfig:
    """Trace-based STDP rule for directed synapses."""

    a_plus: float = 0.02
    a_minus: float = 0.018
    tau_pre: float = 20.0
    tau_post: float = 20.0
    dt: float = 1.0
    enabled: bool = True

    def __post_init__(self) -> None:
        if self.a_plus < 0.0 or self.a_minus < 0.0:
            raise ValueError("STDP amplitudes must be non-negative")
        if self.tau_pre <= 0.0 or self.tau_post <= 0.0:
            raise ValueError("STDP time constants must be positive")
        if self.dt <= 0.0:
            raise ValueError("STDP dt must be positive")


@dataclass(frozen=True)
class DynamicCouplingConfig:
    """Dynamic recurrent-coupling settings."""

    learning_rate: float = 0.05
    decay_rate: float = 0.01
    coherence_gain: float = 0.25
    min_weight: float = 0.0
    max_weight: float = 1.0
    enabled: bool = True

    def __post_init__(self) -> None:
        if self.learning_rate < 0.0:
            raise ValueError("learning_rate must be non-negative")
        if not 0.0 <= self.decay_rate <= 1.0:
            raise ValueError("decay_rate must be in [0, 1]")
        if self.coherence_gain < 0.0:
            raise ValueError("coherence_gain must be non-negative")
        if self.max_weight <= self.min_weight:
            raise ValueError("max_weight must exceed min_weight")


@dataclass
class TraceSTDPState:
    """Mutable pre/post traces for one directed synapse bank."""

    n_pre: int
    n_post: int
    pre_trace: np.ndarray | None = None
    post_trace: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.n_pre <= 0 or self.n_post <= 0:
            raise ValueError("trace dimensions must be positive")
        if self.pre_trace is None:
            self.pre_trace = np.zeros(self.n_pre, dtype=np.float64)
        else:
            self.pre_trace = _as_finite_vector("pre_trace", self.pre_trace, (self.n_pre,))
        if self.post_trace is None:
            self.post_trace = np.zeros(self.n_post, dtype=np.float64)
        else:
            self.post_trace = _as_finite_vector("post_trace", self.post_trace, (self.n_post,))

    def decay(self, dt: float, tau_pre: float, tau_post: float) -> None:
        """Apply exponential trace decay in-place."""

        if dt <= 0.0:
            raise ValueError("dt must be positive")
        if tau_pre <= 0.0 or tau_post <= 0.0:
            raise ValueError("trace time constants must be positive")
        assert self.pre_trace is not None
        assert self.post_trace is not None
        self.pre_trace *= np.exp(-dt / tau_pre)
        self.post_trace *= np.exp(-dt / tau_post)

    def update(self, pre_spikes: np.ndarray, post_spikes: np.ndarray) -> None:
        """Accumulate binary spike events into traces."""

        pre = _as_finite_vector("pre_spikes", pre_spikes, (self.n_pre,))
        post = _as_finite_vector("post_spikes", post_spikes, (self.n_post,))
        if np.any((pre < 0.0) | (pre > 1.0)) or np.any((post < 0.0) | (post > 1.0)):
            raise ValueError("STDP spikes must be in [0, 1]")
        assert self.pre_trace is not None
        assert self.post_trace is not None
        self.pre_trace += pre
        self.post_trace += post


@dataclass(frozen=True)
class NeuromorphicStepResult:
    """Single bridge-step result."""

    membrane: FloatArray
    spike_probabilities: FloatArray
    spikes: IntArray
    input_current: FloatArray
    synaptic_current: FloatArray
    input_weights: FloatArray
    recurrent_weights: FloatArray
    coupling_delta: FloatArray
    quantum_circuit: QuantumCircuit
    claim_boundary: str = CLAIM_BOUNDARY


class QuantumNeuromorphicBridge:
    """Time-stepped quantum LIF + STDP + dynamic coupling engine.

    Weight shapes follow dense-layer convention: input weights are
    ``(n_neurons, n_inputs)`` and recurrent weights are
    ``(n_neurons, n_neurons)``. Recurrent self-loops are always removed.
    """

    def __init__(
        self,
        n_inputs: int,
        n_neurons: int,
        *,
        lif: QuantumLIFConfig | None = None,
        stdp: TraceSTDPConfig | None = None,
        coupling: DynamicCouplingConfig | None = None,
        input_weights: np.ndarray | None = None,
        recurrent_weights: np.ndarray | None = None,
        seed: int | None = None,
        deterministic: bool = True,
    ) -> None:
        if n_inputs <= 0:
            raise ValueError("n_inputs must be positive")
        if n_neurons <= 0:
            raise ValueError("n_neurons must be positive")
        self.n_inputs = int(n_inputs)
        self.n_neurons = int(n_neurons)
        self.lif = lif or QuantumLIFConfig()
        self.stdp = stdp or TraceSTDPConfig()
        self.coupling = coupling or DynamicCouplingConfig()
        self.deterministic = bool(deterministic)
        self.rng = np.random.default_rng(seed)

        if input_weights is None:
            input_weights = self.rng.uniform(0.05, 0.35, size=(self.n_neurons, self.n_inputs))
        self.input_weights: FloatArray = _as_finite_matrix(
            "input_weights",
            input_weights,
            (self.n_neurons, self.n_inputs),
        )
        self.input_weights = self._clip_weights(self.input_weights)

        if recurrent_weights is None:
            recurrent_weights = self.rng.uniform(0.0, 0.05, size=(self.n_neurons, self.n_neurons))
        self.recurrent_weights: FloatArray = _as_finite_matrix(
            "recurrent_weights",
            recurrent_weights,
            (self.n_neurons, self.n_neurons),
        )
        self.recurrent_weights = self._clip_weights(self.recurrent_weights)
        np.fill_diagonal(self.recurrent_weights, 0.0)

        self.membrane: FloatArray = cast(
            FloatArray, np.full(self.n_neurons, self.lif.v_rest, dtype=np.float64)
        )
        self.last_spikes: FloatArray = cast(FloatArray, np.zeros(self.n_neurons, dtype=np.float64))
        self.refractory_count: IntArray = cast(IntArray, np.zeros(self.n_neurons, dtype=np.int64))
        self.input_trace = TraceSTDPState(self.n_inputs, self.n_neurons)
        self.recurrent_trace = TraceSTDPState(self.n_neurons, self.n_neurons)
        self._last_circuit = self._build_quantum_circuit(np.zeros(self.n_neurons))

    def _clip_weights(self, weights: np.ndarray) -> FloatArray:
        clipped = np.clip(weights, self.coupling.min_weight, self.coupling.max_weight).astype(
            np.float64,
            copy=False,
        )
        return cast(FloatArray, clipped)

    def _membrane_to_angles(self) -> FloatArray:
        norm = (self.membrane - self.lif.v_rest) / (self.lif.v_threshold - self.lif.v_rest)
        return cast(FloatArray, np.pi * np.clip(norm, 0.0, 1.0))

    @staticmethod
    def _probabilities_from_angles(theta: np.ndarray) -> FloatArray:
        return cast(FloatArray, np.sin(theta / 2.0) ** 2)

    def _build_quantum_circuit(self, theta: np.ndarray) -> QuantumCircuit:
        circuit = QuantumCircuit(self.n_neurons)
        for idx, angle in enumerate(theta):
            circuit.ry(float(angle), idx)
        for src in range(self.n_neurons):
            for dst in range(self.n_neurons):
                if src == dst:
                    continue
                weight = float(self.recurrent_weights[dst, src])
                if weight <= 0.0:
                    continue
                circuit.cry(np.pi * weight, src, dst)
        return circuit

    def _sample_spikes(self, probabilities: np.ndarray) -> FloatArray:
        refractory = self.refractory_count > 0
        effective = np.where(refractory, 0.0, probabilities)
        if self.deterministic or self.lif.n_shots == 0:
            spikes = (effective > 0.5).astype(np.float64)
        else:
            samples = self.rng.binomial(1, effective, size=(self.lif.n_shots, self.n_neurons))
            spikes = (np.mean(samples, axis=0) > 0.5).astype(np.float64)
        return cast(FloatArray, spikes)

    def _apply_lif(self, synaptic_current: np.ndarray) -> tuple[FloatArray, FloatArray]:
        active = self.refractory_count <= 0
        leak = -(self.membrane - self.lif.v_rest) * (self.lif.dt / self.lif.tau_mem)
        drive = self.lif.resistance * synaptic_current * self.lif.dt
        self.membrane = cast(
            FloatArray,
            np.asarray(
                np.where(active, self.membrane + leak + drive, self.lif.v_reset),
                dtype=np.float64,
            ),
        )

        theta = self._membrane_to_angles()
        probabilities = self._probabilities_from_angles(theta)
        spikes = self._sample_spikes(probabilities)

        self.refractory_count = cast(
            IntArray,
            np.asarray(np.maximum(self.refractory_count - 1, 0), dtype=np.int64),
        )
        fired = spikes > 0.0
        self.membrane = cast(
            FloatArray,
            np.asarray(np.where(fired, self.lif.v_reset, self.membrane), dtype=np.float64),
        )
        if self.lif.refractory_steps > 0:
            self.refractory_count = cast(
                IntArray,
                np.asarray(
                    np.where(fired, self.lif.refractory_steps, self.refractory_count),
                    dtype=np.int64,
                ),
            )

        self._last_circuit = self._build_quantum_circuit(theta)
        return probabilities, spikes

    def _stdp_delta(
        self,
        state: TraceSTDPState,
        pre_spikes: np.ndarray,
        post_spikes: np.ndarray,
    ) -> FloatArray:
        state.decay(self.stdp.dt, self.stdp.tau_pre, self.stdp.tau_post)
        assert state.pre_trace is not None
        assert state.post_trace is not None
        potentiation = self.stdp.a_plus * np.outer(post_spikes, state.pre_trace)
        depression = self.stdp.a_minus * np.outer(state.post_trace, pre_spikes)
        state.update(pre_spikes, post_spikes)
        return cast(FloatArray, potentiation - depression)

    def apply_plasticity(self, pre_spikes: np.ndarray, post_spikes: np.ndarray) -> None:
        """Apply input-synapse trace STDP without running a LIF step."""

        pre = _as_finite_vector("pre_spikes", pre_spikes, (self.n_inputs,))
        post = _as_finite_vector("post_spikes", post_spikes, (self.n_neurons,))
        if not self.stdp.enabled:
            return
        delta = self._stdp_delta(self.input_trace, pre, post)
        self.input_weights = self._clip_weights(self.input_weights + delta)

    def _update_recurrent_coupling(
        self,
        spikes: np.ndarray,
        probabilities: np.ndarray,
    ) -> FloatArray:
        if not self.coupling.enabled:
            return cast(FloatArray, np.zeros_like(self.recurrent_weights))

        coactivity = np.outer(spikes, self.last_spikes)
        coherence_vector = np.sqrt(np.clip(probabilities * (1.0 - probabilities), 0.0, 1.0))
        coherence = np.outer(coherence_vector, coherence_vector)
        drive = coactivity + self.coupling.coherence_gain * coherence
        np.fill_diagonal(drive, 0.0)

        updated = (1.0 - self.coupling.decay_rate) * self.recurrent_weights
        updated += self.coupling.learning_rate * drive
        updated = self._clip_weights(updated)
        np.fill_diagonal(updated, 0.0)

        delta = updated - self.recurrent_weights
        self.recurrent_weights = updated
        return cast(FloatArray, delta)

    def step(self, external_current: np.ndarray) -> NeuromorphicStepResult:
        """Advance the bridge by one time step."""

        external = _as_finite_vector("external_current", external_current, (self.n_inputs,))
        synaptic_current = (
            self.input_weights @ external + self.recurrent_weights @ self.last_spikes
        )
        probabilities, spikes = self._apply_lif(synaptic_current)

        if self.stdp.enabled:
            input_delta = self._stdp_delta(self.input_trace, external, spikes)
            recurrent_delta = self._stdp_delta(self.recurrent_trace, self.last_spikes, spikes)
            self.input_weights = self._clip_weights(self.input_weights + input_delta)
            self.recurrent_weights = self._clip_weights(self.recurrent_weights + recurrent_delta)
            np.fill_diagonal(self.recurrent_weights, 0.0)

        coupling_delta = self._update_recurrent_coupling(spikes, probabilities)
        self.last_spikes = spikes.copy()

        return NeuromorphicStepResult(
            membrane=self.membrane.copy(),
            spike_probabilities=probabilities.copy(),
            spikes=spikes.astype(int),
            input_current=external.copy(),
            synaptic_current=synaptic_current.copy(),
            input_weights=self.input_weights.copy(),
            recurrent_weights=self.recurrent_weights.copy(),
            coupling_delta=coupling_delta.copy(),
            quantum_circuit=self._last_circuit.copy(),
        )

    def run(self, external_currents: np.ndarray) -> list[NeuromorphicStepResult]:
        """Run a sequence of external currents with shape ``(steps, n_inputs)``."""

        currents = np.asarray(external_currents, dtype=np.float64)
        if currents.ndim != 2 or currents.shape[1] != self.n_inputs:
            raise ValueError(
                f"external_currents shape must be (steps, {self.n_inputs}), got {currents.shape}"
            )
        if not np.all(np.isfinite(currents)):
            raise ValueError("external_currents must contain only finite values")
        return [self.step(row) for row in currents]

    def get_circuit(self) -> QuantumCircuit:
        """Return the latest quantum LIF/coupling circuit."""

        return self._last_circuit.copy()

    def reset(self) -> None:
        """Reset dynamic membrane, spike, refractory, and trace state."""

        self.membrane.fill(self.lif.v_rest)
        self.last_spikes.fill(0.0)
        self.refractory_count.fill(0)
        self.input_trace = TraceSTDPState(self.n_inputs, self.n_neurons)
        self.recurrent_trace = TraceSTDPState(self.n_neurons, self.n_neurons)
        self._last_circuit = self._build_quantum_circuit(np.zeros(self.n_neurons))


__all__ = [
    "CLAIM_BOUNDARY",
    "DynamicCouplingConfig",
    "NeuromorphicStepResult",
    "QuantumLIFConfig",
    "QuantumNeuromorphicBridge",
    "TraceSTDPConfig",
    "TraceSTDPState",
]
