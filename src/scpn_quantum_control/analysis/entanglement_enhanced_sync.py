# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Entangled Initial-State Synchronisation Study
"""Compare initial-state coherence under bounded Kuramoto-XY evolution.

The local phase order is defined only when at least one qubit has measurable
transverse visibility. States with vanishing local Bloch vectors therefore no
longer acquire the false value ``R=1`` through ``atan2(0, 0)``. A second,
explicit observable reports pairwise transverse-exchange coherence.

Computational-basis dephasing supplies a population-matched coherence control
for each pure initial state. Differences from that control are descriptive
coherence effects, not causal proof of an entanglement-specific mechanism or a
quantum-advantage claim.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from scipy.linalg import expm

from ..advantage_language_protocol import issue_no_advantage_certificate
from ..bridge.knm_hamiltonian import knm_to_dense_matrix
from ..dense_budget import require_dense_allocation

_VISIBILITY_TOLERANCE: Final[float] = 1e-12
_COMPARISON_PROTOCOL: Final[str] = "protocol:entanglement.initial_state_observation"


class InitialState(Enum):
    """Supported pure initial-state families."""

    PRODUCT = "product"
    BELL_PAIRS = "bell_pairs"
    GHZ = "ghz"
    W_STATE = "w_state"


@dataclass(slots=True)
class SyncTrajectory:
    """Observable trajectory for one initial state.

    ``R_values`` and ``final_R`` are compatibility names for the local phase
    order. A zero value is returned when local transverse visibility vanishes;
    callers must inspect ``phase_defined_values`` before interpreting phase
    order. ``exchange_coherence_values`` is a separate pair-correlation score.
    """

    initial_state: str
    times: list[float]
    R_values: list[float]
    final_R: float
    n_qubits: int
    local_visibility_values: list[float] = field(default_factory=list)
    phase_defined_values: list[bool] = field(default_factory=list)
    exchange_coherence_values: list[float] = field(default_factory=list)

    @property
    def final_local_visibility(self) -> float:
        """Return the final mean transverse visibility, or zero for legacy rows."""
        return self.local_visibility_values[-1] if self.local_visibility_values else 0.0

    @property
    def final_phase_defined(self) -> bool:
        """Return whether the final local phase order is observable."""
        return self.phase_defined_values[-1] if self.phase_defined_values else False

    @property
    def final_exchange_coherence(self) -> float:
        """Return the final pairwise transverse-exchange coherence score."""
        return self.exchange_coherence_values[-1] if self.exchange_coherence_values else 0.0

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready trajectory record."""
        return {
            "initial_state": self.initial_state,
            "times": self.times,
            "local_phase_order_values": self.R_values,
            "local_visibility_values": self.local_visibility_values,
            "phase_defined_values": self.phase_defined_values,
            "exchange_coherence_values": self.exchange_coherence_values,
            "final_local_phase_order": self.final_R,
            "final_local_visibility": self.final_local_visibility,
            "final_phase_defined": self.final_phase_defined,
            "final_exchange_coherence": self.final_exchange_coherence,
            "n_qubits": self.n_qubits,
        }


@dataclass(frozen=True, slots=True)
class InitialStateControlComparison:
    """Pure-state versus population-matched dephased-control comparison."""

    initial_state: str
    initial_mean_single_qubit_linear_entropy: float
    final_exchange_coherence: float
    control_final_exchange_coherence: float
    delta_final_exchange_coherence: float
    mean_exchange_coherence: float
    control_mean_exchange_coherence: float
    delta_mean_exchange_coherence: float
    final_local_phase_order: float
    control_final_local_phase_order: float
    final_phase_defined: bool
    control_final_phase_defined: bool
    attribution_status: str
    entanglement_specific_effect_supported: bool
    language_status: str
    no_advantage_certificate: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready comparison record."""
        return {
            "initial_state": self.initial_state,
            "initial_mean_single_qubit_linear_entropy": (
                self.initial_mean_single_qubit_linear_entropy
            ),
            "final_exchange_coherence": self.final_exchange_coherence,
            "control_final_exchange_coherence": self.control_final_exchange_coherence,
            "delta_final_exchange_coherence": self.delta_final_exchange_coherence,
            "mean_exchange_coherence": self.mean_exchange_coherence,
            "control_mean_exchange_coherence": self.control_mean_exchange_coherence,
            "delta_mean_exchange_coherence": self.delta_mean_exchange_coherence,
            "final_local_phase_order": self.final_local_phase_order,
            "control_final_local_phase_order": self.control_final_local_phase_order,
            "final_phase_defined": self.final_phase_defined,
            "control_final_phase_defined": self.control_final_phase_defined,
            "attribution_status": self.attribution_status,
            "entanglement_specific_effect_supported": (
                self.entanglement_specific_effect_supported
            ),
            "language_status": self.language_status,
            "no_advantage_certificate": self.no_advantage_certificate,
        }


def _validate_qubit_count(n_qubits: int) -> None:
    """Validate a positive integral qubit count."""
    if isinstance(n_qubits, bool) or not isinstance(n_qubits, int) or n_qubits < 1:
        raise ValueError("n_qubits must be a positive integer")


def _validate_initial_state_inputs(
    n_qubits: int,
    omega: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Validate state-preparation dimensions and frequencies."""
    _validate_qubit_count(n_qubits)
    omega_array = np.asarray(omega, dtype=np.float64)
    if omega_array.shape != (n_qubits,):
        raise ValueError(f"omega must have shape ({n_qubits},)")
    if not np.all(np.isfinite(omega_array)):
        raise ValueError("omega must contain only finite values")
    return omega_array


def prepare_initial_state(
    n_qubits: int,
    state_type: InitialState,
    omega: NDArray[np.float64],
) -> QuantumCircuit:
    """Prepare one supported pure initial state.

    Parameters
    ----------
    n_qubits
        Positive number of qubits.
    state_type
        Initial-state family.
    omega
        Finite frequency vector used for product-state rotation angles and an
        unpaired Bell-family qubit.

    Returns
    -------
    QuantumCircuit
        State-preparation circuit.

    """
    omega_array = _validate_initial_state_inputs(n_qubits, omega)
    if not isinstance(state_type, InitialState):
        raise TypeError("state_type must be an InitialState")
    circuit = QuantumCircuit(n_qubits)

    if state_type is InitialState.PRODUCT:
        for index in range(n_qubits):
            circuit.ry(float(omega_array[index]) % (2 * np.pi), index)
    elif state_type is InitialState.BELL_PAIRS:
        for index in range(0, n_qubits - 1, 2):
            circuit.h(index)
            circuit.cx(index, index + 1)
        if n_qubits % 2 == 1:
            circuit.ry(float(omega_array[-1]) % (2 * np.pi), n_qubits - 1)
    elif state_type is InitialState.GHZ:
        circuit.h(0)
        for index in range(n_qubits - 1):
            circuit.cx(index, index + 1)
    else:
        _prepare_w_state(circuit, n_qubits)
    return circuit


def _prepare_w_state(circuit: QuantumCircuit, n_qubits: int) -> None:
    """Prepare the equal-amplitude one-excitation Dicke state."""
    circuit.x(0)
    for index in range(n_qubits - 1):
        theta = 2 * np.arccos(np.sqrt(1.0 / (n_qubits - index)))
        circuit.cry(theta, index, index + 1)
        circuit.cx(index + 1, index)


def _validate_simulation_inputs(
    coupling: NDArray[np.float64],
    omega: NDArray[np.float64],
    t_max: float,
    n_steps: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Validate one dense exact-simulation contract."""
    coupling_array = np.asarray(coupling, dtype=np.float64)
    if coupling_array.ndim != 2 or coupling_array.shape[0] != coupling_array.shape[1]:
        raise ValueError("K must be a square matrix")
    n_qubits = coupling_array.shape[0]
    omega_array = _validate_initial_state_inputs(n_qubits, omega)
    if not np.all(np.isfinite(coupling_array)):
        raise ValueError("K must contain only finite values")
    if not np.allclose(coupling_array, coupling_array.T, atol=1e-12, rtol=0.0):
        raise ValueError("K must be symmetric")
    if not math.isfinite(t_max) or t_max <= 0.0:
        raise ValueError("t_max must be positive and finite")
    if isinstance(n_steps, bool) or not isinstance(n_steps, int) or n_steps < 1:
        raise ValueError("n_steps must be a positive integer")
    return coupling_array, omega_array


def _build_evolution(
    coupling: NDArray[np.float64],
    omega: NDArray[np.float64],
    t_max: float,
    n_steps: int,
    max_dense_gib: float | None,
) -> tuple[int, list[float], NDArray[np.complex128]]:
    """Build one budgeted exact propagator shared across initial states."""
    coupling_array, omega_array = _validate_simulation_inputs(
        coupling,
        omega,
        t_max,
        n_steps,
    )
    n_qubits = coupling_array.shape[0]
    require_dense_allocation(
        n_qubits,
        dtype=np.complex128,
        rank=2,
        object_count=5,
        max_gib=max_dense_gib,
        label="entangled initial-state dense evolution workspace",
    )
    require_dense_allocation(
        n_qubits,
        dtype=np.complex128,
        rank=1,
        object_count=2,
        max_gib=max_dense_gib,
        label="entangled initial-state dense vector workspace",
    )
    hamiltonian = np.asarray(
        knm_to_dense_matrix(coupling_array, omega_array, max_dense_gib=max_dense_gib),
        dtype=np.complex128,
    )
    propagator = np.asarray(expm(-1j * hamiltonian * (t_max / n_steps)), dtype=np.complex128)
    times = [float(value) for value in np.linspace(0.0, t_max, n_steps + 1)]
    return n_qubits, times, propagator


def _as_density_matrix(
    state: NDArray[np.complex128],
    n_qubits: int,
) -> NDArray[np.complex128]:
    """Return a normalised density matrix from a statevector or density input."""
    _validate_qubit_count(n_qubits)
    array = np.asarray(state, dtype=np.complex128)
    dimension = 1 << n_qubits
    if array.shape == (dimension,):
        norm = float(np.real(np.vdot(array, array)))
        if not math.isfinite(norm) or norm <= 0.0:
            raise ValueError("statevector must have positive finite norm")
        vector = array / np.sqrt(norm)
        return np.outer(vector, vector.conj())
    if array.shape != (dimension, dimension):
        raise ValueError(f"state must have shape ({dimension},) or ({dimension}, {dimension})")
    if not np.all(np.isfinite(array)):
        raise ValueError("density matrix must contain only finite values")
    trace = complex(np.trace(array))
    if abs(trace - 1.0) > 1e-10:
        raise ValueError("density matrix must have unit trace")
    if not np.allclose(array, array.conj().T, atol=1e-10, rtol=0.0):
        raise ValueError("density matrix must be Hermitian")
    return array.copy()


def local_phase_observables(
    state: NDArray[np.complex128],
    n_qubits: int,
    *,
    visibility_tolerance: float = _VISIBILITY_TOLERANCE,
) -> tuple[float, float, bool]:
    """Return visibility-aware local phase order, visibility, and defined flag.

    The local complex amplitudes are ``<X_i> + i<Y_i>``. Their vector sum is
    normalised by their total magnitude. If that magnitude is below the
    tolerance, local phase is unobservable and the returned order is zero.
    """
    if not math.isfinite(visibility_tolerance) or visibility_tolerance < 0.0:
        raise ValueError("visibility_tolerance must be finite and non-negative")
    density = _as_density_matrix(state, n_qubits)
    dimension = 1 << n_qubits
    amplitudes = np.empty(n_qubits, dtype=np.complex128)
    for qubit in range(n_qubits):
        bit = 1 << qubit
        coherence = sum(
            density[index | bit, index] for index in range(dimension) if index & bit == 0
        )
        amplitudes[qubit] = 2.0 * coherence
    total_visibility = float(np.sum(np.abs(amplitudes)))
    mean_visibility = total_visibility / n_qubits
    if total_visibility <= visibility_tolerance:
        return 0.0, mean_visibility, False
    phase_order = float(abs(np.sum(amplitudes)) / total_visibility)
    return float(np.clip(phase_order, 0.0, 1.0)), mean_visibility, True


def transverse_exchange_coherence(
    state: NDArray[np.complex128],
    n_qubits: int,
) -> float:
    """Return mean pairwise ``2|<sigma_i^+ sigma_j^->|`` in ``[0, 1]``."""
    density = _as_density_matrix(state, n_qubits)
    if n_qubits == 1:
        return 0.0
    dimension = 1 << n_qubits
    total = 0.0
    pair_count = n_qubits * (n_qubits - 1) // 2
    for left in range(n_qubits):
        left_bit = 1 << left
        for right in range(left + 1, n_qubits):
            right_bit = 1 << right
            coherence = sum(
                density[index, (index ^ left_bit) ^ right_bit]
                for index in range(dimension)
                if index & left_bit and index & right_bit == 0
            )
            total += 2.0 * abs(coherence)
    return float(np.clip(total / pair_count, 0.0, 1.0))


def mean_single_qubit_linear_entropy(
    statevector: NDArray[np.complex128],
    n_qubits: int,
) -> float:
    """Return mean normalised one-qubit linear entropy for a pure state."""
    if np.asarray(statevector).ndim != 1:
        raise ValueError("statevector must be a pure statevector")
    density = _as_density_matrix(statevector, n_qubits)
    dimension = 1 << n_qubits
    entropies: list[float] = []
    for qubit in range(n_qubits):
        bit = 1 << qubit
        zero_indices = [index for index in range(dimension) if index & bit == 0]
        population_zero = float(np.real(sum(density[index, index] for index in zero_indices)))
        population_one = 1.0 - population_zero
        coherence = sum(density[index, index | bit] for index in zero_indices)
        purity = population_zero**2 + population_one**2 + 2.0 * abs(coherence) ** 2
        entropies.append(float(np.clip(2.0 * (1.0 - purity), 0.0, 1.0)))
    return float(np.mean(entropies))


def _trajectory_from_density(
    initial_density: NDArray[np.complex128],
    initial_state: str,
    n_qubits: int,
    times: list[float],
    propagator: NDArray[np.complex128],
) -> SyncTrajectory:
    """Evolve one density matrix and collect both observables."""
    density = initial_density.copy()
    propagator_adjoint = propagator.conj().T
    local_order_values: list[float] = []
    local_visibility_values: list[float] = []
    phase_defined_values: list[bool] = []
    exchange_values: list[float] = []
    for step in range(len(times)):
        if step:
            density = propagator @ density @ propagator_adjoint
        local_order, visibility, phase_defined = local_phase_observables(density, n_qubits)
        local_order_values.append(local_order)
        local_visibility_values.append(visibility)
        phase_defined_values.append(phase_defined)
        exchange_values.append(transverse_exchange_coherence(density, n_qubits))
    return SyncTrajectory(
        initial_state=initial_state,
        times=times.copy(),
        R_values=local_order_values,
        final_R=local_order_values[-1],
        n_qubits=n_qubits,
        local_visibility_values=local_visibility_values,
        phase_defined_values=phase_defined_values,
        exchange_coherence_values=exchange_values,
    )


def simulate_sync_trajectory(
    K: NDArray[np.float64],
    omega: NDArray[np.float64],
    state_type: InitialState,
    t_max: float = 2.0,
    n_steps: int = 20,
    *,
    max_dense_gib: float | None = None,
) -> SyncTrajectory:
    """Evolve one pure state and report bounded local and pair observables."""
    n_qubits, times, propagator = _build_evolution(
        K,
        omega,
        t_max,
        n_steps,
        max_dense_gib,
    )
    circuit = prepare_initial_state(n_qubits, state_type, np.asarray(omega, dtype=np.float64))
    vector = np.asarray(Statevector.from_instruction(circuit), dtype=np.complex128)
    density = _as_density_matrix(vector, n_qubits)
    return _trajectory_from_density(density, state_type.value, n_qubits, times, propagator)


def compare_all_initial_states(
    K: NDArray[np.float64],
    omega: NDArray[np.float64],
    t_max: float = 2.0,
    n_steps: int = 20,
    *,
    max_dense_gib: float | None = None,
) -> dict[str, SyncTrajectory]:
    """Evolve every pure initial-state family through one shared propagator."""
    n_qubits, times, propagator = _build_evolution(
        K,
        omega,
        t_max,
        n_steps,
        max_dense_gib,
    )
    omega_array = np.asarray(omega, dtype=np.float64)
    results: dict[str, SyncTrajectory] = {}
    for state_type in InitialState:
        circuit = prepare_initial_state(n_qubits, state_type, omega_array)
        vector = np.asarray(Statevector.from_instruction(circuit), dtype=np.complex128)
        density = _as_density_matrix(vector, n_qubits)
        results[state_type.value] = _trajectory_from_density(
            density,
            state_type.value,
            n_qubits,
            times,
            propagator,
        )
    return results


def compare_initial_states_with_dephased_controls(
    K: NDArray[np.float64],
    omega: NDArray[np.float64],
    t_max: float = 2.0,
    n_steps: int = 20,
    *,
    max_dense_gib: float | None = None,
) -> dict[str, InitialStateControlComparison]:
    """Compare each pure state with its population-matched dephased control.

    The separable product row is retained as an attribution control. If it also
    differs from its dephased counterpart, the study cannot identify
    entanglement as the unique cause of a coherence-score difference.
    """
    n_qubits, times, propagator = _build_evolution(
        K,
        omega,
        t_max,
        n_steps,
        max_dense_gib,
    )
    omega_array = np.asarray(omega, dtype=np.float64)
    raw_rows: dict[
        str,
        tuple[SyncTrajectory, SyncTrajectory, float],
    ] = {}
    for state_type in InitialState:
        circuit = prepare_initial_state(n_qubits, state_type, omega_array)
        vector = np.asarray(Statevector.from_instruction(circuit), dtype=np.complex128)
        density = _as_density_matrix(vector, n_qubits)
        dephased = np.diag(np.diag(density)).astype(np.complex128)
        pure_trajectory = _trajectory_from_density(
            density,
            state_type.value,
            n_qubits,
            times,
            propagator,
        )
        control_trajectory = _trajectory_from_density(
            dephased,
            f"{state_type.value}_dephased_control",
            n_qubits,
            times,
            propagator,
        )
        raw_rows[state_type.value] = (
            pure_trajectory,
            control_trajectory,
            mean_single_qubit_linear_entropy(vector, n_qubits),
        )

    product, product_control, _ = raw_rows[InitialState.PRODUCT.value]
    product_mean_delta = _time_average_difference(product, product_control)
    attribution_status = (
        "coherence_effect_not_entanglement_specific"
        if abs(product_mean_delta) > 1e-9
        else "entanglement_specificity_not_established"
    )
    certificate = issue_no_advantage_certificate(
        context="entanglement_initial_state_comparison",
        protocol_id=_COMPARISON_PROTOCOL,
    ).to_dict()
    comparisons: dict[str, InitialStateControlComparison] = {}
    for state_name, (trajectory, control, linear_entropy) in raw_rows.items():
        mean_score = _time_average(trajectory)
        control_mean_score = _time_average(control)
        comparisons[state_name] = InitialStateControlComparison(
            initial_state=state_name,
            initial_mean_single_qubit_linear_entropy=linear_entropy,
            final_exchange_coherence=trajectory.final_exchange_coherence,
            control_final_exchange_coherence=control.final_exchange_coherence,
            delta_final_exchange_coherence=(
                trajectory.final_exchange_coherence - control.final_exchange_coherence
            ),
            mean_exchange_coherence=mean_score,
            control_mean_exchange_coherence=control_mean_score,
            delta_mean_exchange_coherence=mean_score - control_mean_score,
            final_local_phase_order=trajectory.final_R,
            control_final_local_phase_order=control.final_R,
            final_phase_defined=trajectory.final_phase_defined,
            control_final_phase_defined=control.final_phase_defined,
            attribution_status=attribution_status,
            entanglement_specific_effect_supported=False,
            language_status="research_observation",
            no_advantage_certificate=certificate,
        )
    return comparisons


def _time_average(trajectory: SyncTrajectory) -> float:
    """Return the trapezoidal time average of exchange coherence."""
    values = np.asarray(trajectory.exchange_coherence_values, dtype=np.float64)
    times = np.asarray(trajectory.times, dtype=np.float64)
    widths = np.diff(times)
    integral = np.sum((values[:-1] + values[1:]) * widths * 0.5)
    return float(integral / (times[-1] - times[0]))


def _time_average_difference(
    trajectory: SyncTrajectory,
    control: SyncTrajectory,
) -> float:
    """Return pure-state minus control mean exchange coherence."""
    return _time_average(trajectory) - _time_average(control)


def _state_order_parameter(
    state: NDArray[np.complex128],
    n_qubits: int,
) -> float:
    """Return the visibility-aware local phase order compatibility value."""
    phase_order, _visibility, _defined = local_phase_observables(state, n_qubits)
    return phase_order


def entanglement_advantage(results: dict[str, SyncTrajectory]) -> dict[str, Any]:
    """Return a legacy descriptive comparison under a no-advantage certificate.

    This compatibility helper does not establish an entanglement-specific
    effect. New studies should call
    :func:`compare_initial_states_with_dephased_controls`.
    """
    product = results.get(InitialState.PRODUCT.value)
    if product is None:
        return {}
    certificate = issue_no_advantage_certificate(
        context="legacy_entanglement_advantage_helper",
        protocol_id=_COMPARISON_PROTOCOL,
    ).to_dict()
    comparisons: dict[str, Any] = {}
    for name, trajectory in results.items():
        if name == InitialState.PRODUCT.value:
            continue
        comparisons[name] = {
            "delta_R_final": trajectory.final_R - product.final_R,
            "delta_exchange_coherence_final": (
                trajectory.final_exchange_coherence - product.final_exchange_coherence
            ),
            "language_status": "no_advantage_default",
            "claim_status": "legacy_descriptive_comparison_only",
            "entanglement_specific_effect_supported": False,
            "no_advantage_certificate": certificate,
        }
    return comparisons
