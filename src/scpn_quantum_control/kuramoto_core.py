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
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from numpy.typing import NDArray
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Statevector

from .bridge.knm_hamiltonian import knm_to_dense_matrix, knm_to_hamiltonian
from .phase.xy_kuramoto import QuantumKuramotoSolver

if TYPE_CHECKING:
    from oscillatools.accel.kuramoto_system import KuramotoSystem

    from .hardware.analog_kuramoto import AnalogKuramotoPlatform, AnalogKuramotoProgram
    from .hardware.hybrid_digital_analog import HybridDigitalAnalogProgram
    from .phase.kuramoto_variants import KuramotoVariantResult
    from .scientific_design import ScientificDesign

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
        """Validate inputs and freeze defensive copies after construction."""
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


def validate_scientific_design(problem: KuramotoProblem, design: ScientificDesign) -> None:
    """Validate a scientific declaration against this original problem owner.

    Parameters
    ----------
    problem
        Existing finite symmetric coupling matrix and intrinsic frequencies.
    design
        Immutable explicit model, units, graph, initial/history state and objective.

    Raises
    ------
    ValueError
        Units, indices, shapes or declared model semantics are unsupported.

    Notes
    -----
    No state is saved or submitted and no unit or phase-to-spin conversion occurs.

    """
    from .scientific_design import DesignObjective, ScientificDesign, ScientificUnits

    if not isinstance(problem, KuramotoProblem):
        raise ValueError("problem requires KuramotoProblem")
    problem.validate()
    if not isinstance(design, ScientificDesign):
        raise ValueError("design requires ScientificDesign")
    if design.model not in ("phase_kuramoto", "quantum_xy"):
        raise ValueError("model must be phase_kuramoto or quantum_xy")
    if design.normalisation not in ("pairwise_sum", "population_mean"):
        raise ValueError("normalisation must be pairwise_sum or population_mean")
    if design.coordinate_space not in ("logical", "physical"):
        raise ValueError("coordinate_space must be logical or physical")
    if not isinstance(design.units, ScientificUnits):
        raise ValueError("units requires explicit ScientificUnits")
    units = design.units
    if (units.time, units.frequency, units.coupling) not in (
        ("s", "rad/s", "rad/s"),
        ("1", "1", "1"),
    ) or units.observable != "1":
        raise ValueError("units must declare one consistent dimensional or dimensionless regime")
    n = problem.n_oscillators
    phase = design.model == "phase_kuramoto"
    if units.state != ("rad" if phase else "1"):
        raise ValueError("units.state must distinguish phase angles from quantum amplitudes")
    expected_observable = "phase_order_parameter" if phase else "spin_z"
    if design.observable != expected_observable:
        raise ValueError("observable must match the declared phase or quantum-spin model")
    if design.observable_weights.shape != (n,):
        raise ValueError(f"observable_weights must have shape ({n},)")
    if phase and (
        np.any(design.observable_weights < 0.0) or not np.any(design.observable_weights > 0.0)
    ):
        raise ValueError("phase observable_weights must be nonnegative and have positive total")
    if not phase and design.normalisation != "pairwise_sum":
        raise ValueError("quantum_xy requires the original pairwise_sum convention")
    state_shape = (n,) if phase else (2**n,)
    if design.initial_state.shape != state_shape:
        raise ValueError(f"initial_state must have shape {state_shape}")
    if not phase and not np.isclose(
        np.vdot(design.initial_state, design.initial_state).real, 1.0, rtol=0.0, atol=1e-12
    ):
        raise ValueError("quantum initial_state must have unit norm")
    edges: set[tuple[int, int]] = set()
    for edge in design.topology:
        if len(edge) != 2 or any(not isinstance(i, int) or isinstance(i, bool) for i in edge):
            raise ValueError("topology requires integer index pairs")
        first, second = edge
        if not 0 <= first < second < n or edge in edges:
            raise ValueError("topology requires unique ordered undirected edges")
        edges.add(edge)
    if tuple(sorted(edges)) != design.topology:
        raise ValueError("topology edges must be sorted")
    for first in range(n):
        for second in range(first + 1, n):
            if (problem.K_nm[first, second] != 0.0 or problem.K_nm[second, first] != 0.0) and (
                first,
                second,
            ) not in edges:
                raise ValueError("topology must cover every nonzero coupling")
    times, states = design.history_times, design.history_states
    if (times is None) != (states is None):
        raise ValueError("history_times and history_states must be supplied together")
    if times is not None and states is not None:
        if not phase:
            raise ValueError("quantum_xy phase history is unsupported")
        if times.ndim != 1 or len(times) == 0 or states.shape != (len(times), n):
            raise ValueError("history must have shapes (H,) and (H,N) with H > 0")
        if times[-1] != 0.0 or np.any(np.diff(times) <= 0.0):
            raise ValueError("history_times must increase strictly and end at zero")
        if not np.array_equal(states[-1], design.initial_state):
            raise ValueError("history_states must end at initial_state")
    objective = design.objective
    if not isinstance(objective, DesignObjective):
        raise ValueError("objective requires DesignObjective")
    if objective.kind not in (
        "simulate",
        "synchronise",
        "maximise_observable",
        "minimise_gate_cost",
    ):
        raise ValueError("objective kind is unsupported")
    expected_unit = "gate" if objective.kind == "minimise_gate_cost" else "1"
    if objective.unit != expected_unit:
        raise ValueError("objective unit does not match its estimand")
    target = objective.target
    if target is not None and (
        isinstance(target, bool) or not isinstance(target, (int, float)) or not np.isfinite(target)
    ):
        raise ValueError("objective target must be a finite scalar")
    if objective.kind == "simulate" and target is not None:
        raise ValueError("simulate objective has no target")
    if objective.kind == "synchronise" and (
        not phase or target is None or not 0.0 <= target <= 1.0
    ):
        raise ValueError("synchronise requires a phase-order target in [0,1]")
    if objective.kind == "minimise_gate_cost" and (
        phase or (target is not None and (target < 0.0 or target != int(target)))
    ):
        raise ValueError(
            "minimise_gate_cost requires a quantum model and nonnegative integer target"
        )
    allowed = {"omega", "K_nm", "initial_state_real", "observable_weights"}
    if not phase:
        allowed.add("initial_state_imag")
    if times is not None:
        allowed.update(("history_times", "history_states"))
    if len(set(design.trainable)) != len(design.trainable) or any(
        key not in allowed for key in design.trainable
    ):
        raise ValueError("trainable requires unique declared parameter keys")


def build_scientific_phase_system(
    problem: KuramotoProblem,
    design: ScientificDesign,
    *,
    dt: float,
    model: str = "finite_networked",
    scheme: str = "rk4",
) -> KuramotoSystem:
    r"""Bind an explicit finite phase design to the original numerical system.

    Parameters
    ----------
    problem
        Original ``(N, N)`` symmetric coupling/frequency owner. Frequencies
        are in radians per declared time unit, without unit conversion.
    design
        Explicit phase model, ``(N,)`` initial phases in radians, units,
        topology and coupling normalisation. The original scientific validator
        checks the binding before constructing a numerical system.
    dt
        Finite positive step in the declared time unit.
    model
        ``finite_networked`` uses pairwise matrix coefficients;
        ``finite_mean_field`` requires uniform off-diagonal coefficients and
        projects them to the original scalar ``K/N`` mean-field owner.
    scheme
        Original ``euler`` or ``rk4`` integrator, without phase wrapping.

    Returns
    -------
    oscillatools.accel.kuramoto_system.KuramotoSystem
        Original system with independent state and parameter arrays, time zero
        and the supplied step. Its trajectory includes the initial state.

    Raises
    ------
    ValueError
        The binding, model/solver pair, time step or mean-field topology is
        unsupported. Supplied history is refused by this instantaneous owner.

    Notes
    -----
    The phase rule is :math:`\dot\theta_j=\omega_j+
    \sum_k C_{jk}\sin(\theta_k-\theta_j)`. No phase-to-spin conversion,
    history truncation, unit conversion, new kernel or backend selection occurs.
    Existing numerical owners retain their dispatch and resource policies.

    """
    from oscillatools.accel.kuramoto_system import KuramotoSystem

    from .kuramoto_model_conventions import kuramoto_model_convention

    validate_scientific_design(problem, design)
    convention = kuramoto_model_convention(model, scheme)
    if convention.model not in ("finite_networked", "finite_mean_field"):
        raise ValueError("this factory requires an instantaneous finite phase model")
    if isinstance(dt, bool) or not isinstance(dt, (int, float)) or not np.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be a finite positive scalar")
    if design.model != "phase_kuramoto":
        raise ValueError("this factory requires phase_kuramoto scientific inputs")
    if design.history_times is not None:
        raise ValueError("this instantaneous phase model does not consume history")
    phases = cast(NDArray[np.float64], design.initial_state)
    count = problem.n_oscillators
    coupling = problem.K_nm
    if design.normalisation == "population_mean":
        coupling = coupling / count
    if model == "finite_networked":
        return KuramotoSystem.networked(
            phases, problem.omega, coupling, dt=float(dt), scheme=scheme
        )
    off_diagonal = coupling[~np.eye(count, dtype=np.bool_)]
    coefficient = float(off_diagonal[0]) if count > 1 else 0.0
    if np.any(off_diagonal != coefficient):
        raise ValueError("finite_mean_field requires uniform off-diagonal coupling")
    scalar_coupling = count * coefficient
    if not np.isfinite(scalar_coupling):
        raise ValueError("finite_mean_field scalar coupling must remain finite")
    return KuramotoSystem.mean_field(
        phases, problem.omega, scalar_coupling, dt=float(dt), scheme=scheme
    )


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
