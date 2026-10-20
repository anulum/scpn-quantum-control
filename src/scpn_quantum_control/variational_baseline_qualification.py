# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — variational baseline qualification
"""Qualify the existing positive-amplitude RBM without changing its optimiser.

Exact enumeration removes Monte Carlo sampling error. The remaining energy
gap cannot generally identify representation and optimisation errors separately.
Only diagonal problems and the two-spin zero-field family have an analytic
ansatz-infimum certificate here. Infima refer to the closure of the real RBM
family; they need not be attained by finite parameters.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from time import perf_counter
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .bridge.knm_hamiltonian import knm_to_dense_matrix
from .jax_nqs_baseline_product import (
    JAXNQSBaselineProduct,
    JAXNQSBaselineSpec,
    run_jax_nqs_baseline,
)
from .phase.jax_nqs import jax_rbm_energy, jax_vmc_ground_state
from .phase.nqs_ansatz import RBMWavefunction

EvaluationProblems = Mapping[str, tuple[NDArray[np.float64], NDArray[np.float64]]]
_MAX_PARAMETERS = 4096
_MAX_EVALUATIONS = 8
_ABSOLUTE_TOLERANCE = 1e-5
_RELATIVE_TOLERANCE = 1e-4


@dataclass(frozen=True, slots=True)
class VariationalEvaluationResult:
    """Observe one frozen state under an explicitly identified Hamiltonian.

    Attributes
    ----------
    label : str
        Caller evaluation label, or ``training`` for the fitted problem.
    hamiltonian_sha256 : str
        Digest of the effective dense complex128 matrix, including shape.
    exact_ground_energy : float
        Small dense reference minimum in the Hamiltonian's energy units.
    variational_energy, numpy_energy : float
        Public JAX and independently evaluated NumPy RBM expectations.
    reference_subspace_overlap, numpy_overlap : float
        Squared projections onto the declared lowest-energy reference subspace.
    reference_subspace_dimension : int
        Number of eigenvalues admitted by the recorded spectral tolerance.
    reference_subspace_energy_tolerance : float
        Absolute energy window above the reference minimum; not a gap bound.

    """

    label: str
    hamiltonian_sha256: str
    exact_ground_energy: float
    variational_energy: float
    numpy_energy: float
    reference_subspace_overlap: float
    numpy_overlap: float
    reference_subspace_dimension: int
    reference_subspace_energy_tolerance: float


@dataclass(frozen=True, slots=True)
class VariationalErrorBudget:
    """Separate certified ansatz limits from unresolved optimisation error.

    Attributes
    ----------
    certificate : str
        Analytic family establishing the infimum, or an explicit unknown reason.
    ansatz_energy_infimum : float or None
        Infimum in the closure of the real positive RBM family, if established.
    representation_bias : float or None
        Ansatz infimum minus exact ground energy; unknown without a certificate.
    optimisation_excess : float or None
        Observed energy minus the certified infimum, without numerical clipping.
    combined_energy_excess : float
        Signed observed energy minus exact ground energy, including round-off.
    monte_carlo_standard_error : float
        Zero for exhaustive configuration summation; not zero total uncertainty.

    """

    certificate: str
    ansatz_energy_infimum: float | None
    representation_bias: float | None
    optimisation_excess: float | None
    combined_energy_excess: float
    monte_carlo_standard_error: float


@dataclass(frozen=True, slots=True)
class VariationalQualificationCost:
    """Record functional costs, including compilation and a real seeded replay.

    Attributes
    ----------
    training_seconds, replay_seconds, evaluation_seconds : float
        Observed wall times on this process; not isolated performance evidence.
    training_energy_evaluations : int
        Explicit optimiser loss calls in the two training runs, excluding AD work.
    training_gradient_evaluations : int
        Explicit gradient-transform calls across the two training runs.
    evaluation_observable_calls : int
        Additional JAX energy/projector calls on frozen parameters.
    exact_configurations_per_observable : int
        Enumerated configuration count; no stochastic sample interpretation.

    """

    training_seconds: float
    replay_seconds: float
    evaluation_seconds: float
    training_energy_evaluations: int
    training_gradient_evaluations: int
    evaluation_observable_calls: int
    exact_configurations_per_observable: int


@dataclass(frozen=True, slots=True)
class VariationalBaselineQualification:
    """Keep the unchanged raw baseline beside independent qualification evidence.

    Attributes
    ----------
    baseline : JAXNQSBaselineProduct
        Original v2 record, retaining its original codec and claim boundary.
    training : VariationalEvaluationResult
        Frozen-state observations for the original fitted Hamiltonian.
    held_out : tuple of VariationalEvaluationResult
        Predeclared problems evaluated without another parameter update.
    errors : VariationalErrorBudget
        Identifiable and unknown error components for the training problem.
    costs : VariationalQualificationCost
        Actual functional wall times and explicitly defined call counts.
    parameters_sha256 : str
        Shape/dtype/value digest of the real replay's fitted parameters.
    parameter_shapes : tuple
        Named visible bias, hidden bias and weight shapes.
    parameter_dtypes : tuple
        Observed parameter dtypes; global X64 configuration is in the raw record.
    replay_max_energy_difference : float
        Largest difference between the complete original and replay histories.

    """

    baseline: JAXNQSBaselineProduct
    training: VariationalEvaluationResult
    held_out: tuple[VariationalEvaluationResult, ...]
    errors: VariationalErrorBudget
    costs: VariationalQualificationCost
    parameters_sha256: str
    parameter_shapes: tuple[tuple[str, tuple[int, ...]], ...]
    parameter_dtypes: tuple[tuple[str, str], ...]
    replay_max_energy_difference: float

    @property
    def global_optimum_claimed(self) -> bool:
        """Return false; an improving finite-budget iterate proves no optimum.

        Returns
        -------
        bool
            False even when an analytic reference is available.

        """
        return False

    @property
    def performance_advantage_claimed(self) -> bool:
        """Return false for these non-isolated functional observations.

        Returns
        -------
        bool
            False; timings include compilation and different diagnostic work.

        """
        return False

    def to_dict(self) -> dict[str, object]:
        """Serialise the companion while retaining the original raw codec.

        Returns
        -------
        dict
            JSON-ready qualification with a digest over the complete payload.

        Raises
        ------
        ValueError
            If a manually constructed record contains a non-finite number.

        """
        payload: dict[str, object] = {
            "schema": "variational_baseline_qualification.v1",
            "baseline": self.baseline.to_dict(),
            "training": asdict(self.training),
            "held_out": [asdict(row) for row in self.held_out],
            "errors": asdict(self.errors),
            "costs": asdict(self.costs),
            "parameters_sha256": self.parameters_sha256,
            "parameter_shapes": {name: list(shape) for name, shape in self.parameter_shapes},
            "parameter_dtypes": dict(self.parameter_dtypes),
            "replay_max_energy_difference": self.replay_max_energy_difference,
            "global_optimum_claimed": self.global_optimum_claimed,
            "performance_advantage_claimed": self.performance_advantage_claimed,
            "sampling_mode": "exact_enumeration",
            "cost_classification": "functional_non_isolated",
            "observable_absolute_tolerance": _ABSOLUTE_TOLERANCE,
            "observable_relative_tolerance": _RELATIVE_TOLERANCE,
            "claim_boundary": (
                "Bounded real positive-amplitude RBM qualification; reference-subspace "
                "overlap is measured separately from energy. Unknown error components "
                "remain unknown. No sampled VMC, general fidelity, global optimum, "
                "hardware, scaling or performance advantage qualification."
            ),
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        return {**payload, "evidence_sha256": hashlib.sha256(encoded.encode()).hexdigest()}


def qualify_jax_nqs_baseline(
    spec: JAXNQSBaselineSpec,
    *,
    held_out: EvaluationProblems | None = None,
) -> VariationalBaselineQualification:
    """Train, replay and evaluate the existing bounded JAX RBM through public APIs.

    Parameters
    ----------
    spec : JAXNQSBaselineSpec
        Original validated training request, restricted to at most 4096 parameters.
    held_out : mapping or None
        Up to eight non-training labelled ``(coupling, omega)`` pairs. Arrays are
        copied and validated before training, retain energy units and must have
        the same visible shape. ``training`` is reserved. The fitted parameters
        never change during evaluation; this is Hamiltonian transfer, not a
        statistical generalisation claim about an unseen dataset.

    Returns
    -------
    VariationalBaselineQualification
        Unchanged raw evidence, actual seeded replay, independent state checks,
        separate error components and functional execution costs.

    Raises
    ------
    ValueError
        If evaluation admission, shape, finiteness or resource bounds fail.
    ImportError
        If the required real JAX runtime is unavailable; no numerical fallback.
    RuntimeError
        If the original runner violates its contract, replay differs, or JAX
        observables disagree with independent NumPy expectations.

    """
    count = spec.n_qubits + spec.n_hidden + spec.n_qubits * spec.n_hidden
    if count > _MAX_PARAMETERS:
        raise ValueError("qualification requires at most 4096 RBM parameters")
    coupling = np.asarray(spec.coupling, dtype=np.float64)
    omega = np.asarray(spec.omega, dtype=np.float64)
    training_h = knm_to_dense_matrix(coupling, omega, max_dense_gib=spec.max_dense_gib)
    evaluations: list[tuple[str, NDArray[np.complex128]]] = []
    seen = {_matrix_digest(training_h)}
    if held_out is not None:
        if len(held_out) > _MAX_EVALUATIONS:
            raise ValueError("qualification permits at most eight held-out problems")
        for label, (evaluation_k, evaluation_omega) in held_out.items():
            if not isinstance(label, str) or not label.strip() or label == "training":
                raise ValueError("held-out labels must be non-empty strings other than training")
            copied = JAXNQSBaselineSpec.from_arrays(
                evaluation_k, evaluation_omega, max_dense_gib=spec.max_dense_gib
            )
            if copied.n_qubits != spec.n_qubits:
                raise ValueError("held-out visible shape must match the trained state")
            matrix = knm_to_dense_matrix(
                np.asarray(copied.coupling),
                np.asarray(copied.omega),
                max_dense_gib=spec.max_dense_gib,
            )
            digest = _matrix_digest(matrix)
            if digest in seen:
                raise ValueError("held-out effective Hamiltonians must be distinct from training")
            seen.add(digest)
            evaluations.append((label, matrix))

    started = perf_counter()
    baseline = run_jax_nqs_baseline(spec)
    training_seconds = perf_counter() - started
    started = perf_counter()
    replay = jax_vmc_ground_state(
        coupling,
        omega,
        n_hidden=spec.n_hidden,
        learning_rate=spec.learning_rate,
        n_iterations=spec.n_iterations,
        seed=spec.seed,
        max_dense_gib=spec.max_dense_gib,
    )
    replay_seconds = perf_counter() - started
    history = np.asarray(replay["energy_history"], dtype=np.float64)
    original = np.asarray(baseline.comparison.energy_history, dtype=np.float64)
    if history.shape != original.shape or not np.array_equal(history, original):
        raise RuntimeError("seeded replay differs from the original complete energy history")
    if replay["n_params"] != count or float(replay["energy"]) != history[-1]:
        raise RuntimeError("seeded replay returned inconsistent energy or parameter count")
    shapes = {"a": (spec.n_qubits,), "b": (spec.n_hidden,), "W": (spec.n_hidden, spec.n_qubits)}
    if not isinstance(replay["params"], dict) or set(replay["params"]) != set(shapes):
        raise RuntimeError("seeded replay returned an incomplete real parameter mapping")
    params = {name: np.asarray(replay["params"][name]) for name in shapes}
    for name, shape in shapes.items():
        value = params[name]
        if value.shape != shape or value.dtype.kind != "f" or not np.all(np.isfinite(value)):
            raise RuntimeError("seeded replay returned malformed or non-finite real parameters")
    encoded_params = json.dumps(
        {
            name: {"shape": list(value.shape), "dtype": str(value.dtype), "values": value.tolist()}
            for name, value in params.items()
        },
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    started = perf_counter()
    numpy_state = RBMWavefunction(spec.n_qubits, spec.n_hidden, seed=spec.seed)
    numpy_state.a = np.asarray(params["a"], dtype=np.complex128)
    numpy_state.b = np.asarray(params["b"], dtype=np.complex128)
    numpy_state.W = np.asarray(params["W"], dtype=np.complex128)
    state = numpy_state.all_amplitudes()
    if not np.all(np.isfinite(state)) or not math.isclose(float(np.vdot(state, state).real), 1.0):
        raise RuntimeError("independent RBM state is not finite and normalised")
    training = _observe("training", training_h, params, state, spec.n_qubits)
    if not _agrees(training.variational_energy, baseline.comparison.variational_energy):
        raise RuntimeError("fitted parameters do not reproduce the original final energy")
    observations = tuple(
        _observe(label, h, params, state, spec.n_qubits) for label, h in evaluations
    )
    errors = _error_budget(spec, training_h, training)
    elapsed = perf_counter() - started
    return VariationalBaselineQualification(
        baseline=baseline,
        training=training,
        held_out=observations,
        errors=errors,
        costs=VariationalQualificationCost(
            training_seconds,
            replay_seconds,
            elapsed,
            2 * (spec.n_iterations + 1),
            2 * spec.n_iterations,
            2 * (len(observations) + 1),
            2**spec.n_qubits,
        ),
        parameters_sha256=hashlib.sha256(encoded_params.encode()).hexdigest(),
        parameter_shapes=tuple(shapes.items()),
        parameter_dtypes=tuple((name, str(value.dtype)) for name, value in params.items()),
        replay_max_energy_difference=float(np.max(np.abs(history - original))),
    )


def _matrix_digest(matrix: NDArray[np.complex128]) -> str:
    """Bind shape and little-endian complex128 bytes of the effective operator."""
    encoded = np.ascontiguousarray(np.where(matrix == 0, 0.0, matrix), dtype="<c16").tobytes()
    return hashlib.sha256(str(matrix.shape).encode() + b"\n" + encoded).hexdigest()


def _agrees(actual: float, reference: float) -> bool:
    """Apply the declared float32 absolute-plus-relative observable budget."""
    return math.isfinite(actual) and abs(actual - reference) <= (
        _ABSOLUTE_TOLERANCE + _RELATIVE_TOLERANCE * abs(reference)
    )


def _observe(
    label: str,
    hamiltonian: NDArray[np.complex128],
    params: dict[str, Any],
    state: NDArray[np.complex128],
    n: int,
) -> VariationalEvaluationResult:
    """Compare public JAX energy/projector expectations with an independent state."""
    eigenvalues, vectors = np.linalg.eigh(hamiltonian)
    minimum = float(eigenvalues[0])
    window = 1e-10 + 1e-8 * abs(minimum)
    basis = vectors[:, eigenvalues <= minimum + window]
    projector = basis @ basis.conj().T
    energy = float(jax_rbm_energy(params, hamiltonian.real, n))
    overlap = float(jax_rbm_energy(params, projector.real, n))
    numpy_energy = float(np.vdot(state, hamiltonian @ state).real)
    numpy_overlap = float(np.linalg.norm(basis.conj().T @ state) ** 2)
    if not _agrees(energy, numpy_energy) or not _agrees(overlap, numpy_overlap):
        raise RuntimeError("JAX frozen observables disagree with the independent NumPy state")
    return VariationalEvaluationResult(
        label,
        _matrix_digest(hamiltonian),
        minimum,
        energy,
        numpy_energy,
        overlap,
        numpy_overlap,
        basis.shape[1],
        window,
    )


def _error_budget(
    spec: JAXNQSBaselineSpec,
    hamiltonian: NDArray[np.complex128],
    training: VariationalEvaluationResult,
) -> VariationalErrorBudget:
    """Use an analytic infimum only for an admitted effective Hamiltonian family."""
    off_diagonal = hamiltonian - np.diag(np.diag(hamiltonian))
    infimum: float | None
    if not np.any(off_diagonal):
        infimum = float(np.min(hamiltonian.diagonal().real))
        certificate = "diagonal_product_state_infimum"
    elif spec.n_qubits == 2 and not np.any(hamiltonian.diagonal()):
        infimum = min(float(hamiltonian[1, 2].real), 0.0)
        certificate = "two_spin_zero_field_positive_amplitude_infimum"
    else:
        infimum = None
        certificate = "unknown_ansatz_infimum; representation_and_optimisation_not_identifiable"
    return VariationalErrorBudget(
        certificate,
        infimum,
        None if infimum is None else infimum - training.exact_ground_energy,
        None if infimum is None else training.variational_energy - infimum,
        training.variational_energy - training.exact_ground_energy,
        0.0,
    )


__all__ = [
    "EvaluationProblems",
    "VariationalBaselineQualification",
    "VariationalEvaluationResult",
    "VariationalErrorBudget",
    "VariationalQualificationCost",
    "qualify_jax_nqs_baseline",
]
