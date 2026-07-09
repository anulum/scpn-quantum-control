# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Open-System Differentiable Objectives
"""Bounded Lindblad and MCWF objective certificates."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from .lindblad import LindbladKuramotoSolver
from .tensor_jump import mcwf_ensemble

OPEN_SYSTEM_OBJECTIVE_EVIDENCE_CLASS = "functional_non_isolated"
OPEN_SYSTEM_OBJECTIVE_CLAIM_BOUNDARY = (
    "Bounded open-system objective evidence uses scipy Lindblad density-matrix "
    "evolution and seeded MCWF trajectory ensembles on small local systems. "
    "Gradients are deterministic central finite differences over scalar coupling "
    "and damping scales; they are not hardware gradients, adjoint Lindblad "
    "gradients, unbiased stochastic-gradient estimators, or isolated performance "
    "benchmarks."
)

OpenSystemBackend = Literal["lindblad_density", "mcwf_ensemble"]
BoundaryStatus = Literal["hard_gap"]


def _as_float_array(name: str, values: object, *, ndim: int) -> NDArray[np.float64]:
    """Return a finite float64 array with the required rank."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}-dimensional.")
    if array.size == 0:
        raise ValueError(f"{name} must be non-empty.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    return np.array(array, dtype=np.float64, copy=True)


def _as_positive_float(name: str, value: object) -> float:
    """Return a finite positive scalar."""
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite positive scalar.")
    raw = np.asarray(value)
    if raw.shape != () or raw.dtype.kind in {"b", "O", "S", "U", "c"}:
        raise ValueError(f"{name} must be a finite positive scalar.")
    scalar = float(raw.item())
    if not np.isfinite(scalar) or scalar <= 0.0:
        raise ValueError(f"{name} must be finite and positive.")
    return scalar


def _as_nonnegative_float(name: str, value: object) -> float:
    """Return a finite non-negative scalar."""
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite non-negative scalar.")
    raw = np.asarray(value)
    if raw.shape != () or raw.dtype.kind in {"b", "O", "S", "U", "c"}:
        raise ValueError(f"{name} must be a finite non-negative scalar.")
    scalar = float(raw.item())
    if not np.isfinite(scalar) or scalar < 0.0:
        raise ValueError(f"{name} must be finite and non-negative.")
    return scalar


def _as_positive_int(name: str, value: object) -> int:
    """Return a positive integer without bool coercion."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return value


def _parameter_vector(values: object) -> NDArray[np.float64]:
    """Validate the two-parameter open-system objective vector."""
    params = _as_float_array("params", values, ndim=1)
    if params.shape != (2,):
        raise ValueError("params must have shape (2,) for coupling and damping scales.")
    if np.any(params <= 0.0):
        raise ValueError("params must contain positive coupling and damping scales.")
    return params


@dataclass(frozen=True)
class BoundedOpenSystemObjectiveCase:
    """Definition of a small open-system objective case."""

    case_id: str
    n_oscillators: int
    coupling_matrix: NDArray[np.float64]
    omega: NDArray[np.float64]
    gamma_amp: float
    gamma_deph: float
    initial_params: NDArray[np.float64]
    target_order_parameter: float
    target_purity: float
    t_max: float
    dt: float
    n_trajectories: int
    seed: int
    finite_difference_step: float = 1.0e-3
    trace_tolerance: float = 5.0e-8
    hermiticity_tolerance: float = 5.0e-8
    positivity_tolerance: float = 5.0e-8
    reproducibility_tolerance: float = 1.0e-12

    def __post_init__(self) -> None:
        """Validate immutable case metadata."""
        if not self.case_id.strip():
            raise ValueError("case_id must be non-empty.")
        n_oscillators = _as_positive_int("n_oscillators", self.n_oscillators)
        coupling = _as_float_array("coupling_matrix", self.coupling_matrix, ndim=2)
        omega = _as_float_array("omega", self.omega, ndim=1)
        if coupling.shape != (n_oscillators, n_oscillators):
            raise ValueError("coupling_matrix must have shape (n_oscillators, n_oscillators).")
        if omega.shape != (n_oscillators,):
            raise ValueError("omega must have shape (n_oscillators,).")
        if not np.allclose(coupling, coupling.T, atol=1.0e-12, rtol=1.0e-12):
            raise ValueError("coupling_matrix must be symmetric.")
        gamma_amp = _as_nonnegative_float("gamma_amp", self.gamma_amp)
        gamma_deph = _as_nonnegative_float("gamma_deph", self.gamma_deph)
        initial_params = _parameter_vector(self.initial_params)
        target_order_parameter = _as_nonnegative_float(
            "target_order_parameter", self.target_order_parameter
        )
        if target_order_parameter > 1.0:
            raise ValueError("target_order_parameter must be at most 1.")
        target_purity = _as_positive_float("target_purity", self.target_purity)
        if target_purity > 1.0:
            raise ValueError("target_purity must be at most 1.")
        t_max = _as_nonnegative_float("t_max", self.t_max)
        dt = _as_positive_float("dt", self.dt)
        n_trajectories = _as_positive_int("n_trajectories", self.n_trajectories)
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a non-negative integer.")
        finite_difference_step = _as_positive_float(
            "finite_difference_step", self.finite_difference_step
        )
        if np.any(initial_params - finite_difference_step <= 0.0):
            raise ValueError("finite_difference_step must keep perturbed params positive.")
        object.__setattr__(self, "n_oscillators", n_oscillators)
        object.__setattr__(self, "coupling_matrix", coupling)
        object.__setattr__(self, "omega", omega)
        object.__setattr__(self, "gamma_amp", gamma_amp)
        object.__setattr__(self, "gamma_deph", gamma_deph)
        object.__setattr__(self, "initial_params", initial_params)
        object.__setattr__(self, "target_order_parameter", target_order_parameter)
        object.__setattr__(self, "target_purity", target_purity)
        object.__setattr__(self, "t_max", t_max)
        object.__setattr__(self, "dt", dt)
        object.__setattr__(self, "n_trajectories", n_trajectories)
        object.__setattr__(self, "finite_difference_step", finite_difference_step)
        object.__setattr__(
            self, "trace_tolerance", _as_positive_float("trace_tolerance", self.trace_tolerance)
        )
        object.__setattr__(
            self,
            "hermiticity_tolerance",
            _as_positive_float("hermiticity_tolerance", self.hermiticity_tolerance),
        )
        object.__setattr__(
            self,
            "positivity_tolerance",
            _as_positive_float("positivity_tolerance", self.positivity_tolerance),
        )
        object.__setattr__(
            self,
            "reproducibility_tolerance",
            _as_positive_float("reproducibility_tolerance", self.reproducibility_tolerance),
        )

    def scaled_inputs(
        self, params: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], float, float]:
        """Return scaled coupling and damping rates for ``params``."""
        checked = _parameter_vector(params)
        coupling_scale = float(checked[0])
        damping_scale = float(checked[1])
        return (
            self.coupling_matrix * coupling_scale,
            self.gamma_amp * damping_scale,
            self.gamma_deph * damping_scale,
        )

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready case definition."""
        return {
            "case_id": self.case_id,
            "n_oscillators": self.n_oscillators,
            "coupling_matrix": self.coupling_matrix.tolist(),
            "omega": self.omega.tolist(),
            "gamma_amp": self.gamma_amp,
            "gamma_deph": self.gamma_deph,
            "initial_params": self.initial_params.tolist(),
            "target_order_parameter": self.target_order_parameter,
            "target_purity": self.target_purity,
            "t_max": self.t_max,
            "dt": self.dt,
            "n_trajectories": self.n_trajectories,
            "seed": self.seed,
            "finite_difference_step": self.finite_difference_step,
        }


@dataclass(frozen=True)
class DensityMatrixInvariantCertificate:
    """Trace, Hermiticity, and positivity certificate for a density matrix."""

    trace_error: float
    hermiticity_error: float
    min_eigenvalue: float
    purity: float
    finite: bool
    passed: bool

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready density-matrix invariant evidence."""
        return {
            "trace_error": self.trace_error,
            "hermiticity_error": self.hermiticity_error,
            "min_eigenvalue": self.min_eigenvalue,
            "purity": self.purity,
            "finite": self.finite,
            "passed": self.passed,
        }


@dataclass(frozen=True)
class MCWFReproducibilityCertificate:
    """Seeded trajectory-batch reproducibility certificate."""

    n_trajectories: int
    seed: int
    trajectory_shape: tuple[int, int]
    same_seed_max_abs_diff: float
    final_mean_order_parameter: float
    final_std_order_parameter: float
    total_jumps: int
    finite: bool
    passed: bool

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready trajectory reproducibility evidence."""
        return {
            "n_trajectories": self.n_trajectories,
            "seed": self.seed,
            "trajectory_shape": list(self.trajectory_shape),
            "same_seed_max_abs_diff": self.same_seed_max_abs_diff,
            "final_mean_order_parameter": self.final_mean_order_parameter,
            "final_std_order_parameter": self.final_std_order_parameter,
            "total_jumps": self.total_jumps,
            "finite": self.finite,
            "passed": self.passed,
        }


@dataclass(frozen=True)
class OpenSystemObjectiveRecord:
    """One differentiable open-system objective evaluation record."""

    case_id: str
    backend: OpenSystemBackend
    params: tuple[float, float]
    value: float
    gradient: tuple[float, float]
    evaluations: int
    final_order_parameter: float
    invariant_certificate: DensityMatrixInvariantCertificate | None
    reproducibility_certificate: MCWFReproducibilityCertificate | None
    evidence_class: str = OPEN_SYSTEM_OBJECTIVE_EVIDENCE_CLASS
    claim_boundary: str = OPEN_SYSTEM_OBJECTIVE_CLAIM_BOUNDARY

    @property
    def passed(self) -> bool:
        """Return whether all backend-specific certificates passed."""
        invariant_passed = (
            True if self.invariant_certificate is None else self.invariant_certificate.passed
        )
        reproducibility_passed = (
            True
            if self.reproducibility_certificate is None
            else self.reproducibility_certificate.passed
        )
        finite_values = np.isfinite([self.value, *self.gradient, self.final_order_parameter])
        return bool(invariant_passed and reproducibility_passed and np.all(finite_values))

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready objective record evidence."""
        return {
            "case_id": self.case_id,
            "backend": self.backend,
            "params": list(self.params),
            "value": self.value,
            "gradient": list(self.gradient),
            "evaluations": self.evaluations,
            "final_order_parameter": self.final_order_parameter,
            "invariant_certificate": (
                None
                if self.invariant_certificate is None
                else self.invariant_certificate.to_dict()
            ),
            "reproducibility_certificate": (
                None
                if self.reproducibility_certificate is None
                else self.reproducibility_certificate.to_dict()
            ),
            "passed": self.passed,
            "evidence_class": self.evidence_class,
            "claim_boundary": self.claim_boundary,
        }


@dataclass(frozen=True)
class OpenSystemObjectiveBoundaryRow:
    """Non-executable boundary row for open-system objective claims."""

    case_id: str
    backend: str
    status: BoundaryStatus
    failure_class: str
    setup_instructions: str
    evidence_class: str = "hard_gap"
    claim_boundary: str = OPEN_SYSTEM_OBJECTIVE_CLAIM_BOUNDARY

    def __post_init__(self) -> None:
        """Validate boundary-row metadata."""
        if not self.case_id.strip():
            raise ValueError("case_id must be non-empty.")
        if not self.backend.strip():
            raise ValueError("backend must be non-empty.")
        if self.status != "hard_gap":
            raise ValueError("status must be hard_gap.")
        if not self.failure_class.strip():
            raise ValueError("failure_class must be non-empty.")
        if not self.setup_instructions.strip():
            raise ValueError("setup_instructions must be non-empty.")

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready boundary-row evidence."""
        return {
            "case_id": self.case_id,
            "backend": self.backend,
            "status": self.status,
            "failure_class": self.failure_class,
            "setup_instructions": self.setup_instructions,
            "evidence_class": self.evidence_class,
            "claim_boundary": self.claim_boundary,
        }


@dataclass(frozen=True)
class OpenSystemObjectiveSuiteResult:
    """BL-16 open-system objective suite result."""

    cases: tuple[BoundedOpenSystemObjectiveCase, ...]
    records: tuple[OpenSystemObjectiveRecord, ...]
    boundary_rows: tuple[OpenSystemObjectiveBoundaryRow, ...]
    evidence_class: str = OPEN_SYSTEM_OBJECTIVE_EVIDENCE_CLASS
    claim_boundary: str = OPEN_SYSTEM_OBJECTIVE_CLAIM_BOUNDARY

    @property
    def passed(self) -> bool:
        """Return whether all executable objective records passed."""
        return all(record.passed for record in self.records)

    @property
    def case_count(self) -> int:
        """Return the number of objective cases."""
        return len(self.cases)

    @property
    def record_count(self) -> int:
        """Return the number of executable objective rows."""
        return len(self.records)

    @property
    def backend_names(self) -> tuple[str, ...]:
        """Return the backend names present in executable rows."""
        return tuple(dict.fromkeys(record.backend for record in self.records))

    def records_for_case(self, case_id: str) -> tuple[OpenSystemObjectiveRecord, ...]:
        """Return all executable records for ``case_id``."""
        records = tuple(record for record in self.records if record.case_id == case_id)
        if not records:
            raise KeyError(f"unknown open-system objective case: {case_id}")
        return records

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready suite evidence."""
        return {
            "passed": self.passed,
            "case_count": self.case_count,
            "record_count": self.record_count,
            "backend_names": list(self.backend_names),
            "evidence_class": self.evidence_class,
            "claim_boundary": self.claim_boundary,
            "cases": [case.to_dict() for case in self.cases],
            "records": [record.to_dict() for record in self.records],
            "boundary_rows": [row.to_dict() for row in self.boundary_rows],
        }


def default_open_system_objective_cases() -> tuple[BoundedOpenSystemObjectiveCase, ...]:
    """Return deterministic bounded open-system objective cases."""
    return (
        BoundedOpenSystemObjectiveCase(
            case_id="two_qubit_relaxing_sync",
            n_oscillators=2,
            coupling_matrix=np.array([[0.0, 0.32], [0.32, 0.0]], dtype=np.float64),
            omega=np.array([0.7, 1.15], dtype=np.float64),
            gamma_amp=0.2,
            gamma_deph=0.02,
            initial_params=np.array([1.0, 1.0], dtype=np.float64),
            target_order_parameter=0.42,
            target_purity=0.78,
            t_max=0.5,
            dt=0.1,
            n_trajectories=12,
            seed=1701,
        ),
        BoundedOpenSystemObjectiveCase(
            case_id="two_qubit_dephasing_balance",
            n_oscillators=2,
            coupling_matrix=np.array([[0.0, 0.24], [0.24, 0.0]], dtype=np.float64),
            omega=np.array([0.55, 0.95], dtype=np.float64),
            gamma_amp=0.12,
            gamma_deph=0.08,
            initial_params=np.array([0.9, 1.1], dtype=np.float64),
            target_order_parameter=0.50,
            target_purity=0.82,
            t_max=0.5,
            dt=0.1,
            n_trajectories=12,
            seed=1702,
        ),
    )


def evaluate_lindblad_objective(
    case: BoundedOpenSystemObjectiveCase,
    params: NDArray[np.float64],
) -> tuple[float, float, DensityMatrixInvariantCertificate]:
    """Evaluate one density-matrix Lindblad objective."""
    checked = _parameter_vector(params)
    coupling, gamma_amp, gamma_deph = case.scaled_inputs(checked)
    solver = LindbladKuramotoSolver(
        case.n_oscillators,
        coupling,
        case.omega,
        gamma_amp=gamma_amp,
        gamma_deph=gamma_deph,
    )
    result = solver.run(case.t_max, case.dt)
    final_r = float(np.asarray(result["R"], dtype=np.float64)[-1])
    rho = np.asarray(result["rho_final"], dtype=np.complex128)
    invariant = certify_density_matrix_invariants(case, rho)
    objective = (final_r - case.target_order_parameter) ** 2 + 0.05 * (
        invariant.purity - case.target_purity
    ) ** 2
    return float(objective), final_r, invariant


def evaluate_mcwf_objective(
    case: BoundedOpenSystemObjectiveCase,
    params: NDArray[np.float64],
) -> tuple[float, float, MCWFReproducibilityCertificate]:
    """Evaluate one seeded MCWF trajectory-ensemble objective."""
    checked = _parameter_vector(params)
    coupling, gamma_amp, gamma_deph = case.scaled_inputs(checked)
    first = mcwf_ensemble(
        coupling,
        case.omega,
        gamma_amp=gamma_amp,
        gamma_deph=gamma_deph,
        t_max=case.t_max,
        dt=case.dt,
        n_trajectories=case.n_trajectories,
        seed=case.seed,
    )
    second = mcwf_ensemble(
        coupling,
        case.omega,
        gamma_amp=gamma_amp,
        gamma_deph=gamma_deph,
        t_max=case.t_max,
        dt=case.dt,
        n_trajectories=case.n_trajectories,
        seed=case.seed,
    )
    certificate = certify_mcwf_reproducibility(case, first, second)
    final_r = certificate.final_mean_order_parameter
    objective = (final_r - case.target_order_parameter) ** 2 + 0.05 * (
        certificate.final_std_order_parameter**2
    )
    return float(objective), final_r, certificate


def certify_density_matrix_invariants(
    case: BoundedOpenSystemObjectiveCase,
    rho: NDArray[np.complex128],
) -> DensityMatrixInvariantCertificate:
    """Certify trace preservation, Hermiticity, and positivity for ``rho``."""
    matrix = np.asarray(rho, dtype=np.complex128)
    if matrix.shape != (2**case.n_oscillators, 2**case.n_oscillators):
        raise ValueError("rho shape does not match the objective case dimension.")
    finite = bool(np.all(np.isfinite(matrix)))
    trace_error = float(abs(np.trace(matrix) - 1.0))
    hermiticity_error = float(np.max(np.abs(matrix - matrix.conj().T)))
    eigenvalues = np.linalg.eigvalsh((matrix + matrix.conj().T) / 2.0)
    min_eigenvalue = float(np.min(eigenvalues))
    purity = float(np.real(np.trace(matrix @ matrix)))
    passed = (
        finite
        and trace_error <= case.trace_tolerance
        and hermiticity_error <= case.hermiticity_tolerance
        and min_eigenvalue >= -case.positivity_tolerance
        and 0.0 <= purity <= 1.0 + case.trace_tolerance
    )
    return DensityMatrixInvariantCertificate(
        trace_error=trace_error,
        hermiticity_error=hermiticity_error,
        min_eigenvalue=min_eigenvalue,
        purity=purity,
        finite=finite,
        passed=passed,
    )


def certify_mcwf_reproducibility(
    case: BoundedOpenSystemObjectiveCase,
    first: dict[str, object],
    second: dict[str, object],
) -> MCWFReproducibilityCertificate:
    """Certify same-seed MCWF ensemble replay and trajectory batching."""
    first_trajectories = np.asarray(first["R_trajectories"], dtype=np.float64)
    second_trajectories = np.asarray(second["R_trajectories"], dtype=np.float64)
    if first_trajectories.shape != second_trajectories.shape:
        raise ValueError("same-seed MCWF trajectory batches must share shape.")
    if first_trajectories.ndim != 2:
        raise ValueError("MCWF trajectory batch must be two-dimensional.")
    if first_trajectories.shape[0] != case.n_trajectories:
        raise ValueError("MCWF trajectory batch row count does not match n_trajectories.")
    first_mean = np.asarray(first["R_mean"], dtype=np.float64)
    first_std = np.asarray(first["R_std"], dtype=np.float64)
    if first_mean.shape != first_std.shape:
        raise ValueError("MCWF mean and standard-deviation histories must share shape.")
    same_seed_max_abs_diff = float(np.max(np.abs(first_trajectories - second_trajectories)))
    finite = bool(
        np.all(np.isfinite(first_trajectories))
        and np.all(np.isfinite(second_trajectories))
        and np.all(np.isfinite(first_mean))
        and np.all(np.isfinite(first_std))
    )
    total_jumps_object = first["total_jumps"]
    if isinstance(total_jumps_object, bool) or not isinstance(total_jumps_object, int):
        raise ValueError("MCWF total_jumps must be an integer.")
    total_jumps = total_jumps_object
    passed = (
        finite
        and same_seed_max_abs_diff <= case.reproducibility_tolerance
        and first_trajectories.shape[0] == case.n_trajectories
        and first_trajectories.shape[1] == first_mean.shape[0]
        and total_jumps >= 0
    )
    return MCWFReproducibilityCertificate(
        n_trajectories=case.n_trajectories,
        seed=case.seed,
        trajectory_shape=(int(first_trajectories.shape[0]), int(first_trajectories.shape[1])),
        same_seed_max_abs_diff=same_seed_max_abs_diff,
        final_mean_order_parameter=float(first_mean[-1]),
        final_std_order_parameter=float(first_std[-1]),
        total_jumps=total_jumps,
        finite=finite,
        passed=passed,
    )


def _finite_difference_gradient(
    evaluator: Literal["lindblad_density", "mcwf_ensemble"],
    case: BoundedOpenSystemObjectiveCase,
    params: NDArray[np.float64],
) -> tuple[NDArray[np.float64], int]:
    checked = _parameter_vector(params)
    gradient = np.zeros_like(checked)
    for index in range(checked.size):
        shift = np.zeros_like(checked)
        shift[index] = case.finite_difference_step
        if evaluator == "lindblad_density":
            plus, _, _ = evaluate_lindblad_objective(case, checked + shift)
            minus, _, _ = evaluate_lindblad_objective(case, checked - shift)
        else:
            plus, _, _ = evaluate_mcwf_objective(case, checked + shift)
            minus, _, _ = evaluate_mcwf_objective(case, checked - shift)
        gradient[index] = (plus - minus) / (2.0 * case.finite_difference_step)
    return gradient, 1 + 2 * checked.size


def _record_for_backend(
    case: BoundedOpenSystemObjectiveCase,
    backend: OpenSystemBackend,
) -> OpenSystemObjectiveRecord:
    if backend == "lindblad_density":
        value, final_r, invariant = evaluate_lindblad_objective(case, case.initial_params)
        gradient, evaluations = _finite_difference_gradient(backend, case, case.initial_params)
        return OpenSystemObjectiveRecord(
            case_id=case.case_id,
            backend=backend,
            params=(float(case.initial_params[0]), float(case.initial_params[1])),
            value=value,
            gradient=(float(gradient[0]), float(gradient[1])),
            evaluations=evaluations,
            final_order_parameter=final_r,
            invariant_certificate=invariant,
            reproducibility_certificate=None,
        )
    value, final_r, reproducibility = evaluate_mcwf_objective(case, case.initial_params)
    gradient, evaluations = _finite_difference_gradient(backend, case, case.initial_params)
    return OpenSystemObjectiveRecord(
        case_id=case.case_id,
        backend=backend,
        params=(float(case.initial_params[0]), float(case.initial_params[1])),
        value=value,
        gradient=(float(gradient[0]), float(gradient[1])),
        evaluations=evaluations,
        final_order_parameter=final_r,
        invariant_certificate=None,
        reproducibility_certificate=reproducibility,
    )


def open_system_objective_boundary_rows() -> tuple[OpenSystemObjectiveBoundaryRow, ...]:
    """Return approximation and promotion boundary rows for BL-16."""
    return (
        OpenSystemObjectiveBoundaryRow(
            case_id="adjoint_lindblad_gradient_boundary",
            backend="lindblad_adjoint",
            status="hard_gap",
            failure_class="unsupported_adjoint_lindblad_gradient",
            setup_instructions=(
                "Only central finite differences over bounded scalar scales are executed. "
                "Continuous adjoint Lindblad sensitivities require a separate validated "
                "solver and invariant-preserving gradient checks."
            ),
        ),
        OpenSystemObjectiveBoundaryRow(
            case_id="hardware_open_system_gradient_boundary",
            backend="qpu_open_system_gradient",
            status="hard_gap",
            failure_class="no_live_provider_attestation",
            setup_instructions=(
                "No hardware-submitted open-system gradient or provider attestation is "
                "included. Provider execution remains behind the live-ticket gate."
            ),
        ),
    )


def run_open_system_objective_suite(
    cases: Sequence[BoundedOpenSystemObjectiveCase] | None = None,
    *,
    backends: Iterable[OpenSystemBackend] = ("lindblad_density", "mcwf_ensemble"),
    include_boundary_rows: bool = True,
) -> OpenSystemObjectiveSuiteResult:
    """Run bounded Lindblad and MCWF objective evidence rows."""
    selected_cases = tuple(cases) if cases is not None else default_open_system_objective_cases()
    if not selected_cases:
        raise ValueError("at least one open-system objective case is required.")
    backend_names = tuple(dict.fromkeys(backends))
    if not backend_names:
        raise ValueError("at least one backend is required.")
    unknown_backends = sorted(set(backend_names) - {"lindblad_density", "mcwf_ensemble"})
    if unknown_backends:
        raise ValueError(f"unknown open-system objective backend: {unknown_backends[0]}")

    records = tuple(
        _record_for_backend(case, backend) for case in selected_cases for backend in backend_names
    )
    boundaries = open_system_objective_boundary_rows() if include_boundary_rows else ()
    return OpenSystemObjectiveSuiteResult(
        cases=selected_cases,
        records=records,
        boundary_rows=boundaries,
    )


__all__ = [
    "OPEN_SYSTEM_OBJECTIVE_CLAIM_BOUNDARY",
    "OPEN_SYSTEM_OBJECTIVE_EVIDENCE_CLASS",
    "BoundedOpenSystemObjectiveCase",
    "DensityMatrixInvariantCertificate",
    "MCWFReproducibilityCertificate",
    "OpenSystemObjectiveBoundaryRow",
    "OpenSystemObjectiveRecord",
    "OpenSystemObjectiveSuiteResult",
    "certify_density_matrix_invariants",
    "certify_mcwf_reproducibility",
    "default_open_system_objective_cases",
    "evaluate_lindblad_objective",
    "evaluate_mcwf_objective",
    "open_system_objective_boundary_rows",
    "run_open_system_objective_suite",
]
