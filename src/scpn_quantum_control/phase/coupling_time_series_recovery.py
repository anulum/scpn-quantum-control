# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Coupling Time-Series Recovery
"""Recover bounded Kuramoto/XY coupling matrices from synthetic time series."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final, Literal, TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .coupling_learning import coupling_matrix_from_edge_vector

FloatArray: TypeAlias = NDArray[np.float64]
Edge = tuple[int, int]
CouplingRecoveryFamily = Literal["kuramoto_phase", "xy_pair_energy"]
BoundaryStatus = Literal["hard_gap"]

COUPLING_RECOVERY_EVIDENCE_CLASS: Final[str] = "functional_non_isolated"
"""Evidence class for local synthetic recovery runs."""

COUPLING_RECOVERY_CLAIM_BOUNDARY: Final[str] = (
    "bounded synthetic Kuramoto phase and XY pair-energy time-series recovery "
    "with known ground truth; not hardware Hamiltonian learning, provider "
    "execution, isolated timing, or arbitrary partial-observation inference"
)
"""Claim boundary attached to BL-17 coupling-recovery records."""


def _as_finite_scalar(name: str, value: object) -> float:
    raw = np.asarray(value)
    if raw.shape != () or raw.dtype.kind in {"b", "c", "O", "S", "U"}:
        raise ValueError(f"{name} must be a finite real scalar")
    scalar = float(raw.item())
    if not np.isfinite(scalar):
        raise ValueError(f"{name} must be a finite real scalar")
    return scalar


def _as_finite_vector(name: str, values: ArrayLike) -> FloatArray:
    vector = np.asarray(values, dtype=float)
    if vector.ndim != 1 or vector.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain only finite values")
    return vector.astype(np.float64, copy=True)


def _as_coupling_matrix(name: str, values: ArrayLike) -> FloatArray:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"{name} must be a square matrix")
    if matrix.shape[0] < 2:
        raise ValueError(f"{name} must contain at least two nodes")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must contain only finite values")
    if not np.allclose(matrix, matrix.T, atol=1e-12, rtol=1e-12):
        raise ValueError(f"{name} must be symmetric")
    if not np.allclose(np.diag(matrix), 0.0, atol=1e-12, rtol=0.0):
        raise ValueError(f"{name} diagonal must be zero")
    return matrix.astype(np.float64, copy=True)


def _normalise_edges(edges: Sequence[Sequence[int]] | None, n_nodes: int) -> tuple[Edge, ...]:
    if n_nodes < 2:
        raise ValueError("n_nodes must be at least two")
    if edges is None:
        return tuple((row, col) for row in range(n_nodes) for col in range(row + 1, n_nodes))
    normalised: list[Edge] = []
    seen: set[Edge] = set()
    for raw_edge in edges:
        if len(raw_edge) != 2:
            raise ValueError("each edge must contain exactly two node indices")
        left = int(raw_edge[0])
        right = int(raw_edge[1])
        if left == right:
            raise ValueError("coupling-recovery edges must not contain self edges")
        if left < 0 or right < 0 or left >= n_nodes or right >= n_nodes:
            raise ValueError("coupling-recovery edge index is out of bounds")
        edge = (left, right) if left < right else (right, left)
        if edge in seen:
            raise ValueError("coupling-recovery edges must be unique")
        seen.add(edge)
        normalised.append(edge)
    if not normalised:
        raise ValueError("edges must contain at least one trainable coupling")
    return tuple(normalised)


def _phase_delta(left: FloatArray, right: FloatArray) -> FloatArray:
    delta = np.angle(np.exp(1j * (right - left))).astype(np.float64, copy=False)
    return np.asarray(delta, dtype=np.float64)


def _kuramoto_derivative(
    theta: FloatArray, omega: FloatArray, couplings: FloatArray
) -> FloatArray:
    phase_delta = theta[None, :] - theta[:, None]
    derivative: FloatArray = omega + np.sum(couplings * np.sin(phase_delta), axis=1)
    return derivative


@dataclass(frozen=True)
class CouplingRecoveryCase:
    """Synthetic known-ground-truth recovery case.

    Parameters
    ----------
    case_id:
        Stable case identifier used in evidence artefacts.
    family:
        Recovery family: ``"kuramoto_phase"`` or ``"xy_pair_energy"``.
    true_couplings:
        Symmetric zero-diagonal ground-truth coupling matrix.
    omega:
        Natural frequencies for the phase trajectory generator.
    theta0:
        Initial phases for the phase trajectory generator.
    dt:
        Fixed time step for synthetic trajectory generation.
    n_steps:
        Number of integration steps.
    noise_std:
        Additive Gaussian observation-noise standard deviation.
    missing_fraction:
        Fraction of observations replaced by ``NaN``.
    seed:
        Seed used for the noise/missing mask.
    tolerance:
        Maximum allowed absolute coupling error for the case.
    """

    case_id: str
    family: CouplingRecoveryFamily
    true_couplings: FloatArray
    omega: FloatArray
    theta0: FloatArray
    dt: float
    n_steps: int
    noise_std: float
    missing_fraction: float
    seed: int
    tolerance: float

    def __post_init__(self) -> None:
        if not self.case_id:
            raise ValueError("case_id must be non-empty")
        if self.family not in ("kuramoto_phase", "xy_pair_energy"):
            raise ValueError("family must be kuramoto_phase or xy_pair_energy")
        true_couplings = _as_coupling_matrix("true_couplings", self.true_couplings)
        omega = _as_finite_vector("omega", self.omega)
        theta0 = _as_finite_vector("theta0", self.theta0)
        if omega.shape != theta0.shape or true_couplings.shape != (omega.size, omega.size):
            raise ValueError("true_couplings, omega, and theta0 dimensions must agree")
        dt = _as_finite_scalar("dt", self.dt)
        noise_std = _as_finite_scalar("noise_std", self.noise_std)
        missing_fraction = _as_finite_scalar("missing_fraction", self.missing_fraction)
        tolerance = _as_finite_scalar("tolerance", self.tolerance)
        if dt <= 0.0:
            raise ValueError("dt must be finite and positive")
        if not isinstance(self.n_steps, int) or self.n_steps < 2:
            raise ValueError("n_steps must be an integer >= 2")
        if noise_std < 0.0:
            raise ValueError("noise_std must be non-negative")
        if not 0.0 <= missing_fraction < 1.0:
            raise ValueError("missing_fraction must be in [0, 1)")
        if not isinstance(self.seed, int):
            raise ValueError("seed must be an integer")
        if tolerance <= 0.0:
            raise ValueError("tolerance must be finite and positive")
        object.__setattr__(self, "true_couplings", true_couplings)
        object.__setattr__(self, "omega", omega)
        object.__setattr__(self, "theta0", theta0)
        object.__setattr__(self, "dt", dt)
        object.__setattr__(self, "noise_std", noise_std)
        object.__setattr__(self, "missing_fraction", missing_fraction)
        object.__setattr__(self, "tolerance", tolerance)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready case description."""
        return {
            "case_id": self.case_id,
            "family": self.family,
            "true_couplings": self.true_couplings.tolist(),
            "omega": self.omega.tolist(),
            "theta0": self.theta0.tolist(),
            "dt": self.dt,
            "n_steps": self.n_steps,
            "noise_std": self.noise_std,
            "missing_fraction": self.missing_fraction,
            "seed": self.seed,
            "tolerance": self.tolerance,
        }


@dataclass(frozen=True)
class CouplingRecoveryRecord:
    """Known-ground-truth coupling recovery certificate."""

    case_id: str
    family: CouplingRecoveryFamily
    learned_couplings: FloatArray
    true_couplings: FloatArray
    abs_error: FloatArray
    max_abs_error: float
    rmse: float
    valid_fraction: float
    design_rank: int
    condition_number: float
    noise_std: float
    missing_fraction: float
    tolerance: float
    passed: bool
    claim_boundary: str = COUPLING_RECOVERY_CLAIM_BOUNDARY

    def __post_init__(self) -> None:
        if not self.case_id:
            raise ValueError("case_id must be non-empty")
        if self.family not in ("kuramoto_phase", "xy_pair_energy"):
            raise ValueError("family must be kuramoto_phase or xy_pair_energy")
        learned = _as_coupling_matrix("learned_couplings", self.learned_couplings)
        truth = _as_coupling_matrix("true_couplings", self.true_couplings)
        abs_error = np.asarray(self.abs_error, dtype=float)
        if learned.shape != truth.shape or abs_error.shape != truth.shape:
            raise ValueError("learned, truth, and error matrices must share one shape")
        if not np.all(np.isfinite(abs_error)) or np.any(abs_error < 0.0):
            raise ValueError("abs_error must contain finite non-negative values")
        max_abs_error = _as_finite_scalar("max_abs_error", self.max_abs_error)
        rmse = _as_finite_scalar("rmse", self.rmse)
        valid_fraction = _as_finite_scalar("valid_fraction", self.valid_fraction)
        condition_number = _as_finite_scalar("condition_number", self.condition_number)
        noise_std = _as_finite_scalar("noise_std", self.noise_std)
        missing_fraction = _as_finite_scalar("missing_fraction", self.missing_fraction)
        tolerance = _as_finite_scalar("tolerance", self.tolerance)
        if max_abs_error < 0.0 or rmse < 0.0:
            raise ValueError("error metrics must be non-negative")
        if not 0.0 <= valid_fraction <= 1.0:
            raise ValueError("valid_fraction must be in [0, 1]")
        if self.design_rank < 1:
            raise ValueError("design_rank must be positive")
        if condition_number < 1.0:
            raise ValueError("condition_number must be >= 1")
        if noise_std < 0.0 or not 0.0 <= missing_fraction < 1.0:
            raise ValueError("noise and missing metadata are invalid")
        if tolerance <= 0.0:
            raise ValueError("tolerance must be finite and positive")
        if not isinstance(self.passed, bool):
            raise ValueError("passed must be a boolean")
        if not self.claim_boundary:
            raise ValueError("claim_boundary must be non-empty")
        object.__setattr__(self, "learned_couplings", learned)
        object.__setattr__(self, "true_couplings", truth)
        object.__setattr__(self, "abs_error", abs_error.astype(np.float64, copy=True))
        object.__setattr__(self, "max_abs_error", max_abs_error)
        object.__setattr__(self, "rmse", rmse)
        object.__setattr__(self, "valid_fraction", valid_fraction)
        object.__setattr__(self, "condition_number", condition_number)
        object.__setattr__(self, "noise_std", noise_std)
        object.__setattr__(self, "missing_fraction", missing_fraction)
        object.__setattr__(self, "tolerance", tolerance)

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready recovery evidence."""
        return {
            "case_id": self.case_id,
            "family": self.family,
            "learned_couplings": self.learned_couplings.tolist(),
            "true_couplings": self.true_couplings.tolist(),
            "abs_error": self.abs_error.tolist(),
            "max_abs_error": self.max_abs_error,
            "rmse": self.rmse,
            "valid_fraction": self.valid_fraction,
            "design_rank": self.design_rank,
            "condition_number": self.condition_number,
            "noise_std": self.noise_std,
            "missing_fraction": self.missing_fraction,
            "tolerance": self.tolerance,
            "passed": self.passed,
            "claim_boundary": self.claim_boundary,
        }


@dataclass(frozen=True)
class CouplingRecoveryBoundaryRow:
    """Fail-closed boundary for non-covered coupling-recovery routes."""

    boundary_id: str
    status: BoundaryStatus
    reason: str
    claim_boundary: str = COUPLING_RECOVERY_CLAIM_BOUNDARY

    def __post_init__(self) -> None:
        if not self.boundary_id:
            raise ValueError("boundary_id must be non-empty")
        if self.status != "hard_gap":
            raise ValueError("status must be hard_gap")
        if not self.reason:
            raise ValueError("reason must be non-empty")
        if not self.claim_boundary:
            raise ValueError("claim_boundary must be non-empty")

    def to_dict(self) -> dict[str, str]:
        """Return a JSON-ready boundary row."""
        return {
            "boundary_id": self.boundary_id,
            "status": self.status,
            "reason": self.reason,
            "claim_boundary": self.claim_boundary,
        }


@dataclass(frozen=True)
class CouplingRecoverySuiteResult:
    """Suite result for BL-17 coupling recovery evidence."""

    records: tuple[CouplingRecoveryRecord, ...]
    boundary_rows: tuple[CouplingRecoveryBoundaryRow, ...]
    evidence_class: str = COUPLING_RECOVERY_EVIDENCE_CLASS
    claim_boundary: str = COUPLING_RECOVERY_CLAIM_BOUNDARY

    def __post_init__(self) -> None:
        if not self.records:
            raise ValueError("records must be non-empty")
        if not self.boundary_rows:
            raise ValueError("boundary_rows must be non-empty")
        if not self.evidence_class:
            raise ValueError("evidence_class must be non-empty")
        if not self.claim_boundary:
            raise ValueError("claim_boundary must be non-empty")

    @property
    def passed(self) -> bool:
        """Return whether every recovery case satisfied its tolerance."""
        return all(record.passed for record in self.records)

    def records_for_family(
        self, family: CouplingRecoveryFamily
    ) -> tuple[CouplingRecoveryRecord, ...]:
        """Return recovery records for one recovery family."""
        return tuple(record for record in self.records if record.family == family)

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready suite evidence."""
        return {
            "evidence_class": self.evidence_class,
            "claim_boundary": self.claim_boundary,
            "passed": self.passed,
            "records": [record.to_dict() for record in self.records],
            "boundary_rows": [row.to_dict() for row in self.boundary_rows],
        }


def simulate_kuramoto_phase_time_series(
    couplings: ArrayLike,
    omega: ArrayLike,
    theta0: ArrayLike,
    *,
    dt: float,
    n_steps: int,
) -> FloatArray:
    """Generate a fixed-step RK4 Kuramoto phase trajectory."""
    matrix = _as_coupling_matrix("couplings", couplings)
    omega_arr = _as_finite_vector("omega", omega)
    theta = _as_finite_vector("theta0", theta0)
    if matrix.shape != (omega_arr.size, omega_arr.size) or theta.shape != omega_arr.shape:
        raise ValueError("couplings, omega, and theta0 dimensions must agree")
    dt_value = _as_finite_scalar("dt", dt)
    if dt_value <= 0.0:
        raise ValueError("dt must be finite and positive")
    if not isinstance(n_steps, int) or n_steps < 2:
        raise ValueError("n_steps must be an integer >= 2")

    trajectory = np.empty((n_steps + 1, theta.size), dtype=np.float64)
    trajectory[0] = theta
    current = theta.copy()
    for index in range(n_steps):
        k1 = _kuramoto_derivative(current, omega_arr, matrix)
        k2 = _kuramoto_derivative(current + 0.5 * dt_value * k1, omega_arr, matrix)
        k3 = _kuramoto_derivative(current + 0.5 * dt_value * k2, omega_arr, matrix)
        k4 = _kuramoto_derivative(current + dt_value * k3, omega_arr, matrix)
        current = current + (dt_value / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        trajectory[index + 1] = current
    return trajectory


def simulate_xy_pair_energy_time_series(couplings: ArrayLike, phases: ArrayLike) -> FloatArray:
    """Generate synthetic edge-resolved XY pair-energy observations."""
    matrix = _as_coupling_matrix("couplings", couplings)
    phase_series = np.asarray(phases, dtype=float)
    if phase_series.ndim != 2 or phase_series.shape[1] != matrix.shape[0]:
        raise ValueError("phases must have shape (n_times, n_nodes)")
    if not np.all(np.isfinite(phase_series)):
        raise ValueError("phases must contain only finite values")
    delta = phase_series[:, :, None] - phase_series[:, None, :]
    pair_energy: FloatArray = matrix[None, :, :] * np.cos(delta)
    return pair_energy.astype(np.float64, copy=True)


def inject_time_series_noise_and_missing(
    values: ArrayLike,
    *,
    noise_std: float,
    missing_fraction: float,
    seed: int,
) -> FloatArray:
    """Add deterministic Gaussian noise and ``NaN`` missing observations."""
    noise_value = _as_finite_scalar("noise_std", noise_std)
    missing_value = _as_finite_scalar("missing_fraction", missing_fraction)
    if noise_value < 0.0:
        raise ValueError("noise_std must be non-negative")
    if not 0.0 <= missing_value < 1.0:
        raise ValueError("missing_fraction must be in [0, 1)")
    if not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    observed = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(observed)):
        raise ValueError("values must be finite before noise/missing injection")
    rng = np.random.default_rng(seed)
    noisy = observed.astype(np.float64, copy=True)
    if noise_value > 0.0:
        noisy = noisy + rng.normal(0.0, noise_value, size=noisy.shape)
    if missing_value > 0.0:
        mask = rng.random(noisy.shape) < missing_value
        noisy[mask] = np.nan
    return noisy


def _solve_ridge(
    design: FloatArray, target: FloatArray, ridge: float
) -> tuple[FloatArray, int, float]:
    if design.ndim != 2 or target.ndim != 1 or design.shape[0] != target.size:
        raise ValueError("design and target dimensions are inconsistent")
    if design.shape[0] == 0 or design.shape[1] == 0:
        raise ValueError("design matrix must be non-empty")
    ridge_value = _as_finite_scalar("ridge", ridge)
    if ridge_value < 0.0:
        raise ValueError("ridge must be non-negative")
    gram = design.T @ design
    if ridge_value > 0.0:
        gram = gram + ridge_value * np.eye(design.shape[1], dtype=np.float64)
    rhs = design.T @ target
    try:
        solution = np.linalg.solve(gram, rhs)
    except np.linalg.LinAlgError:
        solution = np.linalg.lstsq(design, target, rcond=None)[0]
    singular_values = np.linalg.svd(design, compute_uv=False)
    max_singular = float(max([float(value) for value in singular_values], default=0.0))
    rank_tolerance = float(max(design.shape) * np.finfo(np.float64).eps * max_singular)
    rank = sum(float(value) > rank_tolerance for value in singular_values)
    gram_singular_values = np.linalg.svd(gram, compute_uv=False)
    positive_gram_values = [float(value) for value in gram_singular_values if float(value) > 0.0]
    if len(positive_gram_values) != gram_singular_values.size:
        condition_number = float(np.finfo(np.float64).max)
    else:
        condition_number = max(positive_gram_values) / min(positive_gram_values)
    return solution.astype(np.float64, copy=True), rank, condition_number


def _record_from_solution(
    *,
    case_id: str,
    family: CouplingRecoveryFamily,
    solution: FloatArray,
    true_couplings: FloatArray,
    edges: tuple[Edge, ...],
    valid_rows: int,
    possible_rows: int,
    design_rank: int,
    condition_number: float,
    noise_std: float,
    missing_fraction: float,
    tolerance: float,
) -> CouplingRecoveryRecord:
    learned = coupling_matrix_from_edge_vector(
        solution, n_nodes=true_couplings.shape[0], edges=edges
    )
    abs_error = np.abs(learned - true_couplings)
    upper_errors = np.array([abs_error[row, col] for row, col in edges], dtype=np.float64)
    max_abs_error = float(np.max(upper_errors)) if upper_errors.size else 0.0
    rmse = float(np.sqrt(np.mean(upper_errors * upper_errors))) if upper_errors.size else 0.0
    valid_fraction = float(valid_rows / possible_rows) if possible_rows else 0.0
    return CouplingRecoveryRecord(
        case_id=case_id,
        family=family,
        learned_couplings=learned,
        true_couplings=true_couplings,
        abs_error=abs_error,
        max_abs_error=max_abs_error,
        rmse=rmse,
        valid_fraction=valid_fraction,
        design_rank=design_rank,
        condition_number=condition_number,
        noise_std=noise_std,
        missing_fraction=missing_fraction,
        tolerance=tolerance,
        passed=max_abs_error <= tolerance,
    )


def recover_kuramoto_couplings_from_time_series(
    phases: ArrayLike,
    omega: ArrayLike,
    true_couplings: ArrayLike,
    *,
    dt: float,
    case_id: str = "kuramoto_time_series",
    edges: Sequence[Sequence[int]] | None = None,
    ridge: float = 1.0e-9,
    noise_std: float = 0.0,
    missing_fraction: float = 0.0,
    tolerance: float = 1.0e-2,
) -> CouplingRecoveryRecord:
    """Recover Kuramoto couplings from phase time series with known truth."""
    phase_series = np.asarray(phases, dtype=float)
    truth = _as_coupling_matrix("true_couplings", true_couplings)
    omega_arr = _as_finite_vector("omega", omega)
    if phase_series.ndim != 2 or phase_series.shape[1] != truth.shape[0]:
        raise ValueError("phases must have shape (n_times, n_nodes)")
    if omega_arr.shape != (truth.shape[0],):
        raise ValueError("omega dimension must match true_couplings")
    dt_value = _as_finite_scalar("dt", dt)
    if dt_value <= 0.0:
        raise ValueError("dt must be finite and positive")
    tolerance_value = _as_finite_scalar("tolerance", tolerance)
    if tolerance_value <= 0.0:
        raise ValueError("tolerance must be finite and positive")
    edge_tuple = _normalise_edges(edges, truth.shape[0])
    design_rows: list[list[float]] = []
    targets: list[float] = []
    possible_rows = max(int(phase_series.shape[0] - 1), 0) * truth.shape[0]
    for time_index in range(phase_series.shape[0] - 1):
        theta = phase_series[time_index]
        theta_next = phase_series[time_index + 1]
        if not np.all(np.isfinite(theta)):
            continue
        for node in range(truth.shape[0]):
            if not np.isfinite(theta_next[node]):
                continue
            row: list[float] = []
            for left, right in edge_tuple:
                if node == left:
                    row.append(float(np.sin(theta[right] - theta[left])))
                elif node == right:
                    row.append(float(np.sin(theta[left] - theta[right])))
                else:
                    row.append(0.0)
            derivative = float(
                _phase_delta(theta[node : node + 1], theta_next[node : node + 1])[0] / dt_value
            )
            design_rows.append(row)
            targets.append(derivative - float(omega_arr[node]))
    if not design_rows:
        raise ValueError("no valid Kuramoto time-series rows remain after missing-data filtering")
    design = np.asarray(design_rows, dtype=np.float64)
    target = np.asarray(targets, dtype=np.float64)
    solution, rank, condition_number = _solve_ridge(design, target, ridge)
    return _record_from_solution(
        case_id=case_id,
        family="kuramoto_phase",
        solution=solution,
        true_couplings=truth,
        edges=edge_tuple,
        valid_rows=design.shape[0],
        possible_rows=possible_rows,
        design_rank=rank,
        condition_number=condition_number,
        noise_std=noise_std,
        missing_fraction=missing_fraction,
        tolerance=tolerance_value,
    )


def recover_xy_couplings_from_pair_energy_series(
    pair_energy: ArrayLike,
    phases: ArrayLike,
    true_couplings: ArrayLike,
    *,
    case_id: str = "xy_pair_energy",
    edges: Sequence[Sequence[int]] | None = None,
    ridge: float = 1.0e-9,
    noise_std: float = 0.0,
    missing_fraction: float = 0.0,
    tolerance: float = 1.0e-2,
) -> CouplingRecoveryRecord:
    """Recover XY couplings from edge-resolved pair-energy observations."""
    energy = np.asarray(pair_energy, dtype=float)
    phase_series = np.asarray(phases, dtype=float)
    truth = _as_coupling_matrix("true_couplings", true_couplings)
    if phase_series.ndim != 2 or phase_series.shape[1] != truth.shape[0]:
        raise ValueError("phases must have shape (n_times, n_nodes)")
    if energy.shape != (phase_series.shape[0], truth.shape[0], truth.shape[0]):
        raise ValueError("pair_energy must have shape (n_times, n_nodes, n_nodes)")
    tolerance_value = _as_finite_scalar("tolerance", tolerance)
    if tolerance_value <= 0.0:
        raise ValueError("tolerance must be finite and positive")
    edge_tuple = _normalise_edges(edges, truth.shape[0])
    design_rows: list[list[float]] = []
    targets: list[float] = []
    possible_rows = phase_series.shape[0] * len(edge_tuple)
    for time_index, theta in enumerate(phase_series):
        if not np.all(np.isfinite(theta)):
            continue
        for edge_index, (left, right) in enumerate(edge_tuple):
            observed = energy[time_index, left, right]
            if not np.isfinite(observed):
                continue
            row = [0.0] * len(edge_tuple)
            row[edge_index] = float(np.cos(theta[left] - theta[right]))
            design_rows.append(row)
            targets.append(float(observed))
    if not design_rows:
        raise ValueError("no valid XY pair-energy rows remain after missing-data filtering")
    design = np.asarray(design_rows, dtype=np.float64)
    target = np.asarray(targets, dtype=np.float64)
    solution, rank, condition_number = _solve_ridge(design, target, ridge)
    return _record_from_solution(
        case_id=case_id,
        family="xy_pair_energy",
        solution=solution,
        true_couplings=truth,
        edges=edge_tuple,
        valid_rows=design.shape[0],
        possible_rows=possible_rows,
        design_rank=rank,
        condition_number=condition_number,
        noise_std=noise_std,
        missing_fraction=missing_fraction,
        tolerance=tolerance_value,
    )


def coupling_recovery_boundary_rows() -> tuple[CouplingRecoveryBoundaryRow, ...]:
    """Return fail-closed BL-17 boundary rows."""
    return (
        CouplingRecoveryBoundaryRow(
            boundary_id="partial_observation_inference_boundary",
            status="hard_gap",
            reason=(
                "arbitrary partial-observation oscillator inference requires "
                "identifiability analysis beyond this bounded synthetic suite"
            ),
        ),
        CouplingRecoveryBoundaryRow(
            boundary_id="hardware_hamiltonian_learning_boundary",
            status="hard_gap",
            reason=(
                "provider-backed XY Hamiltonian learning requires raw counts, "
                "calibration, and owner-approved hardware tickets"
            ),
        ),
    )


def default_coupling_recovery_cases() -> tuple[CouplingRecoveryCase, ...]:
    """Return deterministic BL-17 recovery cases."""
    true = np.array(
        [
            [0.0, 0.42, 0.18],
            [0.42, 0.0, 0.31],
            [0.18, 0.31, 0.0],
        ],
        dtype=np.float64,
    )
    omega = np.array([-0.18, 0.04, 0.21], dtype=np.float64)
    theta0 = np.array([0.2, 1.1, 2.4], dtype=np.float64)
    return (
        CouplingRecoveryCase(
            case_id="kuramoto_clean_three_node",
            family="kuramoto_phase",
            true_couplings=true,
            omega=omega,
            theta0=theta0,
            dt=0.015,
            n_steps=180,
            noise_std=0.0,
            missing_fraction=0.0,
            seed=2701,
            tolerance=0.025,
        ),
        CouplingRecoveryCase(
            case_id="kuramoto_noisy_missing_three_node",
            family="kuramoto_phase",
            true_couplings=true,
            omega=omega,
            theta0=theta0,
            dt=0.015,
            n_steps=220,
            noise_std=0.0005,
            missing_fraction=0.03,
            seed=2702,
            tolerance=0.085,
        ),
        CouplingRecoveryCase(
            case_id="xy_pair_energy_noisy_missing_three_node",
            family="xy_pair_energy",
            true_couplings=true,
            omega=omega,
            theta0=theta0,
            dt=0.015,
            n_steps=160,
            noise_std=0.002,
            missing_fraction=0.05,
            seed=2703,
            tolerance=0.03,
        ),
    )


def _run_case(case: CouplingRecoveryCase) -> CouplingRecoveryRecord:
    phases = simulate_kuramoto_phase_time_series(
        case.true_couplings,
        case.omega,
        case.theta0,
        dt=case.dt,
        n_steps=case.n_steps,
    )
    if case.family == "kuramoto_phase":
        observed_phases = inject_time_series_noise_and_missing(
            phases,
            noise_std=case.noise_std,
            missing_fraction=case.missing_fraction,
            seed=case.seed,
        )
        return recover_kuramoto_couplings_from_time_series(
            observed_phases,
            case.omega,
            case.true_couplings,
            dt=case.dt,
            case_id=case.case_id,
            noise_std=case.noise_std,
            missing_fraction=case.missing_fraction,
            tolerance=case.tolerance,
        )
    pair_energy = simulate_xy_pair_energy_time_series(case.true_couplings, phases)
    observed_pair_energy = inject_time_series_noise_and_missing(
        pair_energy,
        noise_std=case.noise_std,
        missing_fraction=case.missing_fraction,
        seed=case.seed,
    )
    return recover_xy_couplings_from_pair_energy_series(
        observed_pair_energy,
        phases,
        case.true_couplings,
        case_id=case.case_id,
        noise_std=case.noise_std,
        missing_fraction=case.missing_fraction,
        tolerance=case.tolerance,
    )


def run_coupling_recovery_suite(
    cases: Sequence[CouplingRecoveryCase] | None = None,
) -> CouplingRecoverySuiteResult:
    """Run the deterministic BL-17 coupling-recovery evidence suite."""
    selected_cases = tuple(default_coupling_recovery_cases() if cases is None else cases)
    if not selected_cases:
        raise ValueError("cases must be non-empty")
    records = tuple(_run_case(case) for case in selected_cases)
    return CouplingRecoverySuiteResult(
        records=records,
        boundary_rows=coupling_recovery_boundary_rows(),
    )


__all__ = [
    "BoundaryStatus",
    "COUPLING_RECOVERY_CLAIM_BOUNDARY",
    "COUPLING_RECOVERY_EVIDENCE_CLASS",
    "CouplingRecoveryBoundaryRow",
    "CouplingRecoveryCase",
    "CouplingRecoveryFamily",
    "CouplingRecoveryRecord",
    "CouplingRecoverySuiteResult",
    "coupling_recovery_boundary_rows",
    "default_coupling_recovery_cases",
    "inject_time_series_noise_and_missing",
    "recover_kuramoto_couplings_from_time_series",
    "recover_xy_couplings_from_pair_energy_series",
    "run_coupling_recovery_suite",
    "simulate_kuramoto_phase_time_series",
    "simulate_xy_pair_energy_time_series",
]
