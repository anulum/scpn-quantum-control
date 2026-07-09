# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Synchronisation Witness Suite
"""Order-parameter and persistent-homology synchronisation witnesses.

Provides bounded synchronisation witnesses over synthetic phase clouds: harmonic
Kuramoto order parameters, exact Vietoris--Rips persistent homology (Betti curves
and persistence diagrams in dimensions ``0`` and ``1``) over geodesic phase
distances, bootstrap uncertainty on the order parameter, and deterministic
synchronised, desynchronised, and clustered reference regimes. The persistence
computation is the standard reduction of the boundary matrix over ``GF(2)`` and
is exact for the small phase clouds used here; it is not an accelerated timing
kernel, hardware phase tomography, or high-dimensional manifold inference.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final, Literal, TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray: TypeAlias = NDArray[np.float64]
PhaseCloudRegime = Literal["synchronised", "desynchronised", "clustered"]
BoundaryStatus = Literal["hard_gap"]

SYNC_WITNESS_EVIDENCE_CLASS: Final[str] = "functional_non_isolated"
"""Evidence class for local synthetic synchronisation-witness runs."""

SYNC_WITNESS_CLAIM_BOUNDARY: Final[str] = (
    "bounded synthetic phase-cloud synchronisation witnesses (harmonic Kuramoto "
    "order parameters and exact Vietoris-Rips persistent homology in dimensions "
    "0 and 1 over geodesic phase distances) with known reference regimes; not "
    "hardware phase tomography, provider execution, isolated timing, or "
    "high-dimensional manifold inference"
)
"""Claim boundary attached to BL-18 synchronisation-witness records."""


def _as_finite_scalar(name: str, value: object) -> float:
    raw = np.asarray(value)
    if raw.shape != () or raw.dtype.kind in {"b", "c", "O", "S", "U"}:
        raise ValueError(f"{name} must be a finite real scalar")
    scalar = float(raw.item())
    if not np.isfinite(scalar):
        raise ValueError(f"{name} must be a finite real scalar")
    return scalar


def _as_phase_vector(name: str, values: ArrayLike) -> FloatArray:
    vector = np.asarray(values, dtype=float)
    if vector.ndim != 1 or vector.size < 2:
        raise ValueError(f"{name} must be a one-dimensional array with at least two phases")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain only finite values")
    return vector.astype(np.float64, copy=True)


def _as_threshold_vector(name: str, values: ArrayLike) -> FloatArray:
    vector = np.asarray(values, dtype=float)
    if vector.ndim != 1 or vector.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain only finite values")
    if np.any(vector < 0.0):
        raise ValueError(f"{name} must contain only non-negative thresholds")
    if np.any(np.diff(vector) <= 0.0):
        raise ValueError(f"{name} must be strictly increasing")
    return vector.astype(np.float64, copy=True)


def _as_distance_matrix(name: str, values: ArrayLike) -> FloatArray:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"{name} must be a square matrix")
    if matrix.shape[0] < 2:
        raise ValueError(f"{name} must contain at least two points")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must contain only finite values")
    if np.any(matrix < 0.0):
        raise ValueError(f"{name} must be non-negative")
    if not np.allclose(matrix, matrix.T, atol=1e-12, rtol=0.0):
        raise ValueError(f"{name} must be symmetric")
    if not np.allclose(np.diag(matrix), 0.0, atol=1e-12, rtol=0.0):
        raise ValueError(f"{name} diagonal must be zero")
    return matrix.astype(np.float64, copy=True)


def harmonic_order_parameter(phases: ArrayLike, *, harmonic: int = 1) -> float:
    """Return the magnitude of the ``harmonic``-th Kuramoto order parameter.

    Parameters
    ----------
    phases : array_like
        One-dimensional array of oscillator phases in radians.
    harmonic : int, optional
        Positive harmonic index ``m`` of the Daido order parameter
        ``|mean(exp(i * m * phases))|``. ``harmonic=1`` recovers the global
        Kuramoto order parameter; ``harmonic=2`` witnesses two-cluster
        (anti-phase) structure.

    Returns
    -------
    float
        The order-parameter magnitude in ``[0, 1]``.
    """
    vector = _as_phase_vector("phases", phases)
    if harmonic < 1:
        raise ValueError("harmonic must be a positive integer")
    complex_mean = np.mean(np.exp(1j * harmonic * vector))
    return float(np.abs(complex_mean))


def geodesic_phase_distance_matrix(phases: ArrayLike) -> FloatArray:
    """Return the pairwise geodesic (arc-length) phase-distance matrix.

    Parameters
    ----------
    phases : array_like
        One-dimensional array of oscillator phases in radians.

    Returns
    -------
    numpy.ndarray
        Symmetric zero-diagonal matrix of arc distances in ``[0, pi]`` between
        phases wrapped onto the unit circle.
    """
    vector = _as_phase_vector("phases", phases)
    delta = vector[:, None] - vector[None, :]
    wrapped = np.angle(np.exp(1j * delta))
    distance = np.abs(wrapped).astype(np.float64, copy=False)
    np.fill_diagonal(distance, 0.0)
    return np.asarray(distance, dtype=np.float64)


def _rips_simplices(
    matrix: FloatArray, *, max_dimension: int
) -> list[tuple[float, int, frozenset[int]]]:
    n = matrix.shape[0]
    simplices: list[tuple[float, int, frozenset[int]]] = [
        (0.0, 0, frozenset((vertex,))) for vertex in range(n)
    ]
    for left in range(n):
        for right in range(left + 1, n):
            simplices.append((float(matrix[left, right]), 1, frozenset((left, right))))
    if max_dimension >= 1:
        for left in range(n):
            for middle in range(left + 1, n):
                for right in range(middle + 1, n):
                    birth = float(
                        max(matrix[left, middle], matrix[left, right], matrix[middle, right])
                    )
                    simplices.append((birth, 2, frozenset((left, middle, right))))
    return simplices


def vietoris_rips_persistence(
    distance: ArrayLike, *, max_dimension: int = 1
) -> dict[int, FloatArray]:
    """Return exact Vietoris--Rips persistence pairs by homology dimension.

    The boundary matrix over the filtered Rips complex is reduced over ``GF(2)``
    with the standard lowest-one algorithm. Essential classes (never filled by a
    higher simplex) are reported with an infinite death. The result is exact for
    the small point clouds used by the synchronisation-witness suite.

    Parameters
    ----------
    distance : array_like
        Symmetric zero-diagonal non-negative pairwise distance matrix.
    max_dimension : int, optional
        Highest homology dimension to certify. ``0`` builds edges only (H0);
        ``1`` also builds triangles so that H1 loops can be filled.

    Returns
    -------
    dict of int to numpy.ndarray
        Mapping from homology dimension to an ``(k, 2)`` array of
        ``(birth, death)`` pairs. Deaths may be ``inf`` for essential classes.
    """
    matrix = _as_distance_matrix("distance", distance)
    if max_dimension not in (0, 1):
        raise ValueError("max_dimension must be 0 or 1")
    simplices = _rips_simplices(matrix, max_dimension=max_dimension)
    order = sorted(
        range(len(simplices)),
        key=lambda index: (simplices[index][0], simplices[index][1], sorted(simplices[index][2])),
    )
    ordered = [simplices[index] for index in order]
    index_of: dict[frozenset[int], int] = {
        simplex[2]: position for position, simplex in enumerate(ordered)
    }
    columns: list[set[int]] = []
    for _birth, dimension, vertices in ordered:
        if dimension == 0:
            columns.append(set())
            continue
        vertex_list = sorted(vertices)
        faces = {
            index_of[frozenset(vertex_list[:omit] + vertex_list[omit + 1 :])]
            for omit in range(len(vertex_list))
        }
        columns.append(faces)

    low_to_column: dict[int, int] = {}
    reduced_low: list[int | None] = [None] * len(columns)
    for column_index in range(len(columns)):
        column = columns[column_index]
        lowest = max(column) if column else None
        while lowest is not None and lowest in low_to_column:
            column ^= columns[low_to_column[lowest]]
            lowest = max(column) if column else None
        columns[column_index] = column
        reduced_low[column_index] = lowest
        if lowest is not None:
            low_to_column[lowest] = column_index

    pairs: dict[int, list[tuple[float, float]]] = {index: [] for index in range(max_dimension + 1)}
    killed: set[int] = set()
    for column_index, lowest in enumerate(reduced_low):
        if lowest is None:
            continue
        killed.add(lowest)
        birth_dimension = ordered[lowest][1]
        pairs[birth_dimension].append((ordered[lowest][0], ordered[column_index][0]))
    for column_index, lowest in enumerate(reduced_low):
        if lowest is not None or column_index in killed:
            continue
        creator_dimension = ordered[column_index][1]
        if creator_dimension in pairs:
            pairs[creator_dimension].append((ordered[column_index][0], float(np.inf)))

    return {
        dimension: (
            np.asarray(sorted(entries), dtype=np.float64).reshape(-1, 2)
            if entries
            else np.empty((0, 2), dtype=np.float64)
        )
        for dimension, entries in pairs.items()
    }


def betti_curve(persistence_pairs: ArrayLike, thresholds: ArrayLike) -> NDArray[np.int64]:
    """Return the Betti curve of one homology dimension over ``thresholds``.

    Parameters
    ----------
    persistence_pairs : array_like
        ``(k, 2)`` array of ``(birth, death)`` pairs for a single dimension.
        An empty array yields an all-zero curve.
    thresholds : array_like
        Strictly increasing non-negative filtration thresholds.

    Returns
    -------
    numpy.ndarray
        Integer Betti number alive at each threshold, ``birth <= t < death``.
    """
    grid = _as_threshold_vector("thresholds", thresholds)
    pairs = np.asarray(persistence_pairs, dtype=float)
    if pairs.size == 0:
        return np.zeros(grid.size, dtype=np.int64)
    if pairs.ndim != 2 or pairs.shape[1] != 2:
        raise ValueError("persistence_pairs must have shape (k, 2)")
    if np.any(pairs[:, 0] < 0.0) or np.any(np.isnan(pairs)):
        raise ValueError("persistence pairs must be non-negative and non-NaN")
    births = pairs[:, 0][:, None]
    deaths = pairs[:, 1][:, None]
    alive = (births <= grid[None, :]) & (grid[None, :] < deaths)
    counts: NDArray[np.int64] = np.sum(alive, axis=0).astype(np.int64)
    return counts


def _dominant_persistence(pairs: FloatArray) -> float:
    if pairs.size == 0:
        return 0.0
    finite = pairs[np.isfinite(pairs[:, 1])]
    if finite.size == 0:
        return 0.0
    return float(np.max(finite[:, 1] - finite[:, 0]))


@dataclass(frozen=True)
class SyncWitnessCase:
    """Deterministic synchronisation-witness reference case.

    Parameters
    ----------
    case_id : str
        Stable identifier used in evidence artefacts.
    regime : str
        Reference regime: ``"synchronised"``, ``"desynchronised"``, or
        ``"clustered"``.
    phases : numpy.ndarray
        Base phase cloud in radians.
    thresholds : numpy.ndarray
        Strictly increasing filtration thresholds for the Betti curves.
    reference_scale : float
        Filtration scale at which the persistent component count is read.
    noise_std : float
        Standard deviation of the bootstrap phase perturbation.
    n_bootstrap : int
        Number of bootstrap perturbations used for the order-parameter
        uncertainty. ``0`` disables the bootstrap.
    seed : int
        Seed for the deterministic bootstrap perturbation.
    min_order_parameter : float
        Lower bound the first-harmonic order parameter must satisfy.
    max_order_parameter : float
        Upper bound the first-harmonic order parameter must satisfy.
    expected_components : int
        Expected persistent component count at ``reference_scale``.
    min_dominant_h1 : float
        Lower bound on the dominant H1 persistence lifetime.
    max_dominant_h1 : float
        Upper bound on the dominant H1 persistence lifetime.
    """

    case_id: str
    regime: PhaseCloudRegime
    phases: FloatArray
    thresholds: FloatArray
    reference_scale: float
    noise_std: float
    n_bootstrap: int
    seed: int
    min_order_parameter: float
    max_order_parameter: float
    expected_components: int
    min_dominant_h1: float
    max_dominant_h1: float

    def __post_init__(self) -> None:
        if not self.case_id:
            raise ValueError("case_id must be non-empty")
        if self.regime not in ("synchronised", "desynchronised", "clustered"):
            raise ValueError("regime must be synchronised, desynchronised, or clustered")
        phases = _as_phase_vector("phases", self.phases)
        thresholds = _as_threshold_vector("thresholds", self.thresholds)
        reference_scale = _as_finite_scalar("reference_scale", self.reference_scale)
        noise_std = _as_finite_scalar("noise_std", self.noise_std)
        min_order = _as_finite_scalar("min_order_parameter", self.min_order_parameter)
        max_order = _as_finite_scalar("max_order_parameter", self.max_order_parameter)
        min_dominant = _as_finite_scalar("min_dominant_h1", self.min_dominant_h1)
        max_dominant = _as_finite_scalar("max_dominant_h1", self.max_dominant_h1)
        if reference_scale <= 0.0:
            raise ValueError("reference_scale must be finite and positive")
        if not thresholds[0] <= reference_scale <= thresholds[-1]:
            raise ValueError("reference_scale must lie within the threshold range")
        if noise_std < 0.0:
            raise ValueError("noise_std must be non-negative")
        if not isinstance(self.n_bootstrap, int) or self.n_bootstrap < 0:
            raise ValueError("n_bootstrap must be a non-negative integer")
        if not isinstance(self.seed, int):
            raise ValueError("seed must be an integer")
        if not 0.0 <= min_order <= max_order <= 1.0:
            raise ValueError("order-parameter bounds must satisfy 0 <= min <= max <= 1")
        if not isinstance(self.expected_components, int) or self.expected_components < 1:
            raise ValueError("expected_components must be a positive integer")
        if not 0.0 <= min_dominant <= max_dominant:
            raise ValueError("H1 persistence bounds must satisfy 0 <= min <= max")
        object.__setattr__(self, "phases", phases)
        object.__setattr__(self, "thresholds", thresholds)
        object.__setattr__(self, "reference_scale", reference_scale)
        object.__setattr__(self, "noise_std", noise_std)
        object.__setattr__(self, "min_order_parameter", min_order)
        object.__setattr__(self, "max_order_parameter", max_order)
        object.__setattr__(self, "min_dominant_h1", min_dominant)
        object.__setattr__(self, "max_dominant_h1", max_dominant)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready case description."""
        return {
            "case_id": self.case_id,
            "regime": self.regime,
            "phases": self.phases.tolist(),
            "thresholds": self.thresholds.tolist(),
            "reference_scale": self.reference_scale,
            "noise_std": self.noise_std,
            "n_bootstrap": self.n_bootstrap,
            "seed": self.seed,
            "min_order_parameter": self.min_order_parameter,
            "max_order_parameter": self.max_order_parameter,
            "expected_components": self.expected_components,
            "min_dominant_h1": self.min_dominant_h1,
            "max_dominant_h1": self.max_dominant_h1,
        }


@dataclass(frozen=True)
class SyncWitnessRecord:
    """Known-regime synchronisation-witness certificate."""

    case_id: str
    regime: PhaseCloudRegime
    n_nodes: int
    order_parameter: float
    order_parameter_harmonic2: float
    order_parameter_std: float
    thresholds: FloatArray
    betti0_curve: NDArray[np.int64]
    betti1_curve: NDArray[np.int64]
    h0_persistence: FloatArray
    h1_persistence: FloatArray
    persistent_component_count: int
    dominant_h1_persistence: float
    reference_scale: float
    n_bootstrap: int
    noise_std: float
    passed: bool
    claim_boundary: str = SYNC_WITNESS_CLAIM_BOUNDARY

    def __post_init__(self) -> None:
        if not self.case_id:
            raise ValueError("case_id must be non-empty")
        if self.regime not in ("synchronised", "desynchronised", "clustered"):
            raise ValueError("regime must be synchronised, desynchronised, or clustered")
        if self.n_nodes < 2:
            raise ValueError("n_nodes must be at least two")
        order_parameter = _as_finite_scalar("order_parameter", self.order_parameter)
        harmonic2 = _as_finite_scalar("order_parameter_harmonic2", self.order_parameter_harmonic2)
        order_std = _as_finite_scalar("order_parameter_std", self.order_parameter_std)
        thresholds = _as_threshold_vector("thresholds", self.thresholds)
        betti0 = np.asarray(self.betti0_curve, dtype=np.int64)
        betti1 = np.asarray(self.betti1_curve, dtype=np.int64)
        if betti0.shape != (thresholds.size,) or betti1.shape != (thresholds.size,):
            raise ValueError("Betti curves must match the threshold length")
        if np.any(betti0 < 0) or np.any(betti1 < 0):
            raise ValueError("Betti curves must be non-negative")
        h0 = _as_persistence_pairs("h0_persistence", self.h0_persistence)
        h1 = _as_persistence_pairs("h1_persistence", self.h1_persistence)
        if not 0.0 <= order_parameter <= 1.0 or not 0.0 <= harmonic2 <= 1.0:
            raise ValueError("order parameters must lie in [0, 1]")
        if order_std < 0.0:
            raise ValueError("order_parameter_std must be non-negative")
        reference_scale = _as_finite_scalar("reference_scale", self.reference_scale)
        dominant = _as_finite_scalar("dominant_h1_persistence", self.dominant_h1_persistence)
        noise_std = _as_finite_scalar("noise_std", self.noise_std)
        if reference_scale <= 0.0:
            raise ValueError("reference_scale must be finite and positive")
        if self.persistent_component_count < 1:
            raise ValueError("persistent_component_count must be positive")
        if dominant < 0.0:
            raise ValueError("dominant_h1_persistence must be non-negative")
        if not isinstance(self.n_bootstrap, int) or self.n_bootstrap < 0:
            raise ValueError("n_bootstrap must be a non-negative integer")
        if noise_std < 0.0:
            raise ValueError("noise_std must be non-negative")
        if not isinstance(self.passed, bool):
            raise ValueError("passed must be a boolean")
        if not self.claim_boundary:
            raise ValueError("claim_boundary must be non-empty")
        object.__setattr__(self, "order_parameter", order_parameter)
        object.__setattr__(self, "order_parameter_harmonic2", harmonic2)
        object.__setattr__(self, "order_parameter_std", order_std)
        object.__setattr__(self, "thresholds", thresholds)
        object.__setattr__(self, "betti0_curve", betti0)
        object.__setattr__(self, "betti1_curve", betti1)
        object.__setattr__(self, "h0_persistence", h0)
        object.__setattr__(self, "h1_persistence", h1)
        object.__setattr__(self, "reference_scale", reference_scale)
        object.__setattr__(self, "dominant_h1_persistence", dominant)
        object.__setattr__(self, "noise_std", noise_std)

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready synchronisation-witness evidence."""
        return {
            "case_id": self.case_id,
            "regime": self.regime,
            "n_nodes": self.n_nodes,
            "order_parameter": self.order_parameter,
            "order_parameter_harmonic2": self.order_parameter_harmonic2,
            "order_parameter_std": self.order_parameter_std,
            "thresholds": self.thresholds.tolist(),
            "betti0_curve": [int(value) for value in self.betti0_curve],
            "betti1_curve": [int(value) for value in self.betti1_curve],
            "h0_persistence": _persistence_to_list(self.h0_persistence),
            "h1_persistence": _persistence_to_list(self.h1_persistence),
            "persistent_component_count": self.persistent_component_count,
            "dominant_h1_persistence": self.dominant_h1_persistence,
            "reference_scale": self.reference_scale,
            "n_bootstrap": self.n_bootstrap,
            "noise_std": self.noise_std,
            "passed": self.passed,
            "claim_boundary": self.claim_boundary,
        }


def _as_persistence_pairs(name: str, values: ArrayLike) -> FloatArray:
    pairs = np.asarray(values, dtype=float)
    if pairs.size == 0:
        return np.empty((0, 2), dtype=np.float64)
    if pairs.ndim != 2 or pairs.shape[1] != 2:
        raise ValueError(f"{name} must have shape (k, 2)")
    if np.any(np.isnan(pairs)) or np.any(pairs[:, 0] < 0.0):
        raise ValueError(f"{name} must be non-negative and non-NaN")
    if np.any(pairs[:, 1] < pairs[:, 0]):
        raise ValueError(f"{name} death must not precede birth")
    return pairs.astype(np.float64, copy=True)


def _persistence_to_list(pairs: FloatArray) -> list[list[float]]:
    result: list[list[float]] = []
    for birth, death in pairs:
        death_value = float(death) if np.isfinite(death) else float("inf")
        result.append([float(birth), death_value])
    return result


@dataclass(frozen=True)
class SyncWitnessBoundaryRow:
    """Fail-closed boundary for non-covered synchronisation-witness routes."""

    boundary_id: str
    status: BoundaryStatus
    reason: str
    claim_boundary: str = SYNC_WITNESS_CLAIM_BOUNDARY

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
class SyncWitnessSuiteResult:
    """Suite result for BL-18 synchronisation-witness evidence."""

    records: tuple[SyncWitnessRecord, ...]
    boundary_rows: tuple[SyncWitnessBoundaryRow, ...]
    evidence_class: str = SYNC_WITNESS_EVIDENCE_CLASS
    claim_boundary: str = SYNC_WITNESS_CLAIM_BOUNDARY

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
        """Return whether every witness case satisfied its regime bounds."""
        return all(record.passed for record in self.records)

    def records_for_regime(self, regime: PhaseCloudRegime) -> tuple[SyncWitnessRecord, ...]:
        """Return witness records for one reference regime."""
        return tuple(record for record in self.records if record.regime == regime)

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready suite evidence."""
        return {
            "evidence_class": self.evidence_class,
            "claim_boundary": self.claim_boundary,
            "passed": self.passed,
            "records": [record.to_dict() for record in self.records],
            "boundary_rows": [row.to_dict() for row in self.boundary_rows],
        }


def _bootstrap_order_parameter_std(
    phases: FloatArray, *, noise_std: float, n_bootstrap: int, seed: int
) -> float:
    if n_bootstrap == 0 or noise_std == 0.0:
        return 0.0
    rng = np.random.default_rng(seed)
    samples = np.empty(n_bootstrap, dtype=np.float64)
    for index in range(n_bootstrap):
        perturbed = phases + rng.normal(0.0, noise_std, size=phases.shape)
        samples[index] = harmonic_order_parameter(perturbed, harmonic=1)
    return float(np.std(samples, ddof=0))


def phase_cloud_synchronisation_witness(
    phases: ArrayLike,
    *,
    thresholds: ArrayLike,
    reference_scale: float,
    case_id: str = "phase_cloud",
    regime: PhaseCloudRegime = "synchronised",
    noise_std: float = 0.0,
    n_bootstrap: int = 0,
    seed: int = 0,
    min_order_parameter: float = 0.0,
    max_order_parameter: float = 1.0,
    expected_components: int = 1,
    min_dominant_h1: float = 0.0,
    max_dominant_h1: float = float(np.pi),
) -> SyncWitnessRecord:
    """Compute the synchronisation witness for one phase cloud.

    Parameters
    ----------
    phases : array_like
        One-dimensional array of oscillator phases in radians.
    thresholds : array_like
        Strictly increasing filtration thresholds for the Betti curves.
    reference_scale : float
        Filtration scale at which the persistent component count is read.
    case_id : str, optional
        Stable identifier used in the returned record.
    regime : str, optional
        Reference regime the witness is checked against.
    noise_std : float, optional
        Standard deviation of the bootstrap phase perturbation.
    n_bootstrap : int, optional
        Number of bootstrap perturbations for the order-parameter uncertainty.
    seed : int, optional
        Seed for the deterministic bootstrap.
    min_order_parameter, max_order_parameter : float, optional
        Inclusive bounds the first-harmonic order parameter must satisfy.
    expected_components : int, optional
        Expected persistent component count at ``reference_scale``.
    min_dominant_h1, max_dominant_h1 : float, optional
        Inclusive bounds on the dominant H1 persistence lifetime.

    Returns
    -------
    SyncWitnessRecord
        The order-parameter and persistent-homology witness certificate.
    """
    vector = _as_phase_vector("phases", phases)
    grid = _as_threshold_vector("thresholds", thresholds)
    reference = _as_finite_scalar("reference_scale", reference_scale)
    if not grid[0] <= reference <= grid[-1]:
        raise ValueError("reference_scale must lie within the threshold range")
    order_parameter = harmonic_order_parameter(vector, harmonic=1)
    harmonic2 = harmonic_order_parameter(vector, harmonic=2)
    order_std = _bootstrap_order_parameter_std(
        vector, noise_std=noise_std, n_bootstrap=n_bootstrap, seed=seed
    )
    distance = geodesic_phase_distance_matrix(vector)
    persistence = vietoris_rips_persistence(distance, max_dimension=1)
    h0 = persistence[0]
    h1 = persistence[1]
    betti0 = betti_curve(h0, grid)
    betti1 = betti_curve(h1, grid)
    component_count = int(betti_curve(h0, np.asarray([reference], dtype=np.float64))[0])
    dominant = _dominant_persistence(h1)
    passed = (
        min_order_parameter <= order_parameter <= max_order_parameter
        and component_count == expected_components
        and min_dominant_h1 <= dominant <= max_dominant_h1
    )
    return SyncWitnessRecord(
        case_id=case_id,
        regime=regime,
        n_nodes=int(vector.size),
        order_parameter=order_parameter,
        order_parameter_harmonic2=harmonic2,
        order_parameter_std=order_std,
        thresholds=grid,
        betti0_curve=betti0,
        betti1_curve=betti1,
        h0_persistence=h0,
        h1_persistence=h1,
        persistent_component_count=component_count,
        dominant_h1_persistence=dominant,
        reference_scale=reference,
        n_bootstrap=int(n_bootstrap),
        noise_std=float(noise_std),
        passed=bool(passed),
    )


def _cluster_phases(centres: Sequence[float], spread: float, per_cluster: int) -> FloatArray:
    offsets = np.linspace(-spread, spread, per_cluster)
    phases = np.concatenate([np.asarray(centre, dtype=np.float64) + offsets for centre in centres])
    return phases.astype(np.float64, copy=True)


def default_sync_witness_cases() -> tuple[SyncWitnessCase, ...]:
    """Return the deterministic BL-18 synchronisation-witness reference cases."""
    two_pi = 2.0 * float(np.pi)
    thresholds = np.linspace(0.05, float(np.pi), 24, dtype=np.float64)
    synchronised = np.array(
        [0.01, -0.01, 0.02, -0.02, 0.0, 0.015, -0.015, 0.005], dtype=np.float64
    )
    desynchronised = np.linspace(0.0, two_pi, 8, endpoint=False).astype(np.float64)
    clustered = _cluster_phases((0.0, two_pi / 3.0, 2.0 * two_pi / 3.0), 0.03, 3)
    return (
        SyncWitnessCase(
            case_id="synchronised_eight_node",
            regime="synchronised",
            phases=synchronised,
            thresholds=thresholds,
            reference_scale=0.5,
            noise_std=0.01,
            n_bootstrap=32,
            seed=3101,
            min_order_parameter=0.999,
            max_order_parameter=1.0,
            expected_components=1,
            min_dominant_h1=0.0,
            max_dominant_h1=1.0e-6,
        ),
        SyncWitnessCase(
            case_id="desynchronised_eight_node",
            regime="desynchronised",
            phases=desynchronised,
            thresholds=thresholds,
            reference_scale=0.5,
            noise_std=0.01,
            n_bootstrap=32,
            seed=3102,
            min_order_parameter=0.0,
            max_order_parameter=0.05,
            expected_components=8,
            min_dominant_h1=1.5,
            max_dominant_h1=float(np.pi),
        ),
        SyncWitnessCase(
            case_id="clustered_three_group",
            regime="clustered",
            phases=clustered,
            thresholds=thresholds,
            reference_scale=0.5,
            noise_std=0.01,
            n_bootstrap=32,
            seed=3103,
            min_order_parameter=0.0,
            max_order_parameter=0.6,
            expected_components=3,
            min_dominant_h1=0.0,
            max_dominant_h1=float(np.pi),
        ),
    )


def sync_witness_boundary_rows() -> tuple[SyncWitnessBoundaryRow, ...]:
    """Return fail-closed BL-18 synchronisation-witness boundary rows."""
    return (
        SyncWitnessBoundaryRow(
            boundary_id="high_dimensional_manifold_boundary",
            status="hard_gap",
            reason=(
                "persistent homology beyond one-dimensional phase clouds requires "
                "manifold-embedding and identifiability analysis outside this suite"
            ),
        ),
        SyncWitnessBoundaryRow(
            boundary_id="hardware_phase_tomography_boundary",
            status="hard_gap",
            reason=(
                "provider-backed phase tomography requires raw counts, calibration, "
                "and owner-approved hardware tickets"
            ),
        ),
    )


def _run_case(case: SyncWitnessCase) -> SyncWitnessRecord:
    return phase_cloud_synchronisation_witness(
        case.phases,
        thresholds=case.thresholds,
        reference_scale=case.reference_scale,
        case_id=case.case_id,
        regime=case.regime,
        noise_std=case.noise_std,
        n_bootstrap=case.n_bootstrap,
        seed=case.seed,
        min_order_parameter=case.min_order_parameter,
        max_order_parameter=case.max_order_parameter,
        expected_components=case.expected_components,
        min_dominant_h1=case.min_dominant_h1,
        max_dominant_h1=case.max_dominant_h1,
    )


def run_sync_witness_suite(
    cases: Sequence[SyncWitnessCase] | None = None,
) -> SyncWitnessSuiteResult:
    """Run the deterministic BL-18 synchronisation-witness evidence suite."""
    selected_cases = tuple(default_sync_witness_cases() if cases is None else cases)
    if not selected_cases:
        raise ValueError("cases must be non-empty")
    records = tuple(_run_case(case) for case in selected_cases)
    return SyncWitnessSuiteResult(
        records=records,
        boundary_rows=sync_witness_boundary_rows(),
    )


__all__ = [
    "BoundaryStatus",
    "PhaseCloudRegime",
    "SYNC_WITNESS_CLAIM_BOUNDARY",
    "SYNC_WITNESS_EVIDENCE_CLASS",
    "SyncWitnessBoundaryRow",
    "SyncWitnessCase",
    "SyncWitnessRecord",
    "SyncWitnessSuiteResult",
    "betti_curve",
    "default_sync_witness_cases",
    "geodesic_phase_distance_matrix",
    "harmonic_order_parameter",
    "phase_cloud_synchronisation_witness",
    "run_sync_witness_suite",
    "sync_witness_boundary_rows",
    "vietoris_rips_persistence",
]
