#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — S19 resource-signature scan generator
"""Generate simulator-only S19 resource-signature scan artefacts."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from qiskit.quantum_info import SparsePauliOp, Statevector

from scpn_quantum_control.analysis.entanglement_entropy import entanglement_at_coupling
from scpn_quantum_control.analysis.krylov_complexity import krylov_vs_coupling
from scpn_quantum_control.analysis.magic_nonstabilizerness import magic_at_coupling
from scpn_quantum_control.analysis.pairing_correlator import pairing_map
from scpn_quantum_control.bridge.knm_hamiltonian import build_knm_paper27, knm_to_dense_matrix
from scpn_quantum_control.dense_budget import require_dense_allocation
from scpn_quantum_control.paper0.topology_schema import (
    build_paper0_topology_schema,
    schema_to_s19_source_boundary,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "s19_resource_signatures"
DEFAULT_PAPER0_TOPOLOGY_BOUNDARY = (
    REPO_ROOT
    / "paper"
    / "gotm_scpn_master_publications"
    / "gotm-scpn_paper-00_the_foundational_framework"
    / "source_validation_artifacts"
    / "paper0_topology_source_boundary_2026-05-13.json"
)
DEFAULT_N_VALUES = (4, 6)
DEFAULT_K_VALUES = tuple(float(x) for x in np.linspace(0.5, 4.5, 9))
DEFAULT_TOPOLOGIES = ("ring",)
TOPOLOGY_PROVENANCE = {
    "chain": "Synthetic nearest-neighbour open chain; simulator-control topology.",
    "ring": "Synthetic nearest-neighbour periodic ring; simulator-control topology.",
    "all_to_all": "Synthetic complete graph with zero diagonal; simulator-control topology.",
    "paper27": (
        "Paper 27 canonical K_nm exponential-decay topology with calibration "
        "anchors, normalised to unit maximum off-diagonal coupling for S19 scans."
    ),
}
NUMERIC_TOPOLOGY_LABELS = {
    "chain": "synthetic_control.chain",
    "ring": "synthetic_control.ring",
    "all_to_all": "synthetic_control.complete",
    "paper27": "legacy.paper27_provisional_not_paper0",
}
SOURCE_BOUNDARY_ONLY_TOPOLOGY = "paper0_source_boundary_only"
CLAIM_BOUNDARY = (
    "Simulator-only finite-size S19 scan. This is not a hardware claim, not a "
    "quantum-advantage claim, and not an edit to any submitted manuscript."
)


def build_topology(n_qubits: int, topology: str) -> np.ndarray:
    """Build a deterministic symmetric coupling-topology matrix."""
    if n_qubits < 2:
        raise ValueError("n_qubits must be at least 2")
    if topology == SOURCE_BOUNDARY_ONLY_TOPOLOGY:
        raise ValueError("Paper 0 source boundary is not a numeric topology")
    matrix = np.zeros((n_qubits, n_qubits), dtype=float)
    if topology == "chain":
        for i in range(n_qubits - 1):
            matrix[i, i + 1] = matrix[i + 1, i] = 1.0
    elif topology == "ring":
        for i in range(n_qubits):
            j = (i + 1) % n_qubits
            matrix[i, j] = matrix[j, i] = 1.0
    elif topology == "all_to_all":
        matrix[:] = 1.0
        np.fill_diagonal(matrix, 0.0)
    elif topology == "paper27":
        matrix = np.asarray(build_knm_paper27(L=n_qubits, K_base=1.0), dtype=float)
        matrix = (matrix + matrix.T) / 2.0
        np.fill_diagonal(matrix, 0.0)
        max_off_diagonal = float(np.max(matrix))
        if max_off_diagonal <= 0.0:
            raise ValueError("paper27 topology has no positive off-diagonal couplings")
        matrix = matrix / max_off_diagonal
    else:
        raise ValueError(f"unsupported topology: {topology}")
    return matrix


def build_omega(
    n_qubits: int,
    *,
    spread: float,
    disorder_seed: int | None = None,
) -> np.ndarray:
    """Build a deterministic zero-mean heterogeneous frequency vector."""
    if n_qubits < 2:
        raise ValueError("n_qubits must be at least 2")
    if spread < 0.0:
        raise ValueError("spread must be non-negative")
    if disorder_seed is not None:
        rng = np.random.default_rng(int(disorder_seed))
        omega = rng.normal(loc=0.0, scale=1.0, size=n_qubits)
        omega -= float(np.mean(omega))
        max_abs = float(np.max(np.abs(omega)))
        if max_abs <= 1e-15:
            return np.zeros(n_qubits, dtype=float)
        return np.asarray(omega * (0.5 * spread / max_abs), dtype=float)
    omega = np.linspace(-0.5 * spread, 0.5 * spread, n_qubits, dtype=float)
    return omega - float(np.mean(omega))


def estimate_synchronization_onset_k(topology: np.ndarray, *, omega_spread: float) -> float:
    """Estimate finite-size synchronisation onset from graph algebraic connectivity.

    For a Kuramoto network with uniform coupling scale multiplying a topology
    matrix, the Laplacian Fiedler value sets a conservative structural
    threshold: larger algebraic connectivity lowers the coupling needed to
    overcome frequency spread. This is an offline reference estimator, not a
    fitted transition claim.
    """
    if omega_spread < 0.0:
        raise ValueError("omega_spread must be non-negative")
    matrix = np.asarray(topology, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("topology must be a square matrix")
    if not np.allclose(matrix, matrix.T):
        raise ValueError("topology must be symmetric")
    degrees = np.diag(np.sum(matrix, axis=1))
    laplacian = degrees - matrix
    eigvals = np.linalg.eigvalsh(laplacian)
    if len(eigvals) < 2:
        raise ValueError("topology must contain at least two nodes")
    fiedler = float(eigvals[1])
    if fiedler <= 1e-12:
        return float("inf")
    return float(omega_spread / fiedler)


def graph_topology_diagnostics(topology: np.ndarray) -> dict[str, Any]:
    """Return finite graph diagnostics used to explain S19 control outcomes."""
    matrix = np.asarray(topology, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("topology must be a square matrix")
    if matrix.shape[0] < 2:
        raise ValueError("topology must contain at least two nodes")
    if not np.allclose(matrix, matrix.T):
        raise ValueError("topology must be symmetric")
    if np.any(matrix < -1e-12):
        raise ValueError("topology weights must be non-negative")

    n_qubits = int(matrix.shape[0])
    upper = matrix[np.triu_indices(n_qubits, k=1)]
    edge_weights = upper[upper > 1e-12]
    possible_edges = n_qubits * (n_qubits - 1) // 2
    edge_count = int(len(edge_weights))
    edge_density = float(edge_count / possible_edges)
    degrees = np.sum(matrix, axis=1)
    laplacian = np.diag(degrees) - matrix
    eigvals = np.linalg.eigvalsh(laplacian)
    algebraic_connectivity = float(eigvals[1])
    laplacian_spectral_radius = float(eigvals[-1])
    degree_mean = float(np.mean(degrees))
    degree_std = float(np.std(degrees, ddof=0))
    weight_mean = float(np.mean(edge_weights)) if edge_count else 0.0
    weight_std = float(np.std(edge_weights, ddof=0)) if edge_count else 0.0
    weights_uniform = bool(edge_count > 0 and np.allclose(edge_weights, edge_weights[0]))

    degree_nonzero = degrees[degrees > 1e-12]
    weighted_degree_cv = (
        float(np.std(degree_nonzero, ddof=0) / np.mean(degree_nonzero))
        if len(degree_nonzero) and float(np.mean(degree_nonzero)) > 1e-12
        else 0.0
    )

    boundary_class = "weighted_heterogeneous"
    if weights_uniform and edge_count == possible_edges:
        boundary_class = "complete"
    elif (
        weights_uniform
        and edge_count == n_qubits
        and np.allclose(degrees, np.full(n_qubits, 2.0 * float(edge_weights[0])))
    ):
        boundary_class = "periodic_ring"
    elif (
        weights_uniform
        and edge_count == n_qubits - 1
        and int(np.sum(np.isclose(degrees, float(edge_weights[0])))) == 2
        and int(np.sum(np.isclose(degrees, 2.0 * float(edge_weights[0])))) == n_qubits - 2
    ):
        boundary_class = "open_chain"
    elif weights_uniform:
        boundary_class = "uniform_sparse"

    return {
        "boundary_class": boundary_class,
        "edge_count": edge_count,
        "possible_edge_count": int(possible_edges),
        "edge_density": edge_density,
        "algebraic_connectivity": algebraic_connectivity,
        "laplacian_spectral_radius": laplacian_spectral_radius,
        "degree_min": float(np.min(degrees)),
        "degree_max": float(np.max(degrees)),
        "degree_mean": degree_mean,
        "degree_std": degree_std,
        "weighted_degree_cv": weighted_degree_cv,
        "edge_weight_min": float(np.min(edge_weights)) if edge_count else 0.0,
        "edge_weight_max": float(np.max(edge_weights)) if edge_count else 0.0,
        "edge_weight_mean": weight_mean,
        "edge_weight_std": weight_std,
        "uniform_edge_weights": weights_uniform,
    }


def refine_k_values_for_onset(
    *,
    base_values: tuple[float, ...],
    onset_k: float,
    half_width: float,
    step: float,
    min_k: float = 0.0,
    max_k: float | None = None,
) -> tuple[float, ...]:
    """Return a deterministic coupling grid refined around a finite onset."""
    if not base_values:
        raise ValueError("base_values must not be empty")
    if half_width < 0.0:
        raise ValueError("half_width must be non-negative")
    if step <= 0.0:
        raise ValueError("step must be positive")
    if min_k < 0.0:
        raise ValueError("min_k must be non-negative")
    if max_k is not None and max_k < min_k:
        raise ValueError("max_k must be greater than or equal to min_k")

    values = {round(float(value), 12) for value in base_values}
    if np.isfinite(onset_k):
        start = max(float(min_k), float(onset_k) - float(half_width))
        stop = float(onset_k) + float(half_width)
        if max_k is not None:
            stop = min(stop, float(max_k))
        n_steps = int(np.floor((stop - start) / step + 1e-12))
        for index in range(n_steps + 1):
            candidate = start + index * step
            if candidate <= stop + 1e-12:
                values.add(round(float(candidate), 12))
    return tuple(sorted(values))


def off_onset_control_centres(
    *,
    onset_k: float,
    half_width: float,
    min_k: float,
    max_k: float,
) -> tuple[float, ...]:
    """Place matched lower/upper local-grid controls away from the onset."""
    if half_width <= 0.0:
        raise ValueError("half_width must be positive")
    if min_k < 0.0:
        raise ValueError("min_k must be non-negative")
    if max_k < min_k:
        raise ValueError("max_k must be greater than or equal to min_k")
    if not np.isfinite(onset_k):
        return ()
    offset = 3.0 * half_width
    centres = []
    lower = float(onset_k) - offset
    upper = float(onset_k) + offset
    if lower - half_width >= min_k:
        centres.append(round(lower, 12))
    if upper + half_width <= max_k:
        centres.append(round(upper, 12))
    return tuple(centres)


def refine_k_values_for_targets(
    *,
    base_values: tuple[float, ...],
    target_centres: tuple[float, ...],
    half_width: float,
    step: float,
    min_k: float = 0.0,
    max_k: float | None = None,
) -> tuple[float, ...]:
    """Return a deterministic coupling grid refined around several targets."""
    values = {round(float(value), 12) for value in base_values}
    for centre in target_centres:
        start = max(float(min_k), float(centre) - float(half_width))
        stop = float(centre) + float(half_width)
        if max_k is not None:
            stop = min(stop, float(max_k))
        n_steps = int(np.floor((stop - start) / step + 1e-12))
        for index in range(n_steps + 1):
            candidate = start + index * step
            if candidate <= stop + 1e-12:
                values.add(round(float(candidate), 12))
    return tuple(sorted(values))


def _pauli_label(n_qubits: int, qubit: int, pauli: str) -> str:
    labels = ["I"] * n_qubits
    labels[qubit] = pauli
    return "".join(reversed(labels))


def _ground_state_sync_order(psi: np.ndarray, n_qubits: int) -> float:
    """Compute the Kuramoto-style XY order parameter from a statevector."""
    state = Statevector(np.ascontiguousarray(psi))
    phase_vectors: list[complex] = []
    for qubit in range(n_qubits):
        x_val = float(
            state.expectation_value(SparsePauliOp(_pauli_label(n_qubits, qubit, "X"))).real
        )
        y_val = float(
            state.expectation_value(SparsePauliOp(_pauli_label(n_qubits, qubit, "Y"))).real
        )
        phase_vectors.append(complex(x_val, y_val))
    return float(abs(sum(phase_vectors) / n_qubits))


def _ground_state_order_and_gap(
    omega: np.ndarray,
    topology: np.ndarray,
    k_base: float,
    *,
    max_dense_gib: float | None,
) -> tuple[float, float]:
    n_qubits = len(omega)
    require_dense_allocation(
        n_qubits,
        dtype=np.complex128,
        rank=2,
        object_count=3,
        max_gib=max_dense_gib,
        label="S19 dense ground-state workspace",
    )
    hamiltonian = knm_to_dense_matrix(k_base * topology, omega, max_dense_gib=max_dense_gib)
    eigenvalues, eigenvectors = np.linalg.eigh(hamiltonian)
    sync_order = _ground_state_sync_order(eigenvectors[:, 0], n_qubits)
    spectral_gap = float(eigenvalues[1] - eigenvalues[0])
    return sync_order, spectral_gap


def _scan_row(
    *,
    n_qubits: int,
    k_base: float,
    topology_name: str,
    numeric_topology_label: str,
    paper0_source_boundary_schema: str,
    topology: np.ndarray,
    omega: np.ndarray,
    omega_spread: float,
    omega_profile: str,
    disorder_seed: int | None,
    synchronization_onset_k: float,
    max_dense_gib: float | None,
    krylov_t_max: float,
    krylov_n_times: int,
    krylov_max_lanczos: int,
) -> dict[str, Any]:
    ent = entanglement_at_coupling(omega, topology, k_base, max_dense_gib=max_dense_gib)
    magic = magic_at_coupling(omega, topology, k_base, max_dense_gib=max_dense_gib)
    pairing = pairing_map(omega, topology, k_base, max_dense_gib=max_dense_gib)
    krylov = krylov_vs_coupling(
        omega,
        topology,
        np.asarray([k_base], dtype=float),
        t_max=krylov_t_max,
        n_times=krylov_n_times,
        max_lanczos=krylov_max_lanczos,
        max_dense_gib=max_dense_gib,
    )
    sync_order, exact_gap = _ground_state_order_and_gap(
        omega, topology, k_base, max_dense_gib=max_dense_gib
    )
    return {
        "n_qubits": int(n_qubits),
        "topology": topology_name,
        "numeric_topology_label": numeric_topology_label,
        "paper0_source_boundary_schema": paper0_source_boundary_schema,
        "omega_profile": omega_profile,
        "disorder_seed": disorder_seed,
        "omega_spread": float(omega_spread),
        "synchronization_onset_K_estimate": float(synchronization_onset_k),
        "K_base": float(k_base),
        "sync_order_ground": sync_order,
        "entropy": float(ent.entropy),
        "schmidt_gap": float(ent.schmidt_gap),
        "spectral_gap": float(exact_gap),
        "entanglement_spectral_gap": float(ent.spectral_gap),
        "magic_sre_m2": float(magic.sre_m2),
        "magic_xi_sum": float(magic.xi_sum),
        "pairing_max": float(pairing.max_pairing),
        "pairing_mean": float(pairing.mean_pairing),
        "pairing_topology_correlation": float(pairing.pairing_topology_correlation),
        "krylov_peak_complexity": float(krylov["peak_complexity"][0]),
        "krylov_n_lanczos": int(krylov["n_lanczos"][0]),
        "krylov_mean_b": float(krylov["mean_b"][0]),
    }


def _jackknife_mean_ci95(values: np.ndarray) -> dict[str, float]:
    """Estimate a deterministic 95% interval for a mean using delete-one jackknife."""
    if values.ndim != 1:
        raise ValueError("values must be one-dimensional")
    if len(values) < 2:
        return {"low": float(values[0]), "high": float(values[0]), "standard_error": 0.0}
    full_mean = float(np.mean(values))
    leave_one_out = np.asarray(
        [float(np.mean(np.delete(values, index))) for index in range(len(values))],
        dtype=float,
    )
    centred = leave_one_out - float(np.mean(leave_one_out))
    variance = ((len(values) - 1) / len(values)) * float(np.sum(centred**2))
    standard_error = float(np.sqrt(max(variance, 0.0)))
    low = max(0.0, full_mean - 1.96 * standard_error)
    high = min(1.0, full_mean + 1.96 * standard_error)
    return {"low": low, "high": high, "standard_error": standard_error}


def curvature_feature_k(
    *,
    k_values: tuple[float, ...],
    observable_values: tuple[float, ...],
) -> float:
    """Locate the strongest interior curvature feature on a coupling grid."""
    if len(k_values) != len(observable_values):
        raise ValueError("k_values and observable_values must have the same length")
    if len(k_values) < 3:
        raise ValueError("at least three points are required for curvature")
    k_array = np.asarray(k_values, dtype=float)
    y_array = np.asarray(observable_values, dtype=float)
    if not np.all(np.isfinite(k_array)) or not np.all(np.isfinite(y_array)):
        raise ValueError("curvature inputs must be finite")
    order = np.argsort(k_array)
    k_sorted = k_array[order]
    y_sorted = y_array[order]
    if np.any(np.diff(k_sorted) <= 0.0):
        raise ValueError("k_values must be unique")
    first = np.gradient(y_sorted, k_sorted, edge_order=1)
    second = np.gradient(first, k_sorted, edge_order=1)
    interior = np.abs(second[1:-1])
    index = int(np.argmax(interior)) + 1
    return float(k_sorted[index])


def _summarise_alignment(
    rows: list[dict[str, Any]],
    *,
    off_onset_control_centres_by_n_topology: dict[str, dict[str, float]] | None = None,
    topology_diagnostics_by_n_topology: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Summarise diagnostic extremum locations without turning them into claims."""
    if off_onset_control_centres_by_n_topology is None:
        off_onset_control_centres_by_n_topology = {}
    if topology_diagnostics_by_n_topology is None:
        topology_diagnostics_by_n_topology = {}
    by_realisation: dict[tuple[int, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        by_realisation.setdefault(
            (
                int(row["n_qubits"]),
                str(row["topology"]),
                str(row.get("omega_profile", "linear")),
            ),
            [],
        ).append(row)

    groups = []
    for (n_qubits, topology, omega_profile), group_rows in sorted(by_realisation.items()):
        ordered = sorted(group_rows, key=lambda item: float(item["K_base"]))
        k_min = float(ordered[0]["K_base"])
        k_max = float(ordered[-1]["K_base"])
        k_span = max(k_max - k_min, 1e-12)

        def arg_k(key: str, fn: Any, ordered_rows: list[dict[str, Any]] = ordered) -> float:
            values = [float(row[key]) for row in ordered_rows]
            index = int(fn(values))
            return float(ordered_rows[index]["K_base"])

        onset_k = float(ordered[0]["synchronization_onset_K_estimate"])
        diagnostic_extrema = {
            "entropy": arg_k("entropy", np.argmax),
            "schmidt_gap": arg_k("schmidt_gap", np.argmin),
            "magic": arg_k("magic_sre_m2", np.argmax),
            "pairing_mean": arg_k("pairing_mean", np.argmax),
            "krylov_peak": arg_k("krylov_peak_complexity", np.argmax),
        }
        curvature_sources = {
            "entropy": "entropy",
            "schmidt_gap": "schmidt_gap",
            "magic": "magic_sre_m2",
            "pairing_mean": "pairing_mean",
            "krylov_peak": "krylov_peak_complexity",
        }
        k_tuple = tuple(float(row["K_base"]) for row in ordered)
        has_curvature_resolution = len(ordered) >= 3
        curvature_features = (
            {
                name: curvature_feature_k(
                    k_values=k_tuple,
                    observable_values=tuple(float(row[key]) for row in ordered),
                )
                for name, key in curvature_sources.items()
            }
            if has_curvature_resolution
            else {name: None for name in curvature_sources}
        )
        distances = {
            name: abs(float(k_value) - onset_k) for name, k_value in diagnostic_extrema.items()
        }
        curvature_distances = (
            {
                name: abs(float(k_value) - onset_k)
                for name, k_value in curvature_features.items()
                if k_value is not None
            }
            if has_curvature_resolution
            else {name: None for name in curvature_sources}
        )
        normalised_distances = {
            name: float(distance / k_span) for name, distance in distances.items()
        }
        normalised_curvature_distances = (
            {
                name: float(distance / k_span)
                for name, distance in curvature_distances.items()
                if distance is not None
            }
            if has_curvature_resolution
            else {name: None for name in curvature_sources}
        )
        mean_normalised_distance = float(np.mean(list(normalised_distances.values())))
        mean_normalised_curvature_distance = (
            float(np.mean(list(normalised_curvature_distances.values())))
            if has_curvature_resolution
            else None
        )
        onset_window = 0.25
        diagnostics_within_onset_window = int(
            sum(distance <= onset_window for distance in normalised_distances.values())
        )
        curvature_features_within_onset_window = (
            int(
                sum(
                    distance <= onset_window
                    for distance in normalised_curvature_distances.values()
                    if distance is not None
                )
            )
            if has_curvature_resolution
            else None
        )
        control_key = f"{n_qubits}:{topology}"
        control_centres = off_onset_control_centres_by_n_topology.get(control_key, {})
        topology_diagnostics = topology_diagnostics_by_n_topology.get(control_key, {})
        off_onset_curvature_control_scores = {}
        off_onset_curvature_control_scores_by_observable: dict[str, dict[str, float]] = {}
        off_onset_curvature_control_distances = {}
        if has_curvature_resolution and control_centres:
            for label, centre in control_centres.items():
                control_distances = {
                    name: abs(float(k_value) - float(centre))
                    for name, k_value in curvature_features.items()
                    if k_value is not None
                }
                mean_control_distance = float(
                    np.mean([distance / k_span for distance in control_distances.values()])
                )
                off_onset_curvature_control_distances[label] = control_distances
                off_onset_curvature_control_scores[label] = float(
                    max(0.0, 1.0 - min(1.0, mean_control_distance))
                )
                off_onset_curvature_control_scores_by_observable[label] = {
                    name: float(max(0.0, 1.0 - min(1.0, distance / k_span)))
                    for name, distance in control_distances.items()
                }
        best_control_score = (
            max(off_onset_curvature_control_scores.values())
            if off_onset_curvature_control_scores
            else None
        )
        curvature_onset_scores_by_observable = (
            {
                name: float(max(0.0, 1.0 - min(1.0, distance)))
                for name, distance in normalised_curvature_distances.items()
                if distance is not None
            }
            if has_curvature_resolution
            else {}
        )
        curvature_onset_minus_best_control_by_observable = {}
        for name, onset_score in curvature_onset_scores_by_observable.items():
            control_scores = [
                score_by_observable[name]
                for score_by_observable in off_onset_curvature_control_scores_by_observable.values()
                if name in score_by_observable
            ]
            if control_scores:
                curvature_onset_minus_best_control_by_observable[name] = float(
                    onset_score - max(control_scores)
                )
        curvature_alignment_score = (
            float(
                max(
                    0.0,
                    1.0 - min(1.0, mean_normalised_curvature_distance),
                )
            )
            if mean_normalised_curvature_distance is not None
            else None
        )

        groups.append(
            {
                "n_qubits": n_qubits,
                "topology": topology,
                "omega_profile": omega_profile,
                "disorder_seed": ordered[0].get("disorder_seed"),
                "synchronization_onset_K_estimate": onset_k,
                "sync_order_max_K": arg_k("sync_order_ground", np.argmax),
                "entropy_max_K": diagnostic_extrema["entropy"],
                "schmidt_gap_min_K": diagnostic_extrema["schmidt_gap"],
                "magic_max_K": diagnostic_extrema["magic"],
                "pairing_mean_max_K": diagnostic_extrema["pairing_mean"],
                "krylov_peak_max_K": diagnostic_extrema["krylov_peak"],
                "diagnostic_distance_from_onset": distances,
                "normalised_diagnostic_distance_from_onset": normalised_distances,
                "curvature_feature_K": curvature_features,
                "curvature_distance_from_onset": curvature_distances,
                "normalised_curvature_distance_from_onset": (normalised_curvature_distances),
                "mean_abs_diagnostic_distance_from_onset": float(
                    np.mean(list(distances.values()))
                ),
                "mean_normalised_diagnostic_distance_from_onset": mean_normalised_distance,
                "alignment_score": float(max(0.0, 1.0 - min(1.0, mean_normalised_distance))),
                "diagnostics_within_onset_window": diagnostics_within_onset_window,
                "mean_normalised_curvature_distance_from_onset": (
                    mean_normalised_curvature_distance
                ),
                "curvature_alignment_score": curvature_alignment_score,
                "curvature_features_within_onset_window": (curvature_features_within_onset_window),
                "curvature_status": (
                    "computed" if has_curvature_resolution else "insufficient_points"
                ),
                "off_onset_curvature_control_centres": control_centres,
                "off_onset_curvature_control_distances": (off_onset_curvature_control_distances),
                "off_onset_curvature_control_scores": off_onset_curvature_control_scores,
                "off_onset_curvature_control_scores_by_observable": (
                    off_onset_curvature_control_scores_by_observable
                ),
                "best_off_onset_curvature_control_score": best_control_score,
                "curvature_onset_minus_best_control": (
                    float(curvature_alignment_score - best_control_score)
                    if curvature_alignment_score is not None and best_control_score is not None
                    else None
                ),
                "curvature_onset_minus_best_control_by_observable": (
                    curvature_onset_minus_best_control_by_observable
                ),
                "topology_diagnostics": topology_diagnostics,
            }
        )
    ensemble_groups = []
    by_n_topology: dict[tuple[int, str], list[dict[str, Any]]] = {}
    for group in groups:
        by_n_topology.setdefault((int(group["n_qubits"]), str(group["topology"])), []).append(
            group
        )
    for (n_qubits, topology), topology_groups in sorted(by_n_topology.items()):
        scores = np.asarray(
            [float(group["alignment_score"]) for group in topology_groups], dtype=float
        )
        distances = np.asarray(
            [
                float(group["mean_normalised_diagnostic_distance_from_onset"])
                for group in topology_groups
            ],
            dtype=float,
        )
        finite_curvature_scores = [
            float(group["curvature_alignment_score"])
            for group in topology_groups
            if group["curvature_alignment_score"] is not None
        ]
        finite_curvature_distances = [
            float(group["mean_normalised_curvature_distance_from_onset"])
            for group in topology_groups
            if group["mean_normalised_curvature_distance_from_onset"] is not None
        ]
        curvature_scores = np.asarray(finite_curvature_scores, dtype=float)
        curvature_distances = np.asarray(finite_curvature_distances, dtype=float)
        has_curvature_ensemble = len(curvature_scores) > 0
        finite_control_scores = [
            float(group["best_off_onset_curvature_control_score"])
            for group in topology_groups
            if group["best_off_onset_curvature_control_score"] is not None
        ]
        finite_onset_minus_control = [
            float(group["curvature_onset_minus_best_control"])
            for group in topology_groups
            if group["curvature_onset_minus_best_control"] is not None
        ]
        control_scores = np.asarray(finite_control_scores, dtype=float)
        onset_minus_control = np.asarray(finite_onset_minus_control, dtype=float)
        has_control_ensemble = len(onset_minus_control) > 0
        observable_deltas: dict[str, list[float]] = {}
        for group in topology_groups:
            for name, delta in group["curvature_onset_minus_best_control_by_observable"].items():
                observable_deltas.setdefault(name, []).append(float(delta))
        observable_delta_means = {
            name: float(np.mean(values)) for name, values in sorted(observable_deltas.items())
        }
        failing_observable_families = [
            name for name, value in observable_delta_means.items() if value < 0.0
        ]
        passing_observable_families = [
            name for name, value in observable_delta_means.items() if value >= 0.0
        ]
        if not observable_delta_means:
            control_status = "no_off_onset_control"
        elif failing_observable_families:
            control_status = "fails_off_onset_control"
        else:
            control_status = "passes_off_onset_control"
        topology_key = f"{n_qubits}:{topology}"
        ensemble_groups.append(
            {
                "n_qubits": n_qubits,
                "topology": topology,
                "realisation_count": len(topology_groups),
                "mean_alignment_score": float(np.mean(scores)),
                "std_alignment_score": float(np.std(scores, ddof=0)),
                "jackknife_alignment_score_ci95": _jackknife_mean_ci95(scores),
                "mean_curvature_alignment_score": (
                    float(np.mean(curvature_scores)) if has_curvature_ensemble else None
                ),
                "std_curvature_alignment_score": float(np.std(curvature_scores, ddof=0))
                if has_curvature_ensemble
                else None,
                "jackknife_curvature_alignment_score_ci95": (
                    _jackknife_mean_ci95(curvature_scores) if has_curvature_ensemble else None
                ),
                "mean_normalised_distance": float(np.mean(distances)),
                "std_normalised_distance": float(np.std(distances, ddof=0)),
                "mean_normalised_curvature_distance": float(np.mean(curvature_distances))
                if has_curvature_ensemble
                else None,
                "std_normalised_curvature_distance": float(np.std(curvature_distances, ddof=0))
                if has_curvature_ensemble
                else None,
                "mean_best_off_onset_curvature_control_score": (
                    float(np.mean(control_scores)) if has_control_ensemble else None
                ),
                "mean_curvature_onset_minus_best_control": (
                    float(np.mean(onset_minus_control)) if has_control_ensemble else None
                ),
                "mean_curvature_onset_minus_best_control_by_observable": (observable_delta_means),
                "control_status": control_status,
                "failing_observable_families": failing_observable_families,
                "passing_observable_families": passing_observable_families,
                "topology_diagnostics": topology_diagnostics_by_n_topology.get(topology_key, {}),
                "std_curvature_onset_minus_best_control": (
                    float(np.std(onset_minus_control, ddof=0)) if has_control_ensemble else None
                ),
                "jackknife_curvature_onset_minus_best_control_ci95": (
                    _jackknife_mean_ci95(onset_minus_control) if has_control_ensemble else None
                ),
                "omega_profiles": [str(group["omega_profile"]) for group in topology_groups],
            }
        )
    return {
        "group_count": len(groups),
        "groups": groups,
        "ensemble_group_count": len(ensemble_groups),
        "ensemble_groups": ensemble_groups,
    }


def run_scan(
    *,
    n_values: tuple[int, ...] = DEFAULT_N_VALUES,
    k_values: tuple[float, ...] = DEFAULT_K_VALUES,
    topologies: tuple[str, ...] = DEFAULT_TOPOLOGIES,
    omega_spread: float = 0.5,
    disorder_seeds: tuple[int, ...] | None = None,
    refine_onsets: bool = False,
    include_off_onset_controls: bool = False,
    refinement_half_width: float = 0.25,
    refinement_step: float = 0.125,
    max_dense_gib: float | None = 0.5,
    krylov_t_max: float = 4.0,
    krylov_n_times: int = 24,
    krylov_max_lanczos: int = 24,
    paper0_topology_boundary_path: Path | None = DEFAULT_PAPER0_TOPOLOGY_BOUNDARY,
) -> dict[str, Any]:
    """Run the simulator-only S19 scan and return a serialisable summary."""
    if not n_values:
        raise ValueError("n_values must not be empty")
    if not k_values:
        raise ValueError("k_values must not be empty")
    if not topologies:
        raise ValueError("topologies must not be empty")
    paper0_boundary = load_paper0_topology_source_boundary(paper0_topology_boundary_path)
    paper0_boundary_schema = str(paper0_boundary["schema_key"])
    unsupported_labels = sorted(set(topologies) - set(NUMERIC_TOPOLOGY_LABELS))
    if SOURCE_BOUNDARY_ONLY_TOPOLOGY in unsupported_labels:
        raise ValueError("Paper 0 source boundary is not a numeric topology")
    if unsupported_labels:
        raise ValueError(f"unsupported topology: {unsupported_labels}")
    rows: list[dict[str, Any]] = []
    effective_k_values_by_n_topology: dict[str, list[float]] = {}
    control_centres_by_n_topology: dict[str, dict[str, float]] = {}
    topology_diagnostics_by_n_topology: dict[str, dict[str, Any]] = {}
    numeric_topology_labels_by_topology = {
        topology: NUMERIC_TOPOLOGY_LABELS[topology] for topology in topologies
    }
    omega_realisations: tuple[int | None, ...] = (
        (None,) if disorder_seeds is None else tuple(int(seed) for seed in disorder_seeds)
    )
    if not omega_realisations:
        raise ValueError("disorder_seeds must not be empty when provided")
    for topology_name in topologies:
        numeric_topology_label = numeric_topology_labels_by_topology[topology_name]
        for n_qubits in n_values:
            coupling_topology = build_topology(int(n_qubits), topology_name)
            topology_diagnostics_by_n_topology[f"{int(n_qubits)}:{topology_name}"] = (
                graph_topology_diagnostics(coupling_topology)
            )
            onset_k = estimate_synchronization_onset_k(
                coupling_topology, omega_spread=omega_spread
            )
            effective_k_values = (
                refine_k_values_for_onset(
                    base_values=k_values,
                    onset_k=onset_k,
                    half_width=refinement_half_width,
                    step=refinement_step,
                    min_k=0.0,
                    max_k=max(k_values),
                )
                if refine_onsets
                else tuple(float(value) for value in k_values)
            )
            control_centres = ()
            if refine_onsets and include_off_onset_controls:
                control_centres = off_onset_control_centres(
                    onset_k=onset_k,
                    half_width=refinement_half_width,
                    min_k=0.0,
                    max_k=max(k_values),
                )
                effective_k_values = refine_k_values_for_targets(
                    base_values=effective_k_values,
                    target_centres=control_centres,
                    half_width=refinement_half_width,
                    step=refinement_step,
                    min_k=0.0,
                    max_k=max(k_values),
                )
            control_map = {}
            for centre in control_centres:
                label = "lower" if float(centre) < onset_k else "upper"
                control_map[label] = float(centre)
            if control_map:
                control_centres_by_n_topology[f"{int(n_qubits)}:{topology_name}"] = control_map
            effective_k_values_by_n_topology[f"{int(n_qubits)}:{topology_name}"] = [
                float(value) for value in effective_k_values
            ]
            for disorder_seed in omega_realisations:
                omega_profile = (
                    "linear" if disorder_seed is None else f"disorder_seed_{int(disorder_seed)}"
                )
                omega = build_omega(
                    int(n_qubits), spread=omega_spread, disorder_seed=disorder_seed
                )
                for k_base in effective_k_values:
                    rows.append(
                        _scan_row(
                            n_qubits=int(n_qubits),
                            k_base=float(k_base),
                            topology_name=topology_name,
                            numeric_topology_label=numeric_topology_label,
                            paper0_source_boundary_schema=paper0_boundary_schema,
                            topology=coupling_topology,
                            omega=omega,
                            omega_spread=omega_spread,
                            omega_profile=omega_profile,
                            disorder_seed=disorder_seed,
                            synchronization_onset_k=onset_k,
                            max_dense_gib=max_dense_gib,
                            krylov_t_max=krylov_t_max,
                            krylov_n_times=krylov_n_times,
                            krylov_max_lanczos=krylov_max_lanczos,
                        )
                    )
    return {
        "schema": "scpn_s19_resource_signature_scan_v1",
        "active_lane": "S19_resource_signatures",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "submission_status": "not_submitted",
        "hardware_spend_authorised": False,
        "separation_from_submitted_papers": {
            "submitted_papers_modified": False,
            "submitted_paper_claims_reused": False,
            "note": "S19 artefacts are a new lane and do not modify the four submitted papers.",
        },
        "parameters": {
            "n_values": [int(value) for value in n_values],
            "k_values": [float(value) for value in k_values],
            "effective_k_values_by_n_topology": effective_k_values_by_n_topology,
            "topologies": list(topologies),
            "omega_spread": float(omega_spread),
            "disorder_seeds": (
                None if disorder_seeds is None else [int(seed) for seed in disorder_seeds]
            ),
            "max_dense_gib": max_dense_gib,
            "krylov_t_max": float(krylov_t_max),
            "krylov_n_times": int(krylov_n_times),
            "krylov_max_lanczos": int(krylov_max_lanczos),
            "refinement": {
                "enabled": bool(refine_onsets),
                "method": "graph_fiedler_onset_local_grid" if refine_onsets else "none",
                "half_width": float(refinement_half_width),
                "step": float(refinement_step),
                "off_onset_controls_enabled": bool(include_off_onset_controls),
            },
            "off_onset_control_centres_by_n_topology": control_centres_by_n_topology,
            "topology_diagnostics_by_n_topology": topology_diagnostics_by_n_topology,
            "paper0_topology_source_boundary_schema": paper0_boundary_schema,
            "numeric_topology_labels_by_topology": numeric_topology_labels_by_topology,
        },
        "paper0_topology_source_boundary": paper0_boundary,
        "topology_provenance": build_topology_provenance(topologies),
        "row_count": len(rows),
        "rows": rows,
        "alignment_summary": _summarise_alignment(
            rows,
            off_onset_control_centres_by_n_topology=control_centres_by_n_topology,
            topology_diagnostics_by_n_topology=topology_diagnostics_by_n_topology,
        ),
        "claim_boundary": CLAIM_BOUNDARY,
    }


def load_paper0_topology_source_boundary(path: Path | None) -> dict[str, Any]:
    """Load or build the Paper 0 topology boundary used to label S19 scans."""
    if path is not None and path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
    else:
        payload = schema_to_s19_source_boundary(build_paper0_topology_schema())
    _validate_paper0_boundary_payload(payload)
    return payload


def build_topology_provenance(topologies: tuple[str, ...]) -> dict[str, dict[str, Any]]:
    """Return provenance records that separate source boundaries from numeric topologies."""
    return {
        topology: {
            "description": TOPOLOGY_PROVENANCE[topology],
            "numeric_topology_label": NUMERIC_TOPOLOGY_LABELS[topology],
            "paper0_topology_claim": False,
            "paper0_source_boundary_role": "provenance_boundary_only",
        }
        for topology in topologies
    }


def _validate_paper0_boundary_payload(payload: dict[str, Any]) -> None:
    if payload.get("schema_key") != "paper0.topology.source_boundary.v1":
        raise ValueError("Paper 0 topology boundary schema_key is invalid")
    if payload.get("provider_ready") is not False:
        raise ValueError("Paper 0 topology boundary must not be provider ready")
    if payload.get("numeric_coupling_matrix") is not None:
        raise ValueError("Paper 0 source boundary must not contain a numeric coupling matrix")
    if payload.get("hardware_status") != "source_boundary_only_no_provider_submission":
        raise ValueError("Paper 0 topology boundary has invalid hardware status")


def write_outputs(
    summary: dict[str, Any],
    *,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    date_tag: str | None = None,
) -> dict[str, Path]:
    """Write JSON, CSV, and markdown claim-boundary artefacts."""
    if date_tag is None:
        date_tag = date.today().isoformat()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / f"s19_scan_manifest_{date_tag}.json"
    rows_path = output_dir / f"s19_resource_rows_{date_tag}.csv"
    claim_path = output_dir / f"s19_claim_boundary_{date_tag}.md"

    manifest_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    rows = list(summary["rows"])
    if rows:
        with rows_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    else:
        rows_path.write_text("", encoding="utf-8")
    claim_path.write_text(_claim_markdown(summary), encoding="utf-8")
    return {"manifest": manifest_path, "rows": rows_path, "claim_boundary": claim_path}


def _claim_markdown(summary: dict[str, Any]) -> str:
    def fmt_optional(value: float | None) -> str:
        return "n/a" if value is None else f"{value:.6g}"

    lines = [
        "# S19 Resource-Signature Claim Boundary",
        "",
        f"- Schema: `{summary['schema']}`",
        f"- Row count: `{summary['row_count']}`",
        "- Status: simulator-only, not submitted to hardware.",
        "- Separation: this does not edit any submitted manuscript.",
        "",
        "## Boundary",
        "",
        summary["claim_boundary"],
        "",
        "## Paper 0 Source Boundary",
        "",
        "The Paper 0 topology source boundary is attached as provenance only. The",
        "source boundary is not a numeric coupling matrix and is not provider-ready.",
        "Every simulated row must carry a separate numeric-topology label.",
        "",
        "| topology | numeric topology label | Paper 0 topology claim |",
        "|---|---|---:|",
    ]
    for topology, provenance in summary["topology_provenance"].items():
        lines.append(
            f"| {topology} | {provenance['numeric_topology_label']} | "
            f"{provenance['paper0_topology_claim']} |"
        )
    lines.extend(
        [
            "",
            "This is not a hardware claim, not a thermodynamic-limit claim, and not a",
            "quantum-advantage claim. It is an offline finite-size scan used to decide",
            "whether an S19 paper lane deserves larger simulation and later IBM gating.",
            "",
            "## Alignment Summary",
            "",
            "| n | topology | omega profile | onset estimate | entropy max K | Schmidt min K | magic max K | pairing max K | Krylov max K | mean distance |",
            "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for group in summary["alignment_summary"]["groups"]:
        lines.append(
            f"| {group['n_qubits']} | {group['topology']} | "
            f"{group.get('omega_profile', 'linear')} | "
            f"{group['synchronization_onset_K_estimate']:.6g} | "
            f"{group['entropy_max_K']:.6g} | "
            f"{group['schmidt_gap_min_K']:.6g} | {group['magic_max_K']:.6g} | "
            f"{group['pairing_mean_max_K']:.6g} | {group['krylov_peak_max_K']:.6g} | "
            f"{group['mean_abs_diagnostic_distance_from_onset']:.6g} |"
        )
    lines.extend(
        [
            "",
            "## Ensemble Alignment",
            "",
            "| n | topology | boundary | realisations | extremum score | extremum CI95 | curvature score | curvature CI95 | best control | onset-control | control status | failing observables |",
            "|---:|---|---|---:|---:|---|---:|---|---:|---:|---|---|",
        ]
    )
    for group in summary["alignment_summary"].get("ensemble_groups", []):
        ci95 = group.get(
            "jackknife_alignment_score_ci95",
            {
                "low": group["mean_alignment_score"],
                "high": group["mean_alignment_score"],
            },
        )
        curvature_ci95 = group.get("jackknife_curvature_alignment_score_ci95")
        curvature_ci_text = (
            "n/a"
            if curvature_ci95 is None
            else f"{curvature_ci95['low']:.6g}--{curvature_ci95['high']:.6g}"
        )
        topology_diagnostics = group.get("topology_diagnostics", {})
        failing_observables = ", ".join(group.get("failing_observable_families", []))
        if not failing_observables:
            failing_observables = "none"
        lines.append(
            f"| {group['n_qubits']} | {group['topology']} | "
            f"{topology_diagnostics.get('boundary_class', 'unknown')} | "
            f"{group['realisation_count']} | {group['mean_alignment_score']:.6g} | "
            f"{ci95['low']:.6g}--{ci95['high']:.6g} | "
            f"{fmt_optional(group.get('mean_curvature_alignment_score'))} | "
            f"{curvature_ci_text} | "
            f"{fmt_optional(group.get('mean_best_off_onset_curvature_control_score'))} | "
            f"{fmt_optional(group.get('mean_curvature_onset_minus_best_control'))} | "
            f"{group.get('control_status', 'not_evaluated')} | "
            f"{failing_observables} |"
        )
    return "\n".join(lines) + "\n"


def _parse_csv_floats(raw: str) -> tuple[float, ...]:
    return tuple(float(part.strip()) for part in raw.split(",") if part.strip())


def _parse_csv_ints(raw: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in raw.split(",") if part.strip())


def _parse_csv_strings(raw: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in raw.split(",") if part.strip())


def main() -> int:
    """Run the command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-values", default="4,6")
    parser.add_argument(
        "--k-values", default=",".join(f"{value:.6g}" for value in DEFAULT_K_VALUES)
    )
    parser.add_argument(
        "--topology",
        choices=("ring", "chain", "all_to_all", "paper27", SOURCE_BOUNDARY_ONLY_TOPOLOGY),
        default="ring",
    )
    parser.add_argument("--topologies", default=None)
    parser.add_argument("--omega-spread", type=float, default=0.5)
    parser.add_argument("--disorder-seeds", default=None)
    parser.add_argument("--refine-onsets", action="store_true")
    parser.add_argument("--include-off-onset-controls", action="store_true")
    parser.add_argument("--refinement-half-width", type=float, default=0.25)
    parser.add_argument("--refinement-step", type=float, default=0.125)
    parser.add_argument("--max-dense-gib", type=float, default=0.5)
    parser.add_argument("--krylov-t-max", type=float, default=4.0)
    parser.add_argument("--krylov-n-times", type=int, default=24)
    parser.add_argument("--krylov-max-lanczos", type=int, default=24)
    parser.add_argument(
        "--paper0-topology-boundary",
        type=Path,
        default=DEFAULT_PAPER0_TOPOLOGY_BOUNDARY,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--date-tag", default=date.today().isoformat())
    args = parser.parse_args()

    summary = run_scan(
        n_values=_parse_csv_ints(args.n_values),
        k_values=_parse_csv_floats(args.k_values),
        topologies=_parse_csv_strings(args.topologies) if args.topologies else (args.topology,),
        omega_spread=args.omega_spread,
        disorder_seeds=(
            _parse_csv_ints(args.disorder_seeds) if args.disorder_seeds is not None else None
        ),
        refine_onsets=args.refine_onsets,
        include_off_onset_controls=args.include_off_onset_controls,
        refinement_half_width=args.refinement_half_width,
        refinement_step=args.refinement_step,
        max_dense_gib=args.max_dense_gib,
        krylov_t_max=args.krylov_t_max,
        krylov_n_times=args.krylov_n_times,
        krylov_max_lanczos=args.krylov_max_lanczos,
        paper0_topology_boundary_path=args.paper0_topology_boundary,
    )
    paths = write_outputs(summary, output_dir=args.output_dir, date_tag=args.date_tag)
    for key, path in paths.items():
        print(f"wrote_{key}={path}")
    print(f"row_count={summary['row_count']}")
    print("submission_status=not_submitted")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
