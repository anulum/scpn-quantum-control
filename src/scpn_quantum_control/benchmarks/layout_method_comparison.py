# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Layout-method comparison benchmark (KT-3)
"""Honest comparison of layout methods for the Kuramoto XY-Trotter circuit.

Benchmarks three layout methods on the same problem, coupling map, and
calibration data:

* ``dynq`` — the DynQ community-detection layout from
  :func:`~scpn_quantum_control.hardware.qubit_mapper.dynq_initial_layout`,
  routed as-is;
* ``dynq+kuramoto_opt`` — the KT-3 discrete optimiser
  (:func:`~scpn_quantum_control.hardware.kuramoto_layout_optimiser.optimise_kuramoto_layout`)
  seeded by the DynQ layout and searching the candidate region;
* ``sabre`` — Qiskit's SABRE layout + routing.

With ``LayoutComparisonConfig.include_relaxation`` a fourth, **research** row
is added:

* ``dynq+sinkhorn_relaxation`` — the KT-4 annealed Sinkhorn relaxation
  (:func:`~scpn_quantum_control.hardware.kuramoto_layout_relaxation.relax_kuramoto_layout`),
  seeded by the DynQ layout, searching the same candidate region, and with
  its true-cost evaluation budget **bound to the discrete optimiser's
  ``n_evaluations`` on the same instance** — the preregistered budget match
  of the KT-4 protocol. The row carries the research label; it is an
  experiment observation, never a promoted capability.

Both search arms share the candidate region selected by
``LayoutComparisonConfig.candidate_region``: the DynQ selected region
(default, the KT-3 baseline setting) or the full calibrated device (the
preregistered ``m ≥ 2n`` setting where relocations dominate).

Metrics and their honest labels
-------------------------------
* **Routed depth and two-qubit gate count** are *measured* on the transpiled
  circuit.
* **Estimated success probability** is an *analytic model, not a hardware
  measurement*: the product of ``1 − gate_error`` over every routed two-qubit
  gate, using the per-edge calibration errors. It is layout-aware (each method
  pays for the physical edges its routed circuit actually uses).
* **R proxy** is the ideal dynamical order parameter ``R`` of the compiled
  logical circuit — which is *method-independent*, since every layout routes
  the same unitary — degraded by the estimated success probability under a
  global depolarising model: ``R_proxy = p · R_ideal``. It is a model, not a
  hardware measurement.
* **Selection wall-times** are measured on this host; the artifact carries the
  host-isolation verdict from
  :func:`~.isolated_host_readiness.capture_host_readiness`, and on a shared
  host the timings are advisory, never decision-grade.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, replace
from math import isfinite
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

from ..hardware.kuramoto_layout_cost import (
    DepthProvider,
    dynq_mean_gate_fidelity,
    routed_layout_depth,
)
from ..hardware.kuramoto_layout_optimiser import (
    LayoutSearchConfig,
    optimise_kuramoto_layout,
)
from ..hardware.kuramoto_layout_relaxation import (
    RESEARCH_LABEL,
    SinkhornRelaxationConfig,
    relax_kuramoto_layout,
)
from ..hardware.qubit_mapper import dynq_initial_layout
from ..phase.xy_compiler import compile_xy_trotter
from .decisive_run_harness import command_line, dependency_versions, git_commit
from .isolated_host_readiness import HostReadiness, capture_host_readiness

FloatArray = NDArray[np.float64]
GateErrors = dict[tuple[int, int], float]

SCHEMA_VERSION = "1.1"

#: Transpilation basis shared by every compared method.
_BASIS_GATES: tuple[str, ...] = ("cx", "rz", "rx", "ry")

#: Candidate-region choices for the search arms.
_CANDIDATE_REGIONS: tuple[str, ...] = ("dynq_region", "full_device")

_PROXY_NOTE = (
    "estimated_success_probability and r_noisy_proxy are analytic models "
    "(per-edge calibration product, global depolarising), not hardware measurements"
)
_R_IDEAL_NOTE = "r_ideal is method-independent: every layout routes the same logical unitary"
_RELAXATION_NOTE = (
    "dynq+sinkhorn_relaxation is a RESEARCH row (KT-4): its true-cost budget is "
    "bound to the discrete optimiser's n_evaluations on this instance (the "
    "preregistered budget match); no gain over the discrete baseline is a valid outcome"
)


class RProvider(Protocol):
    """Callable returning the ideal dynamical order parameter of the circuit."""

    def __call__(self, K: FloatArray, omega: FloatArray, *, t: float, reps: int) -> float:
        """Return the ideal ``R`` of the compiled XY-Trotter circuit."""
        ...


class MetricsProvider(Protocol):
    """Callable routing the XY-Trotter circuit and measuring layout metrics."""

    def __call__(
        self,
        K: FloatArray,
        omega: FloatArray,
        gate_errors: GateErrors,
        *,
        t: float,
        reps: int,
        initial_layout: tuple[int, ...] | None,
        layout_method: str | None,
        optimization_level: int,
        seed: int,
    ) -> tuple[RoutedLayoutMetrics, tuple[int, ...]]:
        """Return the routed metrics and the physical layout actually used."""
        ...


@dataclass(frozen=True)
class RoutedLayoutMetrics:
    """Measured routing metrics plus the analytic success-probability model."""

    routed_depth: int
    two_qubit_gates: int
    estimated_success_probability: float

    def to_dict(self) -> dict[str, float | int]:
        """Return a JSON-serialisable mapping of the metrics."""
        return {
            "routed_depth": self.routed_depth,
            "two_qubit_gates": self.two_qubit_gates,
            "estimated_success_probability": self.estimated_success_probability,
        }


@dataclass(frozen=True)
class MethodRow:
    """One compared layout method with its measured and modelled metrics."""

    method: str
    layout: tuple[int, ...]
    routed_depth: int
    two_qubit_gates: int
    estimated_success_probability: float
    r_ideal: float
    r_noisy_proxy: float
    selection_time_s: float
    notes: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable mapping of the row."""
        return {
            "method": self.method,
            "layout": list(self.layout),
            "routed_depth": self.routed_depth,
            "two_qubit_gates": self.two_qubit_gates,
            "estimated_success_probability": self.estimated_success_probability,
            "r_ideal": self.r_ideal,
            "r_noisy_proxy": self.r_noisy_proxy,
            "selection_time_s": self.selection_time_s,
            "notes": list(self.notes),
        }


@dataclass(frozen=True)
class LayoutComparisonConfig:
    """Configuration of the layout-method comparison run.

    Parameters
    ----------
    t, reps, order
        Evolution time, Trotter repetitions, and product-formula order for the
        compiled circuit and the optimiser cost.
    seed
        Seed shared by DynQ community detection, the optimiser restarts, and
        the SABRE transpiler.
    optimization_level
        Qiskit optimisation level shared by every routed method.
    reserved_core
        CPU core whose isolation state grades the measured wall-times.
    dynq_resolution, dynq_min_qubits
        Louvain resolution and minimum region size for the DynQ baseline.
    search
        Optimiser search configuration; ``None`` derives one from ``t``,
        ``reps``, ``order``, and ``seed``.
    include_relaxation
        Add the ``dynq+sinkhorn_relaxation`` research row (KT-4). Off by
        default so the promoted comparison surface never silently carries a
        research arm.
    candidate_region
        Candidate set searched by both search arms: ``"dynq_region"`` (the
        KT-3 baseline setting) or ``"full_device"`` (every calibrated qubit;
        the preregistered ``m ≥ 2n`` setting).
    relaxation
        Relaxation configuration; ``None`` derives one from ``t``, ``reps``,
        ``order``, and ``seed``. Its ``max_true_cost_evaluations`` is always
        overridden by the run with the discrete optimiser's ``n_evaluations``
        — the preregistered budget match.
    """

    t: float = 0.1
    reps: int = 5
    order: int = 1
    seed: int = 0
    optimization_level: int = 1
    reserved_core: int = 0
    dynq_resolution: float = 1.0
    dynq_min_qubits: int = 3
    search: LayoutSearchConfig | None = None
    include_relaxation: bool = False
    candidate_region: str = "dynq_region"
    relaxation: SinkhornRelaxationConfig | None = None

    def __post_init__(self) -> None:
        """Validate the configuration.

        Raises
        ------
        ValueError
            If ``t`` is not finite and positive, ``reps`` is not positive, or
            ``candidate_region`` is not a known region choice.
        """
        if not isfinite(self.t) or self.t <= 0.0:
            raise ValueError("t must be finite and positive")
        if self.reps < 1:
            raise ValueError("reps must be a positive integer")
        if self.candidate_region not in _CANDIDATE_REGIONS:
            raise ValueError(f"candidate_region must be one of {_CANDIDATE_REGIONS}")

    def search_config(self) -> LayoutSearchConfig:
        """Return the optimiser configuration, deriving the default when unset."""
        if self.search is not None:
            return self.search
        return LayoutSearchConfig(seed=self.seed, t=self.t, reps=self.reps, order=self.order)

    def relaxation_config(self) -> SinkhornRelaxationConfig:
        """Return the relaxation configuration, deriving the default when unset."""
        if self.relaxation is not None:
            return self.relaxation
        return SinkhornRelaxationConfig(seed=self.seed, t=self.t, reps=self.reps, order=self.order)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable mapping of the configuration."""
        return {
            "t": self.t,
            "reps": self.reps,
            "order": self.order,
            "seed": self.seed,
            "optimization_level": self.optimization_level,
            "reserved_core": self.reserved_core,
            "dynq_resolution": self.dynq_resolution,
            "dynq_min_qubits": self.dynq_min_qubits,
            "search": self.search_config().to_dict(),
            "include_relaxation": self.include_relaxation,
            "candidate_region": self.candidate_region,
            "relaxation": self.relaxation_config().to_dict() if self.include_relaxation else None,
        }


@dataclass(frozen=True)
class LayoutComparisonArtifact:
    """Full comparison artifact: rows, provenance, and honest labelling."""

    rows: tuple[MethodRow, ...]
    r_ideal: float
    timing_grade: str
    host: dict[str, Any]
    config: dict[str, Any]
    provenance: dict[str, Any]
    notes: tuple[str, ...]
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable mapping of the artifact."""
        return {
            "schema_version": self.schema_version,
            "rows": [row.to_dict() for row in self.rows],
            "r_ideal": self.r_ideal,
            "timing_grade": self.timing_grade,
            "host": self.host,
            "config": self.config,
            "provenance": self.provenance,
            "notes": list(self.notes),
        }

    def render_markdown_table(self) -> str:
        """Render the comparison rows as a GitHub-flavoured Markdown table."""
        lines = [
            "| Method | Layout | Routed depth | 2Q gates | Est. success prob. "
            "| R proxy | Selection time (s) |",
            "|---|---|---|---|---|---|---|",
        ]
        for row in self.rows:
            layout = ", ".join(str(q) for q in row.layout)
            lines.append(
                f"| {row.method} | [{layout}] | {row.routed_depth} | {row.two_qubit_gates} "
                f"| {row.estimated_success_probability:.4f} | {row.r_noisy_proxy:.4f} "
                f"| {row.selection_time_s:.3f} |"
            )
        return "\n".join(lines)


def coupling_map_from_gate_errors(gate_errors: GateErrors) -> Any:
    """Build a symmetric Qiskit ``CouplingMap`` from calibration edges.

    Parameters
    ----------
    gate_errors
        Per-edge two-qubit gate errors, ``{(i, j): error}``.

    Returns
    -------
    CouplingMap
        The bidirectional coupling map over every calibrated edge.
    """
    from qiskit.transpiler import CouplingMap

    edges: set[tuple[int, int]] = set()
    for i, j in gate_errors:
        edges.add((i, j))
        edges.add((j, i))
    return CouplingMap(sorted(edges))


def _edge_error(gate_errors: GateErrors, i: int, j: int) -> float:
    """Return the calibration error of edge ``(i, j)`` in either direction.

    Raises
    ------
    ValueError
        If the routed circuit uses an edge absent from the calibration data —
        the analytic success model then has no defensible value (fail-closed).
    """
    if (i, j) in gate_errors:
        return gate_errors[(i, j)]
    if (j, i) in gate_errors:
        return gate_errors[(j, i)]
    raise ValueError(f"routed two-qubit gate on uncalibrated edge ({i}, {j})")


def ideal_xy_order_parameter(K: FloatArray, omega: FloatArray, *, t: float, reps: int) -> float:
    """Return the ideal dynamical order parameter of the compiled circuit.

    Simulates the compiled XY-Trotter circuit with an exact statevector and
    computes ``R`` from the single-qubit ``X``/``Y`` expectations. The value is
    layout-independent: routing permutes qubits and inserts SWAPs but preserves
    the logical unitary.

    Parameters
    ----------
    K, omega
        Coupling matrix and frequency vector for the XY problem.
    t, reps
        Evolution time and Trotter repetitions of the compiled circuit.

    Returns
    -------
    float
        The ideal order parameter ``R`` in ``[0, 1]``.
    """
    from qiskit.quantum_info import Statevector

    from ..phase.floquet_kuramoto import _order_parameter

    circuit = compile_xy_trotter(K, omega, t, reps)
    statevector = Statevector.from_instruction(circuit)
    psi = np.ascontiguousarray(np.asarray(statevector.data, dtype=np.complex128))
    return _order_parameter(psi, int(K.shape[0]))


def routed_layout_metrics(
    K: FloatArray,
    omega: FloatArray,
    gate_errors: GateErrors,
    *,
    t: float,
    reps: int,
    initial_layout: tuple[int, ...] | None,
    layout_method: str | None,
    optimization_level: int,
    seed: int,
) -> tuple[RoutedLayoutMetrics, tuple[int, ...]]:
    """Route the XY-Trotter circuit and measure the layout metrics.

    Exactly one of ``initial_layout`` (fixed-layout methods) and
    ``layout_method`` (Qiskit layout passes such as ``"sabre"``) must be given.

    Parameters
    ----------
    K, omega
        Coupling matrix and frequency vector for the XY problem.
    gate_errors
        Per-edge calibration errors; they define the coupling map and price
        the analytic success model.
    t, reps
        Evolution time and Trotter repetitions of the compiled circuit.
    initial_layout
        Fixed physical placement, or ``None`` when ``layout_method`` chooses.
    layout_method
        Qiskit layout method name, or ``None`` when ``initial_layout`` fixes it.
    optimization_level
        Qiskit optimisation level for routing.
    seed
        Transpiler seed (routing determinism).

    Returns
    -------
    tuple of (RoutedLayoutMetrics, tuple of int)
        The measured metrics and the physical layout actually used.

    Raises
    ------
    ValueError
        If both or neither of ``initial_layout`` and ``layout_method`` are
        given, or a routed gate lands on an uncalibrated edge.
    """
    from qiskit import transpile

    if (initial_layout is None) == (layout_method is None):
        raise ValueError("exactly one of initial_layout and layout_method must be given")

    circuit = compile_xy_trotter(K, omega, t, reps)
    routed = transpile(
        circuit,
        coupling_map=coupling_map_from_gate_errors(gate_errors),
        initial_layout=None if initial_layout is None else list(initial_layout),
        layout_method=layout_method,
        routing_method=None if layout_method is None else layout_method,
        basis_gates=list(_BASIS_GATES),
        optimization_level=optimization_level,
        seed_transpiler=seed,
    )

    two_qubit_gates = 0
    success_probability = 1.0
    for instruction in routed.data:
        if len(instruction.qubits) != 2:
            continue
        two_qubit_gates += 1
        physical_i, physical_j = (routed.find_bit(qubit).index for qubit in instruction.qubits)
        success_probability *= 1.0 - _edge_error(gate_errors, physical_i, physical_j)

    used_layout: tuple[int, ...]
    if initial_layout is not None:
        used_layout = tuple(initial_layout)
    else:
        used_layout = tuple(routed.layout.initial_index_layout(filter_ancillas=True))

    metrics = RoutedLayoutMetrics(
        routed_depth=int(routed.depth()),
        two_qubit_gates=two_qubit_gates,
        estimated_success_probability=success_probability,
    )
    return metrics, used_layout


def _validate_problem(K: FloatArray, omega: FloatArray, gate_errors: GateErrors) -> int:
    """Validate the problem inputs and return the logical qubit count.

    Raises
    ------
    ValueError
        If ``K`` is not square, ``omega`` has the wrong shape, the calibration
        set is empty, or any calibration error lies outside ``[0, 1)``.
    """
    if K.ndim != 2 or K.shape[0] != K.shape[1]:
        raise ValueError("K must be a square matrix")
    n = int(K.shape[0])
    if omega.shape != (n,):
        raise ValueError(f"omega must have shape {(n,)}, got {omega.shape}")
    if not gate_errors:
        raise ValueError("gate_errors must not be empty")
    for edge, error in gate_errors.items():
        if not isfinite(error) or not 0.0 <= error < 1.0:
            raise ValueError(f"gate error on edge {edge} must lie in [0, 1)")
    return n


def run_layout_method_comparison(
    gate_errors: GateErrors,
    K: FloatArray,
    omega: FloatArray,
    *,
    readout_errors: dict[int, float] | None = None,
    config: LayoutComparisonConfig | None = None,
    host_readiness: HostReadiness | None = None,
    r_provider: RProvider = ideal_xy_order_parameter,
    metrics_provider: MetricsProvider = routed_layout_metrics,
    depth_provider: DepthProvider = routed_layout_depth,
) -> LayoutComparisonArtifact:
    """Compare DynQ, DynQ+Kuramoto-optimiser, and SABRE on one problem.

    With ``config.include_relaxation`` a fourth, research-labelled
    ``dynq+sinkhorn_relaxation`` row is added, its true-cost budget bound to
    the discrete optimiser's ``n_evaluations`` on this instance (the
    preregistered KT-4 budget match).

    Parameters
    ----------
    gate_errors
        Per-edge two-qubit calibration errors defining the coupling map.
    K, omega
        Coupling matrix and frequency vector for the XY problem.
    readout_errors
        Optional per-qubit readout errors for the DynQ baseline.
    config
        Run configuration; ``None`` selects :class:`LayoutComparisonConfig`
        defaults.
    host_readiness
        Pre-captured host-isolation verdict; when ``None`` the live host is
        assessed via :func:`~.isolated_host_readiness.capture_host_readiness`.
    r_provider
        Ideal-order-parameter callable (injectable for tests); defaults to
        :func:`ideal_xy_order_parameter`.
    metrics_provider
        Routing-metrics callable (injectable for tests); defaults to
        :func:`routed_layout_metrics`.
    depth_provider
        Depth callable forwarded to the optimiser; defaults to
        :func:`~scpn_quantum_control.hardware.kuramoto_layout_cost.routed_layout_depth`,
        which the run wraps with its own ``optimization_level`` and transpiler
        seed so the optimiser cost landscape is reproducible. An injected
        provider is used as-is.

    Returns
    -------
    LayoutComparisonArtifact
        The three compared rows with provenance and honest labelling.

    Raises
    ------
    ValueError
        If the problem inputs are malformed or no DynQ region fits the
        circuit width (fail-closed: without a DynQ baseline the comparison
        has no defensible reference).
    """
    config = config or LayoutComparisonConfig()
    n = _validate_problem(K, omega, gate_errors)

    started = time.perf_counter()
    mapping = dynq_initial_layout(
        gate_errors,
        n,
        readout_errors=readout_errors,
        resolution=config.dynq_resolution,
        min_qubits=config.dynq_min_qubits,
        seed=config.seed,
    )
    dynq_time = time.perf_counter() - started
    if mapping is None:
        raise ValueError(f"no DynQ region fits a {n}-qubit circuit; comparison is undefined")

    fidelity = dynq_mean_gate_fidelity(mapping)
    dynq_layout = tuple(mapping.initial_layout)
    if config.candidate_region == "dynq_region":
        candidate_qubits = tuple(sorted(mapping.selected_region.qubits))
    else:
        candidate_qubits = tuple(sorted({qubit for edge in gate_errors for qubit in edge}))
    coupling_map = coupling_map_from_gate_errors(gate_errors)

    if depth_provider is routed_layout_depth:
        # Qiskit routing is stochastic when unseeded; bind the run seed so the
        # optimiser cost landscape — and hence the artifact — is reproducible.
        def _seeded_routed_depth(
            layout: tuple[int, ...],
            K: FloatArray,
            omega: FloatArray,
            coupling_map: Any,
            *,
            t: float,
            reps: int,
        ) -> int:
            return routed_layout_depth(
                layout,
                K,
                omega,
                coupling_map,
                t=t,
                reps=reps,
                optimization_level=config.optimization_level,
                seed_transpiler=config.seed,
            )

        optimiser_depth_provider: DepthProvider = _seeded_routed_depth
    else:
        optimiser_depth_provider = depth_provider

    started = time.perf_counter()
    search = optimise_kuramoto_layout(
        K,
        omega,
        coupling_map,
        candidate_qubits,
        mean_gate_fidelity=fidelity,
        config=config.search_config(),
        initial_layout=dynq_layout,
        depth_provider=optimiser_depth_provider,
    )
    optimiser_time = time.perf_counter() - started

    relaxation = None
    relaxation_time = 0.0
    if config.include_relaxation:
        # Preregistered budget match: the relaxation may spend at most as many
        # true-cost evaluations as the discrete baseline did on this instance.
        relaxation_settings = replace(
            config.relaxation_config(), max_true_cost_evaluations=search.n_evaluations
        )
        started = time.perf_counter()
        relaxation = relax_kuramoto_layout(
            K,
            omega,
            coupling_map,
            candidate_qubits,
            mean_gate_fidelity=fidelity,
            config=relaxation_settings,
            initial_layout=dynq_layout,
            depth_provider=optimiser_depth_provider,
        )
        relaxation_time = time.perf_counter() - started

    started = time.perf_counter()
    sabre_metrics, sabre_layout = metrics_provider(
        K,
        omega,
        gate_errors,
        t=config.t,
        reps=config.reps,
        initial_layout=None,
        layout_method="sabre",
        optimization_level=config.optimization_level,
        seed=config.seed,
    )
    sabre_time = time.perf_counter() - started

    r_ideal = r_provider(K, omega, t=config.t, reps=config.reps)

    def fixed_layout_row(
        method: str,
        layout: tuple[int, ...],
        seconds: float,
        extra_notes: tuple[str, ...] = (),
    ) -> MethodRow:
        metrics, used_layout = metrics_provider(
            K,
            omega,
            gate_errors,
            t=config.t,
            reps=config.reps,
            initial_layout=layout,
            layout_method=None,
            optimization_level=config.optimization_level,
            seed=config.seed,
        )
        return _method_row(method, used_layout, metrics, r_ideal, seconds, extra_notes)

    rows = [
        fixed_layout_row("dynq", dynq_layout, dynq_time),
        fixed_layout_row("dynq+kuramoto_opt", search.best_layout, dynq_time + optimiser_time),
        _method_row("sabre", sabre_layout, sabre_metrics, r_ideal, sabre_time),
    ]

    readiness = host_readiness or capture_host_readiness(config.reserved_core)
    timing_grade = "isolated_measured" if readiness.ready else "advisory_shared_host"
    notes = [_PROXY_NOTE, _R_IDEAL_NOTE]
    provenance: dict[str, Any] = {
        "git_commit": git_commit(),
        "command": command_line(),
        "dependencies": dependency_versions(),
        "optimiser": search.to_dict(),
    }
    if relaxation is not None:
        rows.append(
            fixed_layout_row(
                "dynq+sinkhorn_relaxation",
                relaxation.best_layout,
                dynq_time + relaxation_time,
                extra_notes=(RESEARCH_LABEL,),
            )
        )
        notes.append(_RELAXATION_NOTE)
        provenance["relaxation"] = {
            **relaxation.to_dict(),
            "budget": search.n_evaluations,
        }
    if not readiness.ready:
        notes.append("selection wall-times measured on a shared host: advisory only")

    return LayoutComparisonArtifact(
        rows=tuple(rows),
        r_ideal=r_ideal,
        timing_grade=timing_grade,
        host=asdict(readiness),
        config=config.to_dict(),
        provenance=provenance,
        notes=tuple(notes),
    )


def _method_row(
    method: str,
    layout: tuple[int, ...],
    metrics: RoutedLayoutMetrics,
    r_ideal: float,
    selection_time_s: float,
    extra_notes: tuple[str, ...] = (),
) -> MethodRow:
    """Assemble one comparison row from routed metrics and the R model."""
    return MethodRow(
        method=method,
        layout=layout,
        routed_depth=metrics.routed_depth,
        two_qubit_gates=metrics.two_qubit_gates,
        estimated_success_probability=metrics.estimated_success_probability,
        r_ideal=r_ideal,
        r_noisy_proxy=metrics.estimated_success_probability * r_ideal,
        selection_time_s=selection_time_s,
        notes=(_PROXY_NOTE, *extra_notes),
    )
