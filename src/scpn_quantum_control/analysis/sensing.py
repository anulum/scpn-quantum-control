# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- S11 quantum sensing readiness
"""No-submit S11 sync-order quantum-sensing readiness model."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

from .bkt_analysis import fiedler_eigenvalue
from .phase_diagram import critical_coupling_finite_graph, order_parameter_steady_state
from .qfi import compute_qfi
from .qfi_criticality import qfi_vs_coupling
from .qfi_geometric_crosscheck import crosscheck_qfi_geometric

FloatArray: TypeAlias = NDArray[np.float64]

QUANTUM_SENSING_SCHEMA = "s11_quantum_sensing_readiness_v1"
GAIN_SCAN_SCHEMA = "s11_quantum_sensing_gain_scan_v1"
CRITICALITY_TAIL_SCHEMA = "s11_qfi_criticality_sensing_tail_v1"
CLAIM_BOUNDARY = (
    "QFI and sync-order sensing readiness estimate only; no hardware submission "
    "and no sensing-advantage claim"
)
ROW_BOUNDARY = "readiness estimate only; not hardware evidence"
TAIL_BOUNDARY = (
    "QFI-criticality operating-point recommendation only; no probe has been run "
    "on hardware and no sensing-advantage claim is allowed"
)
FALSIFIER = (
    "ratio of QFI-based Fisher information to classical Fisher information is "
    "below 1 on the pre-registered perturbation benchmark"
)
TAIL_FALSIFIER = (
    "the QFI peak does not survive the spectral/geometric cross-check or the "
    "pre-registered perturbation benchmark"
)


@dataclass(frozen=True)
class QuantumSensingReadinessConfig:
    """Configuration for the deterministic S11 sensing-readiness scan."""

    readout_variance_floor: float = 0.05
    finite_difference_delta: float = 1e-3
    max_dense_gib: float = 0.01

    def __post_init__(self) -> None:
        _require_positive(self.readout_variance_floor, "readout_variance_floor")
        _require_positive(self.finite_difference_delta, "finite_difference_delta")
        _require_positive(self.max_dense_gib, "max_dense_gib")

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible config data."""
        return asdict(self)


@dataclass(frozen=True)
class SensingGainRow:
    """One row in the S11 QFI/classical-Fisher readiness scan."""

    k_value: float
    qfi_value: float
    spectral_gap: float
    sync_order_parameter: float
    classical_fisher_proxy: float
    gain_ratio: float
    claim_boundary: str = ROW_BOUNDARY
    hardware_submission_allowed: bool = False
    sensing_advantage_claim_allowed: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible row data."""
        return asdict(self)


@dataclass(frozen=True)
class SensingGainScan:
    """No-submit QFI/classical-Fisher readiness scan."""

    schema: str
    k_values: tuple[float, ...]
    rows: tuple[SensingGainRow, ...]
    peak_k: float
    peak_qfi: float
    best_gain_ratio: float
    optimal_row: SensingGainRow
    falsifier: str
    hardware_submission_allowed: bool = False
    sensing_advantage_claim_allowed: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible scan data."""
        return {
            "schema": self.schema,
            "k_values": list(self.k_values),
            "rows": [row.to_dict() for row in self.rows],
            "peak_k": self.peak_k,
            "peak_qfi": self.peak_qfi,
            "best_gain_ratio": self.best_gain_ratio,
            "optimal_row": self.optimal_row.to_dict(),
            "falsifier": self.falsifier,
            "hardware_submission_allowed": self.hardware_submission_allowed,
            "sensing_advantage_claim_allowed": self.sensing_advantage_claim_allowed,
        }


@dataclass(frozen=True)
class CriticalitySensingTail:
    """QFI-peak operating-point recommendation for follow-up sensing probes."""

    schema: str
    operating_k: float
    selected_pair: tuple[int, int]
    qfi_value: float
    qfi_trace: float
    spectral_gap: float
    measurements: int
    cramer_rao_variance_bound: float
    cramer_rao_std_bound: float
    gap_min_k: float
    peak_gap_delta: float
    geometric_crosscheck_agrees: bool
    geometric_crosscheck_max_rel_difference: float
    claim_boundary: str
    falsifier: str
    hardware_submission_allowed: bool = False
    sensing_advantage_claim_allowed: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible criticality-tail data."""
        return {
            "schema": self.schema,
            "operating_k": self.operating_k,
            "selected_pair": list(self.selected_pair),
            "qfi_value": self.qfi_value,
            "qfi_trace": self.qfi_trace,
            "spectral_gap": self.spectral_gap,
            "measurements": self.measurements,
            "cramer_rao_variance_bound": self.cramer_rao_variance_bound,
            "cramer_rao_std_bound": self.cramer_rao_std_bound,
            "gap_min_k": self.gap_min_k,
            "peak_gap_delta": self.peak_gap_delta,
            "geometric_crosscheck_agrees": self.geometric_crosscheck_agrees,
            "geometric_crosscheck_max_rel_difference": (
                self.geometric_crosscheck_max_rel_difference
            ),
            "claim_boundary": self.claim_boundary,
            "falsifier": self.falsifier,
            "hardware_submission_allowed": self.hardware_submission_allowed,
            "sensing_advantage_claim_allowed": self.sensing_advantage_claim_allowed,
        }


def metrological_gain_vs_k(
    omega: FloatArray,
    topology: FloatArray,
    k_grid: FloatArray,
    *,
    config: QuantumSensingReadinessConfig | None = None,
) -> SensingGainScan:
    """Scan QFI-based sensing gain against a classical sync-order proxy."""
    frequencies, graph, grid = _validate_inputs(omega, topology, k_grid)
    cfg = config or QuantumSensingReadinessConfig()
    qfi_result = qfi_vs_coupling(
        frequencies,
        graph,
        k_range=grid,
        max_dense_gib=cfg.max_dense_gib,
    )
    k_critical = critical_coupling_finite_graph(frequencies, fiedler_eigenvalue(graph))
    rows: list[SensingGainRow] = []
    for k_value, qfi_value, gap in zip(
        qfi_result.k_values,
        qfi_result.max_qfi,
        qfi_result.spectral_gap,
    ):
        r_value = order_parameter_steady_state(float(k_value), k_critical)
        classical_fisher = _classical_fisher_proxy(
            float(k_value),
            k_critical,
            cfg.readout_variance_floor,
            cfg.finite_difference_delta,
        )
        gain_ratio = float(qfi_value / max(classical_fisher, 1e-12))
        rows.append(
            SensingGainRow(
                k_value=float(k_value),
                qfi_value=float(qfi_value),
                spectral_gap=float(gap),
                sync_order_parameter=r_value,
                classical_fisher_proxy=classical_fisher,
                gain_ratio=gain_ratio,
            )
        )
    optimal = max(rows, key=lambda row: row.gain_ratio)
    return SensingGainScan(
        schema=GAIN_SCAN_SCHEMA,
        k_values=tuple(float(k_value) for k_value in qfi_result.k_values),
        rows=tuple(rows),
        peak_k=float(qfi_result.peak_k),
        peak_qfi=float(qfi_result.peak_qfi),
        best_gain_ratio=float(optimal.gain_ratio),
        optimal_row=optimal,
        falsifier=FALSIFIER,
    )


def optimal_sensing_k(
    omega: FloatArray,
    topology: FloatArray,
    k_grid: FloatArray,
    *,
    config: QuantumSensingReadinessConfig | None = None,
) -> SensingGainRow:
    """Return the readiness row with the largest QFI/classical-Fisher ratio."""
    return metrological_gain_vs_k(omega, topology, k_grid, config=config).optimal_row


def qfi_criticality_sensing_tail(
    omega: FloatArray,
    topology: FloatArray,
    k_grid: FloatArray,
    *,
    measurements: int = 10000,
    geometric_epsilon: float = 0.005,
    run_geometric_crosscheck: bool = True,
    config: QuantumSensingReadinessConfig | None = None,
) -> CriticalitySensingTail:
    """Select the QFI-criticality operating point for a sensing follow-up.

    The scan first finds the coupling value with the largest diagonal QFI,
    then recomputes the full QFI matrix at that point to identify the most
    informative coupling-pair generator. The returned Cramer-Rao bound is a
    local readiness estimate for that selected pair under the supplied
    measurement budget; it is not hardware evidence.
    """
    frequencies, graph, grid = _validate_inputs(omega, topology, k_grid)
    _require_coupled_topology(graph)
    _require_positive_integer(measurements, "measurements")
    _require_positive(geometric_epsilon, "geometric_epsilon")
    cfg = config or QuantumSensingReadinessConfig()

    qfi_scan = qfi_vs_coupling(
        frequencies,
        graph,
        k_range=grid,
        max_dense_gib=cfg.max_dense_gib,
    )
    peak_index = int(np.argmax(qfi_scan.max_qfi))
    gap_min_index = int(np.argmin(qfi_scan.spectral_gap))
    operating_k = float(qfi_scan.k_values[peak_index])
    K_operating = operating_k * graph

    qfi_at_peak = compute_qfi(
        K_operating,
        frequencies,
        max_dense_gib=cfg.max_dense_gib,
    )
    qfi_diagonal = np.diag(qfi_at_peak.qfi_matrix)
    pair_index = int(np.argmax(qfi_diagonal))
    qfi_value = float(qfi_diagonal[pair_index])
    variance_bound = _cramer_rao_variance_bound(qfi_value, measurements)

    if run_geometric_crosscheck:
        crosscheck = crosscheck_qfi_geometric(
            K_operating,
            frequencies,
            epsilon=geometric_epsilon,
            max_dense_gib=cfg.max_dense_gib,
        )
        geometric_agrees = crosscheck.agrees
        geometric_max_rel = crosscheck.max_rel_difference
    else:
        geometric_agrees = False
        geometric_max_rel = float("inf")

    return CriticalitySensingTail(
        schema=CRITICALITY_TAIL_SCHEMA,
        operating_k=operating_k,
        selected_pair=qfi_at_peak.coupling_pairs[pair_index],
        qfi_value=qfi_value,
        qfi_trace=float(np.trace(qfi_at_peak.qfi_matrix)),
        spectral_gap=float(qfi_scan.spectral_gap[peak_index]),
        measurements=measurements,
        cramer_rao_variance_bound=variance_bound,
        cramer_rao_std_bound=float(np.sqrt(variance_bound)),
        gap_min_k=float(qfi_scan.k_values[gap_min_index]),
        peak_gap_delta=abs(operating_k - float(qfi_scan.k_values[gap_min_index])),
        geometric_crosscheck_agrees=geometric_agrees,
        geometric_crosscheck_max_rel_difference=geometric_max_rel,
        claim_boundary=TAIL_BOUNDARY,
        falsifier=TAIL_FALSIFIER,
    )


def quantum_sensing_payload() -> dict[str, Any]:
    """Return the S11 quantum-sensing readiness payload."""
    omega, topology, k_grid = _default_problem()
    config = QuantumSensingReadinessConfig()
    gain_scan = metrological_gain_vs_k(omega, topology, k_grid, config=config)
    criticality_tail = qfi_criticality_sensing_tail(omega, topology, k_grid, config=config)
    return {
        "schema": QUANTUM_SENSING_SCHEMA,
        "claim_boundary": CLAIM_BOUNDARY,
        "config": config.to_dict(),
        "gain_scan": gain_scan.to_dict(),
        "criticality_tail": criticality_tail.to_dict(),
        "prerequisites": [
            "pre-registered perturbation benchmark and classical Fisher estimator fixed",
            "hardware shot budget and shadow-tomography estimator approved before execution",
            "raw counts and uncertainty intervals archived before sensing-advantage claims",
            "applied EEG or Josephson replay target selected before real-world promotion",
        ],
        "falsifier": gain_scan.falsifier,
        "no_qpu_submission": True,
        "hardware_submission_allowed": False,
        "sensing_advantage_claim_allowed": False,
    }


def quantum_sensing_markdown(payload: dict[str, Any] | None = None) -> str:
    """Render the S11 quantum-sensing readiness note."""
    data = quantum_sensing_payload() if payload is None else payload
    scan = data["gain_scan"]
    lines = [
        "# Quantum Sensing Readiness",
        "",
        "This is the S11 no-submit readiness surface for DLA-driven quantum",
        "sensing via the synchronisation order parameter. It records QFI and",
        "classical Fisher proxy rows without hardware submission or sensing",
        "advantage promotion.",
        "",
        "## Why this page exists",
        "",
        "This page supports teams comparing sensing hypotheses against classical",
        "baselines. It captures reproducible gain estimates and the required",
        "prerequisites before any promotion of sensing-advantage claims.",
        "",
        "## Boundary",
        "",
        str(data["claim_boundary"]),
        "",
        "## Gain Scan",
        "",
        "| K | QFI | spectral gap | R | classical Fisher proxy | gain ratio |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in scan["rows"]:
        lines.append(
            "| {k_value:.6g} | {qfi_value:.6g} | {spectral_gap:.6g} | "
            "{sync_order_parameter:.6g} | {classical_fisher_proxy:.6g} | "
            "{gain_ratio:.6g} |".format(**row)
        )
    optimal = scan["optimal_row"]
    tail = data["criticality_tail"]
    lines.extend(
        [
            "",
            "## Readiness",
            "",
            f"- Peak-QFI K: `{scan['peak_k']}`",
            f"- Optimal readiness K: `{optimal['k_value']}`",
            f"- Best gain ratio estimate: `{scan['best_gain_ratio']:.6g}`",
            "- Hardware submission allowed: `False`",
            "- sensing advantage claim allowed: `False`",
            "",
            "## QFI-Criticality Tail",
            "",
            str(tail["claim_boundary"]),
            "",
            f"- Operating K: `{tail['operating_k']}`",
            f"- Selected coupling pair: `{tuple(tail['selected_pair'])}`",
            f"- Pair QFI: `{tail['qfi_value']:.6g}`",
            f"- Spectral gap at operating point: `{tail['spectral_gap']:.6g}`",
            f"- Cramer-Rao variance bound: `{tail['cramer_rao_variance_bound']:.6g}`",
            f"- Cramer-Rao standard-deviation bound: `{tail['cramer_rao_std_bound']:.6g}`",
            f"- Gap-minimum K: `{tail['gap_min_k']}`",
            f"- QFI/gap K delta: `{tail['peak_gap_delta']:.6g}`",
            f"- Spectral/geometric cross-check agrees: `{tail['geometric_crosscheck_agrees']}`",
            "",
            "## Falsifier",
            "",
            str(data["falsifier"]),
            "",
            str(tail["falsifier"]),
            "",
            "## Prerequisites",
        ]
    )
    lines.extend(f"- {item}" for item in data["prerequisites"])
    lines.extend(
        [
            "",
            "## Gate",
            "",
            "Regenerate and compare this readiness artefact with:",
            "",
            "```bash",
            "scpn-bench s11-quantum-sensing-readiness",
            "```",
        ]
    )
    return "\n".join(lines) + "\n"


def _default_problem() -> tuple[FloatArray, FloatArray, FloatArray]:
    return (
        np.array([-0.20, 0.0, 0.25], dtype=np.float64),
        np.ones((3, 3), dtype=np.float64) - np.eye(3, dtype=np.float64),
        np.array([0.4, 0.8, 1.2], dtype=np.float64),
    )


def _validate_inputs(
    omega: FloatArray,
    topology: FloatArray,
    k_grid: FloatArray,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    frequencies = np.asarray(omega, dtype=np.float64)
    graph = np.asarray(topology, dtype=np.float64)
    grid = np.asarray(k_grid, dtype=np.float64)
    if graph.ndim != 2 or graph.shape[0] != graph.shape[1]:
        raise ValueError("topology must be a square matrix")
    if frequencies.ndim != 1 or frequencies.shape[0] != graph.shape[0]:
        raise ValueError("omega length must match topology dimension")
    if grid.ndim != 1 or grid.size < 2:
        raise ValueError("k_grid must contain at least two values")
    if not np.all(np.isfinite(frequencies)) or not np.all(np.isfinite(graph)):
        raise ValueError("omega and topology must contain finite values")
    if not np.all(np.isfinite(grid)) or np.any(grid <= 0.0):
        raise ValueError("k_grid must contain finite positive values")
    if tuple(np.sort(grid)) != tuple(grid) or len(set(float(k) for k in grid)) != grid.size:
        raise ValueError("k_grid must be strictly increasing")
    if not np.allclose(graph, graph.T):
        raise ValueError("topology must be symmetric")
    return frequencies, graph, grid


def _classical_fisher_proxy(
    k_value: float,
    k_critical: float,
    readout_variance_floor: float,
    finite_difference_delta: float,
) -> float:
    left = max(k_value - finite_difference_delta, 1e-12)
    right = k_value + finite_difference_delta
    derivative = (
        order_parameter_steady_state(right, k_critical)
        - order_parameter_steady_state(left, k_critical)
    ) / (right - left)
    return float((derivative * derivative) / readout_variance_floor)


def _require_positive(value: float, name: str) -> None:
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")


def _require_positive_integer(value: int, name: str) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def _require_coupled_topology(topology: FloatArray) -> None:
    if not np.any(np.abs(np.triu(topology, k=1)) > 1e-12):
        raise ValueError("topology must contain at least one nonzero coupling edge")


def _cramer_rao_variance_bound(qfi_value: float, measurements: int) -> float:
    if qfi_value <= 1e-15:
        return float("inf")
    return float(1.0 / (float(measurements) * qfi_value))


__all__ = [
    "CRITICALITY_TAIL_SCHEMA",
    "QUANTUM_SENSING_SCHEMA",
    "CriticalitySensingTail",
    "QuantumSensingReadinessConfig",
    "SensingGainRow",
    "SensingGainScan",
    "metrological_gain_vs_k",
    "optimal_sensing_k",
    "qfi_criticality_sensing_tail",
    "quantum_sensing_markdown",
    "quantum_sensing_payload",
]
