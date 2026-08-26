# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — IQM layout-transfer per-size powered follow-up
"""Frozen IQM layout-transfer circuit matrix and decision-rule implementation.

Implements the local, provider-free surfaces committed in
``docs/campaigns/iqm_layout_transfer_per_size_prereg_2026-07-22.md``.  The
completed 2026-07-21 layout-transfer builder remains the single source for
logical circuits, layouts, transpilation, exact references, readout
calibration and depth-parity evidence.  IQM layout-transfer repeats each main circuit four
times in deterministic execution order and keeps one readout pair per size.

The analysis resamples every main repetition and readout circuit as a
multinomial, pools repetitions within arm and size, and evaluates all frozen
endpoints.  It performs no I/O and never contacts a provider.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from qiskit import QuantumCircuit
from scipy.stats import chi2

from ..hardware.iqm_lattice_calibration import LatticeCalibration
from .iqm_layout_transfer_benchmark import (
    ARM_NAMES,
    CHAIN_SIZES,
    TRANSPILER_SEED,
    LayoutTransferPlan,
    build_layout_transfer_plan,
    corrected_order_parameter,
    per_qubit_readout_errors,
)

__all__ = [
    "BOOTSTRAP_RESAMPLES",
    "BOOTSTRAP_SEED",
    "CAMPAIGN",
    "REPETITIONS",
    "PerSizeLayoutTransferPlan",
    "analyse_per_size_counts",
    "build_per_size_layout_transfer_plan",
    "holm_adjusted_p_values",
]

CAMPAIGN = "iqm_layout_transfer_per_size_prereg_2026-07-22"
REPETITIONS: tuple[int, ...] = (1, 2, 3, 4)
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20260722
ALPHA = 0.05


@dataclass(frozen=True)
class PerSizeLayoutTransferPlan:
    """Immutable wrapper around the committed per-size layout-transfer plan."""

    base: LayoutTransferPlan

    @property
    def circuit_count(self) -> int:
        """Return 36 mains plus six readout circuits for the frozen matrix."""
        return sum(
            len(block.arms) * len(REPETITIONS) + len(block.readout_circuits)
            for block in self.base.blocks
        )

    @property
    def all_gates_pass(self) -> bool:
        """Return whether every preregistered per-size depth gate passes."""
        return self.base.all_gates_pass

    @property
    def total_shots(self) -> int:
        """Return the frozen full-matrix shot budget."""
        main_circuits = sum(len(block.arms) * len(REPETITIONS) for block in self.base.blocks)
        readout_circuits = sum(len(block.readout_circuits) for block in self.base.blocks)
        return main_circuits * self.base.main_shots + readout_circuits * self.base.readout_shots

    def circuit_manifest(self) -> tuple[tuple[str, QuantumCircuit], ...]:
        """Return the deterministic execution-order circuit matrix."""
        entries: list[tuple[str, QuantumCircuit]] = []
        for block in self.base.blocks:
            for arm in block.arms:
                for repetition in REPETITIONS:
                    entries.append((f"main_n{block.n}_{arm.arm}_rep{repetition}", arm.circuit))
            entries.append((f"readout_n{block.n}_zeros", block.readout_circuits[0]))
            entries.append((f"readout_n{block.n}_ones", block.readout_circuits[1]))
        return tuple(entries)

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-serialisable frozen plan payload."""
        return {
            "campaign": CAMPAIGN,
            "preregistration": (
                "docs/campaigns/iqm_layout_transfer_per_size_prereg_2026-07-22.md"
            ),
            "blocks": [block.to_dict() for block in self.base.blocks],
            "repetitions": list(REPETITIONS),
            "main_shots": self.base.main_shots,
            "main_shots_per_arm_size": self.base.main_shots * len(REPETITIONS),
            "readout_shots": self.base.readout_shots,
            "transpiler_seed": self.base.transpiler_seed,
            "basis_gates": list(self.base.basis_gates),
            "circuit_count": self.circuit_count,
            "total_shots": self.total_shots,
            "all_gates_pass": self.all_gates_pass,
        }


def build_per_size_layout_transfer_plan(
    calibration: LatticeCalibration,
    *,
    sizes: tuple[int, ...] = CHAIN_SIZES,
    seed: int = TRANSPILER_SEED,
) -> PerSizeLayoutTransferPlan:
    """Build the per-size transfer matrix from one calibration snapshot."""
    return PerSizeLayoutTransferPlan(
        build_layout_transfer_plan(calibration, sizes=sizes, seed=seed)
    )


def _resample_counts(counts: Mapping[str, int], rng: np.random.Generator) -> dict[str, int]:
    keys = list(counts)
    values = np.asarray([int(counts[key]) for key in keys], dtype=np.int64)
    if np.any(values < 0) or int(values.sum()) <= 0:
        raise ValueError("counts must contain a positive number of non-negative shots")
    drawn = rng.multinomial(int(values.sum()), values / values.sum())
    return {key: int(value) for key, value in zip(keys, drawn, strict=True) if value}


def _pool_counts(parts: list[Mapping[str, int]]) -> dict[str, int]:
    pooled: dict[str, int] = {}
    for counts in parts:
        for key, value in counts.items():
            count = int(value)
            if count < 0:
                raise ValueError("counts must be non-negative")
            pooled[key] = pooled.get(key, 0) + count
    if not pooled or sum(pooled.values()) <= 0:
        raise ValueError("pooled counts must contain at least one shot")
    return pooled


def _arm_errors(
    block: Mapping[str, Any],
    counts: Mapping[str, Mapping[str, int]],
    rng: np.random.Generator | None,
) -> tuple[dict[str, float], dict[str, float]]:
    n = int(block["n"])
    readout_qubits = tuple(int(q) for q in block["readout_qubits"])
    zeros = counts[f"readout_n{n}_zeros"]
    ones = counts[f"readout_n{n}_ones"]
    if rng is not None:
        zeros = _resample_counts(zeros, rng)
        ones = _resample_counts(ones, rng)
    e01, e10 = per_qubit_readout_errors(zeros, ones, readout_qubits)

    corrected: dict[str, float] = {}
    raw: dict[str, float] = {}
    exact = float(block["exact_reference"])
    for arm_payload in block["arms"]:
        arm = str(arm_payload["arm"])
        measured = tuple(int(q) for q in arm_payload["measured_qubits"])
        repetitions = [counts[f"main_n{n}_{arm}_rep{repetition}"] for repetition in REPETITIONS]
        if rng is not None:
            repetitions = [_resample_counts(part, rng) for part in repetitions]
        pooled = _pool_counts(repetitions)
        identity = dict.fromkeys(measured, 0.0)
        raw[arm] = abs(corrected_order_parameter(pooled, measured, identity, identity) - exact)
        corrected[arm] = abs(corrected_order_parameter(pooled, measured, e01, e10) - exact)
    return corrected, raw


def holm_adjusted_p_values(p_values: Mapping[int, float]) -> dict[int, float]:
    """Return monotone Holm-Bonferroni adjusted p-values."""
    ordered = sorted(
        ((int(key), float(value)) for key, value in p_values.items()), key=lambda x: x[1]
    )
    adjusted: dict[int, float] = {}
    running = 0.0
    total = len(ordered)
    for rank, (key, value) in enumerate(ordered):
        running = max(running, (total - rank) * value)
        adjusted[key] = min(1.0, running)
    return adjusted


def _two_sided_bootstrap_p(samples: NDArray[np.float64]) -> float:
    non_positive = (np.count_nonzero(samples <= 0.0) + 1) / (samples.size + 1)
    non_negative = (np.count_nonzero(samples >= 0.0) + 1) / (samples.size + 1)
    return float(min(1.0, 2.0 * min(non_positive, non_negative)))


def _interval(samples: NDArray[np.float64], confidence: float) -> list[float]:
    tail = (1.0 - confidence) * 50.0
    return [float(np.percentile(samples, tail)), float(np.percentile(samples, 100.0 - tail))]


def _required_labels(plan: Mapping[str, Any]) -> set[str]:
    labels: set[str] = set()
    for block in plan["blocks"]:
        n = int(block["n"])
        labels.update(
            f"main_n{n}_{arm}_rep{repetition}" for arm in ARM_NAMES for repetition in REPETITIONS
        )
        labels.update((f"readout_n{n}_zeros", f"readout_n{n}_ones"))
    return labels


def analyse_per_size_counts(
    plan: Mapping[str, Any],
    counts: Mapping[str, Mapping[str, int]],
    *,
    n_resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Evaluate every frozen endpoint from a complete per-size count matrix."""
    if plan.get("campaign") != CAMPAIGN:
        raise ValueError(f"expected campaign {CAMPAIGN!r}")
    if not bool(plan.get("all_gates_pass")):
        raise ValueError("depth-parity gate failed; per-size analysis is blocked")
    if n_resamples < 2:
        raise ValueError("bootstrap requires at least two resamples")
    required = _required_labels(plan)
    missing = sorted(required.difference(counts))
    unexpected = sorted(set(counts).difference(required))
    if missing or unexpected:
        raise ValueError(f"count matrix mismatch: missing={missing}, unexpected={unexpected}")

    blocks = list(plan["blocks"])
    point: dict[int, tuple[dict[str, float], dict[str, float]]] = {
        int(block["n"]): _arm_errors(block, counts, None) for block in blocks
    }
    rng = np.random.default_rng(seed)
    primary_samples = {int(block["n"]): np.empty(n_resamples) for block in blocks}
    naive_samples = {int(block["n"]): np.empty(n_resamples) for block in blocks}
    raw_samples = {int(block["n"]): np.empty(n_resamples) for block in blocks}
    for index in range(n_resamples):
        for block in blocks:
            n = int(block["n"])
            corrected, raw = _arm_errors(block, counts, rng)
            primary_samples[n][index] = corrected["default"] - corrected["optimised"]
            naive_samples[n][index] = corrected["default"] - corrected["naive"]
            raw_samples[n][index] = raw["default"] - raw["optimised"]

    p_values = {n: _two_sided_bootstrap_p(samples) for n, samples in primary_samples.items()}
    adjusted = holm_adjusted_p_values(p_values)
    per_size: dict[str, Any] = {}
    for n in sorted(point):
        corrected, raw = point[n]
        difference = corrected["default"] - corrected["optimised"]
        per_size[str(n)] = {
            "corrected_errors": corrected,
            "raw_errors": raw,
            "primary_default_minus_optimised": {
                "point": difference,
                "bootstrap_ci95": _interval(primary_samples[n], 0.95),
                "two_sided_p": p_values[n],
                "holm_adjusted_p": adjusted[n],
                "rejects_zero": bool(adjusted[n] < ALPHA),
                "direction": "optimiser_advantage"
                if difference > 0
                else "optimiser_disadvantage"
                if difference < 0
                else "null",
            },
            "s4_default_minus_naive": {
                "point": corrected["default"] - corrected["naive"],
                "bootstrap_ci95": _interval(naive_samples[n], 0.95),
            },
            "s5_raw_default_minus_optimised": {
                "point": raw["default"] - raw["optimised"],
                "bootstrap_ci95": _interval(raw_samples[n], 0.95),
            },
        }

    pooled = np.mean(np.stack(list(primary_samples.values())), axis=0)
    variances = {n: float(np.var(samples, ddof=1)) for n, samples in primary_samples.items()}
    if any(value <= 0.0 or not np.isfinite(value) for value in variances.values()):
        heterogeneity: dict[str, Any] = {
            "analysable": False,
            "reason": "at least one bootstrap variance is non-positive",
            "bootstrap_variances": variances,
        }
    else:
        weights = {n: 1.0 / value for n, value in variances.items()}
        weighted_mean = sum(
            weights[n] * float(np.mean(primary_samples[n])) for n in weights
        ) / sum(weights.values())
        q = sum(
            weights[n] * (float(np.mean(primary_samples[n])) - weighted_mean) ** 2 for n in weights
        )
        degrees = len(weights) - 1
        heterogeneity = {
            "analysable": degrees > 0,
            "cochran_q": q,
            "degrees_of_freedom": degrees,
            "p_value": float(chi2.sf(q, degrees)) if degrees > 0 else None,
            "rejects_homogeneity": bool(degrees > 0 and chi2.sf(q, degrees) < ALPHA),
            "bootstrap_variances": variances,
        }

    return {
        "campaign": CAMPAIGN,
        "kind": "frozen_per_size_decision_rule",
        "matrix_complete": True,
        "primary_all_sizes_holm_significant": all(
            payload["primary_default_minus_optimised"]["rejects_zero"]
            for payload in per_size.values()
        ),
        "per_size": per_size,
        "s2_pooled_default_minus_optimised": {
            "point": float(
                np.mean(
                    [
                        corrected["default"] - corrected["optimised"]
                        for corrected, _raw in point.values()
                    ]
                )
            ),
            "bootstrap_ci90": _interval(pooled, 0.90),
        },
        "s3_cochran_q": heterogeneity,
        "bootstrap": {"resamples": n_resamples, "seed": seed},
        "interpretation_boundary": (
            "The powered per-size run resolves sampled layout effects on one IQM Garnet "
            "calibration window only; no quantum-advantage or cross-device claim"
        ),
    }
