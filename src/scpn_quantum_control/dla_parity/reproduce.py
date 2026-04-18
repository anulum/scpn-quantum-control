# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — DLA parity — statistical reproducer
"""Statistical re-computation of the DLA-parity published numbers.

Recomputes the per-depth Welch statistics, Fisher combined p-value,
peak asymmetry, and mean asymmetry from the raw circuit counts —
*without* trusting the pre-computed ``stats`` blob inside each
circuit record — then compares each scalar against the published
summary in ``figures/phase1/phase1_dla_parity_summary.json`` under
a documented tolerance bundle.

The single public entry point is :func:`reproduce_statistics`.
It returns a :class:`ReproductionResult` and raises
:class:`AssertionError` on any tolerance breach.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy import stats as scipy_stats

from .schema import DlaParityCircuit, DlaParityDataset, StatisticalSummary

DEFAULT_PUBLISHED_SUMMARY = (
    Path(__file__).resolve().parents[3] / "figures" / "phase1" / "phase1_dla_parity_summary.json"
)


@dataclass(frozen=True, slots=True)
class ReproductionTolerance:
    """Per-claim tolerances used by :func:`reproduce_statistics`.

    All tolerances are absolute unless ``_rel`` is in the field name.
    Defaults chosen so that bit-exact IEEE-754 drift and the minor
    re-ordering of float sums across numpy / scipy versions never
    trigger a spurious failure, while any real statistical divergence
    does trigger one.
    """

    leakage_mean_abs: float = 1e-9
    leakage_sem_abs: float = 1e-9
    asymmetry_relative_abs: float = 1e-9
    welch_t_rel: float = 1e-6
    welch_p_rel: float = 1e-6
    fisher_chi2_rel: float = 1e-6
    mean_asymmetry_abs: float = 1e-9
    peak_asymmetry_abs: float = 1e-9


@dataclass(frozen=True, slots=True)
class FisherResult:
    """Outcome of Fisher's combined-p method across depths."""

    chi2: float
    degrees_of_freedom: int
    combined_p: float
    n_depths_significant_at_0_05: int
    n_depths_tested: int


@dataclass(frozen=True, slots=True)
class ReproductionResult:
    """Structured outcome of :func:`reproduce_statistics`."""

    depth_summaries: tuple[StatisticalSummary, ...]
    fisher: FisherResult
    peak_asymmetry_relative: float
    peak_asymmetry_depth: int
    mean_asymmetry_relative: float
    n_circuits_used: int
    published_source: str
    tolerance: ReproductionTolerance
    # Bit-exact mirror of each published scalar against which our
    # re-computed value was checked. Populated even on pass so the
    # caller can audit the comparison, not just the pass/fail bit.
    claims_checked: tuple[tuple[str, float, float, float], ...] = field(default_factory=tuple)


def _popcount(bitstring: str) -> int:
    return bitstring.count("1")


def recompute_parity_leakage(circuit: DlaParityCircuit) -> float:
    """Fraction of shots whose measured parity differs from the initial.

    The published ``stats.parity_leakage`` field is ignored — this
    function recomputes it from raw counts so the reproducer exercises
    the full leaf-to-scalar path.
    """
    initial_parity = _popcount(circuit.meta.initial) % 2
    total = 0
    opposite = 0
    for bitstring, count in circuit.counts.items():
        # Qiskit writes measured bitstrings MSB-first, but parity is
        # bit-order-invariant — a simple popcount suffices.
        total += count
        if _popcount(bitstring) % 2 != initial_parity:
            opposite += count
    if total == 0:
        return float("nan")
    return opposite / total


def _leakage_by_depth_and_sector(
    circuits: tuple[DlaParityCircuit, ...],
) -> dict[int, dict[str, list[float]]]:
    table: dict[int, dict[str, list[float]]] = {}
    for c in circuits:
        if c.meta.n_qubits != 4:
            continue
        if not c.meta.experiment.startswith("A_dla_parity_n4"):
            continue
        if c.meta.sector not in ("even", "odd"):
            continue
        row = table.setdefault(c.meta.depth, {"even": [], "odd": []})
        row[c.meta.sector].append(recompute_parity_leakage(c))
    return table


def _welch_per_depth(depth: int, even: list[float], odd: list[float]) -> StatisticalSummary:
    e = np.asarray(even, dtype=float)
    o = np.asarray(odd, dtype=float)
    mean_e = float(e.mean())
    mean_o = float(o.mean())
    sem_e = float(e.std(ddof=1) / math.sqrt(e.size)) if e.size >= 2 else 0.0
    sem_o = float(o.std(ddof=1) / math.sqrt(o.size)) if o.size >= 2 else 0.0
    if mean_o == 0.0:
        asym_rel = float("nan") if mean_e != 0.0 else 0.0
    else:
        asym_rel = (mean_e - mean_o) / mean_o
    if e.size >= 2 and o.size >= 2:
        t_stat, p_value = scipy_stats.ttest_ind(e, o, equal_var=False)
        welch_t = float(t_stat)
        welch_p = float(p_value)
    else:
        welch_t = float("nan")
        welch_p = float("nan")
    return StatisticalSummary(
        depth=depth,
        leakage_even=mean_e,
        leakage_even_sem=sem_e,
        leakage_odd=mean_o,
        leakage_odd_sem=sem_o,
        asymmetry_relative=asym_rel,
        welch_t=welch_t,
        welch_p=welch_p,
        n_reps_even=int(e.size),
        n_reps_odd=int(o.size),
    )


def _fisher_combined(summaries: tuple[StatisticalSummary, ...]) -> FisherResult:
    valid_p = [
        s.welch_p for s in summaries if 0.0 < s.welch_p <= 1.0 and not math.isnan(s.welch_p)
    ]
    if not valid_p:
        return FisherResult(
            chi2=float("nan"),
            degrees_of_freedom=0,
            combined_p=float("nan"),
            n_depths_significant_at_0_05=0,
            n_depths_tested=len(summaries),
        )
    chi2 = -2.0 * float(np.sum(np.log(valid_p)))
    df = 2 * len(valid_p)
    combined_p = float(1.0 - scipy_stats.chi2.cdf(chi2, df))
    n_sig = sum(1 for p in valid_p if p < 0.05)
    return FisherResult(
        chi2=chi2,
        degrees_of_freedom=df,
        combined_p=combined_p,
        n_depths_significant_at_0_05=n_sig,
        n_depths_tested=len(summaries),
    )


def compute_depth_summaries(dataset: DlaParityDataset) -> tuple[StatisticalSummary, ...]:
    """All per-depth :class:`StatisticalSummary` values, sorted by depth."""
    table = _leakage_by_depth_and_sector(dataset.circuits)
    summaries = [_welch_per_depth(d, row["even"], row["odd"]) for d, row in sorted(table.items())]
    return tuple(summaries)


def _peak_and_mean(summaries: tuple[StatisticalSummary, ...]) -> tuple[float, int, float]:
    if not summaries:
        return (float("nan"), -1, float("nan"))
    pairs = [(s.asymmetry_relative, s.depth) for s in summaries]
    peak_val, peak_depth = max(pairs, key=lambda pair: pair[0])
    mean_val = float(np.mean([s.asymmetry_relative for s in summaries]))
    return (float(peak_val), int(peak_depth), mean_val)


def _as_float(row: Mapping[str, object], key: str) -> float:
    v = row[key]
    if not isinstance(v, (int, float)):
        raise ValueError(
            f"published summary field {key!r}: expected number, got {type(v).__name__}"
        )
    return float(v)


def _load_published_summary(path: Path | None) -> dict[str, object]:
    p = path if path is not None else DEFAULT_PUBLISHED_SUMMARY
    with Path(p).open("r", encoding="utf-8") as fh:
        loaded = json.load(fh)
    if not isinstance(loaded, dict):
        raise ValueError(f"{p}: expected top-level JSON object, got {type(loaded).__name__}")
    return loaded


def _check_claim(
    claims: list[tuple[str, float, float, float]],
    name: str,
    expected: float,
    actual: float,
    tol_abs: float,
    *,
    rel: bool = False,
) -> None:
    if math.isnan(expected) and math.isnan(actual):
        claims.append((name, expected, actual, 0.0))
        return
    if math.isnan(expected) or math.isnan(actual):
        claims.append((name, expected, actual, float("inf")))
        raise AssertionError(
            f"{name}: published={expected!r} vs actual={actual!r} (NaN mismatch)",
        )
    if rel:
        scale = max(abs(expected), abs(actual), 1.0)
        diff = abs(expected - actual) / scale
    else:
        diff = abs(expected - actual)
    claims.append((name, expected, actual, diff))
    if diff > tol_abs:
        raise AssertionError(
            f"{name}: published={expected:.12g} vs actual={actual:.12g}; "
            f"{'relative' if rel else 'absolute'} diff={diff:.3e} > tol={tol_abs:.3e}",
        )


def reproduce_statistics(
    dataset: DlaParityDataset,
    *,
    tolerance: ReproductionTolerance | None = None,
    published_summary: Path | str | None = None,
) -> ReproductionResult:
    """Recompute and cross-check every DLA-parity published scalar.

    Parameters
    ----------
    dataset:
        The loaded :class:`DlaParityDataset`.
    tolerance:
        Per-claim tolerance bundle. Defaults to
        :class:`ReproductionTolerance` defaults.
    published_summary:
        Path to the published summary JSON. Defaults to
        ``figures/phase1/phase1_dla_parity_summary.json`` at the
        repo root.

    Returns
    -------
    :class:`ReproductionResult`
        Structured outcome including every scalar that was checked.

    Raises
    ------
    AssertionError
        If any re-computed scalar falls outside the matching
        tolerance entry in ``tolerance``. The message names the
        offending field, the published value, the re-computed value,
        and the observed vs permitted difference.
    FileNotFoundError
        If ``published_summary`` is missing.
    """
    tol = tolerance if tolerance is not None else ReproductionTolerance()
    summaries = compute_depth_summaries(dataset)
    fisher = _fisher_combined(summaries)
    peak_val, peak_depth, mean_val = _peak_and_mean(summaries)
    n_used = sum(s.n_reps_even + s.n_reps_odd for s in summaries)

    published = _load_published_summary(
        Path(published_summary) if isinstance(published_summary, str) else published_summary,
    )
    depth_rows = published["depth_summaries"]
    if not isinstance(depth_rows, list):
        raise ValueError(
            f"{published_summary or DEFAULT_PUBLISHED_SUMMARY}: "
            f"'depth_summaries' must be a list, got {type(depth_rows).__name__}",
        )
    published_depths: dict[int, Mapping[str, object]] = {}
    for row in depth_rows:
        if not isinstance(row, dict):
            continue
        published_depths[int(row["depth"])] = row
    claims: list[tuple[str, float, float, float]] = []

    for s in summaries:
        row = published_depths.get(s.depth)
        if row is None:
            continue
        _check_claim(
            claims,
            f"depth={s.depth}.leakage_even",
            _as_float(row, "mean_even"),
            s.leakage_even,
            tol.leakage_mean_abs,
        )
        _check_claim(
            claims,
            f"depth={s.depth}.leakage_odd",
            _as_float(row, "mean_odd"),
            s.leakage_odd,
            tol.leakage_mean_abs,
        )
        _check_claim(
            claims,
            f"depth={s.depth}.sem_even",
            _as_float(row, "sem_even"),
            s.leakage_even_sem,
            tol.leakage_sem_abs,
        )
        _check_claim(
            claims,
            f"depth={s.depth}.sem_odd",
            _as_float(row, "sem_odd"),
            s.leakage_odd_sem,
            tol.leakage_sem_abs,
        )
        _check_claim(
            claims,
            f"depth={s.depth}.asymmetry_relative",
            _as_float(row, "asymmetry_relative"),
            s.asymmetry_relative,
            tol.asymmetry_relative_abs,
        )
        _check_claim(
            claims,
            f"depth={s.depth}.welch_t",
            _as_float(row, "welch_t"),
            s.welch_t,
            tol.welch_t_rel,
            rel=True,
        )
        _check_claim(
            claims,
            f"depth={s.depth}.welch_p",
            _as_float(row, "welch_p"),
            s.welch_p,
            tol.welch_p_rel,
            rel=True,
        )

    fisher_pub = published["fisher_combined"]
    if not isinstance(fisher_pub, dict):
        raise ValueError("published summary 'fisher_combined' must be an object")
    _check_claim(
        claims,
        "fisher.chi2",
        _as_float(fisher_pub, "chi2"),
        fisher.chi2,
        tol.fisher_chi2_rel,
        rel=True,
    )
    if "mean_asymmetry_relative" in published:
        _check_claim(
            claims,
            "mean_asymmetry_relative",
            _as_float(published, "mean_asymmetry_relative"),
            mean_val,
            tol.mean_asymmetry_abs,
        )
    if "peak_asymmetry_relative" in published:
        _check_claim(
            claims,
            "peak_asymmetry_relative",
            _as_float(published, "peak_asymmetry_relative"),
            peak_val,
            tol.peak_asymmetry_abs,
        )

    return ReproductionResult(
        depth_summaries=summaries,
        fisher=fisher,
        peak_asymmetry_relative=peak_val,
        peak_asymmetry_depth=peak_depth,
        mean_asymmetry_relative=mean_val,
        n_circuits_used=n_used,
        published_source=str(published_summary or DEFAULT_PUBLISHED_SUMMARY),
        tolerance=tol,
        claims_checked=tuple(claims),
    )
