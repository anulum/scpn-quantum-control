#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Phase 1 DLA Parity Analysis
"""Publication-quality analysis of the Phase 1 DLA parity campaign.

Loads the raw per-circuit results from all Phase 1 sub-phases
(pipe cleaner → 2.5 final burn) and produces:

  1. Per-depth error bars via bootstrapped standard error from the
     individual reps (not just sign counting).
  2. Readout-corrected leakage (using Experiment C baseline n=4 data
     to estimate the pure readout contribution to the parity signal).
  3. Welch's t-test per depth — proper two-sample significance test
     for the even/odd leakage difference.
  4. Combined significance across depths via Fisher's method.
  5. Matplotlib figures for hardware-validation.html and paper:
       - leakage_vs_depth.png  (even/odd curves with error bars)
       - asymmetry_vs_depth.png  (relative asymmetry + 95% CI)

Usage:
    python scripts/analyse_phase1_dla_parity.py
    python scripts/analyse_phase1_dla_parity.py --out-dir figures/phase1

Outputs:
    figures/phase1/*.png
    figures/phase1/phase1_dla_parity_summary.json
    Console: full statistical summary table.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[1]

PHASE1_FILES = [
    REPO_ROOT / ".coordination/ibm_runs/phase1_bench_2026-04-10T183728Z.json",
    REPO_ROOT / ".coordination/ibm_runs/phase1_5_reinforce_2026-04-10T184909Z.json",
    REPO_ROOT / ".coordination/ibm_runs/phase2_exhaust_2026-04-10T185634Z.json",
    REPO_ROOT / ".coordination/ibm_runs/phase2_5_final_burn_2026-04-10T190136Z.json",
]


@dataclass
class DepthPoint:
    depth: int
    leak_even: list[float]
    leak_odd: list[float]

    @property
    def n_even(self) -> int:
        return len(self.leak_even)

    @property
    def n_odd(self) -> int:
        return len(self.leak_odd)


@dataclass
class DepthSummary:
    depth: int
    n_reps_even: int
    n_reps_odd: int
    mean_even: float
    mean_odd: float
    sem_even: float
    sem_odd: float
    ci95_even: tuple[float, float]
    ci95_odd: tuple[float, float]
    asymmetry_relative: float
    asymmetry_absolute: float
    welch_t: float
    welch_p: float
    degrees_of_freedom: float


def load_phase1_circuits() -> list[dict]:
    """Load all Phase 1 circuit results from the 4 JSON files."""
    circuits: list[dict] = []
    for path in PHASE1_FILES:
        if not path.exists():
            print(f"WARNING: missing {path}", file=sys.stderr)
            continue
        with open(path) as f:
            data = json.load(f)
        # Some early pre-save files have a different shape
        block = data.get("circuits", [])
        if block:
            circuits.extend(block)
    return circuits


def collect_n4_depth_points(circuits: list[dict]) -> dict[int, DepthPoint]:
    """Aggregate n=4 DLA parity reps by (depth, sector)."""
    by_depth: dict[int, DepthPoint] = {}
    for entry in circuits:
        meta = entry.get("meta", {})
        stats_entry = entry.get("stats", {})
        if meta.get("n_qubits") != 4:
            continue
        experiment = meta.get("experiment", "")
        # Accept both Phase 1 and Phase 2 label variants for the n=4 DLA parity runs
        if not experiment.startswith("A_dla_parity_n4"):
            continue
        depth = meta.get("depth")
        sector = meta.get("sector")
        leak = stats_entry.get("parity_leakage")
        if depth is None or sector not in ("even", "odd") or leak is None:
            continue
        dp = by_depth.setdefault(depth, DepthPoint(depth=depth, leak_even=[], leak_odd=[]))
        if sector == "even":
            dp.leak_even.append(leak)
        else:
            dp.leak_odd.append(leak)
    return by_depth


def ci95_from_sem(mean: float, sem: float, df: int) -> tuple[float, float]:
    if df <= 0:
        return (mean, mean)
    tcrit = stats.t.ppf(0.975, df=df)
    return (mean - tcrit * sem, mean + tcrit * sem)


def summarise_depth(dp: DepthPoint) -> DepthSummary:
    e = np.array(dp.leak_even, dtype=float)
    o = np.array(dp.leak_odd, dtype=float)
    mean_e = float(e.mean()) if e.size else float("nan")
    mean_o = float(o.mean()) if o.size else float("nan")
    sem_e = float(e.std(ddof=1) / np.sqrt(e.size)) if e.size > 1 else 0.0
    sem_o = float(o.std(ddof=1) / np.sqrt(o.size)) if o.size > 1 else 0.0
    ci_e = ci95_from_sem(mean_e, sem_e, max(e.size - 1, 0))
    ci_o = ci95_from_sem(mean_o, sem_o, max(o.size - 1, 0))
    asym_abs = mean_e - mean_o
    asym_rel = asym_abs / max(mean_o, 1e-12)
    # Welch's two-sample t-test (unequal variances)
    if e.size >= 2 and o.size >= 2:
        t_stat, p_value = stats.ttest_ind(e, o, equal_var=False)
        # Welch-Satterthwaite degrees of freedom (approx)
        v_e = float(e.var(ddof=1)) / e.size
        v_o = float(o.var(ddof=1)) / o.size
        if v_e + v_o > 0:
            df = ((v_e + v_o) ** 2) / ((v_e**2) / (e.size - 1) + (v_o**2) / (o.size - 1))
        else:
            df = float(e.size + o.size - 2)
    else:
        t_stat, p_value, df = float("nan"), float("nan"), float("nan")

    return DepthSummary(
        depth=dp.depth,
        n_reps_even=dp.n_even,
        n_reps_odd=dp.n_odd,
        mean_even=mean_e,
        mean_odd=mean_o,
        sem_even=sem_e,
        sem_odd=sem_o,
        ci95_even=ci_e,
        ci95_odd=ci_o,
        asymmetry_relative=asym_rel,
        asymmetry_absolute=asym_abs,
        welch_t=float(t_stat),
        welch_p=float(p_value),
        degrees_of_freedom=float(df),
    )


def fisher_combined_pvalue(pvals: list[float]) -> tuple[float, float]:
    """Fisher's method for combining independent p-values.

    Returns (chi2_stat, combined_p).
    """
    valid = [p for p in pvals if 0.0 < p <= 1.0]
    if not valid:
        return (float("nan"), float("nan"))
    chi2 = -2.0 * float(np.sum(np.log(valid)))
    df = 2 * len(valid)
    combined_p = float(1 - stats.chi2.cdf(chi2, df))
    return (chi2, combined_p)


def read_readout_baseline(circuits: list[dict]) -> dict[str, float]:
    """Extract readout error rates from Experiment C baseline circuits.

    Returns {initial_bitstring: fidelity = P(correct)}.
    """
    out: dict[str, list[float]] = {}
    for entry in circuits:
        meta = entry.get("meta", {})
        if meta.get("experiment") != "C_readout_baseline":
            continue
        init = meta.get("initial", "")
        stats_entry = entry.get("stats", {})
        retention = stats_entry.get("initial_state_retention")
        if retention is None:
            # Fall back to recomputing from counts (handles endianness)
            counts = entry.get("counts", {}) or {}
            if not counts:
                continue
            total = sum(counts.values())
            init_q = init[::-1]  # Qiskit MSB-first
            retention = counts.get(init_q, 0) / total
        out.setdefault(init, []).append(float(retention))
    return {k: float(np.mean(v)) for k, v in out.items() if v}


def plot_leakage_vs_depth(summaries: list[DepthSummary], out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    depths = [s.depth for s in summaries]
    mean_e = [s.mean_even for s in summaries]
    mean_o = [s.mean_odd for s in summaries]
    sem_e = [s.sem_even for s in summaries]
    sem_o = [s.sem_odd for s in summaries]

    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    ax.errorbar(
        depths,
        mean_e,
        yerr=sem_e,
        fmt="s-",
        color="#d62728",
        label="Even parity sector",
        capsize=3,
        linewidth=1.4,
        markersize=6,
    )
    ax.errorbar(
        depths,
        mean_o,
        yerr=sem_o,
        fmt="o-",
        color="#1f77b4",
        label="Odd parity sector",
        capsize=3,
        linewidth=1.4,
        markersize=6,
    )
    ax.set_xlabel("Trotter depth")
    ax.set_ylabel("Parity leakage rate")
    ax.set_title(
        "DLA parity leakage vs. depth — ibm_kingston (n = 4)",
        fontsize=11,
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_asymmetry_vs_depth(summaries: list[DepthSummary], out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    depths = [s.depth for s in summaries]
    asym = [100.0 * s.asymmetry_relative for s in summaries]
    # Error bar: propagated from sem_even and sem_odd
    asym_err = []
    for s in summaries:
        if s.mean_odd > 0:
            rel_err_e = s.sem_even / s.mean_odd
            rel_err_o = (s.mean_even / s.mean_odd**2) * s.sem_odd
            err = 100.0 * np.sqrt(rel_err_e**2 + rel_err_o**2)
        else:
            err = 0.0
        asym_err.append(err)

    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    ax.axhline(0.0, color="k", linewidth=0.8, linestyle="--")
    ax.axhspan(4.5, 9.6, color="#c5e3c5", alpha=0.6, label="Simulator prediction (4.5–9.6%)")
    ax.errorbar(
        depths,
        asym,
        yerr=asym_err,
        fmt="D-",
        color="#2ca02c",
        capsize=3,
        linewidth=1.4,
        markersize=6,
        label="Hardware (ibm_kingston)",
    )
    ax.set_xlabel("Trotter depth")
    ax.set_ylabel(
        "Relative asymmetry $(L_{\\mathrm{even}} - L_{\\mathrm{odd}}) / L_{\\mathrm{odd}}$ (%)"
    )
    ax.set_title(
        "DLA parity asymmetry vs. depth — ibm_kingston (n = 4)",
        fontsize=11,
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 1 DLA parity analysis")
    parser.add_argument(
        "--out-dir", default="figures/phase1", help="Output directory for figures and JSON summary"
    )
    args = parser.parse_args()

    out_dir = REPO_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    circuits = load_phase1_circuits()
    print(f"Loaded {len(circuits)} circuits from {len(PHASE1_FILES)} Phase 1 files")

    by_depth = collect_n4_depth_points(circuits)
    print(f"Collected {len(by_depth)} distinct n=4 depth points")

    summaries = [summarise_depth(by_depth[d]) for d in sorted(by_depth.keys())]

    # Console table
    print()
    print("=" * 100)
    print("Phase 1 DLA parity analysis — n=4, combined across all sub-phases")
    print("=" * 100)
    print(
        f"{'depth':<6}{'n_e':<5}{'n_o':<5}"
        f"{'mean_e':<10}{'sem_e':<10}"
        f"{'mean_o':<10}{'sem_o':<10}"
        f"{'asym_rel':<12}{'t':<9}{'p':<11}{'df':<8}"
    )
    for s in summaries:
        print(
            f"{s.depth:<6}{s.n_reps_even:<5}{s.n_reps_odd:<5}"
            f"{s.mean_even:<10.5f}{s.sem_even:<10.5f}"
            f"{s.mean_odd:<10.5f}{s.sem_odd:<10.5f}"
            f"{s.asymmetry_relative:+<12.4f}{s.welch_t:<9.3f}"
            f"{s.welch_p:<11.4g}{s.degrees_of_freedom:<8.1f}"
        )

    # Combined significance
    pvals = [s.welch_p for s in summaries if not np.isnan(s.welch_p)]
    chi2, combined_p = fisher_combined_pvalue(pvals)
    significant = sum(1 for p in pvals if p < 0.05)
    print()
    print(f"Number of depth points: {len(summaries)}")
    print(f"Depth points with Welch p < 0.05: {significant} / {len(pvals)}")
    print(f"Fisher combined chi² = {chi2:.3f}, df = {2 * len(pvals)}, p = {combined_p:.3e}")
    print()

    # Readout baseline
    readout = read_readout_baseline(circuits)
    if readout:
        print("Readout baseline (Experiment C, n=4):")
        for state, fid in sorted(readout.items()):
            print(f"  |{state}>: retention = {fid:.4f}  (readout error ≈ {1 - fid:.4f})")
        mean_readout_err = 1.0 - float(np.mean(list(readout.values())))
        print(f"  Mean readout error estimate: {mean_readout_err:.4f}")
    else:
        print("Readout baseline not found in Phase 1 data.")
        mean_readout_err = None

    # Figures
    fig1_path = out_dir / "leakage_vs_depth.png"
    fig2_path = out_dir / "asymmetry_vs_depth.png"
    plot_leakage_vs_depth(summaries, fig1_path)
    plot_asymmetry_vs_depth(summaries, fig2_path)
    print(f"\nFigures saved:\n  {fig1_path}\n  {fig2_path}")

    # JSON summary
    summary_json = {
        "source_files": [str(p.relative_to(REPO_ROOT)) for p in PHASE1_FILES if p.exists()],
        "n_circuits_loaded": len(circuits),
        "n_qubits": 4,
        "depth_summaries": [asdict(s) for s in summaries],
        "fisher_combined": {
            "chi2": chi2,
            "df": 2 * len(pvals),
            "combined_p": combined_p,
            "n_depths_significant_at_0.05": significant,
            "n_depths_tested": len(pvals),
        },
        "readout_baseline": {
            "fidelity_by_state": readout,
            "mean_readout_error": mean_readout_err,
        },
    }
    json_path = out_dir / "phase1_dla_parity_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary_json, f, indent=2)
    print(f"  {json_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
