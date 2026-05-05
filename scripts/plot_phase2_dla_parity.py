#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Phase 2 DLA parity figure generator
"""Generate Phase 2 DLA parity figures from tracked summary JSON."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
PHASE2_SUMMARY = (
    REPO_ROOT / "data" / "phase2_dla_parity" / "phase2_reduced_ag_summary_2026-05-05.json"
)
SCALING_SUMMARY = (
    REPO_ROOT / "data" / "phase2_scaling_bc" / "phase2_scaling_bc_summary_2026-05-05.json"
)
POPCOUNT_SUMMARY = (
    REPO_ROOT
    / "data"
    / "phase2_popcount_control"
    / "phase2_popcount_control_summary_2026-05-05.json"
)
OUT_DIR = REPO_ROOT / "figures" / "phase2"


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _plot_n4(summary: dict[str, Any]) -> None:
    rows = summary["depth_summaries"]
    depths = np.asarray([row["depth"] for row in rows], dtype=float)
    asym = 100 * np.asarray([row["asymmetry_relative"] for row in rows], dtype=float)
    p_values = np.asarray([row["welch_p"] for row in rows], dtype=float)

    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    colours = np.where(p_values < 0.05, "#0b6e4f", "#8c8c8c")
    ax.axhline(0.0, color="#222222", linewidth=1.0)
    ax.fill_between([4, 20], 0, 9.6, color="#cde8d5", alpha=0.35, label="Phase 1 prediction band")
    ax.scatter(depths, asym, s=72, c=colours, edgecolor="#101010", linewidth=0.7, zorder=3)
    ax.plot(depths, asym, color="#124e78", linewidth=1.8, alpha=0.85)
    for depth, value, p_value in zip(depths, asym, p_values):
        if p_value < 0.05:
            ax.annotate(
                "*", (depth, value), xytext=(0, 9), textcoords="offset points", ha="center"
            )
    ax.set_title("Phase 2 reduced A+G: n=4 parity asymmetry replication")
    ax.set_xlabel("Trotter depth")
    ax.set_ylabel("Relative asymmetry A(d) [%]")
    ax.set_xticks(depths)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "phase2_n4_replication_asymmetry.png", dpi=180)
    fig.savefig(OUT_DIR / "phase2_n4_replication_asymmetry.pdf")
    plt.close(fig)


def _plot_scaling(summary: dict[str, Any]) -> None:
    rows = summary["depth_summaries"]
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    ax.axhline(0.0, color="#222222", linewidth=1.0)
    for n_qubits, colour, marker in [(6, "#b23a48", "o"), (8, "#1b6ca8", "s")]:
        group = [row for row in rows if row["n_qubits"] == n_qubits]
        depths = np.asarray([row["depth"] for row in group], dtype=float)
        asym = 100 * np.asarray([row["asymmetry_relative"] for row in group], dtype=float)
        p_values = np.asarray([row["welch_p"] for row in group], dtype=float)
        ax.plot(depths, asym, marker=marker, color=colour, linewidth=1.8, label=f"n={n_qubits}")
        for depth, value, p_value in zip(depths, asym, p_values):
            if p_value < 0.05:
                ax.annotate(
                    "*", (depth, value), xytext=(0, 9), textcoords="offset points", ha="center"
                )
    ax.set_title("Phase 2 B-C scaling: mixed parity-asymmetry evidence")
    ax.set_xlabel("Trotter depth")
    ax.set_ylabel("Relative asymmetry A(d) [%]")
    ax.set_xticks([4, 8, 14, 20])
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "phase2_bc_scaling_mixed_asymmetry.png", dpi=180)
    fig.savefig(OUT_DIR / "phase2_bc_scaling_mixed_asymmetry.pdf")
    plt.close(fig)


def _plot_popcount(summary: dict[str, Any]) -> None:
    rows = summary["state_summaries"]
    styles = {
        "E0_original_even": ("#b23a48", "o", "E0 |0011>, even, k=2"),
        "E1_even_swap": ("#d98c2b", "^", "E1 |0101>, even, k=2"),
        "O0_original_odd": ("#1b6ca8", "s", "O0 |0001>, odd, k=1"),
        "O1_odd_swap": ("#4c956c", "D", "O1 |0010>, odd, k=1"),
        "O3_odd_high_excitation": ("#5f4bb6", "v", "O3 |0111>, odd, k=3"),
    }

    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    for label, (colour, marker, display) in styles.items():
        group = [row for row in rows if row["state_label"] == label]
        depths = np.asarray([row["depth"] for row in group], dtype=float)
        leakage = 100 * np.asarray([row["mean_parity_leakage"] for row in group], dtype=float)
        sem = 100 * np.asarray([row["sem_parity_leakage"] for row in group], dtype=float)
        ax.errorbar(
            depths,
            leakage,
            yerr=sem,
            marker=marker,
            color=colour,
            linewidth=1.7,
            capsize=3,
            label=display,
        )

    ax.set_title("Phase 2 popcount control: state-dependent parity leakage")
    ax.set_xlabel("Trotter depth")
    ax.set_ylabel("Parity leakage [%]")
    ax.set_xticks(sorted({row["depth"] for row in rows}))
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "phase2_popcount_control_leakage.png", dpi=180)
    fig.savefig(OUT_DIR / "phase2_popcount_control_leakage.pdf")
    plt.close(fig)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _plot_n4(_load(PHASE2_SUMMARY))
    _plot_scaling(_load(SCALING_SUMMARY))
    _plot_popcount(_load(POPCOUNT_SUMMARY))
    print(f"Wrote figures to {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
