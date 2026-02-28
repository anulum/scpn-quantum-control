"""Plot VQE convergence for 3 ansatze + final energy gap bar chart (Figure 2).

Produces: figures/vqe_convergence.png
Requires: matplotlib (pip install matplotlib)
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np

FIGURES_DIR = pathlib.Path(__file__).resolve().parent.parent / "figures"


def main():
    from scpn_quantum_control.hardware.classical import classical_exact_diag
    from scpn_quantum_control.phase.ansatz_bench import run_ansatz_benchmark

    n = 4
    exact = classical_exact_diag(n)
    e_exact = exact["ground_energy"]

    print(f"Exact ground energy ({n}q): {e_exact:.6f}")
    print("Running ansatz benchmark (this may take a minute)...")

    results = run_ansatz_benchmark(n_qubits=n, maxiter=300, reps=2)

    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    colors = {"knm_informed": "#2ca02c", "two_local": "#ff7f0e", "efficient_su2": "#1f77b4"}
    labels = {"knm_informed": "Knm-informed", "two_local": "TwoLocal", "efficient_su2": "EfficientSU2"}

    for r in results:
        name = r["ansatz"]
        ax1.plot(
            r["history"],
            color=colors[name],
            label=f"{labels[name]} ({r['n_params']}p)",
            alpha=0.8,
        )

    ax1.axhline(e_exact, ls="--", color="black", lw=1, alpha=0.6, label=f"Exact = {e_exact:.4f}")
    ax1.set_xlabel("VQE Iteration")
    ax1.set_ylabel("Energy")
    ax1.set_title(f"VQE Convergence ({n}-qubit Kuramoto XY)")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.2)

    # Bar chart: final energy gap
    names = [labels[r["ansatz"]] for r in results]
    gaps = [abs(r["energy"] - e_exact) for r in results]
    bar_colors = [colors[r["ansatz"]] for r in results]

    bars = ax2.bar(names, gaps, color=bar_colors, edgecolor="black", linewidth=0.6)
    for bar, gap in zip(bars, gaps):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            gap + max(gaps) * 0.02,
            f"{gap:.4f}",
            ha="center",
            fontsize=10,
        )

    ax2.set_ylabel("|E_VQE - E_exact|")
    ax2.set_title("Final Energy Gap")
    ax2.grid(True, alpha=0.2, axis="y")

    fig.suptitle("Ansatz Comparison — Kuramoto XY Hamiltonian", fontsize=14, y=1.02)
    fig.tight_layout()

    FIGURES_DIR.mkdir(exist_ok=True)
    out = FIGURES_DIR / "vqe_convergence.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
