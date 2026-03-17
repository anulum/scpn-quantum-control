# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Generate publication figures for v1.0 modules."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

FIGURES_DIR = Path(__file__).parent


def pec_overhead_figure() -> None:
    """PEC sampling overhead vs gate count at various error rates."""
    from scpn_quantum_control.mitigation.pec import pauli_twirl_decompose

    fig, ax = plt.subplots(figsize=(7, 4.5))
    gate_counts = np.arange(1, 35)
    colors = ["#2ca02c", "#ff7f0e", "#d62728", "#9467bd"]
    for p, color in zip([0.005, 0.01, 0.02, 0.05], colors):
        coeffs = pauli_twirl_decompose(p)
        gamma = float(np.sum(np.abs(coeffs)))
        overhead = gamma**gate_counts
        ax.semilogy(gate_counts, overhead, "-o", color=color, label=f"p = {p}", markersize=3)

    ax.set_xlabel("Number of gates", fontsize=11)
    ax.set_ylabel("Sampling overhead γⁿ", fontsize=11)
    ax.set_title("PEC Overhead Scaling", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1, 34)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "pec_overhead.png", dpi=200)
    plt.close(fig)
    print("  pec_overhead.png")


def quantum_advantage_figure() -> None:
    """Classical vs quantum wall-clock scaling."""
    from scpn_quantum_control.benchmarks.quantum_advantage import run_scaling_benchmark

    results = run_scaling_benchmark(sizes=[4, 6, 8, 10, 12], t_max=0.3, dt=0.1)
    ns = [r.n_qubits for r in results]
    t_c = [r.t_classical_ms for r in results]
    t_q = [r.t_quantum_ms for r in results]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.semilogy(
        ns, t_c, "o-", color="#d62728", label="Classical (expm)", linewidth=2, markersize=7
    )
    ax.semilogy(
        ns, t_q, "s-", color="#1f77b4", label="Quantum (Trotter)", linewidth=2, markersize=7
    )
    ax.set_xlabel("System size (qubits)", fontsize=11)
    ax.set_ylabel("Wall-clock time (ms)", fontsize=11)
    ax.set_title("Kuramoto XY: Classical vs Quantum Scaling", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "quantum_advantage_scaling.png", dpi=200)
    plt.close(fig)
    print("  quantum_advantage_scaling.png")


def identity_topology_figure() -> None:
    """18-oscillator identity binding spec K_nm heatmap."""
    from scpn_quantum_control.identity.binding_spec import (
        ARCANE_SAPIENCE_SPEC,
        _build_knm_from_spec,
    )

    K, omega = _build_knm_from_spec(ARCANE_SAPIENCE_SPEC)
    layers = ARCANE_SAPIENCE_SPEC["layers"]
    layer_names = [lay["name"] for lay in layers]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    im = ax1.imshow(K, cmap="YlOrRd", aspect="equal")
    ax1.set_title("Identity K_nm (18×18)", fontsize=12)
    ax1.set_xlabel("Oscillator index")
    ax1.set_ylabel("Oscillator index")
    plt.colorbar(im, ax=ax1, shrink=0.8)

    boundaries = [0]
    for lay in layers:
        boundaries.append(boundaries[-1] + len(lay["oscillator_ids"]))
    for b in boundaries[1:-1]:
        ax1.axhline(b - 0.5, color="white", linewidth=0.5)
        ax1.axvline(b - 0.5, color="white", linewidth=0.5)

    ax2.bar(range(18), omega, color="#1f77b4", edgecolor="black", linewidth=0.3)
    ax2.set_xlabel("Oscillator index", fontsize=11)
    ax2.set_ylabel("ω (rad/s)", fontsize=11)
    ax2.set_title("Natural Frequencies", fontsize=12)
    for i, name in enumerate(layer_names):
        center = boundaries[i] + (boundaries[i + 1] - boundaries[i]) / 2
        ax2.text(center, max(omega) * 1.05, name, ha="center", fontsize=7, rotation=30)

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "identity_topology.png", dpi=200)
    plt.close(fig)
    print("  identity_topology.png")


def surface_code_budget_figure() -> None:
    """Physical qubit budget for surface-code UPDE."""
    from scpn_quantum_control.qec.surface_code_upde import SurfaceCodeUPDE

    distances = [3, 5, 7, 9]
    osc_counts = [2, 4, 8, 16]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    for n_osc, color in zip(osc_counts, colors):
        budgets = []
        for d in distances:
            sc = SurfaceCodeUPDE(n_osc=n_osc, code_distance=d)
            budgets.append(sc.total_qubits)
        ax.plot(
            distances,
            budgets,
            "o-",
            color=color,
            label=f"N={n_osc} osc",
            linewidth=2,
            markersize=7,
        )

    ax.set_xlabel("Code distance d", fontsize=11)
    ax.set_ylabel("Physical qubits", fontsize=11)
    ax.set_title("Surface-Code UPDE: Qubit Budget", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "surface_code_budget.png", dpi=200)
    plt.close(fig)
    print("  surface_code_budget.png")


def main() -> None:
    print("Generating v1.0 figures...")
    pec_overhead_figure()
    quantum_advantage_figure()
    identity_topology_figure()
    surface_code_budget_figure()
    print("Done.")


if __name__ == "__main__":
    main()
