# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Generate VQE ansatz comparison figure for publication (Gemini finding 7.1)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from scpn_quantum_control.bridge.knm_hamiltonian import OMEGA_N_16, build_knm_paper27
from scpn_quantum_control.phase.ansatz_bench import run_ansatz_benchmark

FIGURES_DIR = Path(__file__).parent


def main() -> None:
    K = build_knm_paper27(L=4)
    omega = OMEGA_N_16[:4]

    print("Running ansatz benchmark (3 ansatze × 200 iterations)...")
    results = run_ansatz_benchmark(K, omega, maxiter=200, seed=42)

    exact_e = results[0]["exact_energy"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    colors = {"knm": "#1f77b4", "two_local": "#ff7f0e", "efficient_su2": "#2ca02c"}
    labels = {"knm": "K_nm-informed", "two_local": "TwoLocal (linear)", "efficient_su2": "EfficientSU2"}

    # Left: energy error comparison
    names = []
    errors = []
    n_params_list = []
    for r in results:
        name = r["ansatz"]
        err_pct = r["relative_error_pct"]
        n_p = r["n_params"]
        names.append(labels.get(name, name))
        errors.append(err_pct)
        n_params_list.append(n_p)
        print(f"  {labels.get(name, name):>20}: E={r['vqe_energy']:.6f}, err={err_pct:.4f}%, params={n_p}")

    bars = ax1.bar(names, errors, color=[colors.get(r["ansatz"], "#999") for r in results],
                   edgecolor="black", linewidth=0.5)
    ax1.set_ylabel("Relative energy error (%)", fontsize=11)
    ax1.set_title("VQE Ansatz Comparison (4-qubit Kuramoto XY)", fontsize=12)
    ax1.set_ylim(0, max(errors) * 1.3)
    for bar, err, n_p in zip(bars, errors, n_params_list):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                 f"{err:.3f}%\n({n_p} params)", ha="center", va="bottom", fontsize=9)

    # Right: parameter efficiency (error per parameter)
    efficiency = [e / n for e, n in zip(errors, n_params_list)]
    ax2.bar(names, n_params_list, color=[colors.get(r["ansatz"], "#999") for r in results],
            edgecolor="black", linewidth=0.5)
    ax2.set_ylabel("Number of parameters", fontsize=11)
    ax2.set_title("Parameter Count", fontsize=12)
    for bar, n_p in zip(ax2.patches, n_params_list):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                 str(n_p), ha="center", va="bottom", fontsize=10)

    plt.tight_layout()
    out = FIGURES_DIR / "ansatz_comparison.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
