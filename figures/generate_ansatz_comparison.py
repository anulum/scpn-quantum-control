# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Generate VQE ansatz comparison figure for publication (Gemini finding 7.1)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from scpn_quantum_control.bridge.knm_hamiltonian import OMEGA_N_16, build_knm_paper27
from scpn_quantum_control.hardware.classical import classical_exact_diag
from scpn_quantum_control.phase.ansatz_bench import run_ansatz_benchmark

FIGURES_DIR = Path(__file__).parent
LABELS = {
    "knm_informed": "K_nm-informed",
    "two_local": "TwoLocal (linear)",
    "efficient_su2": "EfficientSU2",
}
COLORS = {
    "knm_informed": "#1f77b4",
    "two_local": "#ff7f0e",
    "efficient_su2": "#2ca02c",
}


def main() -> None:
    K = build_knm_paper27(L=4)
    omega = OMEGA_N_16[:4]
    exact = classical_exact_diag(4, K=K, omega=omega)
    exact_e = exact["ground_energy"]

    print("Running ansatz benchmark (3 ansatze x 200 iterations)...")
    results = run_ansatz_benchmark(n_qubits=4, maxiter=200)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    names = []
    errors = []
    n_params_list = []
    for r in results:
        name = r["ansatz"]
        vqe_e = r["energy"]
        err_pct = abs(vqe_e - exact_e) / abs(exact_e) * 100
        n_p = r["n_params"]
        names.append(LABELS.get(name, name))
        errors.append(err_pct)
        n_params_list.append(n_p)
        print(
            f"  {LABELS.get(name, name):>20}: E={vqe_e:.6f}, exact={exact_e:.6f}, err={err_pct:.4f}%, params={n_p}"
        )

    bars = ax1.bar(
        names,
        errors,
        color=[COLORS.get(r["ansatz"], "#999") for r in results],
        edgecolor="black",
        linewidth=0.5,
    )
    ax1.set_ylabel("Relative energy error (%)", fontsize=11)
    ax1.set_title("VQE Ansatz Comparison (4-qubit Kuramoto XY)", fontsize=12)
    ax1.set_ylim(0, max(errors) * 1.4)
    for bar, err, n_p in zip(bars, errors, n_params_list):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(errors) * 0.02,
            f"{err:.3f}%\n({n_p} params)",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax2.bar(
        names,
        n_params_list,
        color=[COLORS.get(r["ansatz"], "#999") for r in results],
        edgecolor="black",
        linewidth=0.5,
    )
    ax2.set_ylabel("Number of parameters", fontsize=11)
    ax2.set_title("Parameter Count", fontsize=12)
    for bar, n_p in zip(ax2.patches, n_params_list):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            str(n_p),
            ha="center",
            va="bottom",
            fontsize=10,
        )

    plt.tight_layout()
    out = FIGURES_DIR / "ansatz_comparison.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
