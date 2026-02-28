"""Plot the 12-point decoherence scaling curve from IBM Heron r2 hardware runs.

Produces: figures/decoherence_curve.png
Requires: matplotlib (pip install matplotlib)
"""
from __future__ import annotations

import json
import pathlib

import numpy as np

RESULTS_DIR = pathlib.Path(__file__).resolve().parent.parent / "results"
FIGURES_DIR = pathlib.Path(__file__).resolve().parent.parent / "figures"

# Master decoherence data: (depth, hw_R, exact_R_or_nan, n_qubits, label)
DECOHERENCE_DATA = [
    (5, 0.8054, 0.8060, 4, "noise baseline"),
    (13, 0.7369, np.nan, 2, "2-osc minimal"),
    (25, 0.7727, np.nan, 4, "depth-25 probe"),
    (85, 0.7427, 0.8015, 4, "1 Trotter rep"),
    (147, 0.4822, 0.5317, 6, "6-osc"),
    (149, 0.6662, 0.8015, 4, "2 Trotter reps"),
    (233, 0.4648, 0.5816, 8, "8-osc"),
    (290, 0.6252, 0.8015, 4, "4 Trotter reps"),
    (395, 0.4224, 0.6417, 10, "10-osc"),
    (469, 0.3574, 0.5644, 12, "12-osc"),
    (747, 0.3814, 0.60, 14, "14-osc"),
    (770, 0.3321, 0.56, 16, "UPDE-16"),
]

# Points where both hw and exact are available (for error %)
ERROR_DATA = [
    (d, 100.0 * abs(hw - ex) / ex, n, lab)
    for d, hw, ex, n, lab in DECOHERENCE_DATA
    if not np.isnan(ex)
]


def fit_exponential_decay(depths, errors):
    """Fit error(%) = a * (1 - exp(-gamma * depth)) + c."""
    from scipy.optimize import curve_fit

    def model(d, a, gamma, c):
        return a * (1.0 - np.exp(-gamma * np.asarray(d))) + c

    depths_arr = np.asarray(depths, dtype=float)
    errors_arr = np.asarray(errors, dtype=float)
    try:
        popt, _ = curve_fit(
            model, depths_arr, errors_arr,
            p0=[50.0, 0.005, 0.0],
            bounds=([0, 0, -5], [100, 0.1, 10]),
            maxfev=5000,
        )
        return popt, model
    except Exception:
        return None, None


def plot_decoherence():
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    fig, ax = plt.subplots(figsize=(10, 6))

    # Regime backgrounds
    ax.axvspan(0, 150, alpha=0.08, color="green", label="Publishable (< 10%)")
    ax.axvspan(150, 400, alpha=0.08, color="orange", label="Mitigable (15-35%)")
    ax.axvspan(400, 850, alpha=0.08, color="red", label="Qualitative (> 35%)")

    # Error data points
    depths = [d for d, e, n, _ in ERROR_DATA]
    errors = [e for d, e, n, _ in ERROR_DATA]
    n_qubits = [n for d, e, n, _ in ERROR_DATA]

    scatter = ax.scatter(
        depths, errors,
        c=n_qubits, cmap="viridis", s=100, edgecolors="black",
        linewidths=0.8, zorder=5,
    )
    cbar = plt.colorbar(scatter, ax=ax, label="Qubits", pad=0.02)

    # Labels for key points
    annotations = {
        5: ("0.1%\nreadout only", (-30, 15)),
        85: ("7.3%\n1 Trotter", (10, 10)),
        149: ("16.9%\n2 Trotter", (10, -20)),
        290: ("22.0%\n4 Trotter", (10, 5)),
        770: ("46%\nUPDE-16", (-80, 10)),
    }
    for d, e, n, lab in ERROR_DATA:
        if d in annotations:
            text, offset = annotations[d]
            ax.annotate(
                text, (d, e), textcoords="offset points", xytext=offset,
                fontsize=7.5, ha="left", va="bottom",
                arrowprops=dict(arrowstyle="-", color="gray", lw=0.5),
            )

    # Exponential fit
    popt, model = fit_exponential_decay(depths, errors)
    if popt is not None:
        d_fit = np.linspace(1, 800, 300)
        e_fit = model(d_fit, *popt)
        ax.plot(
            d_fit, e_fit, "--", color="gray", alpha=0.6, lw=1.5,
            label=f"Fit: a(1-exp(-{popt[1]:.4f}d))+{popt[2]:.1f}",
        )

    # Regime boundary lines
    ax.axvline(150, color="green", ls=":", lw=0.8, alpha=0.5)
    ax.axvline(400, color="orange", ls=":", lw=0.8, alpha=0.5)

    ax.set_xlabel("Transpiled Circuit Depth", fontsize=12)
    ax.set_ylabel("Relative Error vs Exact (%)", fontsize=12)
    ax.set_title(
        "Decoherence Scaling on IBM Heron r2 (ibm_fez)\n"
        "Kuramoto XY Hamiltonian, order parameter R, Feb 2026",
        fontsize=13,
    )
    ax.set_xlim(-10, 830)
    ax.set_ylim(-2, 55)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.2)

    FIGURES_DIR.mkdir(exist_ok=True)
    out = FIGURES_DIR / "decoherence_curve.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close(fig)


def plot_layer_coherence():
    """Plot per-layer |<X>| vs Knm row sum for the 16-layer UPDE snapshot."""
    import matplotlib.pyplot as plt

    snapshot_path = RESULTS_DIR / "hw_upde_16_snapshot.json"
    with open(snapshot_path) as f:
        data = json.load(f)

    exp_x = np.array(data["exp_x"])
    abs_x = np.abs(exp_x)

    # Build Knm and compute row sums
    from scpn_quantum_control.bridge.knm_hamiltonian import build_knm_paper27
    K = build_knm_paper27(16)
    row_sums = K.sum(axis=1)

    # Spearman correlation
    from scipy.stats import spearmanr
    rho, pval = spearmanr(row_sums, abs_x)

    fig, ax = plt.subplots(figsize=(9, 6))
    labels = [f"L{i+1}" for i in range(16)]

    scatter = ax.scatter(
        row_sums, abs_x, s=120, c=abs_x, cmap="RdYlGn",
        edgecolors="black", linewidths=0.8, zorder=5, vmin=0, vmax=0.7,
    )

    for i, lbl in enumerate(labels):
        offset_x, offset_y = 0.03, 0.015
        if lbl == "L12":
            offset_x, offset_y = 0.03, -0.03
        elif lbl == "L16":
            offset_x, offset_y = -0.15, 0.02
        ax.annotate(lbl, (row_sums[i], abs_x[i]),
                     xytext=(row_sums[i] + offset_x, abs_x[i] + offset_y),
                     fontsize=8, fontweight="bold")

    # Highlight L12 (weakest) and L3 (strongest)
    for idx, color, name in [(11, "red", "L12: weakest coupling"), (2, "green", "L3: strongest coupling")]:
        ax.scatter(
            [row_sums[idx]], [abs_x[idx]], s=250, facecolors="none",
            edgecolors=color, linewidths=2.5, zorder=6, label=name,
        )

    ax.set_xlabel("Knm Row Sum (coupling strength)", fontsize=12)
    ax.set_ylabel("|<X>| (qubit coherence)", fontsize=12)
    ax.set_title(
        f"Per-Layer Coherence vs Coupling Strength (UPDE-16, ibm_fez)\n"
        f"Spearman rho = {rho:.3f}, p = {pval:.4f}",
        fontsize=13,
    )
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.2)

    FIGURES_DIR.mkdir(exist_ok=True)
    out = FIGURES_DIR / "layer_coherence_vs_coupling.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved: {out}")
    print(f"Spearman rho = {rho:.4f}, p-value = {pval:.6f}")
    plt.close(fig)


def plot_trotter_tradeoff():
    """Plot Trotter reps vs error showing diminishing returns on NISQ."""
    import matplotlib.pyplot as plt

    reps = [1, 2, 4]
    depths = [85, 149, 290]
    hw_R = [0.7427, 0.6662, 0.6252]
    exact_R = 0.8015
    errors = [100.0 * abs(r - exact_R) / exact_R for r in hw_R]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left: R vs depth
    ax1.plot(depths, hw_R, "o-", color="#d62728", markersize=10, lw=2, label="Hardware R")
    ax1.axhline(exact_R, ls="--", color="black", lw=1, alpha=0.6, label=f"Exact R = {exact_R}")
    for i, (d, r, rep) in enumerate(zip(depths, hw_R, reps)):
        ax1.annotate(f"{rep} rep{'s' if rep > 1 else ''}", (d, r),
                      textcoords="offset points", xytext=(10, 5), fontsize=9)
    ax1.set_xlabel("Circuit Depth", fontsize=11)
    ax1.set_ylabel("Order Parameter R", fontsize=11)
    ax1.set_title("More Trotter Reps = Worse on NISQ", fontsize=12)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.2)

    # Right: error vs reps
    ax2.bar(reps, errors, color=["#2ca02c", "#ff7f0e", "#d62728"], edgecolor="black", width=0.6)
    for i, (rep, err) in enumerate(zip(reps, errors)):
        ax2.text(rep, err + 0.5, f"{err:.1f}%", ha="center", fontsize=10, fontweight="bold")
    ax2.set_xlabel("Trotter Reps", fontsize=11)
    ax2.set_ylabel("Relative Error (%)", fontsize=11)
    ax2.set_title("Error Grows with Depth (4-osc, t=0.1)", fontsize=12)
    ax2.set_xticks(reps)
    ax2.grid(True, alpha=0.2, axis="y")

    fig.suptitle("Trotter Depth Tradeoff — ibm_fez Heron r2", fontsize=14, y=1.02)
    fig.tight_layout()

    FIGURES_DIR.mkdir(exist_ok=True)
    out = FIGURES_DIR / "trotter_tradeoff.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close(fig)


def plot_upde16_bars():
    """Bar chart of per-layer |<X>| for the 16-layer UPDE snapshot."""
    import matplotlib.pyplot as plt

    snapshot_path = RESULTS_DIR / "hw_upde_16_snapshot.json"
    with open(snapshot_path) as f:
        data = json.load(f)

    abs_x = np.abs(np.array(data["exp_x"]))
    layers = [f"L{i+1}" for i in range(16)]

    # Color by decoherence severity
    colors = []
    for v in abs_x:
        if v >= 0.5:
            colors.append("#2ca02c")   # strong
        elif v >= 0.3:
            colors.append("#ff7f0e")   # moderate
        elif v >= 0.1:
            colors.append("#d62728")   # weak
        else:
            colors.append("#7f7f7f")   # near-dead

    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(layers, abs_x, color=colors, edgecolor="black", linewidth=0.6)

    for bar, v in zip(bars, abs_x):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01,
                f"{v:.2f}", ha="center", fontsize=8)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#2ca02c", edgecolor="black", label="Strong (>0.5)"),
        Patch(facecolor="#ff7f0e", edgecolor="black", label="Moderate (0.3-0.5)"),
        Patch(facecolor="#d62728", edgecolor="black", label="Weak (0.1-0.3)"),
        Patch(facecolor="#7f7f7f", edgecolor="black", label="Near-dead (<0.1)"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=9)

    ax.set_xlabel("SCPN Layer", fontsize=12)
    ax.set_ylabel("|<X>| Expectation", fontsize=12)
    ax.set_title(
        "UPDE-16 Per-Layer Coherence on IBM Heron r2\n"
        "dt=0.05, depth 770, 20k shots",
        fontsize=13,
    )
    ax.set_ylim(0, 0.75)
    ax.grid(True, alpha=0.2, axis="y")

    FIGURES_DIR.mkdir(exist_ok=True)
    out = FIGURES_DIR / "upde16_layer_bars.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close(fig)


if __name__ == "__main__":
    plot_decoherence()
    plot_layer_coherence()
    plot_trotter_tradeoff()
    plot_upde16_bars()
    print("\nAll figures generated in figures/")
