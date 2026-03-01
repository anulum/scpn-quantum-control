"""Generate the 16x16 K_nm coupling matrix heatmap (Paper 27, Eq. 3)."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from scpn_quantum_control.bridge.knm_hamiltonian import build_knm_paper27


def generate_knm_heatmap():
    K = build_knm_paper27(L=16)

    fig, ax = plt.subplots(figsize=(8, 7), dpi=150)
    im = ax.imshow(K, cmap="inferno", origin="upper")
    cbar = fig.colorbar(im, ax=ax, shrink=0.82)
    cbar.set_label("Coupling strength")

    ax.set_xlabel("Layer j")
    ax.set_ylabel("Layer i")
    ax.set_title(r"$K_{nm}$ coupling matrix (Paper 27, Eq. 3)")
    ax.set_xticks(range(16))
    ax.set_yticks(range(16))
    ax.set_xticklabels([f"L{i + 1}" for i in range(16)], fontsize=7)
    ax.set_yticklabels([f"L{i + 1}" for i in range(16)], fontsize=7)

    # Annotate calibration anchors — Paper 27 Table 2
    anchors = {(0, 1): 0.302, (1, 2): 0.201, (2, 3): 0.252, (3, 4): 0.154}
    for (i, j), val in anchors.items():
        ax.text(
            j,
            i,
            f"{val:.3f}",
            ha="center",
            va="center",
            fontsize=6,
            color="white",
            fontweight="bold",
        )
        ax.text(
            i,
            j,
            f"{val:.3f}",
            ha="center",
            va="center",
            fontsize=6,
            color="white",
            fontweight="bold",
        )

    # Annotate cross-hierarchy boosts — Paper 27 S4.3
    boosts = {(0, 15): 0.05, (4, 6): 0.15}
    for (i, j), val in boosts.items():
        ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=6, color="cyan")
        ax.text(i, j, f"{val:.2f}", ha="center", va="center", fontsize=6, color="cyan")

    out = Path(__file__).parent / "knm_heatmap.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Generated {out}")


if __name__ == "__main__":
    generate_knm_heatmap()
