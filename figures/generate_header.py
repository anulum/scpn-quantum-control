"""Generate the 1280x640 repository header image.

Renders a Quantum-Logic Map: stylized Bloch sphere trajectory
sampled into SCPN spike rasters, with quantum interference background.
"""

import matplotlib.pyplot as plt
import numpy as np

WIDTH, HEIGHT = 12.8, 6.4
DPI = 100


def generate_quantum_control_header():
    fig = plt.figure(figsize=(WIDTH, HEIGHT), dpi=DPI, facecolor="#050510")
    ax = fig.add_axes([0, 0, 1, 1], frameon=False)
    ax.set_xlim(0, WIDTH)
    ax.set_ylim(0, HEIGHT)
    ax.set_xticks([])
    ax.set_yticks([])

    # Background: quantum wave interference pattern
    x_wave = np.linspace(0, WIDTH, 400)
    y_wave = np.linspace(0, HEIGHT, 200)
    X, Y = np.meshgrid(x_wave, y_wave)
    Z = (
        np.sin(X * 2)
        * np.cos(Y * 1.5)
        * np.exp(-0.1 * ((X - WIDTH / 2) ** 2 + (Y - HEIGHT / 2) ** 2))
    )
    ax.imshow(
        Z,
        extent=[0, WIDTH, 0, HEIGHT],
        origin="lower",
        cmap="PuBu",
        alpha=0.15,
        aspect="auto",
    )

    # Central visual: projected Bloch sphere
    center_q = (3.5, 3.2)
    radius = 1.8
    circle = plt.Circle(center_q, radius, color="#00d4ff", fill=False, lw=1.5, alpha=0.5)
    ax.add_artist(circle)
    ax.plot(
        [center_q[0] - radius, center_q[0] + radius],
        [center_q[1], center_q[1]],
        color="white",
        alpha=0.2,
        lw=0.5,
    )
    ax.plot(
        [center_q[0], center_q[0]],
        [center_q[1] - radius, center_q[1] + radius],
        color="white",
        alpha=0.2,
        lw=0.5,
    )
    ax.arrow(
        center_q[0],
        center_q[1],
        radius * 0.7,
        radius * 0.5,
        head_width=0.1,
        color="#00d4ff",
        alpha=0.8,
    )

    # SCPN spike rasters (quantum → classical transition)
    rng = np.random.default_rng(42)
    spike_x = np.linspace(5.5, 11.5, 60)
    for i in range(3):
        y_base = 2.0 + i * 1.2
        probs = 0.5 + 0.4 * np.sin(spike_x * 0.8 + i)
        for j, x in enumerate(spike_x):
            if rng.random() < probs[j]:
                ax.plot([x, x], [y_base, y_base + 0.4], color="#ffd700", lw=1.5, alpha=0.6)
        ax.plot(
            [center_q[0] + radius, 5.5],
            [center_q[1], y_base + 0.2],
            color="#00d4ff",
            alpha=0.1,
            lw=1,
        )

    # Branding text
    ax.text(
        1.0,
        5.4,
        "SCPN-QUANTUM-CONTROL",
        color="#ffffff",
        fontsize=34,
        fontweight="bold",
        fontname="monospace",
        alpha=0.9,
    )
    ax.text(
        1.0,
        4.9,
        "KURAMOTO / XY HAMILTONIAN // IBM HERON R2 HARDWARE",
        color="#ffd700",
        fontsize=13,
        fontname="monospace",
        alpha=0.7,
    )
    ax.text(
        1.0,
        4.5,
        "v0.7.0 | 456 TESTS | VQE 0.05% ERROR | 16-LAYER UPDE",
        color="#00d4ff",
        fontsize=10,
        fontname="monospace",
        alpha=0.5,
    )

    # Grid accents
    for x in np.arange(0, WIDTH, 2):
        ax.axvline(x, color="white", alpha=0.03, lw=0.5)

    out = "figures/header.png"
    plt.savefig(out, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    print(f"Generated {out} ({int(WIDTH * DPI)}x{int(HEIGHT * DPI)})")


if __name__ == "__main__":
    generate_quantum_control_header()
