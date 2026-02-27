"""Kuramoto -> XY Hamiltonian demo: 4-oscillator synchronization."""
import numpy as np

from scpn_quantum_control.phase import QuantumKuramotoSolver
from scpn_quantum_control.bridge import OMEGA_N_16, build_knm_paper27


def main():
    n = 4
    K = build_knm_paper27(L=n)
    omega = OMEGA_N_16[:n]

    print(f"Oscillators: {n}")
    print(f"Frequencies: {omega}")
    print(f"Coupling K[0,1]={K[0,1]:.3f}, K[1,2]={K[1,2]:.3f}")

    solver = QuantumKuramotoSolver(n, K, omega)
    result = solver.run(t_max=1.0, dt=0.2, trotter_per_step=3)

    print("\nTime | R (order parameter)")
    print("-" * 30)
    for t, R in zip(result["times"], result["R"]):
        bar = "#" * int(R * 20)
        print(f"{t:5.2f} | {R:.4f} {bar}")


if __name__ == "__main__":
    main()
