"""QAOA-MPC demo: binary coil control over 4 timesteps."""
import numpy as np

from scpn_quantum_control.control import QAOA_MPC


def main():
    B = np.eye(2)
    target = np.array([0.8, 0.6])
    horizon = 4

    print(f"Target state: {target}")
    print(f"Horizon: {horizon} timesteps")
    print(f"QAOA p-layers: 2\n")

    mpc = QAOA_MPC(B, target, horizon=horizon, p_layers=2)
    H = mpc.build_cost_hamiltonian()
    print(f"Cost Hamiltonian: {H.num_qubits} qubits, {len(H)} terms")

    actions = mpc.optimize()
    print(f"\nOptimal action sequence: {actions}")
    print(f"  (1 = coil ON, 0 = coil OFF)")


if __name__ == "__main__":
    main()
