"""Quantum Petri net demo: 3-place, 2-transition token evolution."""
import numpy as np

from scpn_quantum_control.control import QuantumPetriNet


def main():
    W_in = np.array([
        [0.8, 0.0, 0.0],
        [0.0, 0.6, 0.0],
    ])
    W_out = np.array([
        [0.0, 0.7],
        [0.0, 0.0],
        [0.5, 0.4],
    ])
    thresholds = np.array([0.5, 0.5])

    net = QuantumPetriNet(n_places=3, n_transitions=2, W_in=W_in, W_out=W_out, thresholds=thresholds)

    marking = np.array([0.8, 0.5, 0.1])
    print(f"Initial marking: {marking}")

    for step in range(5):
        marking = net.step(marking)
        print(f"Step {step + 1}: {marking.round(4)}")

    print("\nToken densities converge to a quantum superposition-averaged equilibrium.")


if __name__ == "__main__":
    main()
