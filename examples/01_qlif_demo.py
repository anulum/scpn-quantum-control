"""Quantum LIF neuron demo: compare firing rates with classical LIF."""
import numpy as np

from scpn_quantum_control.qsnn import QuantumLIFNeuron


def main():
    neuron = QuantumLIFNeuron(
        v_rest=0.0, v_threshold=1.0, tau_mem=10.0, dt=1.0, n_shots=100
    )

    currents = [0.05, 0.1, 0.15, 0.2, 0.3]
    print("Input Current | Spikes/100 steps | Avg membrane V")
    print("-" * 55)

    for I in currents:
        neuron.reset()
        spikes = 0
        v_sum = 0.0
        n_steps = 100
        for _ in range(n_steps):
            spikes += neuron.step(I)
            v_sum += neuron.v
        print(f"  {I:12.3f} | {spikes:16d} | {v_sum / n_steps:15.4f}")

    print("\nLast circuit:")
    print(neuron.get_circuit())


if __name__ == "__main__":
    main()
