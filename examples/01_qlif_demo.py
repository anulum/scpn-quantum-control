# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Quantum LIF neuron demo: compare firing rates with classical LIF."""

from scpn_quantum_control.qsnn import QuantumLIFNeuron


def main():
    neuron = QuantumLIFNeuron(v_rest=0.0, v_threshold=1.0, tau_mem=10.0, dt=1.0, n_shots=100)

    currents = [0.05, 0.1, 0.15, 0.2, 0.3]
    print("Input Current | Spikes/100 steps | Avg membrane V")
    print("-" * 55)

    for current in currents:
        neuron.reset()
        spikes = 0
        v_sum = 0.0
        n_steps = 100
        for _ in range(n_steps):
            spikes += neuron.step(current)
            v_sum += neuron.v
        print(f"  {current:12.3f} | {spikes:16d} | {v_sum / n_steps:15.4f}")

    print("\nLast circuit:")
    print(neuron.get_circuit())


if __name__ == "__main__":
    main()
