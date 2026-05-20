# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""QSNN quantum neuromorphic bridge demo."""

from __future__ import annotations

import numpy as np

from scpn_quantum_control.qsnn import (
    DynamicCouplingConfig,
    QuantumLIFConfig,
    QuantumNeuromorphicBridge,
    TraceSTDPConfig,
)


def main() -> None:
    bridge = QuantumNeuromorphicBridge(
        n_inputs=2,
        n_neurons=3,
        lif=QuantumLIFConfig(v_threshold=0.35, tau_mem=3.0, dt=1.0),
        stdp=TraceSTDPConfig(a_plus=0.04, a_minus=0.025, tau_pre=6.0, tau_post=8.0),
        coupling=DynamicCouplingConfig(learning_rate=0.08, decay_rate=0.02, coherence_gain=0.5),
        input_weights=np.array([[0.7, 0.1], [0.1, 0.7], [0.4, 0.4]]),
        seed=42,
        deterministic=True,
    )

    currents = np.array(
        [
            [1.0, 0.1],
            [0.8, 0.2],
            [0.2, 1.0],
            [0.1, 0.9],
            [0.7, 0.7],
        ]
    )
    history = bridge.run(currents)

    print("step | spikes | mean P(spike) | recurrent sum")
    print("-" * 52)
    for idx, result in enumerate(history):
        print(
            f"{idx:4d} | {result.spikes.tolist()} | "
            f"{np.mean(result.spike_probabilities):13.4f} | "
            f"{np.sum(result.recurrent_weights):13.4f}"
        )

    print("\nFinal input weights:")
    print(np.round(history[-1].input_weights, 4))
    print("\nLatest quantum circuit:")
    print(bridge.get_circuit())
    print(f"\nClaim boundary: {history[-1].claim_boundary}")


if __name__ == "__main__":
    main()
