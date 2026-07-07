# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Topology-controlled QSNN demo
"""Persistent-H1 topology control of QSNN recurrent coupling.

Runs the quantum neuromorphic bridge twice on the same deterministic input
train — once free-running, once with a ``TopologicalDynamicCouplingPolicy``
projecting the recurrent weights every few steps — and reports the H1 cycle
count of the final coupling graph for both runs.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from scpn_quantum_control.qsnn.quantum_neuromorphic_bridge import QuantumNeuromorphicBridge
from scpn_quantum_control.topology_control import (
    CouplingTopologyObjective,
    NetworkCycleBackend,
    ProjectedSPSAOptimizer,
    TopologicalDynamicCouplingPolicy,
    TopologyConstraintLedger,
    build_coupling_distance_matrix,
)


def _h1_cycles(backend: NetworkCycleBackend, weights: NDArray[np.float64]) -> int:
    summary = backend.compute(build_coupling_distance_matrix(weights))
    return summary.n_h1_persistent


def main() -> None:
    print("Topology-controlled QSNN demo")
    print("=" * 50)

    n_inputs, n_neurons, steps = 3, 8, 40
    rng = np.random.default_rng(11)
    currents = rng.integers(0, 2, size=(steps, n_inputs)).astype(np.float64)

    backend = NetworkCycleBackend(threshold=0.02)
    objective = CouplingTopologyObjective(
        ph_backend=backend,
        ledger=TopologyConstraintLedger(),
        h1_target=1.0,
        allow_approximate_ph_backend=True,
    )
    policy = TopologicalDynamicCouplingPolicy(
        objective=objective,
        optimizer=ProjectedSPSAOptimizer(seed=5, max_steps=8),
    )

    free = QuantumNeuromorphicBridge(n_inputs, n_neurons, seed=11)
    controlled = QuantumNeuromorphicBridge(
        n_inputs,
        n_neurons,
        seed=11,
        topology_policy=policy,
        topology_policy_interval=5,
    )

    for row in currents:
        free.step(row)
        controlled.step(row)

    print(f"steps: {steps}, neurons: {n_neurons}, policy interval: 5")
    print(f"free-running H1 cycles:        {_h1_cycles(backend, free.recurrent_weights)}")
    print(f"topology-controlled H1 cycles: {_h1_cycles(backend, controlled.recurrent_weights)}")
    if policy.last_trace is not None:
        print(f"last optimisation steps:       {len(policy.last_trace.steps)}")
    print("claim boundary: simulator demonstration; no hardware or biological claims.")


if __name__ == "__main__":
    main()
