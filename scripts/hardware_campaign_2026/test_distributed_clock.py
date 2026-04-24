#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Hardware Campaign Tests

import asyncio
import json

import numpy as np

from scpn_quantum_control.analysis import DLAParityWitness, SyncOrderParameter
from scpn_quantum_control.control import StructuredAnsatz
from scpn_quantum_control.hardware import AsyncHardwareRunner


async def run_distributed_clock_test():
    runner_node1 = AsyncHardwareRunner(backend="ibm_heron_r2", shots=10000, mitigation="GUESS")
    runner_node2 = AsyncHardwareRunner(backend="ibm_heron_r2", shots=10000, mitigation="GUESS")
    K_nm = np.load("params/clock_network_16x16.npy")
    omega = np.load("params/clock_omega.npy")
    results = []
    for cycle in range(30):
        print(f"Clock sync cycle {cycle + 1}/30")
        ansatz = StructuredAnsatz.from_kuramoto(
            K_nm=K_nm, omega=omega, trotter_depth=6, mediated_couplings=True
        )
        job1 = runner_node1.submit_circuit_batch(
            ansatz=ansatz,
            observable=[SyncOrderParameter(), DLAParityWitness()],
            job_tags=["clock_node1", f"cycle_{cycle}"],
        )
        job2 = runner_node2.submit_circuit_batch(
            ansatz=ansatz,
            observable=[SyncOrderParameter(), DLAParityWitness()],
            job_tags=["clock_node2", f"cycle_{cycle}"],
        )
        res1, res2 = await asyncio.gather(job1.result(), job2.result())
        combined = {"cycle": cycle, "node1": res1, "node2": res2}
        results.append(combined)
    with open("results/distributed_clock_full.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Distributed clock synchronization test completed.")


if __name__ == "__main__":
    asyncio.run(run_distributed_clock_test())
