#!/usr/bin/env python3
import asyncio
import json

from scpn_quantum_control.hardware import AsyncHardwareRunner


async def run_test():
    runner = AsyncHardwareRunner(backend="ibm_heron_r2", shots=8000, mitigation="GUESS")
    topologies = ["line", "ring", "random", "scale_free"]
    results = {}
    for topo in topologies:
        job = runner.submit_circuit_batch(
            topology=topo, measure_fidelity=True, depths=list(range(50, 600, 50)), n_qubits=12
        )
        results[topo] = await job.result()

    with open("results/coherence_wall_scaling.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Coherence wall scaling law test completed.")


if __name__ == "__main__":
    asyncio.run(run_test())

# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Primary Campaign Tests
