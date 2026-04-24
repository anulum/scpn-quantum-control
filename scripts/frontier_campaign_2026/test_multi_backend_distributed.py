#!/usr/bin/env python3
import asyncio
import json

import mock_injector  # noqa: F401
import numpy as np

from scpn_quantum_control.analysis import DLAParityWitness, SyncOrderParameter
from scpn_quantum_control.control import StructuredAnsatz
from scpn_quantum_control.hardware import AsyncHardwareRunner


async def run_multi_backend():
    runner1 = AsyncHardwareRunner(backend="ibm_heron_r2", shots=10000, mitigation="GUESS")
    runner2 = AsyncHardwareRunner(backend="ibm_heron_r2", shots=10000, mitigation="GUESS")

    K_nm = np.load("params/distributed_Knm_20x20.npy")
    results = []
    for cycle in range(40):
        ansatz = StructuredAnsatz.from_kuramoto(K_nm, mediated_couplings=True, trotter_depth=6)
        job1 = runner1.submit_circuit_batch(
            ansatz=ansatz, observable=[SyncOrderParameter(), DLAParityWitness()]
        )
        job2 = runner2.submit_circuit_batch(
            ansatz=ansatz, observable=[SyncOrderParameter(), DLAParityWitness()]
        )
        res1, res2 = await asyncio.gather(job1.result(), job2.result())
        results.append({"cycle": cycle, "node1": res1, "node2": res2})

    with open("results/multi_backend_distributed.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    asyncio.run(run_multi_backend())

# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Frontier Campaign Tests (Batch 4)
