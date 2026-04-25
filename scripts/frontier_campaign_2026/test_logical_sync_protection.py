#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
import asyncio
import json

import numpy as np

from scpn_quantum_control.analysis import LogicalSyncWitness, SyncOrderParameter
from scpn_quantum_control.control import StructuredAnsatz
from scpn_quantum_control.hardware import AsyncHardwareRunner


async def run_logical_protection():
    runner = AsyncHardwareRunner(backend="ibm_heron_r2", shots=12000, mitigation="GUESS")
    K_nm = np.load("params/logical_Knm_12x12.npy")
    results = []
    for err in np.linspace(0.01, 0.15, 8):
        ansatz = StructuredAnsatz.from_kuramoto(K_nm, trotter_depth=8)
        job = runner.submit_circuit_batch(
            ansatz=ansatz,
            observable=[LogicalSyncWitness(), SyncOrderParameter()],
            error_injection_rate=err,
        )
        result = await job.result()
        results.append(result)

    with open("results/logical_sync_protection.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    asyncio.run(run_logical_protection())
