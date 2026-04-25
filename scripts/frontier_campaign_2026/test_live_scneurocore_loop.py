#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
import asyncio
import json

from scpn_quantum_control.analysis import DLAParityWitness, SyncOrderParameter
from scpn_quantum_control.control import StructuredAnsatz
from scpn_quantum_control.hardware import AsyncHardwareRunner
from scpneurocore.bridge import load_live_stream


async def run_live_scneurocore():
    runner = AsyncHardwareRunner(
        backend="ibm_heron_r2", shots=8000, mitigation="GUESS", real_time_feedback=True
    )
    results = []
    for step in range(50):
        K_nm, omega = load_live_stream(source="eeg_powergrid", step=step)
        ansatz = StructuredAnsatz.from_kuramoto(K_nm, omega, trotter_depth=6)
        job = runner.submit_circuit_batch(
            ansatz=ansatz, observable=[SyncOrderParameter(), DLAParityWitness()]
        )
        result = await job.result()
        results.append(result)
    with open("results/live_scneurocore_loop.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    asyncio.run(run_live_scneurocore())
