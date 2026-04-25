#!/usr/bin/env python3
import asyncio
import json

import numpy as np

from scpn_quantum_control.analysis import DLAParityWitness, SyncOrderParameter
from scpn_quantum_control.control import StructuredAnsatz
from scpn_quantum_control.hardware import AsyncHardwareRunner


# Mock bridge load
def load_live_stream(source, step):
    return np.random.rand(12, 12), np.random.rand(12)


async def run_live_scneurocore():
    runner = AsyncHardwareRunner(
        backend="ibm_heron_r2", shots=8000, mitigation="GUESS", real_time_feedback=True
    )

    results = []
    for step in range(50):
        # Live data from SC-NeuroCore
        K_nm, omega = load_live_stream(source="eeg_powergrid", step=step)

        ansatz = StructuredAnsatz.from_kuramoto(K_nm, omega, trotter_depth=6)
        job = runner.submit_circuit_batch(
            ansatz=ansatz,
            observable=[SyncOrderParameter(), DLAParityWitness()],
            feedback_map={"asymmetry_threshold": 0.08},
            job_tags=["live_scneurocore", f"step_{step}"],
        )
        result = await job.result()
        results.append(result)

    with open("results/live_scneurocore_loop.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    asyncio.run(run_live_scneurocore())

# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Frontier Campaign Tests (Batch 4)
