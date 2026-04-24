#!/usr/bin/env python3
import asyncio
import json

import numpy as np

from scpn_quantum_control.accel import rust_random_state
from scpn_quantum_control.analysis import DLAParityWitness
from scpn_quantum_control.control import StructuredAnsatz
from scpn_quantum_control.hardware import AsyncHardwareRunner


async def run_test():
    runner = AsyncHardwareRunner(backend="ibm_heron_r2", shots=10000, mitigation="GUESS")
    results = []
    K_nm = np.load("params/primary_Knm_12x12.npy")
    omega_vector = np.load("params/primary_omega.npy")
    params = {"n_qubits": 12, "trotter_depth": 6, "K_nm": K_nm, "omega": omega_vector}

    for i in range(50):
        random_state = rust_random_state(n_qubits=12, seed=i)
        job = runner.submit_circuit_batch(
            ansatz=StructuredAnsatz.from_kuramoto(**params),
            initial_state=random_state,
            observable=DLAParityWitness(),
            **params,
        )
        res = await job.result()
        results.append({"seed": i, "result": res})

    with open("results/initial_state_randomization.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Initial state randomization test completed.")


if __name__ == "__main__":
    asyncio.run(run_test())

# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Primary Campaign Tests
