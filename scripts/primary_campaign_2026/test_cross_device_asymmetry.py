#!/usr/bin/env python3
import asyncio
import json

import numpy as np

from scpn_quantum_control.analysis import DLAParityWitness
from scpn_quantum_control.control import StructuredAnsatz
from scpn_quantum_control.hardware import AsyncHardwareRunner


async def run_test():
    backends = ["ibm_fez", "ibm_kingston"]
    results = {}
    K_nm = np.load("params/primary_Knm_12x12.npy")
    omega_vector = np.load("params/primary_omega.npy")
    params = {"n_qubits": 12, "trotter_depth": 6, "K_nm": K_nm, "omega": omega_vector}

    jobs = []
    runners = []
    for backend in backends:
        runner = AsyncHardwareRunner(backend=backend, shots=15000, seed=42)
        runners.append(runner)
        job = runner.submit_circuit_batch(
            ansatz=StructuredAnsatz.from_kuramoto(**params),
            observable=DLAParityWitness(),
            **params,
        )
        jobs.append(job)

    res = await asyncio.gather(*(j.result() for j in jobs))
    for b, r in zip(backends, res):
        results[b] = r

    with open("results/cross_device_asymmetry.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Cross-device verification test completed.")


if __name__ == "__main__":
    asyncio.run(run_test())

# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Primary Campaign Tests
