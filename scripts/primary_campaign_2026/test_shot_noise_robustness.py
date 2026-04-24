#!/usr/bin/env python3
import asyncio
import json

import numpy as np

from scpn_quantum_control.analysis import DLAParityWitness
from scpn_quantum_control.control import StructuredAnsatz
from scpn_quantum_control.hardware import AsyncHardwareRunner


async def run_test():
    base_shots = 5000
    K_nm = np.load("params/primary_Knm_12x12.npy")
    omega = np.load("params/primary_omega.npy")
    params = {"n_qubits": 12, "trotter_depth": 6, "K_nm": K_nm, "omega": omega}

    results = {}
    for multiplier in [1, 2, 4, 8]:
        runner = AsyncHardwareRunner(
            backend="ibm_heron_r2", shots=base_shots * multiplier, mitigation="GUESS"
        )
        job = runner.submit_circuit_batch(
            ansatz=StructuredAnsatz.from_kuramoto(**params),
            observable=DLAParityWitness(),
            **params,
        )
        results[multiplier] = await job.result()

    with open("results/shot_noise_robustness.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Shot-noise + sampling robustness test completed.")


if __name__ == "__main__":
    asyncio.run(run_test())

# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Primary Campaign Tests
