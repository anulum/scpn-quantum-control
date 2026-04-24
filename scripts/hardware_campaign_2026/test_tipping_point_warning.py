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

from scpn_quantum_control.analysis import OTOC, DLAParityWitness
from scpn_quantum_control.control import StructuredAnsatz
from scpn_quantum_control.hardware import AsyncHardwareRunner


async def run_tipping_point_test():
    runner = AsyncHardwareRunner(backend="ibm_heron_r2", shots=12000, mitigation="GUESS")
    K_nm = np.load("params/power_grid_europe_16x16.npy")
    omega = np.load("params/power_grid_omega.npy")
    results = []
    for step in range(25):
        ansatz = StructuredAnsatz.from_kuramoto(K_nm, omega, trotter_depth=8)
        job = runner.submit_circuit_batch(
            ansatz=ansatz,
            observable=[OTOC(), DLAParityWitness()],
            job_tags=["tipping_warning", f"step_{step}"],
        )
        result = await job.result()
        results.append(result)
        if result["otoc_tstar"] < 0.35 and result["dla_asymmetry"] > 0.09:
            print(f"!!! TIPPING POINT PRECURSOR DETECTED at step {step} !!!")
    with open("results/tipping_point_results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    asyncio.run(run_tipping_point_test())
