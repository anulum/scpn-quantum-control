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

from scpn_quantum_control.analysis import SyncOrderParameter, ThermodynamicWitness
from scpn_quantum_control.control import StructuredAnsatz
from scpn_quantum_control.hardware import AsyncHardwareRunner


async def run_thermo_engine_test():
    runner = AsyncHardwareRunner(backend="ibm_heron_r2", shots=12000, mitigation="GUESS")
    K_nm = np.load("params/thermo_Knm_12x12.npy")
    lambda_fim_values = [0.0, 4.0, 8.0]
    results = {}
    for lam in lambda_fim_values:
        ansatz = StructuredAnsatz.from_kuramoto(K_nm, lambda_fim=lam, trotter_depth=8)
        job = runner.submit_circuit_batch(
            ansatz=ansatz,
            observable=[ThermodynamicWitness(), SyncOrderParameter()],
            job_tags=["thermo_engine", f"lambda_{lam}"],
        )
        result = await job.result()
        results[lam] = result
    with open("results/thermo_engine_results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    asyncio.run(run_thermo_engine_test())
