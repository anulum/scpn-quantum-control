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

from scpn_quantum_control.analysis import IntegratedInformationPhi, SyncOrderParameter
from scpn_quantum_control.control import StructuredAnsatz
from scpn_quantum_control.hardware import AsyncHardwareRunner


async def run_biological_fim_test():
    runner = AsyncHardwareRunner(backend="ibm_heron_r2", shots=10000, mitigation="GUESS")
    K_nm_bio = np.load("params/c_elegans_connectome_14x14.npy")
    lambda_values = [0.0, 1.0, 2.0, 2.75, 4.0, 8.0]
    results = {}
    for lam in lambda_values:
        print(f"Running FIM λ = {lam}")
        ansatz = StructuredAnsatz.from_kuramoto(K_nm_bio, lambda_fim=lam, trotter_depth=6)
        job = runner.submit_circuit_batch(
            ansatz=ansatz,
            observable=[SyncOrderParameter(), IntegratedInformationPhi()],
            job_tags=["bio_fim", f"lambda_{lam}"],
        )
        result = await job.result()
        results[lam] = result
    with open("results/bio_fim_results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    asyncio.run(run_biological_fim_test())
