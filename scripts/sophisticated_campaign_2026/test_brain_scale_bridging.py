#!/usr/bin/env python3
import asyncio
import json

import mock_injector  # noqa: F401
import numpy as np

from scpn_quantum_control.accel import rust_kuramoto_classical
from scpn_quantum_control.analysis import DLAParityWitness, IntegratedInformationPhi
from scpn_quantum_control.control import StructuredAnsatz
from scpn_quantum_control.hardware import AsyncHardwareRunner


async def run_brain_scale_bridging():
    runner = AsyncHardwareRunner(backend="ibm_heron_r2", shots=10000, mitigation="GUESS")

    K_nm_small = np.load("params/c_elegans_subnetwork_14x14.npy")
    lambda_fim = 2.75

    results = []
    for scale in range(25):
        ansatz = StructuredAnsatz.from_kuramoto(K_nm_small, lambda_fim=lambda_fim, trotter_depth=6)
        job = runner.submit_circuit_batch(
            ansatz=ansatz,
            observable=[IntegratedInformationPhi(), DLAParityWitness()],
            job_tags=["brain_bridge", f"scale_{scale}"],
        )
        result = await job.result()
        results.append(result)

        # Coarse-grain DLA invariants into classical large-N solver
        # Dummy call for the stub
        rust_kuramoto_classical.run_large_n(
            N=10000, K=1.0, lambda_fim=lambda_fim, delta=0.1, steps=10
        )

    with open("results/brain_scale_bridging.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    asyncio.run(run_brain_scale_bridging())

# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Sophisticated Campaign Tests (Batch 3)
