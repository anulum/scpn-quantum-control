#!/usr/bin/env python3
import asyncio
import json

import mock_injector  # noqa: F401
import numpy as np

from scpn_quantum_control.analysis import QuantumFisherInformation, SyncOrderParameter
from scpn_quantum_control.control import StructuredAnsatz
from scpn_quantum_control.hardware import AsyncHardwareRunner


async def run_distillation():
    runner = AsyncHardwareRunner(backend="ibm_heron_r2", shots=15000, mitigation="GUESS")

    K_nm = np.load("params/distill_Knm_12x12.npy")
    results = {}
    for round_num in range(5):
        ansatz = StructuredAnsatz.from_kuramoto(K_nm, lambda_fim=8.0, trotter_depth=8)
        job = runner.submit_circuit_batch(
            ansatz=ansatz,
            observable=[SyncOrderParameter(), QuantumFisherInformation()],
            job_tags=["distillation", f"round_{round_num}"],
        )
        result = await job.result()
        results[round_num] = result
        # Distillation step: update state based on resource
        K_nm = result.get("resource_updated_Knm", K_nm)  # from analysis module

    with open("results/sync_distillation.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    asyncio.run(run_distillation())

# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Frontier Campaign Tests (Batch 4)
