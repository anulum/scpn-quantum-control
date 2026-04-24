#!/usr/bin/env python3
import asyncio
import json

import mock_injector  # noqa: F401
import numpy as np

from scpn_quantum_control.analysis import (
    QuantumFisherInformation,
    SyncOrderParameter,
    ThermodynamicWitness,
)
from scpn_quantum_control.control import StructuredAnsatz
from scpn_quantum_control.hardware import AsyncHardwareRunner


async def run_resource_distillation():
    runner = AsyncHardwareRunner(backend="ibm_heron_r2", shots=15000, mitigation="GUESS")

    K_nm = np.load("params/resource_Knm_12x12.npy")
    results = {}
    for lam in [0.0, 4.0, 8.0, 12.0]:
        ansatz = StructuredAnsatz.from_kuramoto(K_nm, lambda_fim=lam, trotter_depth=8)
        job = runner.submit_circuit_batch(
            ansatz=ansatz,
            observable=[SyncOrderParameter(), QuantumFisherInformation(), ThermodynamicWitness()],
            job_tags=["resource_theory", f"lam_{lam}"],
        )
        result = await job.result()
        results[lam] = result

    with open("results/sync_resource_theory.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    asyncio.run(run_resource_distillation())

# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Sophisticated Campaign Tests (Batch 3)
