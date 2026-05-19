#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Frontier Campaign Tests (Batch 4)
"""Run the logical synchrony protection frontier campaign check."""

import asyncio
import json

import numpy as np
from campaign_io import parameter_path, result_path

from scpn_quantum_control.analysis import LogicalSyncWitness, SyncOrderParameter
from scpn_quantum_control.control import StructuredAnsatz
from scpn_quantum_control.hardware import AsyncHardwareRunner


async def run_logical_protection():
    """Sweep injected error rates for logical synchrony observables."""
    runner = AsyncHardwareRunner(backend="ibm_heron_r2", shots=12000, mitigation="GUESS")
    K_nm = np.load(parameter_path("logical_Knm_12x12.npy"))
    results = []
    for err in np.linspace(0.01, 0.15, 8):
        ansatz = StructuredAnsatz.from_kuramoto(K_nm, trotter_depth=8)
        job = runner.submit_circuit_batch(
            ansatz=ansatz,
            observable=[LogicalSyncWitness(), SyncOrderParameter()],
            error_injection_rate=err,
        )
        result = await job.result()
        results.append(result)

    with open(result_path("logical_sync_protection.json"), "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    asyncio.run(run_logical_protection())
