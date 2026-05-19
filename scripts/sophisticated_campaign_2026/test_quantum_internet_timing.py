#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Sophisticated Campaign Tests (Batch 3)
"""Run the sophisticated-campaign quantum-internet timing loop."""

import asyncio
import json

import numpy as np
from campaign_io import parameter_path, result_path

from scpn_quantum_control.analysis import DLAParityWitness, SyncOrderParameter
from scpn_quantum_control.control import StructuredAnsatz
from scpn_quantum_control.hardware import AsyncHardwareRunner


async def run_quantum_internet_timing():
    """Submit repeated mediated-coupling timing cycles and persist results."""
    runner = AsyncHardwareRunner(backend="ibm_heron_r2", shots=12000, mitigation="GUESS")

    K_nm = np.load(parameter_path("internet_timing_20x20.npy"))
    results = []
    for cycle in range(40):
        ansatz = StructuredAnsatz.from_kuramoto(K_nm, mediated_couplings=True, trotter_depth=6)
        job = runner.submit_circuit_batch(
            ansatz=ansatz,
            observable=[SyncOrderParameter(), DLAParityWitness()],
            job_tags=["internet_timing", f"cycle_{cycle}"],
        )
        result = await job.result()
        results.append(result)

    with open(result_path("quantum_internet_timing.json"), "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    asyncio.run(run_quantum_internet_timing())
