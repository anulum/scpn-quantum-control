#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Hardware Campaign Tests
"""Run the synchrony-metrology hardware campaign check."""

import asyncio
import json

import numpy as np
from campaign_io import parameter_path, result_path

from scpn_quantum_control.analysis import QuantumFisherInformation, SyncOrderParameter
from scpn_quantum_control.control import StructuredAnsatz
from scpn_quantum_control.hardware import AsyncHardwareRunner


async def run_metrology_test():
    """Sweep coupling strengths for QFI and synchrony metrology observables."""
    runner = AsyncHardwareRunner(backend="ibm_heron_r2", shots=15000, mitigation="GUESS")
    K_nm = np.load(parameter_path("metrology_Knm_12x12.npy"))
    coupling_strengths = np.linspace(0.5, 4.0, 8)
    results = {}
    for K in coupling_strengths:
        ansatz = StructuredAnsatz.from_kuramoto(K_nm, K=K, trotter_depth=6)
        job = runner.submit_circuit_batch(
            ansatz=ansatz,
            observable=[QuantumFisherInformation(), SyncOrderParameter()],
            job_tags=["metrology", f"K_{K:.2f}"],
        )
        result = await job.result()
        results[float(K)] = result
    with open(result_path("metrology_results.json"), "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    asyncio.run(run_metrology_test())
