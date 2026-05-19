#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Primary Campaign Tests
"""Run the primary-campaign BKT scaling hardware sweep."""

import asyncio
import json

import numpy as np
from campaign_io import parameter_path, result_path

from scpn_quantum_control.analysis import SyncOrderParameter
from scpn_quantum_control.control import StructuredAnsatz
from scpn_quantum_control.hardware import AsyncHardwareRunner


async def run_test():
    """Submit BKT scaling circuits across the preregistered system sizes."""
    runner = AsyncHardwareRunner(backend="ibm_heron_r2", shots=20000, mitigation="GUESS")
    Ns = [8, 12, 16, 20, 24, 32]
    results = {}
    for N in Ns:
        print(f"Running BKT scaling for N={N}")
        K_nm = np.load(parameter_path(f"primary_Knm_{N}x{N}.npy"))
        omega_vector = np.load(parameter_path(f"primary_omega_{N}.npy"))
        job = runner.submit_circuit_batch(
            ansatz=StructuredAnsatz.from_kuramoto(K_nm=K_nm, omega=omega_vector, trotter_depth=8),
            observable=SyncOrderParameter(),
            n_qubits=N,
        )
        results[N] = await job.result()

    with open(result_path("bkt_scaling.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("Finer finite-size scaling BKT transition test completed.")


if __name__ == "__main__":
    asyncio.run(run_test())
