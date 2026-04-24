#!/usr/bin/env python3
import asyncio
import json

import numpy as np

from scpn_quantum_control.analysis import OTOC, SyncOrderParameter
from scpn_quantum_control.control import StructuredAnsatz
from scpn_quantum_control.hardware import AsyncHardwareRunner


async def run_test():
    runner = AsyncHardwareRunner(backend="ibm_heron_r2", shots=12000, mitigation="GUESS")
    K_nm = np.load("params/primary_Knm_12x12.npy")
    n_qubits = 12
    deltas = np.linspace(0.0, 1.0, 10)
    results = {}
    for delta in deltas:
        omega = np.random.normal(0, delta, n_qubits)
        job = runner.submit_circuit_batch(
            ansatz=StructuredAnsatz.from_kuramoto(K_nm=K_nm, omega=omega, trotter_depth=6),
            observable=[SyncOrderParameter(dtc_mode=True), OTOC()],
            K_coupling=4.0,
        )
        results[delta] = await job.result()

    with open("results/dtc_survival_sweep.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Heterogeneity vs. DTC survival sweep test completed.")


if __name__ == "__main__":
    asyncio.run(run_test())

# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Primary Campaign Tests
