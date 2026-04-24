#!/usr/bin/env python3
import asyncio
import json

import numpy as np

from scpn_quantum_control.control import StructuredAnsatz
from scpn_quantum_control.hardware import AsyncHardwareRunner
from scpn_quantum_control.phase import vqe_runner


async def run_test():
    runner = AsyncHardwareRunner(backend="ibm_heron_r2", shots=10000, mitigation="GUESS")
    results = {}
    for N in [8, 12, 16, 20]:
        K_nm = np.load(f"params/primary_Knm_{N}x{N}.npy")
        ansatz_informed = StructuredAnsatz.from_kuramoto(K_nm=K_nm)
        res = await vqe_runner.run_vqe(runner=runner, ansatz=ansatz_informed, n_qubits=N)
        results[N] = res

    with open("results/vqe_informed_scaling.json", "w") as f:
        json.dump(results, f, indent=2)
    print("VQE informed ansatz scaling to 20+ qubits test completed.")


if __name__ == "__main__":
    asyncio.run(run_test())

# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Primary Campaign Tests
