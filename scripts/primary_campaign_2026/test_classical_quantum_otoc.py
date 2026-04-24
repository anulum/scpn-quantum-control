#!/usr/bin/env python3
import asyncio
import json

import numpy as np

from scpn_quantum_control.analysis import OTOC
from scpn_quantum_control.control import StructuredAnsatz
from scpn_quantum_control.hardware import AsyncHardwareRunner
from scpn_quantum_control.phase import lindblad_engine


async def run_test():
    runner = AsyncHardwareRunner(backend="ibm_heron_r2", shots=10000, mitigation="GUESS")
    K_nm = np.load("params/primary_Knm_12x12.npy")
    omega = np.load("params/primary_omega.npy")

    ansatz = StructuredAnsatz.from_kuramoto(K_nm=K_nm, omega=omega, trotter_depth=6)

    # Hardware
    job_hw = runner.submit_circuit_batch(ansatz=ansatz, observable=OTOC())
    result_hw = await job_hw.result()

    # Classical
    result_classical = lindblad_engine.run_trotter(
        ansatz=ansatz, observable=OTOC(), noise_model="default"
    )

    results = {"hardware": result_hw, "classical": result_classical}
    with open("results/classical_quantum_otoc.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Classical-quantum OTOC benchmark test completed.")


if __name__ == "__main__":
    asyncio.run(run_test())

# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Primary Campaign Tests
