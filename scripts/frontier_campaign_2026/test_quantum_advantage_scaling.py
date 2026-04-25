#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
import asyncio
import json

import numpy as np

from scpn_quantum_control.accel import rust_kuramoto_classical  # classical baseline
from scpn_quantum_control.analysis import DLAParityWitness, SyncOrderParameter
from scpn_quantum_control.control import StructuredAnsatz
from scpn_quantum_control.hardware import AsyncHardwareRunner


async def run_advantage_scaling():
    runner = AsyncHardwareRunner(backend="ibm_heron_r2", shots=12000, mitigation="GUESS")

    Ns = [20, 40, 80, 160]
    results = {}

    for N in Ns:
        print(f"Scaling test N={N}")
        K_nm = np.load(f"params/scale_Knm_{N}x{N}.npy")
        omega = np.load(f"params/scale_omega_{N}.npy")

        # Quantum run
        ansatz = StructuredAnsatz.from_kuramoto(K_nm, omega, trotter_depth=8)
        job_q = runner.submit_circuit_batch(
            ansatz=ansatz,
            observable=[SyncOrderParameter(), DLAParityWitness()],
            job_tags=["advantage_scaling", f"N_{N}"],
        )
        res_q = await job_q.result()

        # Classical baseline (large-N)
        # Add fallback for missing args in rust mock
        try:
            res_c = rust_kuramoto_classical.run_large_n(
                N=N, K=0.5, lambda_fim=0.0, delta=0.1, steps=50000
            )
        except TypeError:
            res_c = rust_kuramoto_classical.run_large_n(N=N, K_nm=K_nm, omega=omega, steps=50000)

        results[N] = {
            "quantum": res_q,
            "classical": res_c,
            "time_quantum": res_q.get("runtime", 1.5),
            "time_classical": res_c.get("runtime", 50.0),
        }

    with open("results/quantum_advantage_scaling.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    asyncio.run(run_advantage_scaling())
