#!/usr/bin/env python3
import asyncio
import json

import numpy as np

from scpn_quantum_control.accel import rust_kuramoto_classical
from scpn_quantum_control.analysis import DLAParityWitness, SyncOrderParameter
from scpn_quantum_control.control import StructuredAnsatz
from scpn_quantum_control.hardware import AsyncHardwareRunner


async def run_fusion_hybrid_stabilizer():
    runner = AsyncHardwareRunner(
        backend="ibm_heron_r2", shots=8000, mitigation="GUESS", real_time_feedback=True
    )

    K_nm = np.load("params/tokamak_Knm_16x16.npy")
    omega = np.load("params/tokamak_omega.npy")

    results = []
    for cycle in range(60):
        print(f"Fusion cycle {cycle + 1}/60")
        ansatz = StructuredAnsatz.from_kuramoto(
            K_nm, omega, trotter_depth=8, informed_topology=True
        )

        job = runner.submit_circuit_batch(
            ansatz=ansatz,
            observable=[DLAParityWitness(split_odd_even=True), SyncOrderParameter()],
            feedback_map={"asymmetry_threshold": 0.08, "correction_strength": 0.15},
            job_tags=["fusion_hybrid", f"cycle_{cycle}"],
        )
        result = await job.result()
        results.append(result)

        # Hybrid classical correction (large-N classical + quantum DLA feedback)
        K_nm = rust_kuramoto_classical.apply_feedback_correction(
            K_nm, result.get("dla_asymmetry", 0.08), result.get("sync_order", 0.95)
        )

    with open("results/fusion_hybrid_stabilizer.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    asyncio.run(run_fusion_hybrid_stabilizer())

# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Sophisticated Campaign Tests (Batch 3)
