#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Hardware Campaign Tests

import asyncio
import json

import numpy as np

from scpn_quantum_control.accel import rust_kuramoto_classical
from scpn_quantum_control.analysis import DLAParityWitness, SyncOrderParameter
from scpn_quantum_control.control import StructuredAnsatz
from scpn_quantum_control.hardware import AsyncHardwareRunner


async def run_fusion_feedback_test():
    runner = AsyncHardwareRunner(
        backend="ibm_heron_r2",
        shots=8000,
        mitigation="GUESS",
        real_time_feedback=True,
        max_circuit_depth=120,
    )
    K_nm = np.load("params/tokamak_Knm_12x12.npy")
    omega = np.load("params/tokamak_omega.npy")
    results = []
    for cycle in range(40):
        print(f"Feedback cycle {cycle + 1}/40")
        ansatz = StructuredAnsatz.from_kuramoto(
            K_nm=K_nm, omega=omega, trotter_depth=6, informed_topology=True
        )
        job = runner.submit_circuit_batch(
            ansatz=ansatz,
            observable=[DLAParityWitness(split_odd_even=True), SyncOrderParameter()],
            feedback_map={"asymmetry_threshold": 0.075, "correction_strength": 0.12},
            job_tags=["fusion_feedback", f"cycle_{cycle}"],
        )
        result = await job.result()
        results.append(result)
        K_nm = rust_kuramoto_classical.apply_feedback_correction(
            K_nm, result["dla_asymmetry"], result["sync_order"]
        )
    with open("results/fusion_feedback_full.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Fusion feedback test completed. Results saved.")


if __name__ == "__main__":
    asyncio.run(run_fusion_feedback_test())
