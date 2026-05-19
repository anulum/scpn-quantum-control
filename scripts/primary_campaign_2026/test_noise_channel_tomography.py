#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Primary Campaign Tests
"""Run the primary-campaign DLA noise-channel tomography job."""

import asyncio
import json

import numpy as np
from campaign_io import parameter_path, result_path

from scpn_quantum_control.analysis import DLAParityWitness
from scpn_quantum_control.control import StructuredAnsatz
from scpn_quantum_control.hardware import AsyncHardwareRunner


async def run_test():
    """Submit tomography-enabled DLA parity circuits and save the result."""
    runner = AsyncHardwareRunner(
        backend="ibm_heron_r2", shots=15000, mitigation="GUESS", error_mitigation_level=2
    )
    K_nm = np.load(parameter_path("primary_Knm_12x12.npy"))
    omega_vector = np.load(parameter_path("primary_omega.npy"))
    params = {"n_qubits": 12, "trotter_depth": 6, "K_nm": K_nm, "omega": omega_vector}
    job = runner.submit_circuit_batch(
        ansatz=StructuredAnsatz.from_kuramoto(**params),
        observable=DLAParityWitness(split_odd_even=True),
        noise_tomography=True,
        twirling=True,
        num_randomizations=32,
        **params,
    )
    results = await job.result()
    with open(result_path("noise_tomography.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("Noise channel tomography test completed.")


if __name__ == "__main__":
    asyncio.run(run_test())
