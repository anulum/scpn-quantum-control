#!/usr/bin/env python3
import asyncio
import json

import mock_injector  # noqa: F401
import numpy as np

from scpn_quantum_control.analysis import DLAParityWitness, SyncOrderParameter
from scpn_quantum_control.control import StructuredAnsatz
from scpn_quantum_control.hardware import AsyncHardwareRunner


async def run_pt_symmetric():
    runner = AsyncHardwareRunner(backend="ibm_heron_r2", shots=12000, mitigation="GUESS")
    K_nm = np.load("params/pt_Knm_12x12.npy")
    results = {}
    for gain_loss in np.linspace(-0.5, 0.5, 9):
        ansatz = StructuredAnsatz.from_kuramoto(
            K_nm, non_hermitian_gain=gain_loss, trotter_depth=8
        )
        job = runner.submit_circuit_batch(
            ansatz=ansatz, observable=[SyncOrderParameter(), DLAParityWitness()]
        )
        result = await job.result()
        results[float(gain_loss)] = result

    with open("results/pt_symmetric_kuramoto.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    asyncio.run(run_pt_symmetric())

# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Frontier Campaign Tests (Batch 4)
