#!/usr/bin/env python3
import asyncio
import json

import numpy as np

from scpn_quantum_control.analysis import SyncOrderParameter
from scpn_quantum_control.control import StructuredAnsatz
from scpn_quantum_control.hardware import AsyncHardwareRunner


async def run_test():
    runner = AsyncHardwareRunner(backend="ibm_heron_r2", shots=10000, mitigation="GUESS")
    K_nm_bio = np.load("params/c_elegans_connectome_14x14.npy")

    ansatz = StructuredAnsatz.from_kuramoto(K_nm=K_nm_bio, lambda_fim=2.75, trotter_depth=6)
    job = runner.submit_circuit_batch(ansatz=ansatz, observable=SyncOrderParameter())
    result = await job.result()

    with open("results/biological_fim_connectome.json", "w") as f:
        json.dump(result, f, indent=2)
    print("Biological connectome test of FIM completed.")


if __name__ == "__main__":
    asyncio.run(run_test())

# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Primary Campaign Tests
