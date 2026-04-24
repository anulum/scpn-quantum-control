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

from scpn_quantum_control.analysis import SyncOrderParameter
from scpn_quantum_control.control import StructuredAnsatz
from scpn_quantum_control.hardware import AsyncHardwareRunner


async def run_hypergraph_test():
    runner = AsyncHardwareRunner(backend="ibm_heron_r2", shots=10000, mitigation="GUESS")
    K_nm_pairwise = np.load("params/hyper_Knm_pairwise_12x12.npy")
    K_nm_hyper = np.load("params/hyper_Knm_3body.npy")
    results = {}
    for use_hyper in [False, True]:
        ansatz = StructuredAnsatz.from_kuramoto(
            K_nm_pairwise if not use_hyper else K_nm_pairwise,
            hyper_terms=K_nm_hyper if use_hyper else None,
            trotter_depth=6,
        )
        job = runner.submit_circuit_batch(
            ansatz=ansatz,
            observable=SyncOrderParameter(),
            job_tags=["hypergraph", f"hyper_{use_hyper}"],
        )
        result = await job.result()
        results["with_hyper" if use_hyper else "pairwise"] = result
    with open("results/hypergraph_results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    asyncio.run(run_hypergraph_test())
