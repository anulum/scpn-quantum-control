#!/usr/bin/env python3
import asyncio
import json

import mock_injector  # noqa: F401
import numpy as np

from scpn_quantum_control.analysis import SyncOrderParameter
from scpn_quantum_control.control import StructuredAnsatz
from scpn_quantum_control.hardware import AsyncHardwareRunner


async def run_hyper_nonreciprocal():
    runner = AsyncHardwareRunner(backend="ibm_heron_r2", shots=10000, mitigation="GUESS")

    K_pair = np.load("params/hyper_pairwise.npy")
    K_hyper = np.load("params/hyper_3body.npy")
    K_directed = np.load("params/hyper_directed.npy")

    configs = ["pairwise", "hypergraph", "directed", "full"]
    results = {}
    for cfg in configs:
        ansatz = StructuredAnsatz.from_kuramoto(
            K_pair,
            hyper_terms=K_hyper if cfg != "pairwise" else None,
            directed_terms=K_directed if cfg in ["directed", "full"] else None,
            trotter_depth=6,
        )
        job = runner.submit_circuit_batch(
            ansatz=ansatz, observable=SyncOrderParameter(), job_tags=["hypergraph", f"cfg_{cfg}"]
        )
        result = await job.result()
        results[cfg] = result

    with open("results/hyper_nonreciprocal.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    asyncio.run(run_hyper_nonreciprocal())

# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Sophisticated Campaign Tests (Batch 3)
