#!/usr/bin/env python3
import asyncio

from scpn_quantum_control.analysis import RLDiscoveryAgent
from scpn_quantum_control.hardware import AsyncHardwareRunner


async def run_autonomous_discovery():
    runner = AsyncHardwareRunner(backend="ibm_heron_r2", shots=6000, mitigation="GUESS")

    agent = RLDiscoveryAgent(
        runner=runner,
        observables=["DLA_parity", "sync_order", "otoc_tstar", "integrated_information"],
        n_episodes=300,
        reward_function="criticality_novelty",
    )

    await agent.run_discovery_loop()
    agent.save_discovered_phases("results/autonomous_universality_classes.json")


if __name__ == "__main__":
    asyncio.run(run_autonomous_discovery())

# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Sophisticated Campaign Tests (Batch 3)
