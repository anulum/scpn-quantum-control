#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
import asyncio

from scpn_quantum_control.analysis.rl_pulse_optimizer import RLPulseOptimizer
from scpn_quantum_control.hardware import AsyncHardwareRunner


async def run_rl_pulse_opt():
    runner = AsyncHardwareRunner(backend="ibm_heron_r2", shots=6000, mitigation="GUESS")
    optimizer = RLPulseOptimizer(runner=runner, target_sync_order=0.95, episodes=250)
    await optimizer.optimize_pulses()
    optimizer.save_results("results/rl_pulse_optimization.json")


if __name__ == "__main__":
    asyncio.run(run_rl_pulse_opt())
