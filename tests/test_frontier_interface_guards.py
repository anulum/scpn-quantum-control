# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Frontier interface guard tests
"""Tests for fail-fast frontier interfaces."""

from __future__ import annotations

import asyncio
import runpy
from pathlib import Path

import numpy as np
import pytest

from scpn_quantum_control.analysis import RLPulseOptimizer, dla_truncated_tn


def test_dla_truncated_tensor_network_fails_until_implemented():
    K_nm = np.zeros((4, 4), dtype=np.float64)

    with pytest.raises(NotImplementedError, match="not implemented"):
        dla_truncated_tn(K_nm)


def test_rl_pulse_optimizer_fails_until_implemented():
    optimiser = RLPulseOptimizer(runner=object(), target_sync_order=0.5, episodes=1)

    with pytest.raises(NotImplementedError, match="not implemented"):
        asyncio.run(optimiser.optimize_pulses())

    with pytest.raises(NotImplementedError, match="No RL pulse"):
        optimiser.save_results("unused.json")


@pytest.mark.parametrize(
    "relative_path",
    [
        "scripts/frontier_campaign_2026/mock_injector.py",
        "scripts/hardware_campaign_2026/mock_injector.py",
        "scripts/primary_campaign_2026/mock_injector.py",
        "scripts/sophisticated_campaign_2026/mock_injector.py",
    ],
)
def test_retired_campaign_injectors_fail_fast(relative_path: str):
    repo_root = Path(__file__).resolve().parents[1]

    with pytest.raises(RuntimeError, match="Local campaign injectors are retired"):
        runpy.run_path(str(repo_root / relative_path))
