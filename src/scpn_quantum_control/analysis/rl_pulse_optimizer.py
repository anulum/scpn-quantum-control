# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — RL pulse optimiser interface
"""Fail-fast interface for future RL pulse optimisation."""

from typing import Any


class RLPulseOptimizer:
    """
    RL optimizer interface for pulse shaping.

    This class is intentionally not implemented yet. It must fail loudly
    rather than emit synthetic optimisation results.
    """

    def __init__(self, runner: Any, target_sync_order: float = 0.0, episodes: int = 250):
        self.runner = runner
        self.target_sync_order = target_sync_order
        self.episodes = episodes

    async def optimize_pulses(self) -> None:
        raise NotImplementedError(
            "RL pulse optimisation is not implemented. Do not use this path "
            "for QPU campaigns until a real optimiser, objective, and replayable "
            "training trace are added."
        )

    def save_results(self, filepath: str) -> None:
        _ = filepath
        raise NotImplementedError(
            "No RL pulse optimisation results exist because optimisation has not been implemented."
        )
