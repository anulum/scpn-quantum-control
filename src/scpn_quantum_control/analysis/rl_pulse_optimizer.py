# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — RL pulse optimiser interface
"""Fail-fast interface for future RL pulse optimisation."""

from operator import index
from typing import Any

import numpy as np


class RLPulseOptimizer:
    """
    RL optimizer interface for pulse shaping.

    This class is intentionally not implemented yet. It must fail loudly
    rather than emit synthetic optimisation results.
    """

    def __init__(self, runner: Any, target_sync_order: float = 0.0, episodes: int = 250):
        if runner is None:
            raise ValueError("runner must be provided for RL pulse optimisation configuration.")
        target = float(target_sync_order)
        if not np.isfinite(target) or target < 0.0 or target > 1.0:
            raise ValueError("target_sync_order must be a finite value in [0, 1].")
        episode_count = _validate_episode_count(episodes)
        self.runner = runner
        self.target_sync_order = target
        self.episodes = episode_count

    async def optimize_pulses(self) -> None:
        """Fail closed until a real RL pulse optimizer is implemented."""
        raise NotImplementedError(
            "RL pulse optimisation is not implemented. Do not use this path "
            "for QPU campaigns until a real optimiser, objective, and replayable "
            "training trace are added."
        )

    def save_results(self, filepath: str) -> None:
        """Fail closed instead of writing non-existent optimization results."""
        _ = filepath
        raise NotImplementedError(
            "No RL pulse optimisation results exist because optimisation has not been implemented."
        )


def _validate_episode_count(episodes: Any) -> int:
    if isinstance(episodes, bool):
        raise ValueError("episodes must be a positive integer.")
    try:
        episode_count = index(episodes)
    except TypeError as exc:
        raise ValueError("episodes must be a positive integer.") from exc
    if episode_count <= 0:
        raise ValueError("episodes must be a positive integer.")
    return int(episode_count)
