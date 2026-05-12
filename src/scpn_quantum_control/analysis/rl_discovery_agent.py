# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — RL discovery agent compatibility wrapper
"""Compatibility wrapper for automated Kuramoto witness discovery."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .witness_discovery import (
    WitnessDiscoveryResult,
    WitnessDiscoverySpec,
    discover_kuramoto_witnesses,
)

FloatArray = NDArray[np.float64]

_SUPPORTED_OBSERVABLES = ("correlation", "fiedler")
_SUPPORTED_REWARD_FUNCTION = "witness_score"


class RLDiscoveryAgent:
    """Bandit/Bayesian discovery agent for Kuramoto witness candidates."""

    def __init__(
        self,
        runner: Any | None = None,
        observables: list[str] | None = None,
        n_episodes: int = 5,
        reward_function: str = "witness_score",
        *,
        K_nm: FloatArray | None = None,
        omega: FloatArray | None = None,
        theta0: FloatArray | None = None,
        spec: WitnessDiscoverySpec | None = None,
    ) -> None:
        if runner is not None:
            raise ValueError(
                "runner is not supported by RLDiscoveryAgent; configure K_nm, omega, "
                "theta0, and WitnessDiscoverySpec for replayable local discovery."
            )
        configured_observables = tuple(observables or _SUPPORTED_OBSERVABLES)
        if configured_observables != _SUPPORTED_OBSERVABLES:
            raise ValueError(
                "observables must be exactly ['correlation', 'fiedler']; these are the "
                "witnesses wired into the discovery objective."
            )
        if reward_function != _SUPPORTED_REWARD_FUNCTION:
            raise ValueError("reward_function must be 'witness_score'.")
        if n_episodes <= 0:
            raise ValueError("n_episodes must be a positive integer.")
        self.runner = runner
        self.observables = list(configured_observables)
        self.n_episodes = n_episodes
        self.reward_function = reward_function
        self.K_nm = None if K_nm is None else np.array(K_nm, dtype=np.float64, copy=True)
        self.omega = None if omega is None else np.array(omega, dtype=np.float64, copy=True)
        self.theta0 = None if theta0 is None else np.array(theta0, dtype=np.float64, copy=True)
        self.spec = spec
        self.result: WitnessDiscoveryResult | None = None
        self.discovered_phases: list[dict[str, Any]] = []

    async def run_discovery_loop(self) -> WitnessDiscoveryResult:
        """Run the configured witness-discovery loop."""
        if self.K_nm is None or self.omega is None:
            raise NotImplementedError(
                "RLDiscoveryAgent requires K_nm and omega. Compatibility fallback phase "
                "output has been removed; configure a real Kuramoto problem."
            )
        spec = self.spec or WitnessDiscoverySpec(n_iterations=self.n_episodes)
        self.result = discover_kuramoto_witnesses(
            self.K_nm,
            self.omega,
            theta0=self.theta0,
            spec=spec,
        )
        self.discovered_phases = [self.result.best.to_metadata()]
        return self.result

    def get_next_params(self) -> dict[str, float]:
        """Return the best discovered candidate parameters."""
        if self.result is None:
            return {}
        return self.result.best.candidate.to_metadata()

    def update_reward(self, result: dict[str, Any]) -> None:
        """Reject ad hoc reward mutation; discovery traces are recomputed end to end."""
        _ = result
        raise NotImplementedError(
            "External reward mutation is not supported. Use discover_kuramoto_witnesses() "
            "with an explicit WitnessDiscoverySpec for replayable optimisation."
        )

    def save_discovered_phases(self, path: str) -> None:
        """Write discovered candidate metadata to JSON."""
        payload = self.result.to_metadata() if self.result is not None else {"evaluations": []}
        Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
