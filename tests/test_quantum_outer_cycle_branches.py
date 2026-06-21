# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Branch tests for the SSGF quantum outer cycle
"""Branch tests for the SSGF quantum outer-cycle optimiser.

Covers the alpha-range guard, the opt-in classical surrogate cost path, and the
fail-closed guards against a classical cost function returning a non-finite
value or a non-finite finite-difference.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from scpn_quantum_control.ssgf.quantum_outer_cycle import (
    OuterCycleResult,
    classical_cost,
    quantum_outer_cycle,
)


def test_classical_cost_single_oscillator_has_no_off_diagonal() -> None:
    """A 1x1 coupling matrix has no off-diagonal couplings; the surrogate returns 1.0."""
    assert classical_cost(np.zeros((1, 1), dtype=np.float64), allow_surrogate=True) == 1.0


def test_rejects_alpha_out_of_range() -> None:
    """The quantum/classical weight must lie in [0, 1]."""
    with pytest.raises(ValueError, match=r"alpha must be in \[0, 1\]"):
        quantum_outer_cycle(2, alpha=1.5)


def test_classical_surrogate_path_runs_when_opted_in() -> None:
    """With no cost function and the surrogate opt-in, the legacy cost is used."""
    result = quantum_outer_cycle(
        2,
        alpha=0.5,
        allow_classical_surrogate=True,
        max_iterations=1,
        trotter_reps=1,
        seed=0,
    )
    assert isinstance(result, OuterCycleResult)


def test_rejects_non_finite_classical_cost() -> None:
    """A classical cost that is not finite at the base point is rejected."""

    def bad_cost(_matrix: NDArray[np.float64]) -> float:
        return float("inf")

    with pytest.raises(ValueError, match="non-finite value"):
        quantum_outer_cycle(
            2, alpha=0.5, classical_cost_fn=bad_cost, max_iterations=1, trotter_reps=1, seed=0
        )


def test_rejects_non_finite_finite_difference() -> None:
    """A cost finite at the base but not under perturbation is rejected."""
    calls = {"n": 0}

    def flaky_cost(_matrix: NDArray[np.float64]) -> float:
        calls["n"] += 1
        return 1.0 if calls["n"] == 1 else float("nan")

    with pytest.raises(ValueError, match="finite-difference value"):
        quantum_outer_cycle(
            2, alpha=0.5, classical_cost_fn=flaky_cost, max_iterations=1, trotter_reps=1, seed=0
        )
