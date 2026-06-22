# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Branch tests for the synchronisation benchmark runner
"""Guard tests for the synchronisation benchmark runner.

Covers the ring/chain minimum-size guards, the classical ODE solve-failure path
and the XY Hamiltonian shape guards.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from scpn_quantum_control.benchmark_harness.synchronisation_runner import (
    decaying_chain_coupling_matrix,
    ring_coupling_matrix,
    run_classical_reference,
    xy_hamiltonian,
)


def test_ring_requires_three_oscillators() -> None:
    """A ring benchmark needs at least three oscillators."""
    with pytest.raises(ValueError, match="ring benchmark requires at least three"):
        ring_coupling_matrix(n_oscillators=2)


def test_chain_requires_two_oscillators() -> None:
    """A chain benchmark needs at least two oscillators."""
    with pytest.raises(ValueError, match="chain benchmark requires at least two"):
        decaying_chain_coupling_matrix(n_oscillators=1)


def test_classical_reference_raises_on_solve_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """A failed ODE solve surfaces as a runtime error."""

    def _failed(*_args: Any, **_kwargs: Any) -> Any:
        return SimpleNamespace(success=False, message="forced failure")

    monkeypatch.setattr(
        "scpn_quantum_control.benchmark_harness.synchronisation_runner.solve_ivp", _failed
    )
    with pytest.raises(RuntimeError, match="classical ODE solve failed"):
        run_classical_reference()


def test_xy_hamiltonian_rejects_non_square_coupling() -> None:
    """A non-square coupling matrix is rejected."""
    with pytest.raises(ValueError, match="coupling must be square"):
        xy_hamiltonian(np.zeros((2, 3), dtype=np.float64), np.zeros(2, dtype=np.float64))


def test_xy_hamiltonian_rejects_omega_length_mismatch() -> None:
    """A frequency vector of the wrong length is rejected."""
    with pytest.raises(ValueError, match="omega length must match coupling size"):
        xy_hamiltonian(np.zeros((2, 2), dtype=np.float64), np.zeros(3, dtype=np.float64))
