# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Branch tests for the QAOA-MPC circuit builder
"""Branch tests for the QAOA-MPC lazy cost-Hamiltonian construction."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_quantum_control.control.qaoa_mpc import QAOA_MPC


def _mpc() -> QAOA_MPC:
    return QAOA_MPC(
        np.eye(2, dtype=np.float64),
        np.array([0.5, 0.5], dtype=np.float64),
        horizon=2,
        p_layers=1,
    )


def test_build_circuit_lazily_constructs_cost_hamiltonian() -> None:
    """Building the circuit constructs the cost Hamiltonian on demand."""
    circuit = _mpc()._build_qaoa_circuit(
        np.zeros(1, dtype=np.float64), np.zeros(1, dtype=np.float64)
    )
    assert circuit.num_qubits == 2


def test_build_circuit_raises_when_cost_hamiltonian_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A cost Hamiltonian that fails to construct surfaces as a runtime error."""
    mpc = _mpc()
    monkeypatch.setattr(mpc, "build_cost_hamiltonian", lambda: None)
    with pytest.raises(RuntimeError, match="cost Hamiltonian construction failed"):
        mpc._build_qaoa_circuit(np.zeros(1, dtype=np.float64), np.zeros(1, dtype=np.float64))
