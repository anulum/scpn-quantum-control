# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Branch tests for the symmetry-sector compiler validator
"""Validation-blocker tests for the symmetry-sector mitigation planner.

Each test drives one accumulated blocker in the problem validator: qubit count,
coupling shape/finiteness, omega length/finiteness and initial-state length and
basis content.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from scpn_quantum_control.mitigation.symmetry_sector_compiler import (
    SymmetrySectorProblem,
    _validate_problem,
    plan_symmetry_sector_mitigation,
)


def _problem(**overrides: Any) -> SymmetrySectorProblem:
    kwargs: dict[str, Any] = {
        "n_qubits": 2,
        "coupling_matrix": np.array([[0.0, 0.3], [0.3, 0.0]], dtype=np.float64),
        "omega": np.array([0.1, 0.2], dtype=np.float64),
        "initial_state": "01",
        "measurement_basis": "computational",
        "has_raw_counts": True,
    }
    kwargs.update(overrides)
    return SymmetrySectorProblem(**kwargs)


def _blockers(**overrides: Any) -> list[str]:
    return _validate_problem(_problem(**overrides))[2]


def test_rejects_non_positive_qubits() -> None:
    """A non-positive qubit count is flagged."""
    assert "n_qubits must be positive" in _blockers(n_qubits=0)


def test_rejects_non_square_coupling() -> None:
    """A non-square coupling matrix is flagged."""
    blockers = _blockers(coupling_matrix=np.zeros((2, 3), dtype=np.float64))
    assert any("coupling_matrix must be square" in b for b in blockers)


def test_rejects_non_finite_coupling() -> None:
    """A non-finite coupling matrix is flagged."""
    coupling = np.array([[0.0, np.inf], [np.inf, 0.0]], dtype=np.float64)
    assert "coupling_matrix must be finite" in _blockers(coupling_matrix=coupling)


def test_rejects_omega_length_mismatch() -> None:
    """An omega vector of the wrong length is flagged."""
    assert "omega length must match n_qubits" in _blockers(
        omega=np.array([0.1, 0.2, 0.3], dtype=np.float64)
    )


def test_rejects_non_finite_omega() -> None:
    """A non-finite omega vector is flagged."""
    assert "omega must be finite" in _blockers(omega=np.array([0.1, np.inf], dtype=np.float64))


def test_rejects_initial_state_length_mismatch() -> None:
    """An initial state of the wrong length is flagged."""
    assert "initial_state length must match n_qubits" in _blockers(initial_state="011")


def test_rejects_non_binary_initial_state() -> None:
    """A non-computational-basis initial state is flagged."""
    assert "initial_state must be a computational-basis bitstring" in _blockers(initial_state="0x")


def test_valid_problem_plan_serialises() -> None:
    """A valid problem produces a JSON-serialisable mitigation plan."""
    plan = plan_symmetry_sector_mitigation(_problem())
    payload = plan.to_dict()
    assert isinstance(payload, dict)
    assert "claim_boundary" in payload
