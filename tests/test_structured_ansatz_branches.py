# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Branch tests for the structured ansatz builder
"""Guard and accessor tests for the structured Kuramoto ansatz."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_quantum_control.control.structured_ansatz import StructuredAnsatz


def test_from_kuramoto_rejects_non_finite_omega() -> None:
    """A non-finite frequency vector is rejected."""
    with pytest.raises(ValueError, match="omega must contain only finite values"):
        StructuredAnsatz.from_kuramoto(
            np.eye(2, dtype=np.float64), omega=np.array([0.1, np.inf], dtype=np.float64)
        )


def test_from_kuramoto_rejects_non_square_coupling() -> None:
    """A non-square coupling matrix is rejected."""
    with pytest.raises(ValueError, match="K_nm must be a square matrix"):
        StructuredAnsatz.from_kuramoto(np.zeros((2, 3), dtype=np.float64))


def test_from_kuramoto_rejects_non_finite_coupling() -> None:
    """A non-finite coupling matrix is rejected."""
    matrix = np.array([[0.0, np.inf], [np.inf, 0.0]], dtype=np.float64)
    with pytest.raises(ValueError, match="K_nm must contain only finite values"):
        StructuredAnsatz.from_kuramoto(matrix)


def test_from_kuramoto_rejects_omega_shape_mismatch() -> None:
    """A frequency vector of the wrong shape is rejected."""
    with pytest.raises(ValueError, match="omega shape must be"):
        StructuredAnsatz.from_kuramoto(
            np.eye(2, dtype=np.float64), omega=np.zeros(3, dtype=np.float64)
        )


def test_from_kuramoto_with_fim_feedback() -> None:
    """A positive FIM weight adds the feedback rotations to the circuit."""
    ansatz = StructuredAnsatz.from_kuramoto(
        np.array([[0.0, 0.4], [0.4, 0.0]], dtype=np.float64),
        omega=np.array([0.1, 0.2], dtype=np.float64),
        trotter_depth=1,
        lambda_fim=0.5,
    )
    assert ansatz.build_circuit().num_qubits == 2


def test_build_circuit_requires_from_kuramoto() -> None:
    """Building a circuit before from_kuramoto() is rejected."""
    with pytest.raises(ValueError, match=r"Call from_kuramoto\(\) first"):
        StructuredAnsatz().build_circuit()


def test_repr_renders_for_empty_ansatz() -> None:
    """The repr renders even before the ansatz is built."""
    assert "StructuredAnsatz(" in repr(StructuredAnsatz())


def test_from_kuramoto_builds_circuit() -> None:
    """A valid coupling matrix builds a Trotterised circuit."""
    ansatz = StructuredAnsatz.from_kuramoto(
        np.array([[0.0, 0.4], [0.4, 0.0]], dtype=np.float64),
        omega=np.array([0.1, 0.2], dtype=np.float64),
        trotter_depth=1,
    )
    assert ansatz.build_circuit().num_qubits == 2
