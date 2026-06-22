# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Branch tests for zero-noise extrapolation
"""Guard tests for the ZNE gate folding and Richardson extrapolation.

Covers the mid-circuit classical-operation fold guard, the input dimensionality
guard and the order validator's boolean and non-integer rejections.
"""

from __future__ import annotations

from typing import Any, cast

import pytest
from qiskit import QuantumCircuit

from scpn_quantum_control.mitigation.zne import (
    _validate_order,
    gate_fold_circuit,
    zne_extrapolate,
)


def test_gate_fold_rejects_mid_circuit_classical_op() -> None:
    """Folding a circuit with a mid-circuit classical operation is rejected."""
    circuit = QuantumCircuit(1, 1)
    circuit.measure(0, 0)
    circuit.x(0)
    with pytest.raises(ValueError, match="cannot fold circuits with mid-circuit classical"):
        gate_fold_circuit(circuit, 3)


def test_zne_extrapolate_rejects_non_one_dimensional() -> None:
    """Two-dimensional noise/expectation inputs are rejected."""
    with pytest.raises(ValueError, match="must be one-dimensional"):
        zne_extrapolate(cast(Any, [[1, 3]]), cast(Any, [[0.5, 0.4]]))


def test_validate_order_rejects_boolean() -> None:
    """A boolean order is rejected."""
    with pytest.raises(ValueError, match="order must be a non-negative integer"):
        _validate_order(True)


def test_validate_order_rejects_non_integer() -> None:
    """A non-integer order is rejected."""
    with pytest.raises(ValueError, match="order must be a non-negative integer"):
        _validate_order(1.5)
