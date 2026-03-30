# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Tests for Identity Entanglement
"""Tests for identity/entanglement_witness.py."""

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from scpn_quantum_control.identity.entanglement_witness import (
    chsh_from_statevector,
    disposition_entanglement_map,
)


def _bell_state() -> Statevector:
    """Create a maximally entangled Bell state |Phi+>."""
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    return Statevector.from_instruction(qc)


def _product_state() -> Statevector:
    """Create a product (unentangled) state |00>."""
    return Statevector.from_int(0, 4)


def test_bell_state_violates_chsh():
    sv = _bell_state()
    S = chsh_from_statevector(sv, 0, 1)
    assert S > 2.0, f"Bell state should violate CHSH, got S={S}"


def test_bell_state_near_tsirelson():
    sv = _bell_state()
    S = chsh_from_statevector(sv, 0, 1)
    tsirelson = 2 * np.sqrt(2)
    assert tsirelson * 0.9 < S, f"Expected near Tsirelson bound, got S={S}"


def test_product_state_respects_chsh():
    sv = _product_state()
    S = chsh_from_statevector(sv, 0, 1)
    assert S <= 2.0 + 1e-10, f"Product state should satisfy CHSH, got S={S}"


def test_same_qubit_raises():
    sv = _bell_state()
    with pytest.raises(ValueError, match="must differ"):
        chsh_from_statevector(sv, 0, 0)


def test_out_of_range_qubit_raises():
    sv = _bell_state()
    with pytest.raises(ValueError, match="out of range"):
        chsh_from_statevector(sv, 0, 5)


def test_entanglement_map_bell():
    sv = _bell_state()
    result = disposition_entanglement_map(sv)
    assert result["n_pairs"] == 1
    assert result["n_entangled"] == 1
    assert result["max_S"] > 2.0


def test_entanglement_map_product():
    sv = _product_state()
    result = disposition_entanglement_map(sv)
    assert result["n_entangled"] == 0


def test_entanglement_map_with_labels():
    sv = _bell_state()
    result = disposition_entanglement_map(sv, disposition_labels=["verify", "honest_naming"])
    pair = result["pairs"][0]
    assert pair["label_a"] == "verify"
    assert pair["label_b"] == "honest_naming"


def test_entanglement_map_wrong_label_count_raises():
    sv = _bell_state()
    with pytest.raises(ValueError, match="labels"):
        disposition_entanglement_map(sv, disposition_labels=["a", "b", "c"])


def test_integration_metric_bounded():
    sv = _bell_state()
    result = disposition_entanglement_map(sv)
    assert 0.0 <= result["integration_metric"] <= 1.0 + 1e-10


def test_three_qubit_ghz_no_bipartite_entanglement():
    """GHZ entanglement is genuinely tripartite — no bipartite CHSH violation."""
    qc = QuantumCircuit(3)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(1, 2)
    sv = Statevector.from_instruction(qc)
    result = disposition_entanglement_map(sv)
    assert result["n_pairs"] == 3
    assert result["n_entangled"] == 0


def test_three_qubit_two_bell_pairs():
    """Two Bell pairs in 3 qubits: (0,1) entangled, (1,2) entangled."""
    qc = QuantumCircuit(3)
    qc.h(0)
    qc.cx(0, 1)
    # Qubit 2 entangled with qubit 1 via separate Bell pair
    qc.h(2)
    qc.cx(2, 1)
    sv = Statevector.from_instruction(qc)
    result = disposition_entanglement_map(sv)
    assert result["n_pairs"] == 3
