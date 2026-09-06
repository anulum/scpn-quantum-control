# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Computational-Basis Count Mapping Tests
"""Dedicated owner tests for computational-basis count mapping.

Every case drives the mapping through the public entry point
``phase_qnode_computational_basis_fisher_information``. The mapping module's
own names are private and are deliberately never imported here: reaching them
directly would prove the helper rather than the behaviour a caller can rely on.
"""

from __future__ import annotations

import itertools
from collections.abc import Mapping, Sequence

import numpy as np
import pytest

from scpn_quantum_control.phase.qnode_circuit import (
    PhaseQNodeCircuit,
    phase_qnode_computational_basis_fisher_information,
)

_ANGLES: tuple[float, ...] = (0.7, 1.2, 1.5)


def _circuit(n_qubits: int) -> PhaseQNodeCircuit:
    """Build a product ``ry`` circuit with one rotation per logical wire."""
    operations = tuple(("ry", (wire,), wire) for wire in range(n_qubits))
    return PhaseQNodeCircuit(n_qubits=n_qubits, operations=operations, observable="pauli_z")


def _canonical_index(displayed_key: str, wires: Sequence[int]) -> int:
    """Return the canonical basis index of one displayed outcome string.

    Independent of the production implementation: this reassembles the
    canonical qubit-zero-most-significant bitstring by placing each displayed
    character at the position of the logical wire it was declared to carry,
    then parses that string as a base-2 integer. The production owner instead
    accumulates shifted bits, so agreement between the two is real evidence
    rather than a restatement.

    Parameters
    ----------
    displayed_key
        Full-width binary outcome exactly as the provider reported it.
    wires
        Logical wire identifier carried by each character of ``displayed_key``,
        left to right.

    Returns
    -------
    int
        Index of the outcome in canonical qubit-zero-most-significant order.
    """
    canonical = [""] * len(wires)
    for character, wire in zip(displayed_key, wires, strict=True):
        canonical[wire] = character
    return int("".join(canonical), 2)


def _counts(n_qubits: int, wires: Sequence[int]) -> dict[str, int]:
    """Build strictly positive, distinct counts keyed by displayed outcomes.

    Each displayed key is given a count derived from its canonical index, so a
    mapping error cannot be hidden by two outcomes sharing a value.

    Parameters
    ----------
    n_qubits
        Number of logical wires in the circuit.
    wires
        Declared logical wire per displayed character position.

    Returns
    -------
    dict of str to int
        Complete computational-basis record with unique positive counts.
    """
    keys = ["".join(bits) for bits in itertools.product("01", repeat=n_qubits)]
    return {key: 100 + 7 * _canonical_index(key, wires) for key in keys}


@pytest.mark.parametrize("n_qubits", [2, 3])
def test_every_wire_permutation_reorders_counts_to_canonical_order(n_qubits: int) -> None:
    """Exhaust the permutations and compare against the independent oracle."""
    circuit = _circuit(n_qubits)
    parameters = list(_ANGLES[:n_qubits])
    for wires in itertools.permutations(range(n_qubits)):
        counts = _counts(n_qubits, wires)
        result = phase_qnode_computational_basis_fisher_information(
            circuit,
            parameters,
            observed_counts=counts,
            observed_count_wires=list(wires),
            shot_count=sum(counts.values()),
        )
        expected = [0] * 2**n_qubits
        for key, count in counts.items():
            expected[_canonical_index(key, wires)] = count
        assert result.count_record == tuple(expected), wires


def test_identity_wires_preserve_provider_string_order() -> None:
    """The identity permutation must not silently reverse the outcome axis."""
    circuit = _circuit(2)
    counts = {"00": 11, "01": 13, "10": 17, "11": 19}
    result = phase_qnode_computational_basis_fisher_information(
        circuit,
        list(_ANGLES[:2]),
        observed_counts=counts,
        observed_count_wires=[0, 1],
        shot_count=sum(counts.values()),
    )
    assert result.count_record == (11, 13, 17, 19)


def test_mapping_evidence_is_versioned_and_declares_basis_order() -> None:
    """Replay evidence names its schema, wires and axis convention."""
    circuit = _circuit(2)
    wires = [1, 0]
    counts = _counts(2, wires)
    result = phase_qnode_computational_basis_fisher_information(
        circuit,
        list(_ANGLES[:2]),
        observed_counts=counts,
        observed_count_wires=wires,
        shot_count=sum(counts.values()),
    )
    assert result.count_mapping is not None
    assert result.count_mapping.to_dict() == {
        "schema": "phase_qnode.computational_basis_count_mapping.v1",
        "raw_counts": counts,
        "bit_wires": wires,
        "basis_order": "qubit_zero_most_significant",
    }


def test_raw_count_custody_survives_caller_and_consumer_mutation() -> None:
    """Neither the caller's dict nor a returned copy may alter stored evidence."""
    circuit = _circuit(2)
    wires = [1, 0]
    counts = _counts(2, wires)
    original = dict(counts)
    result = phase_qnode_computational_basis_fisher_information(
        circuit,
        list(_ANGLES[:2]),
        observed_counts=counts,
        observed_count_wires=wires,
        shot_count=sum(counts.values()),
    )
    assert result.count_mapping is not None
    counts.clear()
    wires.reverse()
    returned = result.count_mapping.to_dict()
    raw_counts = returned["raw_counts"]
    assert isinstance(raw_counts, dict)
    raw_counts.clear()
    assert isinstance(returned["bit_wires"], list)
    returned["bit_wires"].append(99)
    assert result.count_mapping.to_dict()["raw_counts"] == original
    assert result.count_mapping.bit_wires == (1, 0)


def test_large_integer_counts_are_not_narrowed_to_float() -> None:
    """Totals beyond exact float range keep every unit of the raw record."""
    circuit = _circuit(2)
    base = 2**53
    counts = {"00": base + 1, "01": base + 2, "10": base + 3, "11": base + 4}
    result = phase_qnode_computational_basis_fisher_information(
        circuit,
        list(_ANGLES[:2]),
        observed_counts=counts,
        observed_count_wires=[0, 1],
        shot_count=sum(counts.values()),
    )
    assert result.count_record == (base + 1, base + 2, base + 3, base + 4)
    assert result.count_mapping is not None
    assert result.count_mapping.to_dict()["raw_counts"] == counts


@pytest.mark.parametrize(
    ("counts", "wires", "message"),
    [
        ({"00": 1, "01": 2, "10": 3, "11": 4}, None, "require observed_count_wires"),
        ({"00": 1, "01": 2, "10": 3, "11": 4}, [1, 1], "permutation"),
        ({"00": 1, "01": 2, "10": 3, "11": 4}, [0, 1, 2], "permutation"),
        ({"00": 1, "01": 2, "10": 3, "11": 4}, [-1, 0], "permutation"),
        ({"00": 1, "01": 2, "10": 3, "11": 4}, [True, False], "permutation"),
        ({"00": 1, "01": 2, "11": 4}, [0, 1], "every computational-basis"),
        ({"000": 1, "01": 2, "10": 3, "11": 4}, [0, 1], "full-width binary"),
        ({"0,1": 1, "01": 2, "10": 3, "11": 4}, [0, 1], "full-width binary"),
        ({"02": 1, "01": 2, "10": 3, "11": 4}, [0, 1], "full-width binary"),
        ({"00": 2**53, "01": 2, "10": 3, "11": False}, [0, 1], "positive integers"),
        ({"00": "4", "01": 2, "10": 3, "11": 4}, [0, 1], "positive integers"),
        ({"00": 1, "01": 2, "10": 3, "11": -4}, [0, 1], "positive integers"),
    ],
)
def test_malformed_count_records_are_refused_at_the_public_boundary(
    counts: Mapping[str, object], wires: Sequence[object] | None, message: str
) -> None:
    """Every malformed record raises before any Fisher value is produced."""
    circuit = _circuit(2)
    with pytest.raises(ValueError, match=message):
        phase_qnode_computational_basis_fisher_information(
            circuit,
            list(_ANGLES[:2]),
            observed_counts=counts,  # type: ignore[arg-type]  # deliberately malformed
            observed_count_wires=wires,  # type: ignore[arg-type]  # deliberately malformed
            shot_count=10,
        )


def test_wire_declaration_without_bitstring_counts_is_refused() -> None:
    """A dense count vector must not silently accept wire metadata."""
    circuit = _circuit(2)
    with pytest.raises(ValueError, match="requires bitstring mapping counts"):
        phase_qnode_computational_basis_fisher_information(
            circuit,
            list(_ANGLES[:2]),
            observed_counts=[1, 2, 3, 4],
            observed_count_wires=[0, 1],
            shot_count=10,
        )


def test_mapped_and_dense_routes_agree_on_the_same_evidence() -> None:
    """Bitstring replay and its equivalent dense vector give one Fisher matrix."""
    circuit = _circuit(2)
    parameters = list(_ANGLES[:2])
    wires = [1, 0]
    counts = _counts(2, wires)
    total = sum(counts.values())
    dense = [0] * 4
    for key, count in counts.items():
        dense[_canonical_index(key, wires)] = count
    mapped = phase_qnode_computational_basis_fisher_information(
        circuit,
        parameters,
        observed_counts=counts,
        observed_count_wires=wires,
        shot_count=total,
    )
    plain = phase_qnode_computational_basis_fisher_information(
        circuit, parameters, observed_counts=dense, shot_count=total
    )
    np.testing.assert_array_equal(
        mapped.classical_fisher_information, plain.classical_fisher_information
    )
    assert "count_mapping" not in plain.to_dict()
