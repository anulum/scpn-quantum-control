# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Computational-Basis Count Mapping
"""Explicit bit-position custody for the local Fisher replay consumer."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral


@dataclass(frozen=True)
class _MappedBasisCounts:
    """Owned raw counts, declared wires and canonical Fisher count vector."""

    raw_counts: tuple[tuple[str, int], ...]
    bit_wires: tuple[int, ...]
    count_vector: tuple[int, ...]

    def to_dict(self) -> dict[str, object]:
        """Return versioned mapping evidence without discarding raw bitstrings."""
        return {
            "schema": "phase_qnode.computational_basis_count_mapping.v1",
            "raw_counts": dict(self.raw_counts),
            "bit_wires": list(self.bit_wires),
            "basis_order": "qubit_zero_most_significant",
        }


def _map_basis_counts(
    counts: Mapping[str, object], bit_wires: Sequence[int] | None, n_qubits: int
) -> _MappedBasisCounts:
    """Map full-width binary outcomes using left-to-right logical wire IDs."""
    if bit_wires is None:
        raise ValueError("bitstring counts require observed_count_wires")
    wires = tuple(bit_wires)
    if (
        len(wires) != n_qubits
        or any(isinstance(wire, bool) or not isinstance(wire, Integral) for wire in wires)
        or set(wires) != set(range(n_qubits))
    ):
        raise ValueError("observed_count_wires must be a permutation of circuit qubits")
    # Fisher replay already requires every outcome positive. Check cardinality
    # before allocating the dense vector; never expand a sparse provider record.
    if len(counts) != 2**n_qubits:
        raise ValueError("bitstring counts must include every computational-basis outcome")
    raw: list[tuple[str, int]] = []
    vector = [0] * len(counts)
    for key, count in counts.items():
        if not isinstance(key, str) or len(key) != n_qubits or set(key) - {"0", "1"}:
            raise ValueError("counts require full-width binary keys without register separators")
        if isinstance(count, bool) or not isinstance(count, Integral) or count <= 0:
            raise ValueError("bitstring counts must be strictly positive integers")
        index = sum(int(bit) << (n_qubits - 1 - int(wire)) for bit, wire in zip(key, wires))
        vector[index] = int(count)
        raw.append((key, int(count)))
    return _MappedBasisCounts(tuple(raw), tuple(int(wire) for wire in wires), tuple(vector))
