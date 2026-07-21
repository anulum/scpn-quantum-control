# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Tests for the two-edge-colour XY-Trotter schedule
"""The 2-edge-colour schedule is a valid 2-colouring, conserves excitation
number, and gives constant two-qubit depth per Trotter step (genuine width-2)
while the serial baseline grows with the chain length.
"""

from __future__ import annotations

import pytest

from scpn_quantum_control.analysis.two_colour_schedule import (
    build_two_colour_circuit,
    depth_comparison,
    two_colour_edges,
    two_colour_parity_leakage,
    two_qubit_depth,
)


@pytest.mark.parametrize("n", [4, 5, 8, 9])
def test_two_colouring_is_valid(n: int) -> None:
    colour_a, colour_b = two_colour_edges(n)
    # Every path edge appears exactly once across the two classes.
    all_edges = set(colour_a) | set(colour_b)
    assert all_edges == {(i, i + 1) for i in range(n - 1)}
    assert len(colour_a) + len(colour_b) == n - 1
    # Within a class, edges are vertex-disjoint (so they parallelise).
    for cls in (colour_a, colour_b):
        touched = [q for edge in cls for q in edge]
        assert len(touched) == len(set(touched))


@pytest.mark.parametrize("initial", ["0011", "0001", "01010101", "111000"])
@pytest.mark.parametrize("depth", [0, 1, 5, 20])
def test_schedule_conserves_excitation_number(initial: str, depth: int) -> None:
    assert two_colour_parity_leakage(len(initial), initial, depth) < 1e-9


def test_two_colour_depth_is_constant_in_n() -> None:
    # Genuine width-2: two-qubit depth per step does not grow with the chain.
    depths = {n: depth_comparison(n, depth=5)["two_colour_2q_depth"] for n in (4, 8, 16, 32)}
    assert len(set(depths.values())) == 1


def test_sequential_depth_grows_and_reduction_increases_with_n() -> None:
    small = depth_comparison(8, depth=5)
    large = depth_comparison(32, depth=5)
    assert large["sequential_2q_depth"] > small["sequential_2q_depth"]
    assert large["reduction_factor"] > small["reduction_factor"] > 1.0


def test_two_colour_depth_scales_linearly_with_reps() -> None:
    d1 = depth_comparison(8, depth=1)["two_colour_2q_depth"]
    d3 = depth_comparison(8, depth=3)["two_colour_2q_depth"]
    assert d3 == pytest.approx(3 * d1)


def test_circuit_has_no_measurement() -> None:
    qc = build_two_colour_circuit(6, "000000", 2)
    assert qc.num_clbits == 0
    assert "measure" not in {inst.operation.name for inst in qc.data}
    assert two_qubit_depth(qc) > 0
