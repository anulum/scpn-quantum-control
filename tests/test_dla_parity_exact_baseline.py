# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Tests for the exact DLA-parity statevector baseline
"""The exact XY-Trotter parity leakage is zero (excitation-number conservation),
and the circuit reconstruction is bit-for-bit faithful to the campaign builder.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from qiskit.quantum_info import Statevector

from scpn_quantum_control.analysis.dla_parity_exact_baseline import (
    T_STEP,
    ExactBaselineRow,
    build_statevector_circuit,
    coupling_matrix,
    exact_baseline_grid,
    exact_parity_leakage,
    initial_parity,
)


def test_coupling_matrix_symmetric_zero_diagonal_decaying() -> None:
    k = coupling_matrix(4)
    assert np.allclose(k, k.T)
    assert np.allclose(np.diag(k), 0.0)
    assert k[0, 1] == pytest.approx(0.45 * np.exp(-0.3))
    assert k[0, 2] == pytest.approx(0.45 * np.exp(-0.6))


def test_initial_parity_popcount_mod_two() -> None:
    assert initial_parity("0011") == 0
    assert initial_parity("0001") == 1
    assert initial_parity("000011") == 0
    assert initial_parity("0000") == 0


@pytest.mark.parametrize("initial", ["0011", "0001", "0000", "1111", "0101"])
@pytest.mark.parametrize("depth", [0, 1, 2, 6, 20, 50])
def test_exact_leakage_is_zero_for_all_depths(initial: str, depth: int) -> None:
    # Excitation-number conservation ⇒ no amplitude crosses into the other parity.
    assert exact_parity_leakage(4, initial, depth, T_STEP) < 1e-9


@pytest.mark.parametrize("n", [2, 3, 6, 8])
def test_exact_leakage_zero_across_widths(n: int) -> None:
    initial = "1" + "0" * (n - 1)  # popcount 1 (odd)
    assert exact_parity_leakage(n, initial, depth=8) < 1e-9


def test_statevector_circuit_has_no_measurement_and_scales_with_depth() -> None:
    qc0 = build_statevector_circuit(4, "0011", 0)
    qc2 = build_statevector_circuit(4, "0011", 2)
    assert qc0.num_clbits == 0 and qc2.num_clbits == 0
    assert "measure" not in {inst.operation.name for inst in qc2.data}
    # Two extra oscillators of coupling per step ⇒ more gates at higher depth.
    assert len(qc2.data) > len(qc0.data)


def test_depth_zero_is_the_prepared_state() -> None:
    state = Statevector(build_statevector_circuit(4, "0011", 0))
    probs = state.probabilities_dict()
    # prep sets qubit q from bitstring[q]: '0011' → X on qubits 2,3. Qiskit keys
    # are big-endian (q3 q2 q1 q0), so the prepared basis state prints "1100".
    assert probs.get("1100", 0.0) == pytest.approx(1.0)
    assert set(k.count("1") for k, v in probs.items() if v > 1e-12) == {2}


def test_reconstruction_matches_campaign_builder() -> None:
    # Bit-for-bit fidelity against scripts/phase1_mini_bench_ibm_kingston.
    scripts_dir = str(Path(__file__).resolve().parents[1] / "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    builder = pytest.importorskip("phase1_mini_bench_ibm_kingston")
    ref = builder.build_xy_trotter_circuit(4, "0011", 6, 0.3)
    ref.remove_final_measurements()
    sv_ref = Statevector(ref)
    sv_mine = Statevector(build_statevector_circuit(4, "0011", 6))
    assert abs(sv_ref.inner(sv_mine)) ** 2 == pytest.approx(1.0, abs=1e-9)


def test_exact_baseline_grid_covers_grid() -> None:
    rows = exact_baseline_grid(4, ("0011", "0001"), (2, 4, 6))
    assert len(rows) == 6
    assert all(isinstance(r, ExactBaselineRow) for r in rows)
    assert all(r.exact_leakage < 1e-9 for r in rows)
    assert {r.initial_parity for r in rows} == {0, 1}
