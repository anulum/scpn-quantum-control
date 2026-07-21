# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — IQM layout-transfer benchmark harness tests
"""Hermetic tests for the IQM square-lattice layout-transfer harness."""

from __future__ import annotations

import json

import numpy as np
import pytest

from scpn_quantum_control.benchmarks.iqm_layout_transfer_benchmark import (
    ARM_NAMES,
    CHAIN_SIZES,
    DEPTH_PARITY_TOLERANCE,
    MAIN_SHOTS,
    READOUT_SHOTS,
    TROTTER_DEPTH,
    LayoutTransferPlan,
    build_layout_transfer_plan,
    chain_swap_depth_provider,
    corrected_order_parameter,
    coupling_map_from_calibration,
    depth_parity_gate,
    exact_order_parameter,
    initial_bitstring,
    measured_physical_qubits,
    naive_chain_layout,
    optimised_initial_layout,
    per_qubit_one_probabilities,
    per_qubit_readout_errors,
)
from scpn_quantum_control.hardware.iqm_lattice_calibration import LatticeCalibration


def _grid_calibration(rows: int = 3, cols: int = 4, seed: int = 7) -> LatticeCalibration:
    """Synthetic square-lattice calibration (row-major indexing)."""
    edges: list[tuple[int, int]] = []
    for r in range(rows):
        for c in range(cols):
            q = cols * r + c
            if c < cols - 1:
                edges.append((q, q + 1))
            if r < rows - 1:
                edges.append((q, q + cols))
    rng = np.random.default_rng(seed)
    return LatticeCalibration(
        num_qubits=rows * cols,
        edges=tuple(sorted(edges)),
        edge_fidelity={e: float(0.985 + 0.014 * rng.random()) for e in sorted(edges)},
        readout_error={q: float(0.01 + 0.04 * rng.random()) for q in range(rows * cols)},
    )


class TestPreregisteredConstants:
    def test_frozen_matrix_shape(self) -> None:
        assert CHAIN_SIZES == (8, 12, 16)
        assert TROTTER_DEPTH == 5
        assert MAIN_SHOTS == 2048
        assert READOUT_SHOTS == 1024
        assert DEPTH_PARITY_TOLERANCE == 0.10
        assert ARM_NAMES == ("optimised", "default", "naive")

    def test_preregistered_circuit_count_is_fifteen(self) -> None:
        # 3 sizes × 3 arms + 3 sizes × 2 readout states, per the campaign doc.
        assert len(CHAIN_SIZES) * (len(ARM_NAMES) + 2) == 15


class TestInitialStateAndReference:
    def test_quarter_filling_bitstrings(self) -> None:
        assert initial_bitstring(8) == "10001000"
        assert initial_bitstring(12) == "100010001000"

    @pytest.mark.parametrize("bad", [0, 3, 6, 10])
    def test_rejects_sizes_not_multiple_of_four(self, bad: int) -> None:
        with pytest.raises(ValueError, match="multiple of four"):
            initial_bitstring(bad)

    def test_exact_reference_is_conserved_magnetisation(self) -> None:
        # Total excitation number is conserved, so the mean-Z proxy equals its
        # initial value 0.5 at every depth.
        assert exact_order_parameter(8, depth=0) == pytest.approx(0.5)
        assert exact_order_parameter(8, depth=TROTTER_DEPTH) == pytest.approx(0.5)

    def test_exact_reference_with_explicit_bitstring(self) -> None:
        assert exact_order_parameter(4, bitstring="0000", depth=2) == pytest.approx(1.0)


class TestDepthModelAndOptimiser:
    def test_chain_provider_counts_missing_adjacencies(self) -> None:
        cal = _grid_calibration()
        provider = chain_swap_depth_provider(cal)
        k = np.zeros((4, 4))
        omega = np.zeros(4)
        # 0-1-2-3 runs along the top row: fully adjacent.
        assert provider((0, 1, 2, 3), k, omega, None, t=1.0, reps=5) == 20
        # 3-4 crosses the row boundary (not an edge): one SWAP penalty.
        assert provider((2, 3, 4, 5), k, omega, None, t=1.0, reps=5) == 35

    def test_optimised_layout_is_adjacent_permutation_of_best_region(self) -> None:
        cal = _grid_calibration()
        layout = optimised_initial_layout(cal, 8)
        assert len(set(layout)) == 8
        edges = set(cal.edges)
        for a, b in zip(layout, layout[1:], strict=False):
            assert (min(a, b), max(a, b)) in edges

    def test_naive_chain_is_lexicographically_smallest(self) -> None:
        cal = _grid_calibration()
        assert naive_chain_layout(cal, 4) == (0, 1, 2, 3)
        chain = naive_chain_layout(cal, 12)
        edges = set(cal.edges)
        assert len(set(chain)) == 12
        for a, b in zip(chain, chain[1:], strict=False):
            assert (min(a, b), max(a, b)) in edges

    def test_naive_chain_fails_closed(self) -> None:
        cal = _grid_calibration()
        with pytest.raises(ValueError, match="at least two qubits"):
            naive_chain_layout(cal, 1)
        with pytest.raises(ValueError, match="no connected chain of length 13"):
            naive_chain_layout(cal, 13)

    def test_coupling_map_is_symmetric(self) -> None:
        cal = _grid_calibration()
        cmap = coupling_map_from_calibration(cal)
        pairs = {tuple(edge) for edge in cmap.get_edges()}
        assert (0, 1) in pairs and (1, 0) in pairs
        assert len(pairs) == 2 * len(cal.edges)


class TestCountObservables:
    def test_marginals_use_qiskit_bit_order(self) -> None:
        # Qiskit count keys put clbit 0 last.
        p1 = per_qubit_one_probabilities({"01": 100}, 2)
        assert p1[0] == pytest.approx(1.0)
        assert p1[1] == pytest.approx(0.0)

    def test_marginals_reject_bad_input(self) -> None:
        with pytest.raises(ValueError, match="does not have 2 bits"):
            per_qubit_one_probabilities({"011": 1}, 2)
        with pytest.raises(ValueError, match="non-negative"):
            per_qubit_one_probabilities({"01": -1}, 2)
        with pytest.raises(ValueError, match="empty count dictionary"):
            per_qubit_one_probabilities({}, 2)

    def test_readout_errors_from_calibration_counts(self) -> None:
        qubits = (3, 7)
        e01, e10 = per_qubit_readout_errors(
            {"00": 90, "01": 10},  # clbit 0 (qubit 3) flips to 1 in 10 % of shots
            {"11": 80, "10": 20},  # clbit 0 (qubit 3) drops to 0 in 20 % of shots
            qubits,
        )
        assert e01[3] == pytest.approx(0.10)
        assert e01[7] == pytest.approx(0.0)
        assert e10[3] == pytest.approx(0.20)
        assert e10[7] == pytest.approx(0.0)

    def test_corrected_order_parameter_identity_without_error(self) -> None:
        counts = {"10 00": 60, "0000": 40}  # spaces must be tolerated
        zeros = {q: 0.0 for q in (0, 1, 2, 3)}
        value = corrected_order_parameter(counts, (0, 1, 2, 3), zeros, zeros)
        # p1 = (0, 0, 0, 0.6) → spins (1, 1, 1, -0.2) → |mean| = 0.7
        assert value == pytest.approx(0.7)

    def test_corrected_order_parameter_undoes_known_flip(self) -> None:
        # True state all-zeros; qubit 5 reads 1 in 10 % of shots.
        counts = {"00": 900, "10": 100}
        value = corrected_order_parameter(counts, (4, 5), {4: 0.0, 5: 0.1}, {4: 0.0, 5: 0.0})
        assert value == pytest.approx(1.0)

    def test_corrected_order_parameter_clips_marginals(self) -> None:
        # Observed p1 below e01 would give a negative marginal without clipping.
        counts = {"0": 1000}
        value = corrected_order_parameter(counts, (2,), {2: 0.05}, {2: 0.0})
        assert value == pytest.approx(1.0)

    def test_corrected_order_parameter_fails_closed_on_bad_denominator(self) -> None:
        with pytest.raises(ValueError, match="denominator non-positive"):
            corrected_order_parameter({"0": 1}, (0,), {0: 0.6}, {0: 0.5})


class TestDepthParityGate:
    def test_pass_and_fail(self) -> None:
        ok = depth_parity_gate({"optimised": 40, "default": 42, "naive": 44})
        assert ok.passes and ok.max_over_min == pytest.approx(1.1)
        bad = depth_parity_gate({"optimised": 40, "default": 40, "naive": 64})
        assert not bad.passes
        assert bad.to_dict()["two_qubit_depths"]["naive"] == 64

    def test_rejects_empty_and_non_positive(self) -> None:
        with pytest.raises(ValueError, match="at least one arm"):
            depth_parity_gate({})
        with pytest.raises(ValueError, match="must be positive"):
            depth_parity_gate({"optimised": 0})


class TestMeasuredQubits:
    def test_extracts_clbit_ordered_mapping(self) -> None:
        from qiskit import QuantumCircuit

        qc = QuantumCircuit(3, 2)
        qc.measure(2, 0)
        qc.measure(0, 1)
        assert measured_physical_qubits(qc) == (2, 0)

    def test_rejects_unmeasured_and_double_measured_clbits(self) -> None:
        from qiskit import QuantumCircuit

        unmeasured = QuantumCircuit(2, 2)
        unmeasured.measure(0, 0)
        with pytest.raises(ValueError, match="exactly once"):
            measured_physical_qubits(unmeasured)
        double = QuantumCircuit(2, 1)
        double.measure(0, 0)
        double.measure(1, 0)
        with pytest.raises(ValueError, match="more than once"):
            measured_physical_qubits(double)


class TestPlanAssembly:
    @pytest.fixture(scope="class")
    def plan(self) -> LayoutTransferPlan:
        return build_layout_transfer_plan(_grid_calibration(), sizes=(4, 8))

    def test_matrix_shape_and_manifest(self, plan: LayoutTransferPlan) -> None:
        assert plan.circuit_count == 10
        labels = [label for label, _ in plan.circuit_manifest()]
        assert labels[:5] == [
            "main_n4_optimised",
            "main_n4_default",
            "main_n4_naive",
            "readout_n4_zeros",
            "readout_n4_ones",
        ]

    def test_arms_carry_layouts_and_depths(self, plan: LayoutTransferPlan) -> None:
        block = plan.blocks[1]
        arms = {arm.arm: arm for arm in block.arms}
        # Amendment 1: naive = lexicographically smallest connected chain;
        # on the row-major 3×4 grid that snakes 0-1-2-3 down to row two.
        assert arms["naive"].requested_initial_layout == (0, 1, 2, 3, 7, 6, 5, 4)
        assert arms["default"].requested_initial_layout is None
        assert len(set(arms["optimised"].measured_qubits)) == 8
        assert all(arm.two_qubit_depth > 0 for arm in block.arms)
        assert all(arm.two_qubit_gate_count >= arm.two_qubit_depth for arm in block.arms)

    def test_depth_parity_gate_passes_with_amended_naive_arm(
        self, plan: LayoutTransferPlan
    ) -> None:
        # Amendment 1 keeps every arm a connected chain, so no arm carries
        # SWAP overhead and the validity gate holds at both sizes.
        assert all(block.depth_parity.passes for block in plan.blocks)
        assert plan.all_gates_pass

    def test_readout_circuits_cover_union_of_measured_qubits(
        self, plan: LayoutTransferPlan
    ) -> None:
        for block in plan.blocks:
            union = {q for arm in block.arms for q in arm.measured_qubits}
            assert set(block.readout_qubits) == union
            for circuit in block.readout_circuits:
                assert measured_physical_qubits(circuit) == block.readout_qubits

    def test_exact_reference_present_per_block(self, plan: LayoutTransferPlan) -> None:
        assert [block.exact_reference for block in plan.blocks] == pytest.approx([0.5, 0.5])

    def test_plan_payload_is_json_serialisable(self, plan: LayoutTransferPlan) -> None:
        payload = json.loads(json.dumps(plan.to_dict()))
        assert payload["campaign"].startswith("iqm_layout_transfer")
        assert payload["circuit_count"] == 10
        assert payload["all_gates_pass"] is True
        assert payload["main_shots"] == MAIN_SHOTS
        assert payload["readout_shots"] == READOUT_SHOTS

    def test_readout_state_preparation(self, plan: LayoutTransferPlan) -> None:
        zeros, ones = plan.blocks[0].readout_circuits
        assert sum(1 for i in zeros.data if i.operation.num_qubits == 2) == 0
        prep_ops = [i.operation.name for i in ones.data if i.operation.name != "measure"]
        assert len(prep_ops) == len(plan.blocks[0].readout_qubits)


@pytest.mark.skipif(
    pytest.importorskip("importlib.util").find_spec("iqm") is None,
    reason="iqm optional extra not installed",
)
class TestAgainstRealFakeGarnet:
    def test_full_preregistered_matrix_on_fake_garnet(self) -> None:
        from iqm.qiskit_iqm.fake_backends.fake_garnet import IQMFakeGarnet

        from scpn_quantum_control.hardware.iqm_lattice_calibration import (
            lattice_calibration_from_backend,
        )

        calibration = lattice_calibration_from_backend(IQMFakeGarnet())
        plan = build_layout_transfer_plan(calibration)
        assert plan.circuit_count == 15
        for block in plan.blocks:
            assert block.exact_reference == pytest.approx(0.5)
            optimised = block.arms[0]
            edges = set(calibration.edges)
            for a, b in zip(
                optimised.requested_initial_layout,
                optimised.requested_initial_layout[1:],
                strict=False,
            ):
                assert (min(a, b), max(a, b)) in edges
