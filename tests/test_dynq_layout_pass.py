# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for the DynQ Qiskit layout pass
"""Multi-angle tests for hardware/dynq_layout_pass.py.

Dimensions: calibration extraction (canonicalisation, min-on-duplicate,
None-skipping, readout), layout publication, backend adapter, fail-closed
paths, determinism, and PassManager integration.

Targets are built by hand rather than via ``GenericBackendV2``: the fake
backend's construction fabricates gate circuits that raise under the coverage
trace function (the qiskit×coverage tracer interaction), and hand-built targets
also give deterministic control over Louvain region sizes.
"""

from __future__ import annotations

import itertools

import pytest

pytest.importorskip("networkx")
pytest.importorskip("qiskit")

from qiskit import QuantumCircuit
from qiskit.circuit.library import CXGate, Measure, RZGate
from qiskit.transpiler import (
    InstructionProperties,
    Layout,
    PassManager,
    Target,
    TranspilerError,
)

from scpn_quantum_control.hardware.dynq_layout_pass import (
    DynQLayoutPass,
    calibration_from_target,
)


def _device_target(
    gate_pairs: dict[tuple[int, int], float | None],
    readout: dict[int, float | None],
    num_qubits: int,
) -> Target:
    """Build a Target with CX gate errors and measure (readout) errors."""
    target = Target(num_qubits=num_qubits)
    target.add_instruction(
        CXGate(),
        {pair: InstructionProperties(error=err) for pair, err in gate_pairs.items()},
    )
    target.add_instruction(
        Measure(),
        {(q,): InstructionProperties(error=err) for q, err in readout.items()},
    )
    return target


def _two_cluster_target() -> Target:
    """Two dense low-error clusters {0,1,2,3} and {4,5,6} split by a bad bridge.

    Louvain reliably keeps the size-4 cluster as the top-quality region, so any
    circuit of width ≤ 4 has a fitting region regardless of seed.
    """
    pairs: dict[tuple[int, int], float | None] = {}
    for i, j in itertools.combinations([0, 1, 2, 3], 2):
        pairs[(i, j)] = 0.001
    for i, j in itertools.combinations([4, 5, 6], 2):
        pairs[(i, j)] = 0.002
    pairs[(3, 4)] = 0.3  # high-error bridge keeps the clusters apart
    readout: dict[int, float | None] = {q: 0.01 for q in range(7)}
    readout[0] = 0.004  # qubit 0 is the best-readout qubit in cluster A
    return _device_target(pairs, readout, 7)


def layout_physical(pm: PassManager) -> list[int]:
    """Physical qubits in virtual-qubit order from the published layout."""
    return list(pm.property_set["layout"].get_virtual_bits().values())


def _synthetic_target() -> Target:
    """A 3-qubit target with duplicate/reversed pairs and a None-error readout."""
    return _device_target(
        {
            (0, 1): 0.02,
            (1, 0): 0.01,  # reversed, lower error → replaces (0,1)
            (1, 2): 0.03,
            (2, 1): 0.09,  # reversed, higher error → must NOT replace (1,2)
        },
        {0: 0.05, 1: 0.04, 2: None},  # qubit-2 readout None → skipped
        3,
    )


class TestCalibrationExtraction:
    def test_two_qubit_errors_extracted(self) -> None:
        gate_errors, _ = calibration_from_target(_synthetic_target())
        assert set(gate_errors) == {(0, 1), (1, 2)}

    def test_pairs_are_order_canonicalised_and_min_wins(self) -> None:
        gate_errors, _ = calibration_from_target(_synthetic_target())
        # (0,1) and reversed (1,0) collapse to (0,1) keeping the smaller error.
        assert gate_errors[(0, 1)] == pytest.approx(0.01)
        assert gate_errors[(1, 2)] == pytest.approx(0.03)

    def test_readout_errors_extracted_and_none_skipped(self) -> None:
        _, readout = calibration_from_target(_synthetic_target())
        assert readout == {0: pytest.approx(0.05), 1: pytest.approx(0.04)}
        assert 2 not in readout  # None error skipped

    def test_multi_cluster_target_yields_canonical_calibration(self) -> None:
        gate_errors, readout = calibration_from_target(_two_cluster_target())
        assert gate_errors  # cx errors present
        assert readout  # measure errors present
        assert all(k[0] < k[1] for k in gate_errors)  # every pair canonicalised

    def test_single_qubit_only_target_has_no_gate_errors(self) -> None:
        target = Target(num_qubits=2)
        target.add_instruction(RZGate(0.0), {(0,): InstructionProperties(error=0.0)})
        gate_errors, _ = calibration_from_target(target)
        assert gate_errors == {}

    def test_none_props_and_single_qubit_nonmeasure_ignored(self) -> None:
        # A None-props entry is skipped; a 1-qubit non-measure gate error is not
        # mistaken for a readout error.
        from qiskit.circuit.library import SXGate

        target = Target(num_qubits=2)
        target.add_instruction(CXGate(), {(0, 1): InstructionProperties(error=0.01)})
        target.add_instruction(
            SXGate(),
            {(0,): None, (1,): InstructionProperties(error=0.001)},
        )
        target.add_instruction(Measure(), {(0,): InstructionProperties(error=0.02)})
        gate_errors, readout = calibration_from_target(target)
        assert gate_errors == {(0, 1): pytest.approx(0.01)}
        assert readout == {0: pytest.approx(0.02)}  # SX 1q error is not readout


class TestLayoutPublication:
    def test_layout_published_for_fitting_circuit(self) -> None:
        qc = QuantumCircuit(3)
        pm = PassManager([DynQLayoutPass(_two_cluster_target(), seed=42)])
        pm.run(qc)
        layout = pm.property_set["layout"]
        assert isinstance(layout, Layout)
        physical = sorted(layout.get_virtual_bits().values())
        assert len(physical) == 3
        assert all(0 <= p < 7 for p in physical)

    def test_layout_selects_best_cluster(self) -> None:
        # width 4 must land entirely inside the low-error size-4 cluster {0,1,2,3}.
        qc = QuantumCircuit(4)
        pm = PassManager([DynQLayoutPass(_two_cluster_target(), seed=1)])
        pm.run(qc)
        assert set(layout_physical(pm)) == {0, 1, 2, 3}

    def test_best_readout_qubit_is_placed_first(self) -> None:
        # readout-sorted placement puts qubit 0 (best readout) at virtual index 0.
        qc = QuantumCircuit(4)
        pm = PassManager([DynQLayoutPass(_two_cluster_target(), seed=1)])
        pm.run(qc)
        assert layout_physical(pm)[0] == 0

    def test_mapping_result_attached_order_preserving(self) -> None:
        qc = QuantumCircuit(3)
        pm = PassManager([DynQLayoutPass(_two_cluster_target(), seed=42)])
        pm.run(qc)
        result = pm.property_set["dynq_mapping_result"]
        assert result is not None
        # Layout maps virtual qubit i → initial_layout[i], order-preserving.
        assert result.initial_layout == layout_physical(pm)

    def test_virtual_qubits_map_one_to_one(self) -> None:
        qc = QuantumCircuit(4)
        pm = PassManager([DynQLayoutPass(_two_cluster_target(), seed=7)])
        pm.run(qc)
        virtual = pm.property_set["layout"].get_virtual_bits()
        assert len(virtual) == 4
        assert len(set(virtual.values())) == 4  # distinct physical qubits


class TestBackendAdapter:
    def test_from_backend_builds_working_pass(self) -> None:
        class _Backend:
            target = _two_cluster_target()

        pass_ = DynQLayoutPass.from_backend(_Backend(), seed=1)
        assert isinstance(pass_, DynQLayoutPass)
        pm = PassManager([pass_])
        pm.run(QuantumCircuit(3))
        assert isinstance(pm.property_set["layout"], Layout)

    def test_from_backend_without_target_fails_closed(self) -> None:
        class _NoTarget:
            pass

        with pytest.raises(TranspilerError, match="no usable Target"):
            DynQLayoutPass.from_backend(_NoTarget())


class TestFailClosed:
    def test_target_without_two_qubit_errors_raises(self) -> None:
        target = Target(num_qubits=3)
        target.add_instruction(
            RZGate(0.0), {(q,): InstructionProperties(error=0.0) for q in range(3)}
        )
        pm = PassManager([DynQLayoutPass(target)])
        with pytest.raises(TranspilerError, match="no two-qubit gate error data"):
            pm.run(QuantumCircuit(2))

    def test_circuit_too_wide_for_any_region_raises(self) -> None:
        # width 5 exceeds both clusters (sizes 4 and 3); no region fits.
        pm = PassManager([DynQLayoutPass(_two_cluster_target())])
        with pytest.raises(TranspilerError, match="no execution region"):
            pm.run(QuantumCircuit(5))


class TestDeterminism:
    def test_same_seed_same_layout(self) -> None:
        target = _two_cluster_target()
        qc = QuantumCircuit(4)
        first = PassManager([DynQLayoutPass(target, seed=99)])
        second = PassManager([DynQLayoutPass(target, seed=99)])
        first.run(qc)
        second.run(qc)
        assert layout_physical(first) == layout_physical(second)
