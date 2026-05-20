# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Phase 3 entanglement/tomography submitter tests
"""Tests for the Phase 3 entanglement/tomography IBM runner helpers."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

from qiskit import QuantumCircuit


def _load_module() -> ModuleType:
    script = (
        Path(__file__).resolve().parents[1] / "scripts" / "phase3_entanglement_tomography_ibm.py"
    )
    spec = importlib.util.spec_from_file_location("phase3_entanglement_tomography_ibm", script)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load entanglement/tomography IBM script")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class _Status:
    operational = True
    pending_jobs = 0
    status_msg = "active"


class _CouplingMap:
    def get_edges(self) -> list[tuple[int, int]]:
        return [(i, i + 1) for i in range(8)]


class _Backend:
    name = "fake_heron"
    num_qubits = 156
    coupling_map = _CouplingMap()

    def status(self) -> _Status:
        return _Status()


def test_select_layout_returns_lowest_scoring_connected_window() -> None:
    module = _load_module()

    layout = module.select_layout(_Backend())

    assert layout.layout_id == "L0"
    assert layout.physical_qubits == (0, 1, 2, 3)
    assert layout.score > 0.0


def test_build_circuits_matches_promoted_readiness_scope() -> None:
    module = _load_module()
    layout = module.select_layout(_Backend())

    main, readout = module.build_circuits(layout)

    assert len(main) == 162
    assert len(readout) == 4
    first_meta, first_circuit = main[0]
    assert first_meta["experiment"] == module.EXPERIMENT
    assert first_meta["basis_setting"] == "IIXX"
    assert first_meta["shots"] == 2048
    assert first_meta["physical_qubits"] == [0, 1, 2, 3]
    assert first_circuit.num_clbits == 4


def test_apply_measurement_basis_adds_expected_rotations() -> None:
    module = _load_module()
    circuit = QuantumCircuit(4)

    measured = module.apply_measurement_basis(circuit, "XYZI")

    ops = measured.count_ops()
    assert ops["h"] == 2
    assert ops["sdg"] == 1
    assert ops["measure"] == 4


def test_readiness_rejects_basis_depth_expansion() -> None:
    module = _load_module()
    layout = module.select_layout(_Backend())
    main, _readout = module.build_circuits(layout)

    class _Circuit:
        def __init__(self, depth: int, gates: int) -> None:
            self._depth = depth
            self._gates = gates

        def depth(self) -> int:
            return self._depth

        def count_ops(self) -> dict[str, int]:
            return {"ecr": 2, "rz": self._gates - 2}

    ready = module.readiness(
        _Backend(),
        main[:2],
        [_Circuit(10, 20), _Circuit(10, 20)],
        [_Circuit(13, 24), _Circuit(30, 50)],
        readout_circuits=[],
        readout_isa_circuits=[],
        max_depth=700,
        max_total_gates=1500,
    )

    assert ready["accepted"] is False
    assert "basis expansion" in ready["rejection_reason"]
