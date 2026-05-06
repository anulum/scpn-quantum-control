# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Phase 3 state/layout submitter tests
"""Tests for the Phase 3 state/layout IBM submitter helpers."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType


def _load_module() -> ModuleType:
    script = Path(__file__).resolve().parents[1] / "scripts" / "phase3_state_layout_dla_ibm.py"
    spec = importlib.util.spec_from_file_location("phase3_state_layout_dla_ibm", script)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load state/layout script")
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


def test_select_layouts_returns_three_connected_windows() -> None:
    module = _load_module()

    layouts = module.select_layouts(_Backend(), n_layouts=3)

    assert [layout.physical_qubits for layout in layouts] == [
        (0, 1, 2, 3),
        (1, 2, 3, 4),
        (2, 3, 4, 5),
    ]
    assert all(layout.score > 0 for layout in layouts)


def test_build_circuits_matches_preregistered_scope() -> None:
    module = _load_module()
    layouts = module.select_layouts(_Backend(), n_layouts=3)

    main, readout = module.build_circuits(layouts)

    assert len(main) == 480
    assert len(readout) == 15
    first_meta, first_circuit = main[0]
    assert first_meta["layout_id"] == "L0"
    assert first_meta["physical_qubits"] == [0, 1, 2, 3]
    assert first_meta["shots"] == 4096
    assert first_circuit.num_qubits == 4


def test_readiness_rejects_depth_or_gate_guard_violation() -> None:
    module = _load_module()
    layouts = module.select_layouts(_Backend(), n_layouts=1)
    main, _ = module.build_circuits(layouts)
    circuits = [item for item in main[:2]]

    class _Circuit:
        def __init__(self, depth: int, gates: int) -> None:
            self._depth = depth
            self._gates = gates

        def depth(self) -> int:
            return self._depth

        def count_ops(self) -> dict[str, int]:
            return {"ecr": 4, "rz": self._gates - 4}

    ready = module.readiness(
        _Backend(),
        circuits,
        [_Circuit(10, 20), _Circuit(701, 20)],
        max_depth=700,
        max_total_gates=1500,
    )

    assert ready["accepted"] is False
    assert "max depth" in ready["rejection_reason"]
