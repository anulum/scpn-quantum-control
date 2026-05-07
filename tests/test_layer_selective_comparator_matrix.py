# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- tests for layer-selective comparator matrix
"""Tests for the no-submit layer-selective comparator matrix."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any


def _load_module() -> ModuleType:
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "generate_layer_selective_comparator_matrix.py"
    )
    spec = importlib.util.spec_from_file_location(
        "generate_layer_selective_comparator_matrix", script
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load layer-selective comparator script")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class _FakeStatus:
    operational = True
    pending_jobs = 0
    status_msg = "active"


class _FakeCouplingMap:
    def get_edges(self) -> list[tuple[int, int]]:
        return [(0, 1), (1, 2), (2, 3), (0, 3), (4, 5), (5, 6), (6, 7), (4, 7)]


class _FakeProperties:
    def readout_error(self, qubit: int) -> float:
        return 0.001 if qubit >= 4 else 0.05

    def gate_error(self, gate: str, qubits: list[int]) -> float:
        del gate
        return 0.002 if min(qubits) >= 4 else 0.04


class _FakeBackend:
    num_qubits = 8
    coupling_map = _FakeCouplingMap()

    def name(self) -> str:
        return "fake_backend"

    def status(self) -> _FakeStatus:
        return _FakeStatus()

    def properties(self) -> _FakeProperties:
        return _FakeProperties()


def test_true_layer_layout_prefers_low_error_connected_window() -> None:
    module = _load_module()

    selected = module.select_true_layer_layout(_FakeBackend())

    assert set(selected.physical_qubits) == {4, 5, 6, 7}
    assert selected.score > 0
    assert selected.readout_error_mean == 0.001
    assert selected.two_qubit_error_mean == 0.002


def test_comparator_summary_promotes_only_resource_gain() -> None:
    module = _load_module()
    layer_layout = module.LayerSelectiveLayout((4, 5, 6, 7), 0.1, 0.001, 0.002, 0.1)
    rows: list[dict[str, Any]] = []
    for method, depth, twoq in (
        ("default", 100, 80),
        ("sabre", 90, 75),
        ("layer_selective", 85, 70),
    ):
        rows.append(
            {
                "method": method,
                "initial": "0011",
                "depth": 6,
                "seed": 0,
                "transpiled_depth": depth,
                "total_gates": depth + twoq,
                "two_qubit_gates": twoq,
                "swap_gates": 0,
                "physical_qubits": "4 5 6 7",
            }
        )

    summary = module.build_summary(_FakeBackend(), rows, layer_layout=layer_layout)

    assert summary["hardware_submission"] is False
    assert summary["readiness_decision"] == "promotable_offline_resource_gain"
    assert summary["ready_for_hardware_comparison"] is True


def test_comparator_summary_blocks_no_gain() -> None:
    module = _load_module()
    layer_layout = module.LayerSelectiveLayout((4, 5, 6, 7), 0.1, 0.001, 0.002, 0.1)
    rows: list[dict[str, Any]] = []
    for method in ("default", "sabre", "layer_selective"):
        rows.append(
            {
                "method": method,
                "initial": "0011",
                "depth": 6,
                "seed": 0,
                "transpiled_depth": 100,
                "total_gates": 180,
                "two_qubit_gates": 80,
                "swap_gates": 0,
                "physical_qubits": "4 5 6 7",
            }
        )

    summary = module.build_summary(_FakeBackend(), rows, layer_layout=layer_layout)

    assert summary["readiness_decision"] == "blocked_no_resource_gain"
    assert summary["ready_for_hardware_comparison"] is False
