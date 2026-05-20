# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
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


def _load_hydrator_module() -> ModuleType:
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "hydrate_phase3_entanglement_async_ibm.py"
    )
    spec = importlib.util.spec_from_file_location("hydrate_phase3_entanglement_async_ibm", script)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load entanglement/tomography hydrator script")
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


def test_parse_physical_qubits_requires_four_distinct_indices() -> None:
    module = _load_module()

    assert module.parse_physical_qubits("21,22,23,24") == (21, 22, 23, 24)

    for value in ["21,22,23", "21,22,23,23", "21,22,x,24"]:
        try:
            module.parse_physical_qubits(value)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted invalid qubit list: {value}")


def test_select_pinned_layout_preserves_requested_qubits() -> None:
    module = _load_module()

    layout = module.select_pinned_layout(_Backend(), (2, 3, 4, 5))

    assert layout.layout_id == "pinned_2_3_4_5"
    assert layout.physical_qubits == (2, 3, 4, 5)


def test_pending_job_roles_preserve_main_and_readout_ids() -> None:
    module = _load_module()

    assert module._pending_job_roles("job-main", "job-readout") == {
        "main": "job-main",
        "readout": "job-readout",
    }


def test_async_hydrator_elapsed_seconds_from_runtime_timestamps() -> None:
    module = _load_hydrator_module()

    assert (
        module._elapsed_seconds(
            {
                "timestamps": {
                    "created": "2026-05-20T10:00:00Z",
                    "finished": "2026-05-20T10:02:30Z",
                }
            }
        )
        == 150.0
    )
    assert module._elapsed_seconds({"timestamps": {"created": "bad"}}) is None


def test_async_hydrator_rows_attach_counts_and_metadata_defaults() -> None:
    module = _load_hydrator_module()

    class _Pub:
        metadata = {"custom": "value"}

    rows = module._rows_from_result(
        metas=[{"block": "main"}],
        result=[_Pub()],
        job_id="job-main",
        extract_counts=lambda _pub: {"0000": 7},
        metadata_note="metadata missing",
    )

    assert rows == [
        {
            "meta": {"block": "main"},
            "counts": {"0000": 7},
            "job_id": "job-main",
            "metadata": {
                "custom": "value",
                "depth": None,
                "total_gates": None,
                "ecr_gates": None,
                "recovery_note": "metadata missing",
            },
        }
    ]


def test_build_circuits_can_emit_full_readout_calibration() -> None:
    module = _load_module()
    layout = module.select_layout(_Backend())

    _main, readout = module.build_circuits(layout, full_readout_calibration=True)

    assert len(readout) == 16
    assert [meta["initial"] for meta, _circuit in readout] == [
        "0000",
        "0001",
        "0010",
        "0011",
        "0100",
        "0101",
        "0110",
        "0111",
        "1000",
        "1001",
        "1010",
        "1011",
        "1100",
        "1101",
        "1110",
        "1111",
    ]


def test_zne_subset_selects_dla_transverse_edges_and_one_fim_control() -> None:
    module = _load_module()
    rows = [
        {
            "family": "dla_parity",
            "label": "dla_odd_signal",
            "initial": "0001",
            "depth": "10",
            "lambda_fim": "",
            "basis_setting": "XXII",
            "absolute_deviation": "0.55",
        },
        {
            "family": "dla_parity",
            "label": "dla_odd_signal",
            "initial": "0001",
            "depth": "10",
            "lambda_fim": "",
            "basis_setting": "YYII",
            "absolute_deviation": "0.54",
        },
        {
            "family": "dla_parity",
            "label": "dla_even_signal",
            "initial": "0011",
            "depth": "10",
            "lambda_fim": "",
            "basis_setting": "IIXX",
            "absolute_deviation": "0.90",
        },
        {
            "family": "dla_parity",
            "label": "dla_odd_shallow",
            "initial": "0001",
            "depth": "6",
            "lambda_fim": "",
            "basis_setting": "IIXX",
            "absolute_deviation": "0.50",
        },
        {
            "family": "dla_parity",
            "label": "dla_odd_shallow",
            "initial": "0001",
            "depth": "6",
            "lambda_fim": "",
            "basis_setting": "IIYY",
            "absolute_deviation": "0.49",
        },
        {
            "family": "fim_pair",
            "label": "fim_lambda4_feedback",
            "initial": "0011",
            "depth": "4",
            "lambda_fim": "4.0",
            "basis_setting": "IXXI",
            "absolute_deviation": "0.27",
        },
        {
            "family": "fim_pair",
            "label": "fim_lambda4_feedback",
            "initial": "0011",
            "depth": "4",
            "lambda_fim": "4.0",
            "basis_setting": "IZZI",
            "absolute_deviation": "0.41",
        },
        {
            "family": "fim_pair",
            "label": "fim_lambda0_reference",
            "initial": "0011",
            "depth": "4",
            "lambda_fim": "0.0",
            "basis_setting": "XXII",
            "absolute_deviation": "0.05",
        },
    ]

    selected = module.select_zne_subset_rows(rows)

    assert [(row["label"], row["basis_setting"]) for row in selected] == [
        ("dla_odd_signal", "XXII"),
        ("dla_odd_signal", "YYII"),
        ("dla_odd_shallow", "IIXX"),
        ("dla_odd_shallow", "IIYY"),
        ("fim_lambda4_feedback", "IZZI"),
    ]


def test_build_zne_subset_circuits_adds_odd_scale_metadata() -> None:
    module = _load_module()
    layout = module.select_layout(_Backend())
    selected_rows = [
        {
            "family": "dla_parity",
            "label": "dla_odd_signal",
            "initial": "0001",
            "depth": "10",
            "lambda_fim": "",
            "basis_setting": "XXII",
            "absolute_deviation": "0.55",
        }
    ]

    main, readout = module.build_zne_subset_circuits(
        layout,
        selected_rows,
        noise_scales=(1, 3, 5),
    )

    assert len(main) == 9
    assert len(readout) == 16
    assert {meta["zne_noise_scale"] for meta, _circuit in main} == {1, 3, 5}
    assert {meta["rep"] for meta, _circuit in main} == {0, 1, 2}
    assert all(meta["zne_subset"] is True for meta, _circuit in main)


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
