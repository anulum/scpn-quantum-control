# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- tests for S2 full campaign slice runner
"""Tests for the S2 full scaling campaign slice runner."""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType


def _load_module() -> ModuleType:
    script = Path(__file__).resolve().parents[1] / "scripts" / "run_s2_full_campaign_slice.py"
    spec = importlib.util.spec_from_file_location("run_s2_full_campaign_slice", script)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load S2 full campaign slice runner")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_plan(tmp_path: Path) -> Path:
    data_dir = tmp_path / "data" / "s2_advantage_scaling"
    data_dir.mkdir(parents=True)
    plan_path = data_dir / "s2_full_campaign_plan_2026-05-07.json"
    rows_path = data_dir / "s2_full_campaign_rows_2026-05-07.csv"
    plan_path.write_text(
        json.dumps({"protocol": {"protocol_id": "test"}}),
        encoding="utf-8",
    )
    rows = [
        {
            "protocol_id": "test",
            "n_qubits": "8",
            "baseline": baseline,
            "required": "True",
            "status": "ready_full_campaign",
            "reason": "test",
            "max_qubits": "20",
            "estimated_statevector_bytes": "4096",
            "estimated_dense_matrix_bytes": "1048576",
            "claim_boundary": "test",
        }
        for baseline in (
            "classical_ode",
            "dense_eigh",
            "sparse_eigsh",
            "mps_tensor_network",
            "aer_statevector",
        )
    ]
    rows.append(
        {
            "protocol_id": "test",
            "n_qubits": "8",
            "baseline": "qpu_hardware",
            "required": "False",
            "status": "blocked_optional_hardware",
            "reason": "explicit QPU approval required",
            "max_qubits": "8",
            "estimated_statevector_bytes": "4096",
            "estimated_dense_matrix_bytes": "1048576",
            "claim_boundary": "test",
        }
    )
    with rows_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return plan_path


def test_select_executable_plan_rows_excludes_qpu_and_optional_rows(tmp_path: Path) -> None:
    module = _load_module()
    plan_path = _write_plan(tmp_path)
    module.__dict__["OUT_DIR"] = plan_path.parent

    selected = module.select_executable_plan_rows(module.load_plan_rows(plan_path), (8,))

    assert len(selected) == 5
    assert {row["baseline"] for row in selected} == module.NO_QPU_BASELINES


def test_execute_slice_records_no_hardware_or_advantage(tmp_path: Path) -> None:
    module = _load_module()
    plan_path = _write_plan(tmp_path)
    module.__dict__["OUT_DIR"] = plan_path.parent

    rows, summary = module.execute_slice(
        (4,),
        max_dense_qubits=4,
        max_sparse_qubits=4,
        max_tn_qubits=4,
        max_statevector_qubits=4,
        plan_path=plan_path,
    )

    assert rows
    assert summary["hardware_submission"] is False
    assert summary["advantage_claim"] is False
    assert summary["full_campaign_complete"] is False
    assert summary["slice_decision"] == "completed_no_qpu_campaign_slice"


def test_write_outputs_records_manifest(tmp_path: Path) -> None:
    module = _load_module()
    plan_path = _write_plan(tmp_path)
    module.__dict__["OUT_DIR"] = plan_path.parent
    rows, summary = module.execute_slice(
        (4,),
        max_dense_qubits=4,
        max_sparse_qubits=4,
        max_tn_qubits=4,
        max_statevector_qubits=4,
        plan_path=plan_path,
    )

    json_path, csv_path, md_path = module.write_outputs(
        rows,
        summary,
        out_dir=tmp_path / "out",
        docs_dir=tmp_path / "docs",
    )

    assert json_path.exists()
    assert csv_path.exists()
    manifest = md_path.read_text(encoding="utf-8")
    assert "Hardware submission: `False`" in manifest
    assert "Full campaign complete: `False`" in manifest
