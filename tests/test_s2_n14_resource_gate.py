# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- tests for S2 n=14 resource gate
"""Tests for the S2 n=14 resource gate reporter."""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType


def _load_module() -> ModuleType:
    script = Path(__file__).resolve().parents[1] / "scripts" / "report_s2_n14_resource_gate.py"
    spec = importlib.util.spec_from_file_location("report_s2_n14_resource_gate", script)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load S2 n=14 resource gate reporter")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_inputs(tmp_path: Path) -> tuple[Path, Path]:
    plan_rows = tmp_path / "rows.csv"
    progress = tmp_path / "progress.json"
    rows = [
        {
            "protocol_id": "test",
            "n_qubits": "14",
            "baseline": baseline,
            "required": "True",
            "status": "ready_full_campaign",
            "reason": "test",
            "max_qubits": "20",
            "estimated_statevector_bytes": "262144",
            "estimated_dense_matrix_bytes": "4294967296",
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
    with plan_rows.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    progress.write_text(
        json.dumps(
            {
                "sizes": [8, 10, 12],
                "max_memory_bytes": 2147483648,
                "total_ok_rows": 15,
                "total_executed_rows": 15,
            }
        ),
        encoding="utf-8",
    )
    return plan_rows, progress


def test_resource_gate_blocks_interactive_promotion(tmp_path: Path) -> None:
    module = _load_module()
    plan_rows, progress = _write_inputs(tmp_path)

    report = module.build_report(plan_rows_path=plan_rows, progress_path=progress)

    assert report["resource_gate_decision"] == "blocked_for_scheduled_or_offloaded_no_qpu_run"
    assert report["interactive_n14_promotion"] is False
    assert report["hardware_submission"] is False
    assert report["advantage_claim"] is False
    assert report["estimated_dense_matrix_bytes"] == 4294967296
    assert report["dense_to_prior_memory_ratio"] == 2.0


def test_write_report_records_boundary(tmp_path: Path) -> None:
    module = _load_module()
    plan_rows, progress = _write_inputs(tmp_path)
    report = module.build_report(plan_rows_path=plan_rows, progress_path=progress)

    json_path, md_path = module.write_report(report, out_dir=tmp_path, docs_dir=tmp_path)

    assert json_path.exists()
    text = md_path.read_text(encoding="utf-8")
    assert "Interactive n=14 promotion: `False`" in text
    assert "not an n=14 execution result" in text
