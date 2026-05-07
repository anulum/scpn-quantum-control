# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- tests for S2 slice progress reporter
"""Tests for the S2 slice progress reporter."""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType


def _load_module() -> ModuleType:
    script = Path(__file__).resolve().parents[1] / "scripts" / "report_s2_slice_progress.py"
    spec = importlib.util.spec_from_file_location("report_s2_slice_progress", script)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load S2 progress reporter")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_slice(data_dir: Path, size: int, *, status: str = "ok") -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    summary_path = data_dir / f"s2_full_campaign_slice_n{size}_2026-05-07.json"
    rows_path = data_dir / f"s2_full_campaign_slice_rows_n{size}_2026-05-07.csv"
    summary_path.write_text(
        json.dumps(
            {
                "slice_decision": "completed_no_qpu_campaign_slice",
                "hardware_submission": False,
                "advantage_claim": False,
                "full_campaign_complete": False,
                "executed_rows": 2,
                "skipped_rows": 0 if status == "ok" else 1,
            }
        ),
        encoding="utf-8",
    )
    rows = [
        {
            "protocol_id": "test",
            "n_qubits": str(size),
            "baseline": "classical_ode",
            "status": "ok",
            "wall_time_ms": "10.5",
            "memory_bytes": "1000",
            "metric_payload": "{'hilbert_dim': 0, 'peak_tracemalloc_bytes': 1000}",
            "command": "test",
            "machine": "test",
            "dependencies": "{}",
            "git_commit": "abc",
            "notes": "[]",
        },
        {
            "protocol_id": "test",
            "n_qubits": str(size),
            "baseline": "dense_eigh",
            "status": status,
            "wall_time_ms": "20.0",
            "memory_bytes": "2000",
            "metric_payload": "{'hilbert_dim': 16, 'peak_tracemalloc_bytes': 2000}",
            "command": "test",
            "machine": "test",
            "dependencies": "{}",
            "git_commit": "abc",
            "notes": "[]",
        },
    ]
    with rows_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def test_aggregate_slices_marks_clean_no_qpu_progress(tmp_path: Path) -> None:
    module = _load_module()
    data_dir = tmp_path / "data"
    for size in (8, 10, 12):
        _write_slice(data_dir, size)

    report = module.aggregate_slices((8, 10, 12), data_dir=data_dir)

    assert report["progress_decision"] == "ready_for_next_bounded_no_qpu_slice"
    assert report["total_executed_rows"] == 6
    assert report["total_ok_rows"] == 6
    assert report["hardware_submission"] is False
    assert report["advantage_claim"] is False
    assert report["full_campaign_complete"] is False


def test_aggregate_slices_blocks_when_slice_has_anomaly(tmp_path: Path) -> None:
    module = _load_module()
    data_dir = tmp_path / "data"
    _write_slice(data_dir, 8)
    _write_slice(data_dir, 10, status="skipped")

    report = module.aggregate_slices((8, 10), data_dir=data_dir)

    assert report["progress_decision"] == "blocked_until_slice_anomalies_are_resolved"
    assert report["total_ok_rows"] < report["total_executed_rows"]


def test_write_report_records_claim_boundary(tmp_path: Path) -> None:
    module = _load_module()
    data_dir = tmp_path / "data"
    docs_dir = tmp_path / "docs"
    _write_slice(data_dir, 8)
    report = module.aggregate_slices((8,), data_dir=data_dir)

    json_path, md_path = module.write_report(report, out_dir=data_dir, docs_dir=docs_dir)

    assert json_path.exists()
    manifest = md_path.read_text(encoding="utf-8")
    assert "Hardware submission: `False`" in manifest
    assert "quantum advantage" in manifest
