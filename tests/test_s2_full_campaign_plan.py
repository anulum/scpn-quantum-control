# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- tests for S2 full campaign planner
"""Tests for the S2 full scaling campaign planner."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType


def _load_module() -> ModuleType:
    script = Path(__file__).resolve().parents[1] / "scripts" / "plan_s2_full_scaling_campaign.py"
    spec = importlib.util.spec_from_file_location("plan_s2_full_scaling_campaign", script)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load S2 full campaign planner")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_campaign_rows_cover_full_protocol_grid() -> None:
    module = _load_module()

    rows = module.build_campaign_rows()

    assert len(rows) == 63
    assert {row["n_qubits"] for row in rows} == set(module.FULL_GRID_SIZES)
    assert "blocked_optional_hardware" in {row["status"] for row in rows}
    assert "ready_full_campaign" in {row["status"] for row in rows}


def test_hardware_rows_are_never_promoted_by_planner() -> None:
    module = _load_module()

    rows = module.build_campaign_rows()
    hardware_rows = [row for row in rows if row["baseline"] == "qpu_hardware"]

    assert hardware_rows
    assert all(row["status"] == "blocked_optional_hardware" for row in hardware_rows)
    assert all("explicit QPU approval" in str(row["reason"]) for row in hardware_rows)


def test_summary_blocks_advantage_claim_and_counts_ready_rows() -> None:
    module = _load_module()
    rows = module.build_campaign_rows()

    summary = module.build_summary(rows)

    assert summary["schema"] == "scpn_s2_full_scaling_campaign_plan_v1"
    assert summary["hardware_submission"] is False
    assert summary["advantage_claim"] is False
    assert summary["ready_required_rows"] > 0
    assert summary["campaign_decision"] == "ready_for_deliberate_no_qpu_full_classical_campaign"


def test_write_outputs_records_manifest(tmp_path: Path) -> None:
    module = _load_module()
    rows = module.build_campaign_rows()
    summary = module.build_summary(rows)

    json_path, csv_path, md_path = module.write_outputs(
        rows,
        summary,
        out_dir=tmp_path / "data",
        docs_dir=tmp_path / "docs",
    )

    assert json_path.exists()
    assert csv_path.exists()
    manifest = md_path.read_text(encoding="utf-8")
    assert "Hardware submission: `False`" in manifest
    assert "Advantage claim: `False`" in manifest
