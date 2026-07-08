# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — TN/MPS crossover stage-1 tests
"""Tests for the QWC-5.1 TN/MPS crossover stage-1 gate."""

from __future__ import annotations

import json
import sys
from argparse import Namespace
from importlib import util
from importlib.machinery import ModuleSpec
from pathlib import Path
from types import ModuleType

import pytest

from scpn_quantum_control import benchmarks
from scpn_quantum_control.benchmarks.tn_mps_crossover_stage1 import (
    TN_MPS_CROSSOVER_PROTOCOL_ID,
    TN_MPS_CROSSOVER_REQUIRED_FIELDS,
    TN_MPS_CROSSOVER_STAGE1_SCHEMA,
    TNMPSCrossoverGate,
    TNMPSCrossoverRowSchema,
    TNMPSCrossoverStage1Report,
    build_tn_mps_crossover_stage1,
    render_tn_mps_crossover_stage1_markdown,
    validate_tn_mps_crossover_rows,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "export_tn_mps_crossover_stage1.py"


def _load_export_script() -> ModuleType:
    spec = util.spec_from_file_location("export_tn_mps_crossover_stage1", SCRIPT_PATH)
    if not isinstance(spec, ModuleSpec) or spec.loader is None:
        raise RuntimeError(f"cannot load {SCRIPT_PATH}")
    module = util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _valid_row(n_qubits: int = 30, baseline: str = "mps_tensor_network") -> dict[str, object]:
    return {
        "protocol_id": TN_MPS_CROSSOVER_PROTOCOL_ID,
        "n_qubits": n_qubits,
        "baseline": baseline,
        "status": "ok",
        "wall_time_ms": 1.0,
        "memory_bytes": 1024,
        "max_bond": 64,
        "discarded_weight": 0.0,
        "entropy_proxy": 0.5,
        "truncation_policy": "nearest-neighbour truncation with recorded omitted mass",
        "omitted_coupling_mass": 0.0,
        "command": ("scpn-bench", "s2-tn-crossover-stage1"),
        "machine": "local-test",
        "dependencies": {"quimb": "optional"},
        "git_commit": "test",
        "host_load": {"load1": 0.0},
        "claim_boundary": "measured row only; no broad advantage claim",
        "notes": [],
    }


def test_stage1_report_pins_larger_than_sixteen_row_schema() -> None:
    """The QWC-5.1 report admits N=30-40 schema without executing compute."""
    report = build_tn_mps_crossover_stage1()
    payload = report.to_dict()

    assert report.schema == TN_MPS_CROSSOVER_STAGE1_SCHEMA
    assert report.row_schema.protocol_id == TN_MPS_CROSSOVER_PROTOCOL_ID
    assert report.row_schema.target_sizes == (30, 32, 36, 40)
    assert all(size > 16 for size in report.row_schema.target_sizes)
    assert set(TN_MPS_CROSSOVER_REQUIRED_FIELDS) == set(report.row_schema.required_fields)
    assert report.passed is True
    assert report.stage2_compute_owner_gated is True
    assert report.benchmark_execution_performed is False
    assert report.advantage_claim_allowed is False
    assert payload["passed"] is True


def test_stage1_report_is_exported_from_benchmarks_package() -> None:
    """The public benchmarks namespace exposes the QWC-5.1 stage-1 gate."""
    assert benchmarks.build_tn_mps_crossover_stage1 is build_tn_mps_crossover_stage1
    assert "build_tn_mps_crossover_stage1" in benchmarks.__all__
    assert benchmarks.TN_MPS_CROSSOVER_PROTOCOL_ID == TN_MPS_CROSSOVER_PROTOCOL_ID


def test_row_schema_validation_fails_closed_for_ambiguous_inputs() -> None:
    """Malformed stage-1 schema and gate construction fail before export."""
    with pytest.raises(ValueError, match="greater than 16"):
        TNMPSCrossoverRowSchema(
            protocol_id=TN_MPS_CROSSOVER_PROTOCOL_ID,
            target_sizes=(16,),
            required_fields=TN_MPS_CROSSOVER_REQUIRED_FIELDS,
            required_baselines=("mps_tensor_network",),
            allowed_statuses=("ok",),
            claim_boundary="boundary",
        )
    with pytest.raises(ValueError, match="sorted"):
        TNMPSCrossoverRowSchema(
            protocol_id=TN_MPS_CROSSOVER_PROTOCOL_ID,
            target_sizes=(40, 30),
            required_fields=TN_MPS_CROSSOVER_REQUIRED_FIELDS,
            required_baselines=("mps_tensor_network",),
            allowed_statuses=("ok",),
            claim_boundary="boundary",
        )
    with pytest.raises(ValueError, match="unique"):
        TNMPSCrossoverRowSchema(
            protocol_id=TN_MPS_CROSSOVER_PROTOCOL_ID,
            target_sizes=(30, 30),
            required_fields=TN_MPS_CROSSOVER_REQUIRED_FIELDS,
            required_baselines=("mps_tensor_network",),
            allowed_statuses=("ok",),
            claim_boundary="boundary",
        )
    with pytest.raises(ValueError, match="required_fields"):
        TNMPSCrossoverRowSchema(
            protocol_id=TN_MPS_CROSSOVER_PROTOCOL_ID,
            target_sizes=(30,),
            required_fields=(),
            required_baselines=("mps_tensor_network",),
            allowed_statuses=("ok",),
            claim_boundary="boundary",
        )
    with pytest.raises(ValueError, match="blocker"):
        TNMPSCrossoverGate(
            gate_id="gate",
            passed=False,
            evidence="evidence",
            blocker="",
        )
    report = build_tn_mps_crossover_stage1()
    with pytest.raises(ValueError, match="gates"):
        TNMPSCrossoverStage1Report(
            schema=TN_MPS_CROSSOVER_STAGE1_SCHEMA,
            row_schema=report.row_schema,
            design_schema=report.design_schema,
            gates=(),
            blocked_claims=report.blocked_claims,
            owner_gated_followups=report.owner_gated_followups,
            claim_boundary=report.claim_boundary,
            stage2_compute_owner_gated=True,
            benchmark_execution_performed=False,
            advantage_claim_allowed=False,
        )


def test_validate_rows_accepts_measured_or_explained_skipped_rows() -> None:
    """Future measured/skipped rows are validated through the public stage-1 API."""
    measured = _valid_row()
    skipped = _valid_row(n_qubits=32, baseline="aer_statevector_or_skip")
    skipped["status"] = "skipped"
    skipped["wall_time_ms"] = None
    skipped["memory_bytes"] = None
    skipped["notes"] = ["statevector memory gate above local cap"]

    validation = validate_tn_mps_crossover_rows((measured, skipped))

    assert validation.valid is True
    assert validation.invalid_rows == ()
    assert validation.to_dict() == {"valid": True, "invalid_rows": []}


def test_validate_rows_rejects_missing_metrics_and_duplicate_identity() -> None:
    """The validator rejects incomplete rows and duplicate size/baseline pairs."""
    row = _valid_row()
    duplicate = _valid_row()
    row.pop("host_load")
    row["protocol_id"] = "wrong"
    row["n_qubits"] = 18
    row["baseline"] = "unknown"
    row["status"] = "ok"
    row["wall_time_ms"] = -1.0
    row["memory_bytes"] = -1
    row["discarded_weight"] = -0.1
    row["entropy_proxy"] = -0.1
    row["max_bond"] = 0
    row["omitted_coupling_mass"] = -1.0
    row["truncation_policy"] = ""
    row["command"] = ()
    row["machine"] = ""
    row["dependencies"] = []
    row["git_commit"] = ""
    row["claim_boundary"] = ""
    row["notes"] = "not-a-list"

    validation = validate_tn_mps_crossover_rows((row, duplicate))

    assert validation.valid is False
    assert any("missing fields ['host_load']" in item for item in validation.invalid_rows)
    assert any("protocol_id must be" in item for item in validation.invalid_rows)
    assert any("n_qubits must be one of" in item for item in validation.invalid_rows)
    assert any("baseline must be one of" in item for item in validation.invalid_rows)
    assert any("wall_time_ms must be finite" in item for item in validation.invalid_rows)
    assert any(
        "memory_bytes must be a non-negative integer" in item for item in validation.invalid_rows
    )
    assert any("truncation_policy must be" in item for item in validation.invalid_rows)
    assert any("host_load must be a mapping" in item for item in validation.invalid_rows)
    assert any("notes must be a list" in item for item in validation.invalid_rows)


def test_validate_rows_rejects_bad_status_unexplained_skip_and_duplicate() -> None:
    """Status values, skipped-row notes, and size/baseline identity are strict."""
    bad_status = _valid_row(n_qubits=30, baseline="classical_ode")
    bad_status["status"] = "pending"
    skipped = _valid_row(n_qubits=32, baseline="mps_tensor_network")
    skipped["status"] = "skipped"
    skipped["notes"] = []
    duplicate = _valid_row(n_qubits=36, baseline="mps_tensor_network")

    validation = validate_tn_mps_crossover_rows((bad_status, skipped, duplicate, duplicate))

    assert validation.valid is False
    assert any("status must be one of" in item for item in validation.invalid_rows)
    assert any("skipped row requires notes" in item for item in validation.invalid_rows)
    assert any("duplicate row" in item for item in validation.invalid_rows)


def test_markdown_report_records_regeneration_and_blocked_claims() -> None:
    """The rendered report names the schema, command, and non-claim boundary."""
    report = build_tn_mps_crossover_stage1()
    markdown = render_tn_mps_crossover_stage1_markdown(report)

    assert "# TN/MPS Crossover Stage-1 Gate" in markdown
    assert "scpn-bench s2-tn-crossover-stage1" in markdown
    assert "broad quantum advantage" in markdown
    assert "`30, 32, 36, 40`" in markdown


def test_export_script_writes_stage1_json_and_markdown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The QWC-5.1 artifacts regenerate through the export script."""
    export_script = _load_export_script()
    out_dir = tmp_path / "data"
    doc_path = tmp_path / "tn_mps_crossover_stage1.md"
    monkeypatch.setattr(
        export_script,
        "parse_args",
        lambda _argv=None: Namespace(out_dir=out_dir, doc_path=doc_path),
    )

    assert export_script.main(()) == 0
    json_files = list(out_dir.glob("tn_mps_crossover_stage1_*.json"))
    payload = json.loads(json_files[0].read_text(encoding="utf-8"))

    assert len(json_files) == 1
    assert payload["schema"] == TN_MPS_CROSSOVER_STAGE1_SCHEMA
    assert payload["passed"] is True
    assert "TN/MPS Crossover Stage-1 Gate" in doc_path.read_text(encoding="utf-8")
