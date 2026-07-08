# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- TN/MPS baseline design tests
"""Tests for the QWC-4.2 TN/MPS baseline design manifest."""

from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import pytest

from scpn_quantum_control import benchmarks
from scpn_quantum_control.benchmarks.tn_mps_baseline_design import (
    DEFAULT_TARGET_SIZES,
    TN_MPS_BASELINE_DESIGN_CLAIM_BOUNDARY,
    TN_MPS_BASELINE_DESIGN_SCHEMA,
    TNBaselineAdapter,
    TNBaselineDesign,
    TNBaselineSizePlan,
    build_tn_mps_baseline_design,
    render_tn_mps_baseline_design_markdown,
)
from scripts import export_tn_mps_baseline_design as export_script


def test_design_manifest_records_cpu_first_path_and_blocked_claims() -> None:
    """The QWC-4.2 manifest is explicit about what is and is not evidence."""
    design = build_tn_mps_baseline_design()
    payload = design.to_dict()

    assert design.schema == TN_MPS_BASELINE_DESIGN_SCHEMA
    assert design.target_sizes == DEFAULT_TARGET_SIZES
    assert design.claim_boundary == TN_MPS_BASELINE_DESIGN_CLAIM_BOUNDARY
    assert design.benchmark_execution_performed is False
    assert design.hardware_submission_allowed is False
    assert design.advantage_claim_allowed is False
    assert {adapter.name: adapter.status for adapter in design.adapters} == {
        "quimb_mps_cpu": "optional_dependency",
        "bounded_native_schmidt": "ready",
        "itensor_julia": "blocked",
        "gpu_tn": "owner_gated",
    }
    assert all(row.cpu_first_adapter == "quimb_mps_cpu" for row in design.size_plan)
    assert all(row.qwc5_1_unblocker for row in design.size_plan)
    assert "quantum advantage" in " ".join(design.blocked_claims)
    assert payload["target_sizes"] == list(DEFAULT_TARGET_SIZES)


def test_design_can_target_a_custom_sorted_size_grid() -> None:
    """A custom grid is accepted when it is sorted and unique."""
    design = build_tn_mps_baseline_design((30, 34))

    assert design.target_sizes == (30, 34)
    assert [row.size_class for row in design.size_plan] == ["pilot", "extension"]
    assert all(adapter.max_target_qubits == 34 for adapter in design.adapters)


def test_design_validation_rejects_ambiguous_target_sizes() -> None:
    """Invalid grids fail before a misleading plan can be emitted."""
    with pytest.raises(ValueError, match="non-empty"):
        build_tn_mps_baseline_design(())
    with pytest.raises(ValueError, match=">= 2"):
        build_tn_mps_baseline_design((1,))
    with pytest.raises(ValueError, match="sorted"):
        build_tn_mps_baseline_design((32, 30))
    with pytest.raises(ValueError, match="unique"):
        build_tn_mps_baseline_design((30, 30))


def test_manifest_dataclasses_fail_closed_on_empty_fields() -> None:
    """Direct construction validates all user-visible manifest rows."""
    with pytest.raises(ValueError, match="name"):
        TNBaselineAdapter(
            name="",
            language="Python",
            dependency="none",
            status="ready",
            role="role",
            max_target_qubits=30,
            claim_boundary="boundary",
            notes=("note",),
        )
    with pytest.raises(ValueError, match="max_target_qubits"):
        TNBaselineAdapter(
            name="adapter",
            language="Python",
            dependency="none",
            status="ready",
            role="role",
            max_target_qubits=0,
            claim_boundary="boundary",
            notes=("note",),
        )
    with pytest.raises(ValueError, match="required_rows"):
        TNBaselineSizePlan(
            n_qubits=30,
            size_class="pilot",
            cpu_first_adapter="quimb_mps_cpu",
            required_rows=(),
            acceptance_gates=("gate",),
            blocked_claims=("claim",),
            gpu_followup="defer",
            qwc5_1_unblocker=True,
        )
    with pytest.raises(ValueError, match="size_plan"):
        TNBaselineDesign(
            schema=TN_MPS_BASELINE_DESIGN_SCHEMA,
            target_sizes=(30,),
            decision="decision",
            adapters=(build_tn_mps_baseline_design((30,)).adapters[0],),
            size_plan=(),
            acceptance_gates=("gate",),
            blocked_claims=("claim",),
            qwc5_1_unblocked_by="unblocker",
        )
    with pytest.raises(ValueError, match="adapters"):
        TNBaselineDesign(
            schema=TN_MPS_BASELINE_DESIGN_SCHEMA,
            target_sizes=(30,),
            decision="decision",
            adapters=(),
            size_plan=(
                TNBaselineSizePlan(
                    n_qubits=30,
                    size_class="pilot",
                    cpu_first_adapter="quimb_mps_cpu",
                    required_rows=("row",),
                    acceptance_gates=("gate",),
                    blocked_claims=("claim",),
                    gpu_followup="defer",
                    qwc5_1_unblocker=True,
                ),
            ),
            acceptance_gates=("gate",),
            blocked_claims=("claim",),
            qwc5_1_unblocked_by="unblocker",
        )


def test_markdown_report_and_public_exports_are_wired() -> None:
    """The design is visible through docs rendering and benchmark exports."""
    design = build_tn_mps_baseline_design((30,))
    markdown = render_tn_mps_baseline_design_markdown(design)

    assert "# TN/MPS Baseline Design" in markdown
    assert "scpn-bench s2-tn-mps-baseline-design" in markdown
    assert "quimb_mps_cpu" in markdown
    assert benchmarks.TNBaselineDesign is TNBaselineDesign
    assert benchmarks.build_tn_mps_baseline_design is build_tn_mps_baseline_design
    assert "build_tn_mps_baseline_design" in benchmarks.__all__


def test_export_script_writes_json_and_markdown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The design artifacts regenerate through the export script."""
    out_dir = tmp_path / "data"
    doc_path = tmp_path / "tn_mps_baseline_design.md"
    monkeypatch.setattr(
        export_script,
        "parse_args",
        lambda: Namespace(out_dir=out_dir, doc_path=doc_path),
    )

    assert export_script.main() == 0
    json_files = list(out_dir.glob("tn_mps_baseline_design_*.json"))

    assert len(json_files) == 1
    payload = json.loads(json_files[0].read_text(encoding="utf-8"))
    assert payload["schema"] == TN_MPS_BASELINE_DESIGN_SCHEMA
    assert payload["advantage_claim_allowed"] is False
    assert "TN/MPS Baseline Design" in doc_path.read_text(encoding="utf-8")
