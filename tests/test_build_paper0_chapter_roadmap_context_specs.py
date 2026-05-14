# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 chapter roadmap context spec tests
"""Tests for Paper 0 chapter roadmap context promotion."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

from scpn_quantum_control.paper0.spec_loader import load_chapter_roadmap_context_validation_spec

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "build_paper0_chapter_roadmap_context_specs.py"


def _load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "build_paper0_chapter_roadmap_context_specs", SCRIPT
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load Paper 0 chapter roadmap context spec script")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _record(ledger_id: str, block_index: int, text: str = "source text") -> dict[str, object]:
    return {
        "ledger_id": ledger_id,
        "source_record_id": ledger_id.replace("P0R", "P0B"),
        "source_block_index": block_index,
        "section_path": "Paper 0: The Foundational Framework (ToC)",
        "canonical_category": "context_boundary",
        "math_ids": [],
        "text": text,
    }


def _complete_records() -> list[dict[str, object]]:
    records = [_record(f"P0R{number:05d}", number) for number in range(105, 218)]
    for number in (133, 153, 184, 200, 217):
        records[number - 105]["text"] = ""
    records[161 - 105]["text"] = (
        "Chapter 10: The Spine of the Network - The Unified Phase Dynamics Equation (UPDE)"
    )
    records[176 - 105]["text"] = (
        "Chapter 12: The Coherence Backbone - Multi-Scale Quantum Error Correction (MS-QEC)"
    )
    return records


def test_chapter_roadmap_context_specs_consume_complete_source_span() -> None:
    module = _load_module()

    bundle = module.build_chapter_roadmap_context_specs(_complete_records())

    assert bundle.summary["source_record_count"] == 113
    assert bundle.summary["consumed_source_record_count"] == 113
    assert bundle.summary["coverage_match"] is True
    assert bundle.summary["source_ledger_span"] == ["P0R00105", "P0R00217"]
    assert bundle.summary["spec_count"] == 6
    assert bundle.summary["part_count"] == 5
    assert bundle.summary["chapter_count"] == 18
    assert bundle.summary["blank_marker_count"] == 5
    assert bundle.summary["numbering_inconsistency_present"] is True


def test_chapter_roadmap_specs_preserve_mechanism_targets_without_claim_promotion() -> None:
    module = _load_module()

    bundle = module.build_chapter_roadmap_context_specs(_complete_records())
    specs = {spec.key: spec for spec in bundle.specs}

    assert all(
        spec.source_ledger_ids == tuple(f"P0R{number:05d}" for number in range(105, 218))
        for spec in specs.values()
    )
    assert specs["chapter_roadmap_context.dynamic_spine"].source_equation_ids == (
        "P0R00154-P0R00160:architecture_roadmap",
        "P0R00161-P0R00166:upde_roadmap",
        "P0R00167-P0R00175:quasicriticality_roadmap",
        "P0R00176-P0R00183:ms_qec_roadmap",
    )
    assert "UPDE" in specs["chapter_roadmap_context.dynamic_spine"].source_formulae
    assert (
        "not validation evidence" in specs["chapter_roadmap_context.dynamic_spine"].claim_boundary
    )
    assert (
        "Fisher Information Metric"
        in specs["chapter_roadmap_context.psi_field_physics"].source_formulae
    )


def test_chapter_roadmap_context_builder_rejects_missing_required_source_record() -> None:
    module = _load_module()
    incomplete = [record for record in _complete_records() if record["ledger_id"] != "P0R00161"]

    with pytest.raises(ValueError, match="missing required source ledger ids"):
        module.build_chapter_roadmap_context_specs(incomplete)


def test_chapter_roadmap_context_write_outputs_and_loader_access(tmp_path: Path) -> None:
    module = _load_module()
    bundle = module.build_chapter_roadmap_context_specs(_complete_records())

    outputs = module.write_outputs(bundle, output_dir=tmp_path, date_tag="2099-01-02")

    payload = json.loads(outputs["json"].read_text(encoding="utf-8"))
    report = outputs["report"].read_text(encoding="utf-8")
    loaded = load_chapter_roadmap_context_validation_spec(
        "chapter_roadmap_context.dynamic_spine",
        spec_bundle_path=outputs["json"],
    )

    assert payload["summary"]["coverage_match"] is True
    assert loaded["source_ledger_ids"][0] == "P0R00105"
    assert loaded["source_ledger_ids"][-1] == "P0R00217"
    assert "Paper 0 Chapter Roadmap Context Specs" in report
    assert "not validation evidence" in report
