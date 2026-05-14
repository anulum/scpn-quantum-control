# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 terminal boundary spec tests
"""Tests for Paper 0 terminal taxonomy and EBS promotion."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

from scpn_quantum_control.paper0.spec_loader import load_terminal_boundary_validation_spec

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "build_paper0_terminal_boundary_specs.py"


def _load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("build_paper0_terminal_boundary_specs", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load Paper 0 terminal boundary spec script")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _record(ledger_id: str, block_index: int) -> dict[str, object]:
    return {
        "ledger_id": ledger_id,
        "source_record_id": ledger_id.replace("P0R", "P0B"),
        "source_block_index": block_index,
        "section_path": "New content to be allocated into the document:",
        "canonical_category": "validation_target",
        "table_id": "TBL020" if ledger_id == "P0R07076" else None,
        "math_ids": [],
        "text": "source text",
    }


def _complete_records() -> list[dict[str, object]]:
    return [_record(f"P0R{number:05d}", number) for number in range(7073, 7081)]


def test_terminal_boundary_specs_consume_complete_source_span() -> None:
    module = _load_module()

    bundle = module.build_terminal_boundary_specs(_complete_records())

    assert bundle.summary["source_record_count"] == 8
    assert bundle.summary["consumed_source_record_count"] == 8
    assert bundle.summary["coverage_match"] is True
    assert bundle.summary["source_ledger_span"] == ["P0R07073", "P0R07080"]
    assert bundle.summary["spec_count"] == 4
    assert bundle.summary["terminal_count"] == 7
    assert bundle.summary["table_ids"] == ["TBL020"]


def test_terminal_boundary_specs_preserve_terminal_taxonomy_and_ebs_binding() -> None:
    module = _load_module()

    bundle = module.build_terminal_boundary_specs(_complete_records())
    specs = {spec.key: spec for spec in bundle.specs}

    assert all(
        spec.source_ledger_ids == tuple(f"P0R{number:05d}" for number in range(7073, 7081))
        for spec in specs.values()
    )
    assert specs["terminal_boundary.terminal_taxonomy"].source_equation_ids == (
        "P0R07075:terminal_taxonomy_t1_t7",
        "P0R07076:TBL020_terminal_taxonomy",
        "P0R07080:terminal_categories",
    )
    assert specs["terminal_boundary.ebs_binding"].source_equation_ids == (
        "P0R07074:ebs_boundary_object",
        "P0R07077:ebs_id_hash_binding",
        "P0R07080:reproducible_boundary_conditions",
    )
    assert "EBS ID and hash" in specs["terminal_boundary.claim_traceability"].canonical_statement
    assert all("no unbound empirical claim" in spec.claim_boundary for spec in specs.values())


def test_terminal_boundary_builder_rejects_missing_required_source_record() -> None:
    module = _load_module()
    incomplete = [record for record in _complete_records() if record["ledger_id"] != "P0R07077"]

    with pytest.raises(ValueError, match="missing required source ledger ids"):
        module.build_terminal_boundary_specs(incomplete)


def test_terminal_boundary_write_outputs_and_loader_access(tmp_path: Path) -> None:
    module = _load_module()
    bundle = module.build_terminal_boundary_specs(_complete_records())

    outputs = module.write_outputs(bundle, output_dir=tmp_path, date_tag="2099-01-02")

    payload = json.loads(outputs["json"].read_text(encoding="utf-8"))
    report = outputs["report"].read_text(encoding="utf-8")
    loaded = load_terminal_boundary_validation_spec(
        "terminal_boundary.claim_traceability",
        spec_bundle_path=outputs["json"],
    )

    assert payload["summary"]["coverage_match"] is True
    assert loaded["source_ledger_ids"][0] == "P0R07073"
    assert loaded["source_ledger_ids"][-1] == "P0R07080"
    assert "Paper 0 Terminal Boundary Specs" in report
    assert "no unbound empirical claim" in report
