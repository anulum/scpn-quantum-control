# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 opening foundation spec tests
"""Tests for Paper 0 opening foundation and global-boundary axiom promotion."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

from scpn_quantum_control.paper0.spec_loader import load_opening_foundation_validation_spec

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "build_paper0_opening_foundation_specs.py"


def _load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("build_paper0_opening_foundation_specs", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load Paper 0 opening foundation spec script")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _record(ledger_id: str, block_index: int) -> dict[str, object]:
    return {
        "ledger_id": ledger_id,
        "source_record_id": ledger_id.replace("P0R", "P0B"),
        "source_block_index": block_index,
        "section_path": "",
        "canonical_category": "validation_target",
        "math_ids": [],
        "text": "source text",
    }


def _complete_records() -> list[dict[str, object]]:
    return [_record(f"P0R{number:05d}", number) for number in range(1, 18)]


def test_opening_foundation_specs_consume_complete_source_span() -> None:
    module = _load_module()

    bundle = module.build_opening_foundation_specs(_complete_records())

    assert bundle.summary["source_record_count"] == 17
    assert bundle.summary["consumed_source_record_count"] == 17
    assert bundle.summary["coverage_match"] is True
    assert bundle.summary["source_ledger_span"] == ["P0R00001", "P0R00017"]
    assert bundle.summary["spec_count"] == 5
    assert bundle.summary["boundary_set_size"] == 4
    assert bundle.summary["terminal_count"] == 7


def test_opening_foundation_specs_preserve_global_axiom_equations() -> None:
    module = _load_module()

    bundle = module.build_opening_foundation_specs(_complete_records())
    specs = {spec.key: spec for spec in bundle.specs}

    assert all(
        spec.source_ledger_ids == tuple(f"P0R{number:05d}" for number in range(1, 18))
        for spec in specs.values()
    )
    assert specs["opening_foundation.global_boundary_axiom"].source_equation_ids == (
        "P0R00011:ebs_anchor",
        "P0R00013:C0_boundary_set",
        "P0R00014:F0_terminal_set",
        "P0R00016:beta0_boundary_assertion",
        "P0R00017:no_free_boundary_conditions",
    )
    assert "no free, untracked boundary conditions" in (
        specs["opening_foundation.global_boundary_axiom"].canonical_statement
    )
    assert all(
        "source-bounded opening foundation" in spec.claim_boundary for spec in specs.values()
    )


def test_opening_foundation_builder_rejects_missing_required_source_record() -> None:
    module = _load_module()
    incomplete = [record for record in _complete_records() if record["ledger_id"] != "P0R00013"]

    with pytest.raises(ValueError, match="missing required source ledger ids"):
        module.build_opening_foundation_specs(incomplete)


def test_opening_foundation_write_outputs_and_loader_access(tmp_path: Path) -> None:
    module = _load_module()
    bundle = module.build_opening_foundation_specs(_complete_records())

    outputs = module.write_outputs(bundle, output_dir=tmp_path, date_tag="2099-01-02")

    payload = json.loads(outputs["json"].read_text(encoding="utf-8"))
    report = outputs["report"].read_text(encoding="utf-8")
    loaded = load_opening_foundation_validation_spec(
        "opening_foundation.global_boundary_axiom",
        spec_bundle_path=outputs["json"],
    )

    assert payload["summary"]["coverage_match"] is True
    assert loaded["source_ledger_ids"][0] == "P0R00001"
    assert loaded["source_ledger_ids"][-1] == "P0R00017"
    assert "Paper 0 Opening Foundation Specs" in report
    assert "not empirical validation evidence" in report
