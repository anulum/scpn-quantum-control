# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 category grammar spec tests
"""Tests for Paper 0 integration-synthesis category grammar promotion."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

from scpn_quantum_control.paper0.spec_loader import load_category_grammar_validation_spec

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "build_paper0_category_grammar_specs.py"


def _load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("build_paper0_category_grammar_specs", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load Paper 0 category grammar spec script")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _record(ledger_id: str, block_index: int, text: str = "source text") -> dict[str, object]:
    return {
        "ledger_id": ledger_id,
        "source_record_id": ledger_id.replace("P0R", "P0B"),
        "source_block_index": block_index,
        "section_path": "RAG Inserts",
        "canonical_category": "context",
        "math_ids": [],
        "text": text,
    }


def _complete_records() -> list[dict[str, object]]:
    return [_record(f"P0R{number:05d}", number) for number in range(6815, 6878)]


def test_category_grammar_specs_consume_complete_source_span() -> None:
    module = _load_module()

    bundle = module.build_category_grammar_specs(_complete_records())

    assert bundle.summary["source_record_count"] == 63
    assert bundle.summary["consumed_source_record_count"] == 63
    assert bundle.summary["coverage_match"] is True
    assert bundle.summary["source_ledger_span"] == ["P0R06815", "P0R06877"]
    assert bundle.summary["spec_count"] == 8
    assert bundle.summary["hardware_status"] == "formal_consistency_fixture_no_execution"
    assert bundle.summary["spec_keys"] == [
        "integration_synthesis.category_grammar.block_boundary",
        "integration_synthesis.category_grammar.scpn_category",
        "integration_synthesis.category_grammar.functorial_mappings",
        "integration_synthesis.category_grammar.topos_internal_logic",
        "integration_synthesis.category_grammar.kan_inference_mechanism",
        "integration_synthesis.category_grammar.string_diagram_calculus",
        "integration_synthesis.category_grammar.upde_category_application",
        "integration_synthesis.category_grammar.theorem_obligation_boundary",
    ]


def test_category_grammar_specs_keep_claims_formal_not_empirical() -> None:
    module = _load_module()

    bundle = module.build_category_grammar_specs(_complete_records())
    specs = {spec.key: spec for spec in bundle.specs}

    assert all(
        spec.source_ledger_ids == tuple(f"P0R{number:05d}" for number in range(6815, 6878))
        for spec in specs.values()
    )
    assert all("not empirical evidence" in spec.claim_boundary for spec in specs.values())
    assert all(spec.null_controls for spec in specs.values())
    assert all(spec.executable_validation_targets for spec in specs.values())
    assert specs["integration_synthesis.category_grammar.scpn_category"].source_equation_ids == (
        "P0R06822:projection_morphism",
        "P0R06824:identity",
        "P0R06825:composition",
    )
    assert "Yoneda" in " ".join(
        specs[
            "integration_synthesis.category_grammar.theorem_obligation_boundary"
        ].validation_targets
    )


def test_category_grammar_builder_rejects_missing_required_source_record() -> None:
    module = _load_module()
    incomplete = [record for record in _complete_records() if record["ledger_id"] != "P0R06858"]

    with pytest.raises(ValueError, match="missing required source ledger ids"):
        module.build_category_grammar_specs(incomplete)


def test_category_grammar_write_outputs_and_loader_access(tmp_path: Path) -> None:
    module = _load_module()
    bundle = module.build_category_grammar_specs(_complete_records())

    outputs = module.write_outputs(bundle, output_dir=tmp_path, date_tag="2099-01-02")

    payload = json.loads(outputs["json"].read_text(encoding="utf-8"))
    report = outputs["report"].read_text(encoding="utf-8")
    loaded = load_category_grammar_validation_spec(
        "integration_synthesis.category_grammar.upde_category_application",
        spec_bundle_path=outputs["json"],
    )

    assert payload["summary"]["coverage_match"] is True
    assert loaded["source_ledger_ids"][0] == "P0R06815"
    assert loaded["source_ledger_ids"][-1] == "P0R06877"
    assert "Paper 0 Category Grammar Specs" in report
    assert "not empirical evidence" in report
