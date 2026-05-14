# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 front matter context spec tests
"""Tests for Paper 0 front matter and ToC context promotion."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

from scpn_quantum_control.paper0.spec_loader import load_front_matter_context_validation_spec

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "build_paper0_front_matter_context_specs.py"


def _load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "build_paper0_front_matter_context_specs", SCRIPT
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load Paper 0 front matter context spec script")
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
    records = [_record(f"P0R{number:05d}", number) for number in range(18, 105)]
    for number in range(59, 104):
        records[number - 18]["text"] = ""
    records[104 - 18]["text"] = "Note, this ToC is fragemented and currently incorect!"
    return records


def test_front_matter_context_specs_consume_complete_source_span() -> None:
    module = _load_module()

    bundle = module.build_front_matter_context_specs(_complete_records())

    assert bundle.summary["source_record_count"] == 87
    assert bundle.summary["consumed_source_record_count"] == 87
    assert bundle.summary["coverage_match"] is True
    assert bundle.summary["source_ledger_span"] == ["P0R00018", "P0R00104"]
    assert bundle.summary["spec_count"] == 5
    assert bundle.summary["collection_book_count"] == 5
    assert bundle.summary["layer_monograph_count"] == 16
    assert bundle.summary["blank_placeholder_count"] == 45
    assert bundle.summary["fragmented_toc_warning_present"] is True


def test_front_matter_context_specs_preserve_context_not_evidence_boundary() -> None:
    module = _load_module()

    bundle = module.build_front_matter_context_specs(_complete_records())
    specs = {spec.key: spec for spec in bundle.specs}

    assert all(
        spec.source_ledger_ids == tuple(f"P0R{number:05d}" for number in range(18, 105))
        for spec in specs.values()
    )
    assert specs["front_matter_context.fragmented_toc_warning"].source_equation_ids == (
        "P0R00104:fragmented_toc_warning",
    )
    assert specs["front_matter_context.blank_toc_placeholders"].source_equation_ids == (
        "P0R00059-P0R00103:blank_toc_placeholders",
    )
    assert (
        "not validation evidence"
        in specs["front_matter_context.master_publication_topology"].claim_boundary
    )
    assert (
        "fragemented and currently incorect"
        in specs["front_matter_context.fragmented_toc_warning"].source_formulae
    )


def test_front_matter_context_builder_rejects_missing_required_source_record() -> None:
    module = _load_module()
    incomplete = [record for record in _complete_records() if record["ledger_id"] != "P0R00104"]

    with pytest.raises(ValueError, match="missing required source ledger ids"):
        module.build_front_matter_context_specs(incomplete)


def test_front_matter_context_write_outputs_and_loader_access(tmp_path: Path) -> None:
    module = _load_module()
    bundle = module.build_front_matter_context_specs(_complete_records())

    outputs = module.write_outputs(bundle, output_dir=tmp_path, date_tag="2099-01-02")

    payload = json.loads(outputs["json"].read_text(encoding="utf-8"))
    report = outputs["report"].read_text(encoding="utf-8")
    loaded = load_front_matter_context_validation_spec(
        "front_matter_context.fragmented_toc_warning",
        spec_bundle_path=outputs["json"],
    )

    assert payload["summary"]["coverage_match"] is True
    assert loaded["source_ledger_ids"][0] == "P0R00018"
    assert loaded["source_ledger_ids"][-1] == "P0R00104"
    assert "Paper 0 Front Matter Context Specs" in report
    assert "not validation evidence" in report
