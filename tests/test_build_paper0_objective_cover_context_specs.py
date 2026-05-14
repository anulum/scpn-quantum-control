# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 objective cover context spec tests
"""Tests for Paper 0 objective and cover context promotion."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

from scpn_quantum_control.paper0.spec_loader import load_objective_cover_context_validation_spec

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "build_paper0_objective_cover_context_specs.py"


def _load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "build_paper0_objective_cover_context_specs", SCRIPT
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load Paper 0 objective cover context spec script")
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
    source_text = {
        218: "Objective:",
        219: "Mapping the Multi-Scale Interaction of Consciousness and Biology",
        220: "into a dynamic model of a self-organising, self-optimising, and self-correcting universe.",
        226: "[IMAGE:A black background with a black square]",
        227: "[IMAGE:Ein Bild, das Text, Schrift, Grafiken, Logo enthaelt.]",
        230: "SCPN architecture as a cosmic-scale active inference engine",
        231: "[IMAGE:Ein Bild, das Screenshot, Kreis, Grafiken, Grafikdesign enthaelt.]",
        236: "The model introduces a cyclic operator often referred to as the Meta Metatron Cycle.",
        240: "Book II:",
        241: "The Sentient-Consciousness Projection Network",
        242: "An Architecture for Reality",
    }
    return [
        _record(f"P0R{number:05d}", number, source_text.get(number, "source text"))
        for number in range(218, 249)
    ]


def test_objective_cover_context_specs_consume_complete_source_span() -> None:
    module = _load_module()

    bundle = module.build_objective_cover_context_specs(_complete_records())

    assert bundle.summary["source_record_count"] == 31
    assert bundle.summary["consumed_source_record_count"] == 31
    assert bundle.summary["coverage_match"] is True
    assert bundle.summary["source_ledger_span"] == ["P0R00218", "P0R00248"]
    assert bundle.summary["spec_count"] == 5
    assert bundle.summary["image_marker_count"] == 3
    assert bundle.summary["collection_book_count"] == 5
    assert bundle.summary["positioning_preface_boundary"] == "P0R00249"


def test_objective_cover_specs_preserve_positioning_targets_without_evidence_promotion() -> None:
    module = _load_module()

    bundle = module.build_objective_cover_context_specs(_complete_records())
    specs = {spec.key: spec for spec in bundle.specs}

    assert all(
        spec.source_ledger_ids == tuple(f"P0R{number:05d}" for number in range(218, 249))
        for spec in specs.values()
    )
    assert specs["objective_cover_context.cyclic_operator_positioning"].source_equation_ids == (
        "P0R00230:active_inference_engine_positioning",
        "P0R00232-P0R00235:participatory_recursive_loop_positioning",
        "P0R00236:meta_metatron_cycle_positioning",
    )
    assert any(
        "active inference engine" in formula
        for formula in specs["objective_cover_context.cyclic_operator_positioning"].source_formulae
    )
    assert (
        "not validation evidence"
        in specs["objective_cover_context.cyclic_operator_positioning"].claim_boundary
    )


def test_objective_cover_context_builder_rejects_missing_required_source_record() -> None:
    module = _load_module()
    incomplete = [record for record in _complete_records() if record["ledger_id"] != "P0R00236"]

    with pytest.raises(ValueError, match="missing required source ledger ids"):
        module.build_objective_cover_context_specs(incomplete)


def test_objective_cover_context_write_outputs_and_loader_access(tmp_path: Path) -> None:
    module = _load_module()
    bundle = module.build_objective_cover_context_specs(_complete_records())

    outputs = module.write_outputs(bundle, output_dir=tmp_path, date_tag="2099-01-02")

    payload = json.loads(outputs["json"].read_text(encoding="utf-8"))
    report = outputs["report"].read_text(encoding="utf-8")
    loaded = load_objective_cover_context_validation_spec(
        "objective_cover_context.cyclic_operator_positioning",
        spec_bundle_path=outputs["json"],
    )

    assert payload["summary"]["coverage_match"] is True
    assert loaded["source_ledger_ids"][0] == "P0R00218"
    assert loaded["source_ledger_ids"][-1] == "P0R00248"
    assert "Paper 0 Objective Cover Context Specs" in report
    assert "not validation evidence" in report
