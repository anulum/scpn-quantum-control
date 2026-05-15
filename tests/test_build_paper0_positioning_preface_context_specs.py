# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 positioning preface context spec tests
"""Tests for Paper 0 Positioning Preface context promotion."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

from scpn_quantum_control.paper0.spec_loader import (
    load_positioning_preface_context_validation_spec,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "build_paper0_positioning_preface_context_specs.py"


def _load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "build_paper0_positioning_preface_context_specs", SCRIPT
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load Paper 0 positioning preface context spec script")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _record(ledger_id: str, block_index: int, text: str = "source text") -> dict[str, object]:
    return {
        "ledger_id": ledger_id,
        "source_record_id": ledger_id.replace("P0R", "P0B"),
        "source_block_index": block_index,
        "section_path": "Positioning Preface",
        "canonical_category": "context_boundary",
        "math_ids": [],
        "text": text,
    }


def _complete_records() -> list[dict[str, object]]:
    source_text = {
        249: "Positioning Preface",
        250: "(Book II - The Sentient-Consciousness Projection Network)",
        252: "Consciousness is a structural principle of reality.",
        253: "Field Architecture studies and formalises those structures.",
        254: "Consciousness Engineering translates abstract principles into tools.",
        255: "Noetic Field Theory insists on rigour, explicit equations, and testable couplings.",
        256: "The chapters outline projection layers, VIBRANA, and symbolic operators.",
        259: "Any architecture that ascends layer by layer risks infinite regress.",
        260: "",
        261: "Author's Note",
        262: "This manuscript is designed to operate on two frequencies.",
        263: "The first preface is written in an academic register.",
        264: "The second preface is written in a visionary register.",
        266: "[IMAGE:]",
        267: "",
    }
    return [
        _record(f"P0R{number:05d}", number, source_text.get(number, "source text"))
        for number in range(249, 268)
    ]


def test_positioning_preface_context_specs_consume_complete_source_span() -> None:
    module = _load_module()

    bundle = module.build_positioning_preface_context_specs(_complete_records())

    assert bundle.summary["source_record_count"] == 19
    assert bundle.summary["consumed_source_record_count"] == 19
    assert bundle.summary["coverage_match"] is True
    assert bundle.summary["source_ledger_span"] == ["P0R00249", "P0R00267"]
    assert bundle.summary["spec_count"] == 5
    assert bundle.summary["blank_separator_count"] == 2
    assert bundle.summary["image_marker_count"] == 1
    assert bundle.summary["part_i_boundary"] == "P0R00268"


def test_positioning_preface_specs_preserve_discipline_claim_boundaries() -> None:
    module = _load_module()

    bundle = module.build_positioning_preface_context_specs(_complete_records())
    specs = {spec.key: spec for spec in bundle.specs}

    assert all(
        spec.source_ledger_ids == tuple(f"P0R{number:05d}" for number in range(249, 268))
        for spec in specs.values()
    )
    assert specs["positioning_preface_context.discipline_positioning"].source_equation_ids == (
        "P0R00251-P0R00255:discipline_positioning_claims",
    )
    assert (
        "Field Architecture"
        in specs["positioning_preface_context.discipline_positioning"].source_formulae
    )
    assert (
        "not validation evidence"
        in specs["positioning_preface_context.dual_register_author_note"].claim_boundary
    )


def test_positioning_preface_builder_rejects_missing_required_source_record() -> None:
    module = _load_module()
    incomplete = [record for record in _complete_records() if record["ledger_id"] != "P0R00255"]

    with pytest.raises(ValueError, match="missing required source ledger ids"):
        module.build_positioning_preface_context_specs(incomplete)


def test_positioning_preface_write_outputs_and_loader_access(tmp_path: Path) -> None:
    module = _load_module()
    bundle = module.build_positioning_preface_context_specs(_complete_records())

    outputs = module.write_outputs(bundle, output_dir=tmp_path, date_tag="2099-01-02")

    payload = json.loads(outputs["json"].read_text(encoding="utf-8"))
    report = outputs["report"].read_text(encoding="utf-8")
    loaded = load_positioning_preface_context_validation_spec(
        "positioning_preface_context.dual_register_author_note",
        spec_bundle_path=outputs["json"],
    )

    assert payload["summary"]["coverage_match"] is True
    assert loaded["source_ledger_ids"][0] == "P0R00249"
    assert loaded["source_ledger_ids"][-1] == "P0R00267"
    assert "Paper 0 Positioning Preface Context Specs" in report
    assert "not validation evidence" in report
