# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 Preface II visionary spec tests
"""Tests for Paper 0 Preface II visionary-register promotion."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

from scpn_quantum_control.paper0.spec_loader import load_preface_ii_visionary_validation_spec

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "build_paper0_preface_ii_visionary_specs.py"


def _load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "build_paper0_preface_ii_visionary_specs", SCRIPT
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load Paper 0 Preface II visionary spec script")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _record(ledger_id: str, block_index: int, text: str = "source text") -> dict[str, object]:
    return {
        "ledger_id": ledger_id,
        "source_record_id": ledger_id.replace("P0R", "P0B"),
        "source_block_index": block_index,
        "section_path": "Part I > 1.1 > Preface II (The Visionary Voice)",
        "canonical_category": "validation_target",
        "math_ids": [],
        "text": text,
    }


def _complete_records() -> list[dict[str, object]]:
    source_text = {
        333: "Preface II (The Visionary Voice): The Architecture of Being",
        334: "manifesto; active, structuring principle of reality; living architecture",
        335: "Consciousness Engineering elevates description to creation",
        336: "manual, not commentary; practical guide to architecture of reality",
        337: "master architect, blueprint, and construction crew",
        338: "Field Architecture is the design language",
        339: "Consciousness Engineering tunes the fundamental field",
        340: "manual with diagrams, instructions, and testable plans",
        344: "vast, conscious Generative Model",
        345: "projection lattices and resonance hubs propagate priors",
        346: "intentionally updating the Generative Model; VIBRANA",
        348: "H_int = -lambda * Psi_s * sigma",
        349: "atlas of all possible sigma variables",
        350: "designing novel sigma variables with VIBRANA",
        352: "consciousness is a field",
        353: "projection lattices, resonance hubs, morphogenetic fields",
        354: "vibrational codes, symbolic geometries, biological coupling",
        355: "equations, operators, and diagrams are testable instruments",
        356: "manual; neural coherence, species fields, cosmic embeddings, VIBRANA",
        357: "take up the tools, test them, and extend them",
    }
    return [
        _record(f"P0R{number:05d}", number, source_text.get(number, "source text"))
        for number in range(333, 358)
    ]


def test_preface_ii_visionary_specs_consume_complete_source_span() -> None:
    module = _load_module()

    bundle = module.build_preface_ii_visionary_specs(_complete_records())

    assert bundle.summary["source_record_count"] == 25
    assert bundle.summary["consumed_source_record_count"] == 25
    assert bundle.summary["coverage_match"] is True
    assert bundle.summary["source_ledger_span"] == ["P0R00333", "P0R00357"]
    assert bundle.summary["spec_count"] == 5
    assert bundle.summary["blank_separator_count"] == 1
    assert bundle.summary["interaction_formula_count"] == 1
    assert bundle.summary["status_method_boundary"] == "P0R00358"


def test_preface_ii_visionary_specs_preserve_manifesto_and_sigma_boundaries() -> None:
    module = _load_module()

    bundle = module.build_preface_ii_visionary_specs(_complete_records())
    specs = {spec.key: spec for spec in bundle.specs}

    assert all(
        spec.source_ledger_ids == tuple(f"P0R{number:05d}" for number in range(333, 358))
        for spec in specs.values()
    )
    assert specs["preface_ii_visionary.manifesto_register_boundary"].source_equation_ids == (
        "P0R00333:preface_ii_architecture_of_being",
        "P0R00334:manifesto_structuring_principle",
        "P0R00337-P0R00340:plain_language_manifesto_and_manual",
    )
    assert (
        "vast, conscious Generative Model"
        in specs["preface_ii_visionary.active_inference_mapping"].source_formulae
    )
    assert (
        "H_int = -lambda * Psi_s * sigma"
        in specs["preface_ii_visionary.hamiltonian_mastery_boundary"].source_formulae
    )
    assert (
        "atlas of all possible sigma variables"
        in specs["preface_ii_visionary.sigma_atlas_design_language"].source_formulae
    )


def test_preface_ii_visionary_builder_rejects_missing_required_source_record() -> None:
    module = _load_module()
    incomplete = [record for record in _complete_records() if record["ledger_id"] != "P0R00348"]

    with pytest.raises(ValueError, match="missing required source ledger ids"):
        module.build_preface_ii_visionary_specs(incomplete)


def test_preface_ii_visionary_write_outputs_and_loader_access(tmp_path: Path) -> None:
    module = _load_module()
    bundle = module.build_preface_ii_visionary_specs(_complete_records())

    outputs = module.write_outputs(bundle, output_dir=tmp_path, date_tag="2099-01-02")

    payload = json.loads(outputs["json"].read_text(encoding="utf-8"))
    report = outputs["report"].read_text(encoding="utf-8")
    loaded = load_preface_ii_visionary_validation_spec(
        "preface_ii_visionary.hamiltonian_mastery_boundary",
        spec_bundle_path=outputs["json"],
    )

    assert payload["summary"]["coverage_match"] is True
    assert loaded["source_ledger_ids"][0] == "P0R00333"
    assert loaded["source_ledger_ids"][-1] == "P0R00357"
    assert "Paper 0 Preface II Visionary Specs" in report
    assert "H_int" in report
