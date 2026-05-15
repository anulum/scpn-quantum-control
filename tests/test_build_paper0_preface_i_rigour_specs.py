# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 Preface I rigour spec tests
"""Tests for Paper 0 Preface I methodological-rigour promotion."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

from scpn_quantum_control.paper0.spec_loader import load_preface_i_rigour_validation_spec

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "build_paper0_preface_i_rigour_specs.py"


def _load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("build_paper0_preface_i_rigour_specs", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load Paper 0 Preface I rigour spec script")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _record(ledger_id: str, block_index: int, text: str = "source text") -> dict[str, object]:
    return {
        "ledger_id": ledger_id,
        "source_record_id": ledger_id.replace("P0R", "P0B"),
        "source_block_index": block_index,
        "section_path": "Part I > 1.1 > Preface I (The Academic Voice)",
        "canonical_category": "validation_target",
        "math_ids": [],
        "text": text,
    }


def _complete_records() -> list[dict[str, object]]:
    source_text = {
        307: "Preface I (The Academic Voice): The Mandate for Rigour",
        308: "third path; consciousness as a fundamental, physically real field phenomenon",
        309: "Field Architecture provides the toolkit for Consciousness Engineering",
        310: "Noetic Field Theory distinction; explicit equations and field operators",
        312: "Field Architecture studies consciousness blueprints",
        313: "Consciousness Engineering is practical application",
        318: "Field Architecture and Consciousness Engineering are congruent with HPC",
        319: "Field Architecture is the structure of the Generative Model",
        320: "Consciousness Engineering modulates Prediction Error",
        322: "H_int = -lambda * Psi_s * sigma",
        323: "Field Architecture identifies and characterises sigma",
        324: "Consciousness Engineering designs and controls sigma with VIBRANA",
        327: "active lattice, projection networks, resonance nodes, morphogenetic coupling",
        328: "experiments, simulations, and devices",
        329: "explicit equations, field operators, and layered models",
        330: "neural-morphogenetic synchronisation and VIBRANA",
        331: "discipline open to critique, extension, and integration",
    }
    return [
        _record(f"P0R{number:05d}", number, source_text.get(number, "source text"))
        for number in range(307, 333)
    ]


def test_preface_i_rigour_specs_consume_complete_source_span() -> None:
    module = _load_module()

    bundle = module.build_preface_i_rigour_specs(_complete_records())

    assert bundle.summary["source_record_count"] == 26
    assert bundle.summary["consumed_source_record_count"] == 26
    assert bundle.summary["coverage_match"] is True
    assert bundle.summary["source_ledger_span"] == ["P0R00307", "P0R00332"]
    assert bundle.summary["spec_count"] == 5
    assert bundle.summary["blank_separator_count"] == 2
    assert bundle.summary["interaction_formula_count"] == 1
    assert bundle.summary["preface_ii_boundary"] == "P0R00333"


def test_preface_i_rigour_specs_preserve_formalism_and_hpc_bridge() -> None:
    module = _load_module()

    bundle = module.build_preface_i_rigour_specs(_complete_records())
    specs = {spec.key: spec for spec in bundle.specs}

    assert all(
        spec.source_ledger_ids == tuple(f"P0R{number:05d}" for number in range(307, 333))
        for spec in specs.values()
    )
    assert specs["preface_i_rigour.methodological_third_path"].source_equation_ids == (
        "P0R00307:preface_i_mandate_for_rigour",
        "P0R00308:third_path_field_phenomenon",
        "P0R00311-P0R00314:plain_language_third_path_and_manual_boundary",
    )
    assert (
        "explicit equations" in specs["preface_i_rigour.formalism_noetic_boundary"].source_formulae
    )
    assert (
        "generative model structure"
        in specs["preface_i_rigour.hpc_structure_application_mapping"].source_formulae
    )
    assert (
        "H_int = -lambda * Psi_s * sigma"
        in specs["preface_i_rigour.sigma_programme_bridge"].source_formulae
    )


def test_preface_i_rigour_builder_rejects_missing_required_source_record() -> None:
    module = _load_module()
    incomplete = [record for record in _complete_records() if record["ledger_id"] != "P0R00322"]

    with pytest.raises(ValueError, match="missing required source ledger ids"):
        module.build_preface_i_rigour_specs(incomplete)


def test_preface_i_rigour_write_outputs_and_loader_access(tmp_path: Path) -> None:
    module = _load_module()
    bundle = module.build_preface_i_rigour_specs(_complete_records())

    outputs = module.write_outputs(bundle, output_dir=tmp_path, date_tag="2099-01-02")

    payload = json.loads(outputs["json"].read_text(encoding="utf-8"))
    report = outputs["report"].read_text(encoding="utf-8")
    loaded = load_preface_i_rigour_validation_spec(
        "preface_i_rigour.sigma_programme_bridge",
        spec_bundle_path=outputs["json"],
    )

    assert payload["summary"]["coverage_match"] is True
    assert loaded["source_ledger_ids"][0] == "P0R00307"
    assert loaded["source_ledger_ids"][-1] == "P0R00332"
    assert "Paper 0 Preface I Rigour Specs" in report
    assert "H_int" in report
