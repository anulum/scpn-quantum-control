# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 Foreword coupling spec tests
"""Tests for Paper 0 Foreword predictive-coding and Psi-field coupling promotion."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

from scpn_quantum_control.paper0.spec_loader import load_foreword_coupling_validation_spec

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "build_paper0_foreword_coupling_specs.py"


def _load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("build_paper0_foreword_coupling_specs", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load Paper 0 Foreword coupling spec script")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _record(ledger_id: str, block_index: int, text: str = "source text") -> dict[str, object]:
    return {
        "ledger_id": ledger_id,
        "source_record_id": ledger_id.replace("P0R", "P0B"),
        "source_block_index": block_index,
        "section_path": "Part I > 1.1 > 1.1.1 Foreword",
        "canonical_category": "validation_target",
        "math_ids": [],
        "text": text,
    }


def _complete_records() -> list[dict[str, object]]:
    source_text = {
        268: "Part I: The Foundational Bedrock",
        273: "methodological transition from Book I physics to Book II architecture",
        274: "15-layer hierarchical model for bidirectional flow",
        282: "cosmic-scale Hierarchical Predictive Coding architecture",
        284: "Downward projection as generative model",
        286: "Upward feedback as prediction error flow",
        287: "SCPN is an active inference engine",
        289: "universal Psi-field interacts with material systems",
        290: "H_int = -lambda * Psi_s * sigma.",
        292: "specific collective state variable sigma for each layer",
        293: "Layer 1 microtubule net electric dipole moment",
        294: "Layer 2 gamma global phase synchrony",
        295: "Layer 6 atmospheric oxygen concentration or global temperature",
        296: "empirically testable sigma for corresponding layer",
        305: "[IMAGE:architecture diagram]",
        306: "Layered architecture diagram",
    }
    return [
        _record(f"P0R{number:05d}", number, source_text.get(number, "source text"))
        for number in range(268, 307)
    ]


def test_foreword_coupling_specs_consume_complete_source_span() -> None:
    module = _load_module()

    bundle = module.build_foreword_coupling_specs(_complete_records())

    assert bundle.summary["source_record_count"] == 39
    assert bundle.summary["consumed_source_record_count"] == 39
    assert bundle.summary["coverage_match"] is True
    assert bundle.summary["source_ledger_span"] == ["P0R00268", "P0R00306"]
    assert bundle.summary["spec_count"] == 5
    assert bundle.summary["sigma_layer_example_count"] == 3
    assert bundle.summary["image_marker_count"] == 1
    assert bundle.summary["preface_i_boundary"] == "P0R00307"


def test_foreword_coupling_specs_preserve_h_int_and_hpc_channels() -> None:
    module = _load_module()

    bundle = module.build_foreword_coupling_specs(_complete_records())
    specs = {spec.key: spec for spec in bundle.specs}

    assert all(
        spec.source_ledger_ids == tuple(f"P0R{number:05d}" for number in range(268, 307))
        for spec in specs.values()
    )
    assert specs["foreword_coupling.psi_field_interaction_hamiltonian"].source_equation_ids == (
        "P0R00289:psi_field_material_coupling_question",
        "P0R00290:H_int_formula",
        "P0R00291-P0R00296:layer_sigma_identification_programme",
    )
    assert (
        "H_int = -lambda * Psi_s * sigma"
        in specs["foreword_coupling.psi_field_interaction_hamiltonian"].source_formulae
    )
    assert (
        "downward projection"
        in specs["foreword_coupling.predictive_coding_channels"].source_formulae
    )


def test_foreword_coupling_builder_rejects_missing_required_source_record() -> None:
    module = _load_module()
    incomplete = [record for record in _complete_records() if record["ledger_id"] != "P0R00290"]

    with pytest.raises(ValueError, match="missing required source ledger ids"):
        module.build_foreword_coupling_specs(incomplete)


def test_foreword_coupling_write_outputs_and_loader_access(tmp_path: Path) -> None:
    module = _load_module()
    bundle = module.build_foreword_coupling_specs(_complete_records())

    outputs = module.write_outputs(bundle, output_dir=tmp_path, date_tag="2099-01-02")

    payload = json.loads(outputs["json"].read_text(encoding="utf-8"))
    report = outputs["report"].read_text(encoding="utf-8")
    loaded = load_foreword_coupling_validation_spec(
        "foreword_coupling.psi_field_interaction_hamiltonian",
        spec_bundle_path=outputs["json"],
    )

    assert payload["summary"]["coverage_match"] is True
    assert loaded["source_ledger_ids"][0] == "P0R00268"
    assert loaded["source_ledger_ids"][-1] == "P0R00306"
    assert "Paper 0 Foreword Coupling Specs" in report
    assert "H_int" in report
