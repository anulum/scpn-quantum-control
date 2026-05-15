# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 layer monograph suite spec tests
"""Tests for Paper 0 layer monograph and validation-suite promotion."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

from scpn_quantum_control.paper0.spec_loader import load_layer_monograph_suite_validation_spec

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "build_paper0_layer_monograph_suite_specs.py"


def _load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "build_paper0_layer_monograph_suite_specs", SCRIPT
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load Paper 0 layer monograph suite spec script")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _record(ledger_id: str, block_index: int, text: str = "source text") -> dict[str, object]:
    return {
        "ledger_id": ledger_id,
        "source_record_id": ledger_id.replace("P0R", "P0B"),
        "source_block_index": block_index,
        "section_path": "Part I > 1.1 > Part II and Part III publication map",
        "canonical_category": "validation_target",
        "math_ids": [],
        "text": text,
    }


def _complete_records() -> list[dict[str, object]]:
    source_text = {
        436: "Part II: The 16 Layer-Specific Monographs",
        437: "Series I: Domain I - The Biological Substrate (Layers 1-4)",
        438: "Paper 1: Layer 1 - Quantum Biological",
        439: "Paper 2: Layer 2 - Neurochemical-Neurological",
        440: "Paper 3: Layer 3 - Genomic-Epigenomic-Morphogenetic",
        441: "Paper 4: Layer 4 - Cellular-Tissue Synchronisation",
        442: "Series II: Domain II - Organismal and Planetary Integration (Layers 5-8)",
        443: "Paper 5: Layer 5 - Organismal-Psychoemotional Feedback",
        444: "Paper 6: Layer 6 - Planetary-Biospheric",
        445: "Paper 7: Layer 7 - Geometrical-Symbolic",
        446: "Paper 8: Layer 8 - Cosmic Phase-Locking",
        447: "Series III: Domain III & IV - Memory, Control, and Collective Coherence",
        448: "Paper 9: Layer 9 - Memory Imprint-Existential Holograph",
        449: "Paper 10: Layer 10 - Projective Field Boundary Control",
        450: "Paper 11: Layer 11 - Noospheric-Cultural-Informational",
        451: "Paper 12: Layer 12 - Ecological-Gaian Synchrony",
        452: "Series IV: Domain V - Meta-Universal Integration (Layers 13-15)",
        453: "Paper 13: Layer 13 - Source-Field / Meta-Universal",
        454: "Paper 14: Layer 14 - Transdimensional Resonance",
        455: "Paper 15: Layer 15 - Consilium / Oversoul Integrator",
        456: "Series V: Domain VI - Cybernetic Closure (Meta-Layer 16)",
        457: "Paper 16: Meta-Layer 16",
        458: "Part III: The Critical Validation & Synthesis Suite",
        459: "Paper 17: The Methodological & Experimental Blueprint",
        460: "Paper 18: The Unified Simulation Architecture",
        461: "Paper 19: The Critical Dialogue & Falsifiability Roadmap",
        462: "Paper 20: The Coda - Philosophical Capstone",
    }
    return [
        _record(
            f"P0R{number:05d}",
            number,
            "" if number == 463 else source_text.get(number, "source text"),
        )
        for number in range(436, 464)
    ]


def test_layer_monograph_suite_specs_consume_complete_source_span() -> None:
    module = _load_module()

    bundle = module.build_layer_monograph_suite_specs(_complete_records())

    assert bundle.summary["source_record_count"] == 28
    assert bundle.summary["consumed_source_record_count"] == 28
    assert bundle.summary["coverage_match"] is True
    assert bundle.summary["source_ledger_span"] == ["P0R00436", "P0R00463"]
    assert bundle.summary["spec_count"] == 5
    assert bundle.summary["blank_separator_count"] == 1
    assert bundle.summary["layer_monograph_count"] == 16
    assert bundle.summary["validation_suite_paper_count"] == 4
    assert bundle.summary["next_source_boundary"] == "P0R00464"


def test_layer_monograph_suite_specs_preserve_domains_layers_and_validation_suite() -> None:
    module = _load_module()

    bundle = module.build_layer_monograph_suite_specs(_complete_records())
    specs = {spec.key: spec for spec in bundle.specs}

    assert all(
        spec.source_ledger_ids == tuple(f"P0R{number:05d}" for number in range(436, 464))
        for spec in specs.values()
    )
    assert specs["layer_monograph_suite.biological_substrate_layers"].source_equation_ids == (
        "P0R00437:domain_i_layers_1_4",
        "P0R00438:paper1_layer1_quantum_biological",
        "P0R00441:paper4_layer4_cellular_tissue_synchronisation",
    )
    assert (
        "Domain III & IV - Memory, Control, and Collective Coherence"
        in specs[
            "layer_monograph_suite.memory_control_collective_coherence_layers"
        ].source_formulae
    )
    assert (
        "Meta-Layer 16"
        in specs["layer_monograph_suite.meta_universal_and_cybernetic_layers"].source_formulae
    )
    assert (
        "The Critical Dialogue & Falsifiability Roadmap"
        in specs["layer_monograph_suite.critical_validation_synthesis_suite"].source_formulae
    )


def test_layer_monograph_suite_builder_rejects_missing_required_source_record() -> None:
    module = _load_module()
    incomplete = [record for record in _complete_records() if record["ledger_id"] != "P0R00457"]

    with pytest.raises(ValueError, match="missing required source ledger ids"):
        module.build_layer_monograph_suite_specs(incomplete)


def test_layer_monograph_suite_write_outputs_and_loader_access(tmp_path: Path) -> None:
    module = _load_module()
    bundle = module.build_layer_monograph_suite_specs(_complete_records())

    outputs = module.write_outputs(bundle, output_dir=tmp_path, date_tag="2099-01-02")

    payload = json.loads(outputs["json"].read_text(encoding="utf-8"))
    report = outputs["report"].read_text(encoding="utf-8")
    loaded = load_layer_monograph_suite_validation_spec(
        "layer_monograph_suite.critical_validation_synthesis_suite",
        spec_bundle_path=outputs["json"],
    )

    assert payload["summary"]["coverage_match"] is True
    assert loaded["source_ledger_ids"][0] == "P0R00436"
    assert loaded["source_ledger_ids"][-1] == "P0R00463"
    assert "Paper 0 Layer Monograph Suite Specs" in report
    assert "Paper 18: The Unified Simulation Architecture" in report
