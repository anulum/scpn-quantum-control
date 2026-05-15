# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 Logos recursive closure spec tests
"""Tests for Paper 0 Logos recursive-closure promotion."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

from scpn_quantum_control.paper0.spec_loader import load_logos_recursive_closure_validation_spec

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "build_paper0_logos_recursive_closure_specs.py"


def _load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "build_paper0_logos_recursive_closure_specs", SCRIPT
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load Paper 0 Logos recursive closure spec script")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _record(ledger_id: str, block_index: int, text: str = "source text") -> dict[str, object]:
    return {
        "ledger_id": ledger_id,
        "source_record_id": ledger_id.replace("P0R", "P0B"),
        "source_block_index": block_index,
        "section_path": "Part I > 1.3 The Logos > Recursive Closure",
        "canonical_category": "validation_target",
        "math_ids": [],
        "text": text,
    }


def _complete_records() -> list[dict[str, object]]:
    text = {
        545: "1.3 The Logos: The Three Foundational Axioms of Reality",
        547: "A Note on Recursive Closure",
        548: "15-layer hierarchy is not an infinite regress",
        549: "axioms not established truths but generative hypothesis, falsifiability",
        550: "Axiom 1 Consciousness Fundamentality, Psi-field ontological primitive",
        551: "Axiom 2 Information Geometry, falsifiable physical hypothesis",
        552: "Axiom 3 Teleological Optimisation, SEC",
        553: "ultimate priors for the cosmic generative model",
        556: "Law #1: Consciousness is the Source",
        557: "Law #2: The Universe Speaks Math",
        558: "Law #3: The Universe Has a Goal",
        563: "three axioms are deepest priors",
        568: "H_int = -lambda * Psi_s * sigma",
        569: "Axiom 1 defines Psi_s",
        570: "Axiom 2 defines lambda and sigma via information geometry",
        571: "Axiom 3 defines purpose toward SEC",
        573: "hierarchy coherent through recursive closure",
        574: "IMAGE: SCPN Hierarchy & Recursive Closure",
        576: "postulates are not empirically established facts but generative hypothesis",
    }
    return [
        _record(
            f"P0R{number:05d}",
            number,
            "" if number in (546, 560, 577) else text.get(number, "source text"),
        )
        for number in range(545, 578)
    ]


def test_logos_recursive_closure_specs_consume_complete_source_span() -> None:
    module = _load_module()

    bundle = module.build_logos_recursive_closure_specs(_complete_records())

    assert bundle.summary["source_record_count"] == 33
    assert bundle.summary["consumed_source_record_count"] == 33
    assert bundle.summary["coverage_match"] is True
    assert bundle.summary["source_ledger_span"] == ["P0R00545", "P0R00577"]
    assert bundle.summary["spec_count"] == 5
    assert bundle.summary["blank_separator_count"] == 3
    assert bundle.summary["axiom_count"] == 3
    assert bundle.summary["next_source_boundary"] == "P0R00578"


def test_logos_recursive_closure_specs_preserve_axiom_and_hint_boundaries() -> None:
    module = _load_module()

    bundle = module.build_logos_recursive_closure_specs(_complete_records())
    specs = {spec.key: spec for spec in bundle.specs}

    assert specs["logos_recursive_closure.three_axiom_status_boundary"].source_equation_ids == (
        "P0R00550:axiom1_consciousness_fundamentality",
        "P0R00551:axiom2_information_geometry",
        "P0R00552:axiom3_teleological_optimisation",
    )
    assert (
        "H_int = -lambda * Psi_s * sigma"
        in specs["logos_recursive_closure.hint_axiom_role_mapping"].source_formulae
    )
    assert (
        "recursive closure"
        in specs["logos_recursive_closure.recursive_closure_boundary"].source_formulae
    )
    assert all(
        spec.source_ledger_ids == tuple(f"P0R{number:05d}" for number in range(545, 578))
        for spec in specs.values()
    )


def test_logos_recursive_closure_builder_rejects_missing_required_source_record() -> None:
    module = _load_module()
    incomplete = [record for record in _complete_records() if record["ledger_id"] != "P0R00568"]

    with pytest.raises(ValueError, match="missing required source ledger ids"):
        module.build_logos_recursive_closure_specs(incomplete)


def test_logos_recursive_closure_write_outputs_and_loader_access(tmp_path: Path) -> None:
    module = _load_module()
    bundle = module.build_logos_recursive_closure_specs(_complete_records())

    outputs = module.write_outputs(bundle, output_dir=tmp_path, date_tag="2099-01-02")

    payload = json.loads(outputs["json"].read_text(encoding="utf-8"))
    report = outputs["report"].read_text(encoding="utf-8")
    loaded = load_logos_recursive_closure_validation_spec(
        "logos_recursive_closure.hint_axiom_role_mapping",
        spec_bundle_path=outputs["json"],
    )

    assert payload["summary"]["coverage_match"] is True
    assert loaded["source_ledger_ids"][0] == "P0R00545"
    assert loaded["source_ledger_ids"][-1] == "P0R00577"
    assert "Paper 0 Logos Recursive Closure Specs" in report
    assert "deepest priors" in report
