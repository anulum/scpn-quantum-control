# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 Anulum Collection mandate spec tests
"""Tests for Paper 0 Anulum Collection mandate promotion."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

from scpn_quantum_control.paper0.spec_loader import (
    load_anulum_collection_mandate_validation_spec,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "build_paper0_anulum_collection_mandate_specs.py"


def _load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "build_paper0_anulum_collection_mandate_specs", SCRIPT
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load Paper 0 Anulum Collection mandate spec script")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _record(ledger_id: str, block_index: int, text: str = "source text") -> dict[str, object]:
    return {
        "ledger_id": ledger_id,
        "source_record_id": ledger_id.replace("P0R", "P0B"),
        "source_block_index": block_index,
        "section_path": "Part I > 1.1 > The Anulum Collection & The SCPN Mandate",
        "canonical_category": "validation_target",
        "math_ids": [],
        "text": text,
    }


def _complete_records() -> list[dict[str, object]]:
    source_text = {
        401: "The Anulum Collection & The SCPN Mandate",
        402: "comprehensive multi-year research programme and intellectual architecture",
        403: "Paper 0 establishes universal axioms, fundamental field equations, architectural principles",
        404: "Papers 17-20 provide Critical Validation & Synthesis Suite",
        405: "university course curriculum guide",
        406: "five main Books",
        407: "Book I School of Physics",
        408: "Book II School of Engineering and Architecture",
        409: "Book III School of Philosophy",
        410: "Book IV Graduate Seminar",
        411: "Book V Technical College",
        412: "Paper 0 prerequisite course",
        413: "final exams detail experiments, simulations, toughest questions",
        415: "Meta-Framework Integrations",
        416: "Predictive Coding Integration",
        417: "HPC framework applied to the scientific process",
        418: "Paper 0 as the Deep Priors",
        419: "Papers 1-16 as the Generative Cascade",
        420: "Part III as Prediction Error Minimisation",
        421: "Psi_s Field Coupling Integration",
        422: "H_int = -lambda * Psi_s * sigma across all domains",
        423: "Paper 0 defines universal equation",
        424: "Papers 1-16 isolate and define sigma for each layer",
        425: "Part III provides tools to measure lambda",
        427: "anulum collection and where you are",
        428: "Book I - The Anulum Framework",
        429: "Book II - The Sentient-Consciousness Projection Network",
        430: "Book III - Metatron's Coda",
        431: "Book IV - The Godelian Koans",
        432: "Book V - VIBRANA",
        433: "The SCPN Master Publications - Table of Content",
        434: "Part I: The Foundational Framework",
        435: "Paper 0: The Foundational Framework - You are Here",
    }
    return [
        _record(
            f"P0R{number:05d}",
            number,
            "" if number in (414, 426) else source_text.get(number, "source text"),
        )
        for number in range(401, 436)
    ]


def test_anulum_collection_mandate_specs_consume_complete_source_span() -> None:
    module = _load_module()

    bundle = module.build_anulum_collection_mandate_specs(_complete_records())

    assert bundle.summary["source_record_count"] == 35
    assert bundle.summary["consumed_source_record_count"] == 35
    assert bundle.summary["coverage_match"] is True
    assert bundle.summary["source_ledger_span"] == ["P0R00401", "P0R00435"]
    assert bundle.summary["spec_count"] == 5
    assert bundle.summary["blank_separator_count"] == 2
    assert bundle.summary["book_count"] == 5
    assert bundle.summary["validation_suite_range"] == ["Papers 17", "Papers 20"]


def test_anulum_collection_mandate_specs_preserve_equation_and_process_boundaries() -> None:
    module = _load_module()

    bundle = module.build_anulum_collection_mandate_specs(_complete_records())
    specs = {spec.key: spec for spec in bundle.specs}

    assert all(
        spec.source_ledger_ids == tuple(f"P0R{number:05d}" for number in range(401, 436))
        for spec in specs.values()
    )
    assert specs[
        "anulum_collection_mandate.psi_field_coupling_empirical_plan"
    ].source_equation_ids == (
        "P0R00422:H_int=-lambda*Psi_s*sigma",
        "P0R00424:sigma_layer_isolation",
        "P0R00425:lambda_measurement_tools",
    )
    assert (
        "H_int = -lambda * Psi_s * sigma"
        in specs["anulum_collection_mandate.psi_field_coupling_empirical_plan"].source_formulae
    )
    assert (
        "Part III as Prediction Error Minimisation"
        in specs["anulum_collection_mandate.predictive_coding_research_process"].source_formulae
    )
    assert (
        "Paper 0: The Foundational Framework - You are Here"
        in specs["anulum_collection_mandate.master_publication_map"].source_formulae
    )


def test_anulum_collection_mandate_builder_rejects_missing_required_source_record() -> None:
    module = _load_module()
    incomplete = [record for record in _complete_records() if record["ledger_id"] != "P0R00422"]

    with pytest.raises(ValueError, match="missing required source ledger ids"):
        module.build_anulum_collection_mandate_specs(incomplete)


def test_anulum_collection_mandate_write_outputs_and_loader_access(tmp_path: Path) -> None:
    module = _load_module()
    bundle = module.build_anulum_collection_mandate_specs(_complete_records())

    outputs = module.write_outputs(bundle, output_dir=tmp_path, date_tag="2099-01-02")

    payload = json.loads(outputs["json"].read_text(encoding="utf-8"))
    report = outputs["report"].read_text(encoding="utf-8")
    loaded = load_anulum_collection_mandate_validation_spec(
        "anulum_collection_mandate.psi_field_coupling_empirical_plan",
        spec_bundle_path=outputs["json"],
    )

    assert payload["summary"]["coverage_match"] is True
    assert loaded["source_ledger_ids"][0] == "P0R00401"
    assert loaded["source_ledger_ids"][-1] == "P0R00435"
    assert "Paper 0 Anulum Collection Mandate Specs" in report
    assert "H_int = -lambda * Psi_s * sigma" in report
