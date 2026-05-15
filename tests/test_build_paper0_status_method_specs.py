# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 Status and Method spec tests
"""Tests for Paper 0 Status and Method promotion."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

from scpn_quantum_control.paper0.spec_loader import load_status_method_validation_spec

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "build_paper0_status_method_specs.py"


def _load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("build_paper0_status_method_specs", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load Paper 0 Status and Method spec script")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _record(ledger_id: str, block_index: int, text: str = "source text") -> dict[str, object]:
    return {
        "ledger_id": ledger_id,
        "source_record_id": ledger_id.replace("P0R", "P0B"),
        "source_block_index": block_index,
        "section_path": "Part I > 1.1 > Status and Method",
        "canonical_category": "validation_target",
        "math_ids": [],
        "text": text,
    }


def _complete_records() -> list[dict[str, object]]:
    source_text = {
        358: "Status and Method: A Living Research Programme",
        359: "not static doctrine; version-controlled system of hypotheses",
        360: "Falsifiability first; Hypothesis registry; Tiered status; Versioning and correction",
        362: "working model designed to make predictions; not religion or absolute truths",
        363: "testable; change map when evidence says wrong; public predictions",
        367: "FEP as operating methodology; active inference for scientific discovery",
        369: "theory as generative model with predictions",
        371: "compute, simulate, measure, and compare",
        373: "negative result is quantified prediction error",
        375: "versioning and correction update the generative model",
        378: "H_int = -lambda * Psi_s * sigma quality control",
        380: "falsifiability first requires empirically accessible sigma",
        382: "analogy-class usage requires empirical handle",
        384: "replacement model, tensor sigma_ij instead of scalar sigma",
        387: "research programme, not a finished doctrine",
        389: "structured framework; compute, simulate, measure, and compare",
        390: "What this is not",
    }
    return [
        _record(
            f"P0R{number:05d}",
            number,
            "" if number in (364, 385) else source_text.get(number, "source text"),
        )
        for number in range(358, 391)
    ]


def test_status_method_specs_consume_complete_source_span() -> None:
    module = _load_module()

    bundle = module.build_status_method_specs(_complete_records())

    assert bundle.summary["source_record_count"] == 33
    assert bundle.summary["consumed_source_record_count"] == 33
    assert bundle.summary["coverage_match"] is True
    assert bundle.summary["source_ledger_span"] == ["P0R00358", "P0R00390"]
    assert bundle.summary["spec_count"] == 5
    assert bundle.summary["blank_separator_count"] == 2
    assert bundle.summary["method_commitment_count"] == 4
    assert bundle.summary["next_boundary"] == "P0R00391"


def test_status_method_specs_preserve_fep_and_h_int_quality_control() -> None:
    module = _load_module()

    bundle = module.build_status_method_specs(_complete_records())
    specs = {spec.key: spec for spec in bundle.specs}

    assert all(
        spec.source_ledger_ids == tuple(f"P0R{number:05d}" for number in range(358, 391))
        for spec in specs.values()
    )
    assert specs["status_method.living_research_programme"].source_equation_ids == (
        "P0R00358:status_method_title",
        "P0R00359:version_controlled_hypotheses",
        "P0R00360:operational_commitments",
        "P0R00387-P0R00390:research_programme_not_doctrine",
    )
    assert "falsifiability_first" in specs["status_method.operational_commitments"].variables
    assert (
        "negative result as prediction error"
        in specs["status_method.fep_scientific_methodology"].source_formulae
    )
    assert (
        "H_int = -lambda * Psi_s * sigma"
        in specs["status_method.h_int_quality_control"].source_formulae
    )


def test_status_method_builder_rejects_missing_required_source_record() -> None:
    module = _load_module()
    incomplete = [record for record in _complete_records() if record["ledger_id"] != "P0R00378"]

    with pytest.raises(ValueError, match="missing required source ledger ids"):
        module.build_status_method_specs(incomplete)


def test_status_method_write_outputs_and_loader_access(tmp_path: Path) -> None:
    module = _load_module()
    bundle = module.build_status_method_specs(_complete_records())

    outputs = module.write_outputs(bundle, output_dir=tmp_path, date_tag="2099-01-02")

    payload = json.loads(outputs["json"].read_text(encoding="utf-8"))
    report = outputs["report"].read_text(encoding="utf-8")
    loaded = load_status_method_validation_spec(
        "status_method.h_int_quality_control",
        spec_bundle_path=outputs["json"],
    )

    assert payload["summary"]["coverage_match"] is True
    assert loaded["source_ledger_ids"][0] == "P0R00358"
    assert loaded["source_ledger_ids"][-1] == "P0R00390"
    assert "Paper 0 Status and Method Specs" in report
    assert "H_int" in report
