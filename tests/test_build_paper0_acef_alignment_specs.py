# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 A-CEF alignment spec tests
"""Tests for Paper 0 A-CEF ethical-alignment promotion."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

from scpn_quantum_control.paper0.spec_loader import load_acef_alignment_validation_spec

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "build_paper0_acef_alignment_specs.py"


def _load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("build_paper0_acef_alignment_specs", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load Paper 0 A-CEF alignment spec script")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _record(ledger_id: str, block_index: int) -> dict[str, object]:
    return {
        "ledger_id": ledger_id,
        "source_record_id": ledger_id.replace("P0R", "P0B"),
        "source_block_index": block_index,
        "section_path": "Paper 0 Ethical Implications and A-CEF",
        "math_ids": ["MATH_ACEF"] if ledger_id == "P0R06246" else [],
        "image_ids": [],
        "text": "source text",
    }


def _complete_records() -> list[dict[str, object]]:
    return [_record(f"P0R{number:05d}", number) for number in range(6233, 6251)]


def test_acef_alignment_specs_consume_complete_contiguous_source_span() -> None:
    module = _load_module()

    bundle = module.build_acef_alignment_specs(_complete_records())

    assert bundle.summary["source_record_count"] == 18
    assert bundle.summary["consumed_source_record_count"] == 18
    assert bundle.summary["coverage_match"] is True
    assert bundle.summary["source_ledger_span"] == ["P0R06233", "P0R06250"]
    assert bundle.summary["spec_count"] == 5
    assert bundle.summary["spec_keys"] == [
        "acef_alignment.is_ought_claim_boundary",
        "acef_alignment.governance_quasicriticality_metric",
        "acef_alignment.ai_alignment_risk_boundary",
        "acef_alignment.algorithmic_causal_entropic_force",
        "acef_alignment.consequence_phase_steering",
    ]


def test_acef_alignment_specs_preserve_equation_and_claim_boundary() -> None:
    module = _load_module()

    bundle = module.build_acef_alignment_specs(_complete_records())
    specs = {spec.key: spec for spec in bundle.specs}

    acef = specs["acef_alignment.algorithmic_causal_entropic_force"]

    assert acef.source_equation_ids == ("P0R06246",)
    assert acef.anchor_math_ids == ("MATH_ACEF",)
    assert "F_A-CEF = T_A grad_X S_C(X,tau)" in acef.formal_statement
    assert all(spec.validation_targets for spec in specs.values())
    assert all(spec.null_controls for spec in specs.values())
    assert all(
        spec.implementation_status == "implemented_executable_fixture" for spec in specs.values()
    )
    assert all("not empirical evidence" in spec.claim_boundary for spec in specs.values())


def test_acef_alignment_builder_rejects_missing_required_source_record() -> None:
    module = _load_module()
    incomplete = [record for record in _complete_records() if record["ledger_id"] != "P0R06246"]

    with pytest.raises(ValueError, match="missing required source ledger ids"):
        module.build_acef_alignment_specs(incomplete)


def test_acef_alignment_write_outputs_and_loader_access(tmp_path: Path) -> None:
    module = _load_module()
    bundle = module.build_acef_alignment_specs(_complete_records())

    outputs = module.write_outputs(bundle, output_dir=tmp_path, date_tag="2099-01-02")

    payload = json.loads(outputs["json"].read_text(encoding="utf-8"))
    report = outputs["report"].read_text(encoding="utf-8")
    loaded = load_acef_alignment_validation_spec(
        "acef_alignment.algorithmic_causal_entropic_force",
        spec_bundle_path=outputs["json"],
    )

    assert payload["summary"]["coverage_match"] is True
    assert loaded["source_ledger_ids"] == [f"P0R{number:05d}" for number in range(6233, 6251)]
    assert loaded["source_equation_ids"] == ["P0R06246"]
    assert "Paper 0 A-CEF Alignment Specs" in report
    assert "not empirical evidence" in report
