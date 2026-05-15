# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 Status and Method continuation spec tests
"""Tests for Paper 0 Status and Method continuation promotion."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

from scpn_quantum_control.paper0.spec_loader import (
    load_status_method_continuation_validation_spec,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "build_paper0_status_method_continuation_specs.py"


def _load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "build_paper0_status_method_continuation_specs", SCRIPT
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load Paper 0 Status and Method continuation spec script")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _record(ledger_id: str, block_index: int, text: str = "source text") -> dict[str, object]:
    return {
        "ledger_id": ledger_id,
        "source_record_id": ledger_id.replace("P0R", "P0B"),
        "source_block_index": block_index,
        "section_path": "Part I > 1.1 > Status and Method continuation",
        "canonical_category": "validation_target",
        "math_ids": [],
        "text": text,
    }


def _complete_records() -> list[dict[str, object]]:
    source_text = {
        391: "not absolute truths; not literalising metaphors; analogy-class; empirical standards",
        392: "Commitments (operational)",
        393: "Falsifiability first; Hypothesis registry; Tiered status; Versioning and correction",
        394: "How to read the axioms",
        395: "axioms as generative hypotheses; what follows and what fails",
        396: "How to disagree productively",
        397: "stated prediction and alternative baseline; analogy empirical handle; replacement model",
        398: "Standing invitation",
        399: "foundation stone, not capstone; refutes, reinforces, supersedes",
    }
    return [
        _record(
            f"P0R{number:05d}",
            number,
            "" if number == 400 else source_text.get(number, "source text"),
        )
        for number in range(391, 401)
    ]


def test_status_method_continuation_specs_consume_complete_source_span() -> None:
    module = _load_module()

    bundle = module.build_status_method_continuation_specs(_complete_records())

    assert bundle.summary["source_record_count"] == 10
    assert bundle.summary["consumed_source_record_count"] == 10
    assert bundle.summary["coverage_match"] is True
    assert bundle.summary["source_ledger_span"] == ["P0R00391", "P0R00400"]
    assert bundle.summary["spec_count"] == 5
    assert bundle.summary["blank_separator_count"] == 1
    assert bundle.summary["operational_commitment_count"] == 4
    assert bundle.summary["scp_mandate_boundary"] == "P0R00401"


def test_status_method_continuation_specs_preserve_boundaries_and_disagreement() -> None:
    module = _load_module()

    bundle = module.build_status_method_continuation_specs(_complete_records())
    specs = {spec.key: spec for spec in bundle.specs}

    assert all(
        spec.source_ledger_ids == tuple(f"P0R{number:05d}" for number in range(391, 401))
        for spec in specs.values()
    )
    assert specs[
        "status_method_continuation.not_absolute_truths_boundary"
    ].source_equation_ids == (
        "P0R00391:what_this_is_not_boundary",
        "P0R00392:operational_commitments_header",
    )
    assert (
        "generative hypotheses"
        in specs["status_method_continuation.axioms_as_generative_hypotheses"].source_formulae
    )
    assert (
        "same observables, stricter assumptions, stronger fit"
        in specs["status_method_continuation.productive_disagreement_protocol"].source_formulae
    )
    assert (
        "foundation stone, not a capstone"
        in specs["status_method_continuation.standing_invitation_closure"].source_formulae
    )


def test_status_method_continuation_builder_rejects_missing_required_source_record() -> None:
    module = _load_module()
    incomplete = [record for record in _complete_records() if record["ledger_id"] != "P0R00397"]

    with pytest.raises(ValueError, match="missing required source ledger ids"):
        module.build_status_method_continuation_specs(incomplete)


def test_status_method_continuation_write_outputs_and_loader_access(tmp_path: Path) -> None:
    module = _load_module()
    bundle = module.build_status_method_continuation_specs(_complete_records())

    outputs = module.write_outputs(bundle, output_dir=tmp_path, date_tag="2099-01-02")

    payload = json.loads(outputs["json"].read_text(encoding="utf-8"))
    report = outputs["report"].read_text(encoding="utf-8")
    loaded = load_status_method_continuation_validation_spec(
        "status_method_continuation.productive_disagreement_protocol",
        spec_bundle_path=outputs["json"],
    )

    assert payload["summary"]["coverage_match"] is True
    assert loaded["source_ledger_ids"][0] == "P0R00391"
    assert loaded["source_ledger_ids"][-1] == "P0R00400"
    assert "Paper 0 Status and Method Continuation Specs" in report
    assert "empirical handle" in report
