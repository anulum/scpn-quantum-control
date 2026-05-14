# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 glial slow-control spec tests
"""Tests for Paper 0 glial-neuronal slow-control promotion."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

from scpn_quantum_control.paper0.spec_loader import load_glial_slow_control_validation_spec

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "build_paper0_glial_slow_control_specs.py"


def _load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("build_paper0_glial_slow_control_specs", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load Paper 0 glial slow-control spec script")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _record(ledger_id: str, block_index: int) -> dict[str, object]:
    return {
        "ledger_id": ledger_id,
        "source_record_id": ledger_id.replace("P0R", "P0B"),
        "source_block_index": block_index,
        "section_path": "Paper 0 glial-neuronal slow control",
        "math_ids": [],
        "image_ids": [ledger_id.replace("P0R", "IMG")]
        if ledger_id in {"P0R06417", "P0R06426", "P0R06428"}
        else [],
        "text": "source text",
    }


def _complete_records() -> list[dict[str, object]]:
    return [_record(f"P0R{number:05d}", number) for number in range(6414, 6434)]


def test_glial_slow_control_specs_consume_complete_contiguous_source_span() -> None:
    module = _load_module()

    bundle = module.build_glial_slow_control_specs(_complete_records())

    assert bundle.summary["source_record_count"] == 20
    assert bundle.summary["consumed_source_record_count"] == 20
    assert bundle.summary["coverage_match"] is True
    assert bundle.summary["source_ledger_span"] == ["P0R06414", "P0R06433"]
    assert bundle.summary["structural_source_ledger_ids"] == [
        "P0R06414",
        "P0R06417",
        "P0R06421",
        "P0R06423",
        "P0R06426",
        "P0R06428",
    ]
    assert bundle.summary["caption_source_ledger_ids"] == ["P0R06418", "P0R06426", "P0R06429"]
    assert bundle.summary["protocol_step_ledger_ids"] == [
        "P0R06430",
        "P0R06431",
        "P0R06432",
        "P0R06433",
    ]
    assert bundle.summary["spec_count"] == 4
    assert bundle.summary["spec_keys"] == [
        "glial_slow_control.two_timescale_governor",
        "glial_slow_control.homeostatic_feedback_channels",
        "glial_slow_control.experimental_protocol_catalogue",
        "glial_slow_control.falsification_and_causal_decoupling",
    ]


def test_glial_slow_control_specs_preserve_protocol_and_falsification_mechanisms() -> None:
    module = _load_module()

    bundle = module.build_glial_slow_control_specs(_complete_records())
    specs = {spec.key: spec for spec in bundle.specs}

    assert all(spec.validation_targets for spec in specs.values())
    assert all(spec.null_controls for spec in specs.values())
    assert all("not empirical evidence" in spec.claim_boundary for spec in specs.values())
    assert specs["glial_slow_control.two_timescale_governor"].source_mechanisms == (
        "fast neuronal loop operates at ms-to-100s-ms timescale",
        "slow astrocyte loop operates at seconds-to-minutes timescale",
        "slow glial feedback prevents supercritical and subcritical drift",
    )
    assert specs["glial_slow_control.homeostatic_feedback_channels"].source_mechanisms == (
        "astrocyte Ca2+ waves integrate neuronal activity",
        "gliotransmitters modulate synaptic plasticity and excitability",
        "background ion concentrations and neurotransmitter availability are control channels",
    )
    assert specs["glial_slow_control.experimental_protocol_catalogue"].source_protocol_steps == (
        "dual reporters: GCaMP neurons and jRGECO1a astrocytes with chronic window",
        "simultaneous two-photon Ca2+ imaging and Neuropixels recording",
        "analyse avalanches, estimate tau/sigma, and integrate Ca2+",
        "block gliotransmission and predict decoupling of Ca2+ from criticality metrics",
    )
    assert specs["glial_slow_control.falsification_and_causal_decoupling"].source_formulae == (
        "P(S) proportional_to S^(-tau)",
        "correlate integrated astrocyte Ca2+ with tau or sigma",
        "gliotransmission block predicts Ca2+/criticality decoupling",
    )


def test_glial_slow_control_builder_rejects_missing_source_record() -> None:
    module = _load_module()
    incomplete = [record for record in _complete_records() if record["ledger_id"] != "P0R06427"]

    with pytest.raises(ValueError, match="missing required source ledger ids"):
        module.build_glial_slow_control_specs(incomplete)


def test_glial_slow_control_write_outputs_and_loader_access(tmp_path: Path) -> None:
    module = _load_module()
    bundle = module.build_glial_slow_control_specs(_complete_records())

    outputs = module.write_outputs(bundle, output_dir=tmp_path, date_tag="2099-01-02")

    payload = json.loads(outputs["json"].read_text(encoding="utf-8"))
    report = outputs["report"].read_text(encoding="utf-8")
    loaded = load_glial_slow_control_validation_spec(
        "glial_slow_control.experimental_protocol_catalogue",
        spec_bundle_path=outputs["json"],
    )

    assert payload["summary"]["coverage_match"] is True
    assert loaded["source_ledger_ids"] == [f"P0R{number:05d}" for number in range(6414, 6434)]
    assert "simultaneous two-photon Ca2+ imaging" in loaded["source_protocol_steps"][1]
    assert "Paper 0 Glial Slow-Control Specs" in report
    assert "not empirical evidence" in report
