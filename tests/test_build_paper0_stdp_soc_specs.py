# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 STDP/SOC spec tests
"""Tests for Paper 0 STDP as SOC-engine promotion."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

from scpn_quantum_control.paper0.spec_loader import load_stdp_soc_validation_spec

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "build_paper0_stdp_soc_specs.py"


def _load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("build_paper0_stdp_soc_specs", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load Paper 0 STDP/SOC spec script")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _record(ledger_id: str, block_index: int) -> dict[str, object]:
    return {
        "ledger_id": ledger_id,
        "source_record_id": ledger_id.replace("P0R", "P0B"),
        "source_block_index": block_index,
        "section_path": "Paper 0 STDP as SOC engine",
        "math_ids": [],
        "image_ids": [ledger_id.replace("P0R", "IMG")]
        if ledger_id in {"P0R06404", "P0R06408"}
        else [],
        "text": "source text",
    }


def _complete_records() -> list[dict[str, object]]:
    return [_record(f"P0R{number:05d}", number) for number in range(6402, 6414)]


def test_stdp_soc_specs_consume_complete_contiguous_source_span() -> None:
    module = _load_module()

    bundle = module.build_stdp_soc_specs(_complete_records())

    assert bundle.summary["source_record_count"] == 12
    assert bundle.summary["consumed_source_record_count"] == 12
    assert bundle.summary["coverage_match"] is True
    assert bundle.summary["source_ledger_span"] == ["P0R06402", "P0R06413"]
    assert bundle.summary["structural_source_ledger_ids"] == [
        "P0R06402",
        "P0R06404",
        "P0R06408",
    ]
    assert bundle.summary["caption_source_ledger_ids"] == ["P0R06405", "P0R06409"]
    assert bundle.summary["spec_count"] == 4
    assert bundle.summary["spec_keys"] == [
        "stdp_soc.asymmetric_learning_window",
        "stdp_soc.avalanche_power_law_signature",
        "stdp_soc.quasicritical_relaxation_mapping",
        "stdp_soc.l4_microscopic_engine_boundary",
    ]


def test_stdp_soc_specs_preserve_equations_captions_and_boundary() -> None:
    module = _load_module()

    bundle = module.build_stdp_soc_specs(_complete_records())
    specs = {spec.key: spec for spec in bundle.specs}

    assert all(spec.validation_targets for spec in specs.values())
    assert all(spec.null_controls for spec in specs.values())
    assert all("not empirical evidence" in spec.claim_boundary for spec in specs.values())
    assert specs["stdp_soc.asymmetric_learning_window"].source_equation_ids == (
        "P0R06405:Delta_w_Delta_t",
        "P0R06405:Delta_t_gt_0_LTP",
        "P0R06405:Delta_t_lt_0_LTD",
    )
    assert specs["stdp_soc.avalanche_power_law_signature"].source_formulae == (
        "P(S) proportional_to S^(-tau)",
        "tau approximately 1.5",
    )
    assert specs["stdp_soc.quasicritical_relaxation_mapping"].source_formulae == (
        "d sigma_L / dt = -kappa_L * (sigma_L - 1) + eta_L(t)",
        "sigma tends towards 1",
    )
    assert specs["stdp_soc.l4_microscopic_engine_boundary"].source_mechanisms == (
        "Hebbian STDP reinforces causally effective pathways",
        "anti-Hebbian and depressive plasticity prune ineffective or anti-causal connections",
        "Layer 4 maintains quasicritical cellular-tissue synchronisation",
    )


def test_stdp_soc_builder_rejects_missing_source_record() -> None:
    module = _load_module()
    incomplete = [record for record in _complete_records() if record["ledger_id"] != "P0R06410"]

    with pytest.raises(ValueError, match="missing required source ledger ids"):
        module.build_stdp_soc_specs(incomplete)


def test_stdp_soc_write_outputs_and_loader_access(tmp_path: Path) -> None:
    module = _load_module()
    bundle = module.build_stdp_soc_specs(_complete_records())

    outputs = module.write_outputs(bundle, output_dir=tmp_path, date_tag="2099-01-02")

    payload = json.loads(outputs["json"].read_text(encoding="utf-8"))
    report = outputs["report"].read_text(encoding="utf-8")
    loaded = load_stdp_soc_validation_spec(
        "stdp_soc.quasicritical_relaxation_mapping",
        spec_bundle_path=outputs["json"],
    )

    assert payload["summary"]["coverage_match"] is True
    assert loaded["source_ledger_ids"] == [f"P0R{number:05d}" for number in range(6402, 6414)]
    assert "P0R06410:d_sigma_L_dt" in loaded["source_equation_ids"]
    assert "Paper 0 STDP SOC Specs" in report
    assert "not empirical evidence" in report
