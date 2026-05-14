# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 ethical-imperative runner tests
"""Tests for the Paper 0 Ethical Imperative fixture runner."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "run_paper0_ethical_imperative_fixture.py"


def _load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("run_paper0_ethical_imperative_fixture", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load Paper 0 Ethical Imperative fixture runner")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_write_outputs_records_ethical_imperative_fixture(tmp_path: Path) -> None:
    module = _load_module()
    result = module.run_default_fixture()

    outputs = module.write_outputs(result, output_dir=tmp_path, date_tag="2099-01-02")

    payload = json.loads(outputs["json"].read_text(encoding="utf-8"))
    report = outputs["report"].read_text(encoding="utf-8")

    assert payload["hardware_status"] == "simulator_only_no_provider_submission"
    assert payload["overlap_with_prior_slice"] == "P0R06251-P0R06272"
    assert "collapse_entropy_death" in payload["choice_labels"]
    assert (
        payload["governance"]["governance_score"]
        > payload["config_thresholds"]["governance_threshold"]
    )
    assert payload["feedback_loop_delta"] > 0.0
    assert "ethical_imperative.feedback_loop_tuning_boundary" in payload["spec_keys"]
    assert "not empirical evidence" in payload["claim_boundary"]
    assert "Paper 0 Ethical Imperative Fixture" in report
