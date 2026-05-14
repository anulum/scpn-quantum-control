# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 validation strategy runner tests
"""Tests for the Paper 0 validation-strategy fixture runner."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "run_paper0_validation_strategy_fixture.py"


def _load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("run_paper0_validation_strategy_fixture", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load Paper 0 validation-strategy fixture runner")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_write_outputs_records_simulator_only_validation_strategy_fixture(
    tmp_path: Path,
) -> None:
    module = _load_module()
    result = module.run_default_fixture()

    outputs = module.write_outputs(result, output_dir=tmp_path, date_tag="2099-01-02")

    payload = json.loads(outputs["json"].read_text(encoding="utf-8"))
    report = outputs["report"].read_text(encoding="utf-8")

    assert payload["hardware_status"] == "simulator_only_no_provider_submission"
    assert payload["stage_count"] == 3
    assert payload["domain_count"] == 11
    assert payload["stage_order_valid"] is True
    assert "pathology" in payload["domains"]
    assert "teleology_l15_l8" in payload["domains"]
    assert "not empirical evidence" in payload["claim_boundary"]
    assert "Paper 0 Validation Strategy Fixture" in report
    assert "not empirical evidence" in report
