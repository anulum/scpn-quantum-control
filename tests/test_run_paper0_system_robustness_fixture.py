# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 system-robustness runner tests
"""Tests for the Paper 0 system-robustness fixture runner."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "run_paper0_system_robustness_fixture.py"


def _load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("run_paper0_robustness_fixture", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load Paper 0 system-robustness fixture runner")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_write_outputs_records_simulator_only_system_robustness_fixture(
    tmp_path: Path,
) -> None:
    module = _load_module()
    result = module.run_default_fixture()

    outputs = module.write_outputs(result, output_dir=tmp_path, date_tag="2099-01-02")

    payload = json.loads(outputs["json"].read_text(encoding="utf-8"))
    report = outputs["report"].read_text(encoding="utf-8")

    assert payload["hardware_status"] == "simulator_only_no_provider_submission"
    assert payload["spec_keys"] == [
        "applied.system_robustness.cascading_failure_percolation",
        "applied.system_robustness.critical_slowing_recovery",
        "applied.system_robustness.decoherence_attack_ms_qec_boundary",
    ]
    assert payload["largest_component_loss"] >= 0.0
    assert payload["recovery_time_ratio"] > 1.0
    assert 0.0 <= payload["failure_probability"] < 1.0
    assert "not operational security evidence" in payload["claim_boundary"]
    assert "Paper 0 System-Robustness Fixture" in report
    assert "not operational security evidence" in report
