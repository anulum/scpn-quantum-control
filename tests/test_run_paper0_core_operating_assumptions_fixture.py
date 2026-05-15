# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 core operating assumptions fixture runner tests
"""Tests for the Paper 0 core-operating-assumptions fixture runner."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "run_paper0_core_operating_assumptions_fixture.py"


def _load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "run_paper0_core_operating_assumptions_fixture", SCRIPT
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load Paper 0 core operating assumptions fixture runner")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_core_operating_assumptions_runner_writes_fixture_and_report(tmp_path: Path) -> None:
    module = _load_module()
    output_path = tmp_path / "fixture.json"
    report_path = tmp_path / "fixture.md"

    module.write_outputs(output_path=output_path, report_path=report_path)

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    report = report_path.read_text(encoding="utf-8")

    assert payload["source_ledger_span"] == ["P0R00635", "P0R00669"]
    assert payload["core_assumption_count"] == 5
    assert payload["blank_separator_count"] == 3
    assert payload["hardware_status"] == "source_methodology_no_experiment"
    assert "Paper 0 Core Operating Assumptions Fixture" in report
    assert "ethical_functional_tunes_parameter_not_force" in report
