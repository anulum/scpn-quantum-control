# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 Stuart-Landau precision runner tests
"""Tests for the Paper 0 Stuart-Landau precision fixture runner."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "run_paper0_stuart_landau_precision_fixture.py"


def _load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("run_paper0_sl_precision_fixture", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load Paper 0 Stuart-Landau precision fixture runner")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_write_outputs_records_simulator_only_stuart_landau_precision_fixture(
    tmp_path: Path,
) -> None:
    module = _load_module()
    result = module.run_default_fixture()

    outputs = module.write_outputs(result, output_dir=tmp_path, date_tag="2099-01-02")

    payload = json.loads(outputs["json"].read_text(encoding="utf-8"))
    report = outputs["report"].read_text(encoding="utf-8")

    assert payload["hardware_status"] == "simulator_only_no_provider_submission"
    assert payload["spec_keys"] == [
        "computational.stuart_landau_precision_upgrade",
        "computational.precision_weighted_phase_amplitude_dynamics",
        "computational.salience_radial_precision_control",
    ]
    assert payload["max_complex_polar_residual"] < 1.0e-12
    assert payload["max_phase_ratio_residual"] < 1.0e-12
    assert payload["rho_gain_radius_dot_delta"] > 0.0
    assert payload["null_controls"]["zero_radius_rejection_label"] == 1.0
    assert "Paper 0 Stuart-Landau Precision Fixture" in report
    assert "No provider submission is represented" in report
