# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 computational-unifier fixture runner tests
"""Tests for the Paper 0 computational-unifier fixture runner."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "run_paper0_computational_unifier_fixture.py"


def _load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "run_paper0_computational_unifier_fixture", SCRIPT
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load Paper 0 computational-unifier fixture runner")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_write_outputs_records_simulator_only_computational_fixture(tmp_path: Path) -> None:
    module = _load_module()
    result = module.run_default_fixture()

    outputs = module.write_outputs(result, output_dir=tmp_path, date_tag="2099-01-02")

    payload = json.loads(outputs["json"].read_text(encoding="utf-8"))
    report = outputs["report"].read_text(encoding="utf-8")

    assert payload["hardware_status"] == "simulator_only_no_provider_submission"
    assert payload["cyclic_operator"]["source_equation_ids"] == ["EQ0115"]
    assert payload["tsvf_abl"]["source_equation_ids"] == ["EQ0116"]
    assert payload["information_thermodynamics"]["source_equation_ids"] == [
        "EQ0117",
        "EQ0118",
    ]
    assert payload["cyclic_operator"]["unitarity_error"] < 1.0e-12
    assert payload["cyclic_operator"]["cycle_closure_residual"] < 1.0e-12
    assert payload["tsvf_abl"]["probability_normalisation_error"] < 1.0e-12
    assert payload["information_thermodynamics"]["gsl_margin"] > 0.0
    assert "Paper 0 Computational-Unifier Fixture" in report
    assert "No provider submission is represented" in report
