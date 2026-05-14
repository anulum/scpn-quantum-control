# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 computational verification tools spec tests
"""Tests for Paper 0 computational verification tools promotion."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

from scpn_quantum_control.paper0.spec_loader import (
    load_computational_verification_tools_validation_spec,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "build_paper0_computational_verification_tools_specs.py"


def _load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "build_paper0_computational_verification_tools_specs", SCRIPT
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load Paper 0 computational verification tools spec script")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _record(ledger_id: str, block_index: int) -> dict[str, object]:
    return {
        "ledger_id": ledger_id,
        "source_record_id": ledger_id.replace("P0R", "P0B"),
        "source_block_index": block_index,
        "section_path": "New content to be allocated into the document:",
        "canonical_category": "validation_target",
        "math_ids": [],
        "text": "source text",
    }


def _complete_records() -> list[dict[str, object]]:
    return [_record(f"P0R{number:05d}", number) for number in range(7006, 7073)]


def test_computational_verification_tools_specs_consume_complete_source_span() -> None:
    module = _load_module()

    bundle = module.build_computational_verification_tools_specs(_complete_records())

    assert bundle.summary["source_record_count"] == 67
    assert bundle.summary["consumed_source_record_count"] == 67
    assert bundle.summary["coverage_match"] is True
    assert bundle.summary["source_ledger_span"] == ["P0R07006", "P0R07072"]
    assert bundle.summary["spec_count"] == 5
    assert bundle.summary["tool_count"] == 3
    assert bundle.summary["hardware_status"] == "computational_protocol_no_claimed_execution"


def test_computational_verification_tools_specs_preserve_equations_and_boundaries() -> None:
    module = _load_module()

    bundle = module.build_computational_verification_tools_specs(_complete_records())
    specs = {spec.key: spec for spec in bundle.specs}

    assert all(
        spec.source_ledger_ids == tuple(f"P0R{number:05d}" for number in range(7006, 7073))
        for spec in specs.values()
    )
    assert all(
        "source-bounded computational protocol" in spec.claim_boundary for spec in specs.values()
    )
    assert specs["computational_verification_tools.lattice_hmc_flat_line"].source_equation_ids == (
        "P0R07008:polar_higgs_yukawa_field",
        "P0R07009:lattice_action",
        "P0R07015:pcac_mass_relation",
        "P0R07017:mass_ratio_target",
        "P0R07028:quenched_boundary",
    )
    assert specs["computational_verification_tools.class_goldstone_eos"].source_equation_ids == (
        "P0R07037:oscillatory_equation_of_state",
        "P0R07040:rho_phi_patch",
        "P0R07041:pressure_patch",
        "P0R07043:class_parameters",
        "P0R07051:planck_washout_boundary",
    )
    assert specs["computational_verification_tools.lambda_eff_utility"].source_equation_ids == (
        "P0R07056:lambda_psi_g",
        "P0R07058:natural_units",
        "P0R07059:canonical_lambda_0",
        "P0R07062:reduced_planck_mass",
        "P0R07067:lambda_psi_density",
        "P0R07070:lambda_eff",
    )


def test_computational_verification_tools_builder_rejects_missing_source_record() -> None:
    module = _load_module()
    incomplete = [record for record in _complete_records() if record["ledger_id"] != "P0R07051"]

    with pytest.raises(ValueError, match="missing required source ledger ids"):
        module.build_computational_verification_tools_specs(incomplete)


def test_computational_verification_tools_write_outputs_and_loader_access(tmp_path: Path) -> None:
    module = _load_module()
    bundle = module.build_computational_verification_tools_specs(_complete_records())

    outputs = module.write_outputs(bundle, output_dir=tmp_path, date_tag="2099-01-02")

    payload = json.loads(outputs["json"].read_text(encoding="utf-8"))
    report = outputs["report"].read_text(encoding="utf-8")
    loaded = load_computational_verification_tools_validation_spec(
        "computational_verification_tools.lambda_eff_utility",
        spec_bundle_path=outputs["json"],
    )

    assert payload["summary"]["coverage_match"] is True
    assert loaded["source_ledger_ids"][0] == "P0R07006"
    assert loaded["source_ledger_ids"][-1] == "P0R07072"
    assert "Paper 0 Computational Verification Tools Specs" in report
    assert "not empirical execution evidence" in report
