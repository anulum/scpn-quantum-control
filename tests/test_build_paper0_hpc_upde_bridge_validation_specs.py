# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 HPC/UPDE bridge spec tests
"""Tests for Paper 0 HPC/UPDE bridge validation spec promotion."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

from scpn_quantum_control.paper0.spec_loader import load_hpc_upde_bridge_validation_spec

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "build_paper0_hpc_upde_bridge_validation_specs.py"


def _load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("build_paper0_hpc_upde_bridge_specs", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load Paper 0 HPC/UPDE bridge spec script")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _record(ledger_id: str, block_index: int) -> dict[str, object]:
    return {
        "ledger_id": ledger_id,
        "source_record_id": ledger_id.replace("P0R", "P0B"),
        "source_block_index": block_index,
        "section_path": "Paper 0 HPC/UPDE bridge test section",
        "math_ids": [],
        "text": "source text",
    }


def _complete_records() -> list[dict[str, object]]:
    return [_record(f"P0R{number:05d}", number) for number in range(6156, 6179)]


def test_hpc_upde_bridge_specs_consume_complete_source_span() -> None:
    module = _load_module()

    bundle = module.build_hpc_upde_bridge_validation_specs(_complete_records())

    assert bundle.summary["source_record_count"] == 23
    assert bundle.summary["consumed_source_record_count"] == 23
    assert bundle.summary["coverage_match"] is True
    assert bundle.summary["source_ledger_span"] == ["P0R06156", "P0R06178"]
    assert bundle.summary["spec_count"] == 3
    assert bundle.summary["spec_keys"] == [
        "computational.hpc_bidirectional_flow",
        "computational.upde_phase_prediction_error",
        "computational.upde_free_energy_gradient_bridge",
    ]
    assert bundle.summary["hardware_status"] == "simulator_only_no_provider_submission"


def test_hpc_upde_bridge_specs_have_controls_without_invented_equation_ids() -> None:
    module = _load_module()

    bundle = module.build_hpc_upde_bridge_validation_specs(_complete_records())
    specs = {spec.key: spec for spec in bundle.specs}

    assert all(spec.source_equation_ids == () for spec in specs.values())
    assert all(spec.anchor_math_ids == () for spec in specs.values())
    assert all(spec.null_controls for spec in specs.values())
    assert all(spec.executable_validation_targets for spec in specs.values())
    assert any(
        "wrong-sign" in control
        for control in specs["computational.upde_free_energy_gradient_bridge"].null_controls
    )
    assert any(
        "finite-difference" in target
        for target in specs[
            "computational.upde_free_energy_gradient_bridge"
        ].executable_validation_targets
    )


def test_hpc_upde_bridge_builder_rejects_missing_required_source_record() -> None:
    module = _load_module()
    incomplete = [record for record in _complete_records() if record["ledger_id"] != "P0R06170"]

    with pytest.raises(ValueError, match="missing required source ledger ids"):
        module.build_hpc_upde_bridge_validation_specs(incomplete)


def test_hpc_upde_bridge_write_outputs_and_loader_access(tmp_path: Path) -> None:
    module = _load_module()
    bundle = module.build_hpc_upde_bridge_validation_specs(_complete_records())

    outputs = module.write_outputs(bundle, output_dir=tmp_path, date_tag="2099-01-02")

    payload = json.loads(outputs["json"].read_text(encoding="utf-8"))
    report = outputs["report"].read_text(encoding="utf-8")
    loaded = load_hpc_upde_bridge_validation_spec(
        "computational.upde_phase_prediction_error",
        spec_bundle_path=outputs["json"],
    )

    assert payload["summary"]["coverage_match"] is True
    assert loaded["source_ledger_ids"] == [
        "P0R06159",
        "P0R06160",
        "P0R06161",
        "P0R06162",
        "P0R06163",
    ]
    assert "Paper 0 HPC/UPDE Bridge Validation Specs" in report
    assert "No standalone equation IDs are invented" in report
