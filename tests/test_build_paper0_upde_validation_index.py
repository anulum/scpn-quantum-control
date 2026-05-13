# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Paper 0 UPDE validation index tests
"""Tests for the aggregate Paper 0 UPDE fixture validation index."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "build_paper0_upde_validation_index.py"


def _load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("build_paper0_upde_validation_index", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load Paper 0 UPDE validation index script")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_bundle(tmp_path: Path) -> Path:
    specs = [
        {
            "key": "upde.base_phase",
            "source_equation_ids": ["EQ0003"],
            "source_ledger_ids": ["P0R00001"],
            "null_controls": ["zero coupling"],
            "executable_validation_targets": ["gradient check"],
        },
        {
            "key": "upde.field_coupling",
            "source_equation_ids": ["EQ0034"],
            "source_ledger_ids": ["P0R00002"],
            "null_controls": ["zero field"],
            "executable_validation_targets": ["field check"],
        },
    ]
    path = tmp_path / "specs.json"
    path.write_text(json.dumps({"specs": specs}), encoding="utf-8")
    return path


def _write_result(
    tmp_path: Path,
    *,
    spec_key: str,
    runtime_ms: float,
    hardware_status: str = "simulator_only_no_provider_submission",
) -> Path:
    path = tmp_path / f"{spec_key.replace('.', '_')}_fixture_result.json"
    path.write_text(
        json.dumps(
            {
                "spec_key": spec_key,
                "validation_protocol": f"paper0.{spec_key}",
                "hardware_status": hardware_status,
                "source_equation_ids": ["EQ0003"],
                "source_ledger_ids": ["P0R00001"],
                "null_controls": {"control": 0.0},
                "runtime_ms": runtime_ms,
            }
        ),
        encoding="utf-8",
    )
    return path


def test_build_index_requires_every_promoted_spec_result(tmp_path: Path) -> None:
    module = _load_module()
    bundle = _write_bundle(tmp_path)
    base = _write_result(tmp_path, spec_key="upde.base_phase", runtime_ms=1.0)

    with pytest.raises(ValueError, match="missing fixture results"):
        module.build_validation_index(
            specs_path=bundle,
            result_paths=[base],
        )


def test_build_index_summarises_source_controls_runtime_and_boundary(tmp_path: Path) -> None:
    module = _load_module()
    bundle = _write_bundle(tmp_path)
    base = _write_result(tmp_path, spec_key="upde.base_phase", runtime_ms=1.0)
    field = _write_result(tmp_path, spec_key="upde.field_coupling", runtime_ms=2.5)

    index = module.build_validation_index(specs_path=bundle, result_paths=[base, field])

    assert index["summary"]["spec_count"] == 2
    assert index["summary"]["fixture_result_count"] == 2
    assert index["summary"]["coverage_match"] is True
    assert index["summary"]["hardware_status"] == "simulator_only_no_provider_submission"
    assert index["summary"]["total_runtime_ms"] == pytest.approx(3.5)
    assert index["summary"]["next_recommended_family"] == "paper0.next_mechanism_family"
    assert index["fixtures"][0]["spec_key"] == "upde.base_phase"
    assert index["fixtures"][0]["null_control_count"] == 1
    assert index["fixtures"][1]["runtime_ms"] == pytest.approx(2.5)


def test_build_index_rejects_non_simulator_hardware_status(tmp_path: Path) -> None:
    module = _load_module()
    bundle = _write_bundle(tmp_path)
    base = _write_result(
        tmp_path,
        spec_key="upde.base_phase",
        runtime_ms=1.0,
        hardware_status="hardware_submitted",
    )
    field = _write_result(tmp_path, spec_key="upde.field_coupling", runtime_ms=2.5)

    with pytest.raises(ValueError, match="non-simulator fixture result"):
        module.build_validation_index(specs_path=bundle, result_paths=[base, field])


def test_render_report_lists_each_fixture_and_policy(tmp_path: Path) -> None:
    module = _load_module()
    bundle = _write_bundle(tmp_path)
    base = _write_result(tmp_path, spec_key="upde.base_phase", runtime_ms=1.0)
    field = _write_result(tmp_path, spec_key="upde.field_coupling", runtime_ms=2.5)
    index = module.build_validation_index(specs_path=bundle, result_paths=[base, field])

    report = module.render_validation_report(index)

    assert "Coverage status: `match`" in report
    assert "upde.base_phase" in report
    assert "upde.field_coupling" in report
    assert "No provider submission is represented" in report
