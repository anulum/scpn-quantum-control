# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- tests for layer-selective readiness audit
"""Tests for the layer-selective layout readiness audit."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType


def _load_module() -> ModuleType:
    script = (
        Path(__file__).resolve().parents[1] / "scripts" / "analyse_layer_selective_readiness.py"
    )
    spec = importlib.util.spec_from_file_location("analyse_layer_selective_readiness", script)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load layer-selective readiness script")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_saved_state_layout_payload_blocks_hardware_promotion() -> None:
    module = _load_module()
    payload = json.loads(module.DEFAULT_INPUT.read_text(encoding="utf-8"))

    summary = module.build_readiness(payload, input_path=module.DEFAULT_INPUT)

    assert summary["schema"] == "scpn_phase3_layer_selective_readiness_v1"
    assert summary["backend"] == "ibm_marrakesh"
    assert summary["hardware_submission"] is False
    assert summary["ready_for_hardware_comparison"] is False
    assert summary["readiness_decision"] == "blocked_missing_comparators"
    assert summary["missing_comparator_methods"] == ["default", "sabre", "layer_selective"]


def test_saved_layout_resource_rows_preserve_preregistered_shape() -> None:
    module = _load_module()
    payload = json.loads(module.DEFAULT_INPUT.read_text(encoding="utf-8"))

    rows = module.summarise_saved_layouts(payload)

    assert len(rows) == 3
    assert {row.n_rows for row in rows} == {160}
    assert all(row.max_depth > 0 for row in rows)
    assert all(row.max_total_gates > 0 for row in rows)
    assert all(row.high_priority_cost > 0 for row in rows)


def test_outputs_include_manifest_blocker(tmp_path: Path) -> None:
    module = _load_module()
    payload = json.loads(module.DEFAULT_INPUT.read_text(encoding="utf-8"))
    summary = module.build_readiness(payload, input_path=module.DEFAULT_INPUT)

    json_path, csv_path, md_path = module.write_outputs(
        summary,
        output_dir=tmp_path / "data",
        docs_dir=tmp_path / "docs",
    )

    assert json_path.exists()
    assert csv_path.exists()
    manifest = md_path.read_text(encoding="utf-8")
    assert "blocked_missing_comparators" in manifest
    assert "Hardware submission: `False`" in manifest
