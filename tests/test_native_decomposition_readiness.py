# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- tests for native decomposition readiness
"""Tests for the no-QPU native-decomposition readiness package."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType


def _load_module() -> ModuleType:
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "generate_native_decomposition_readiness.py"
    )
    spec = importlib.util.spec_from_file_location(
        "generate_native_decomposition_readiness", script
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load native decomposition readiness script")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_default_cases_cover_preregistered_sizes_and_families() -> None:
    module = _load_module()

    cases = module.default_cases()

    assert {case.n_qubits for case in cases} == {4, 6, 8}
    assert {"dla_parity", "popcount_control", "stress_depth"}.issubset(
        {case.family for case in cases}
    )


def test_native_targeted_matches_generic_for_n4_case() -> None:
    module = _load_module()
    spec = module.CaseSpec(4, "test", "even", "0011", 2)
    generic = module.build_generic_pauli_circuit(spec)
    native = module.build_native_targeted_circuit(spec)

    distance = module._normalised_unitary_distance(native, generic)

    assert distance <= module.UNITARY_TOLERANCE


def test_readiness_summary_is_bounded_and_non_submitting() -> None:
    module = _load_module()

    resource_rows, eq_rows, summary = module.build_readiness()

    assert resource_rows
    assert eq_rows
    assert summary["schema"] == "scpn_phase3_native_decomposition_readiness_v1"
    assert summary["hardware_submission"] is False
    assert summary["qpu_minutes_spent"] == 0.0
    assert summary["readiness_decision"] in {
        "ready_for_live_backend_transpilation",
        "blocked_native_equivalence_failed",
        "blocked_current_xy_invalid_no_native_gain_vs_generic",
        "blocked_no_resource_gain_vs_current_xy",
    }
    assert "method_summaries" in summary


def test_write_outputs_records_manifest(tmp_path: Path) -> None:
    module = _load_module()
    resource_rows, eq_rows, summary = module.build_readiness()

    json_path, resource_path, eq_path, md_path = module.write_outputs(
        resource_rows,
        eq_rows,
        summary,
        output_dir=tmp_path / "data",
        docs_dir=tmp_path / "docs",
    )

    assert json_path.exists()
    assert resource_path.exists()
    assert eq_path.exists()
    manifest = md_path.read_text(encoding="utf-8")
    assert "Hardware submission: `False`" in manifest
    assert str(summary["readiness_decision"]) in manifest
