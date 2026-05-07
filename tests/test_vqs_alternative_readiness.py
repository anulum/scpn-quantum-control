# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- tests for VQS alternative readiness
"""Tests for the no-QPU VQS alternative-readiness package."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import numpy as np


def _load_module() -> ModuleType:
    script = (
        Path(__file__).resolve().parents[1] / "scripts" / "generate_vqs_alternative_readiness.py"
    )
    spec = importlib.util.spec_from_file_location("generate_vqs_alternative_readiness", script)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load VQS alternative readiness script")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_state_observables_use_qiskit_little_endian_initial_index() -> None:
    module = _load_module()
    state = np.zeros(16, dtype=np.complex128)
    state[int("0011"[::-1], 2)] = 1.0

    observables = module.state_observables(state, "0011")

    assert observables["parity_survival"] == 1.0
    assert observables["exact_state_retention"] == 1.0
    assert observables["magnetisation_expectation"] == 0.0


def test_ansatz_resource_rows_cover_preregistered_n4_n6_n8_matrix() -> None:
    module = _load_module()

    rows = module.resource_rows(module.default_cases())

    labels = {row["label"] for row in rows}
    methods = {row["method"] for row in rows}
    assert {"n4_even_signal", "n6_even_probe", "n8_even_probe"}.issubset(labels)
    assert {
        "trotter_reference",
        "vqs_topology_informed",
        "vqs_efficient_su2",
        "vqs_two_local",
    }.issubset(methods)
    assert all(int(row["transpiled_depth"]) >= 0 for row in rows)
    assert all(int(row["transpiled_two_qubit_gates"]) >= 0 for row in rows)


def test_build_summary_blocks_when_accuracy_rows_fail_promotion_gate() -> None:
    module = _load_module()
    cases = module.default_cases()
    res_rows = module.resource_rows(cases)
    opt_rows = []
    for spec in cases:
        if not spec.optimise:
            continue
        for seed in module.SEEDS:
            opt_rows.append(
                {
                    **spec.to_dict(),
                    "ansatz": "topology_informed",
                    "seed": seed,
                    "fidelity": 0.5,
                    "parity_error": 0.5,
                    "retention_error": 0.5,
                    "promoted_accuracy_gate": False,
                }
            )

    summary = module.build_summary(opt_rows, res_rows)

    assert summary["schema"] == "scpn_phase3_vqs_alternative_readiness_v1"
    assert summary["hardware_submission"] is False
    assert summary["qpu_minutes_spent"] == 0.0
    assert summary["ready_for_optional_hardware"] is False
    assert summary["readiness_decision"] == "blocked_no_vqs_candidate_passed_promotion_gate"


def test_write_outputs_records_manifest_and_hashes(tmp_path: Path) -> None:
    module = _load_module()
    cases = module.default_cases()
    res_rows = module.resource_rows(cases)
    opt_rows = []
    for spec in cases:
        if not spec.optimise:
            continue
        for seed in module.SEEDS:
            opt_rows.append(
                {
                    **spec.to_dict(),
                    "ansatz": "topology_informed",
                    "seed": seed,
                    "fidelity": 0.5,
                    "parity_error": 0.5,
                    "retention_error": 0.5,
                    "promoted_accuracy_gate": False,
                }
            )
    summary = module.build_summary(opt_rows, res_rows)

    json_path, opt_path, resource_path, md_path = module.write_outputs(
        opt_rows,
        res_rows,
        summary,
        output_dir=tmp_path / "data",
        docs_dir=tmp_path / "docs",
    )

    assert json_path.exists()
    assert opt_path.exists()
    assert resource_path.exists()
    manifest = md_path.read_text(encoding="utf-8")
    assert "blocked_no_vqs_candidate_passed_promotion_gate" in manifest
    assert "Hardware submission: `False`" in manifest
