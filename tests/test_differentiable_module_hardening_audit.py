# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Differentiable module hardening audit tests
"""Tests for differentiable module hardening inventory."""

from __future__ import annotations

from pathlib import Path

import scpn_quantum_control as scpn
from scpn_quantum_control.differentiable_module_hardening_audit import (
    DifferentiableModuleHardeningRecord,
    differentiable_module_hardening_registry,
    run_differentiable_module_hardening_audit,
)


def test_differentiable_module_hardening_audit_passes_live_registry() -> None:
    result = run_differentiable_module_hardening_audit()

    assert result.passed, result.errors
    assert result.missing_registry_paths == ()
    assert result.stale_registry_paths == ()
    assert result.missing_test_paths == ()
    assert len(result.records) == len(result.discovered_module_paths)
    assert "does not prove full formal correctness" in result.claim_boundary
    assert result.to_dict()["passed"] is True


def test_differentiable_module_hardening_registry_covers_current_scope() -> None:
    registry = differentiable_module_hardening_registry()
    module_paths = {record.module_path for record in registry}

    assert "src/scpn_quantum_control/differentiable.py" in module_paths
    assert "src/scpn_quantum_control/differentiable_api.py" in module_paths
    assert "src/scpn_quantum_control/phase/qnode_circuit.py" in module_paths
    assert "src/scpn_quantum_control/phase/jax_bridge.py" in module_paths
    assert "src/scpn_quantum_control/benchmarks/differentiable_hardening_gate.py" in (module_paths)
    assert all(record.diagnostic_surfaces for record in registry)


def test_differentiable_module_hardening_audit_reports_missing_tests(
    tmp_path: Path,
) -> None:
    module_path = tmp_path / "src" / "scpn_quantum_control" / "differentiable.py"
    module_path.parent.mkdir(parents=True)
    module_path.write_text("# test module\n", encoding="utf-8")
    result = run_differentiable_module_hardening_audit(
        repo_root=tmp_path,
        registry=(
            DifferentiableModuleHardeningRecord(
                module_path="src/scpn_quantum_control/differentiable.py",
                test_paths=("tests/test_missing.py",),
                diagnostic_surfaces=("shape validation",),
            ),
        ),
    )

    assert not result.passed
    assert result.missing_test_paths == ("tests/test_missing.py",)
    assert "missing test path: tests/test_missing.py" in result.errors


def test_differentiable_module_hardening_audit_rejects_non_specific_tests(
    tmp_path: Path,
) -> None:
    module_path = tmp_path / "src" / "scpn_quantum_control" / "differentiable.py"
    test_path = tmp_path / "tests" / "bucket.py"
    module_path.parent.mkdir(parents=True)
    test_path.parent.mkdir(parents=True)
    module_path.write_text("# test module\n", encoding="utf-8")
    test_path.write_text("# bucket test\n", encoding="utf-8")

    result = run_differentiable_module_hardening_audit(
        repo_root=tmp_path,
        registry=(
            DifferentiableModuleHardeningRecord(
                module_path="src/scpn_quantum_control/differentiable.py",
                test_paths=("tests/bucket.py",),
                diagnostic_surfaces=("shape validation",),
            ),
        ),
    )

    assert not result.passed
    assert (
        "src/scpn_quantum_control/differentiable.py: non-module-specific test path tests/bucket.py"
    ) in result.errors


def test_differentiable_module_hardening_audit_is_exported() -> None:
    assert scpn.run_differentiable_module_hardening_audit is (
        run_differentiable_module_hardening_audit
    )
    assert "run_differentiable_module_hardening_audit" in scpn.__all__
