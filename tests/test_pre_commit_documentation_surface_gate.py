# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — pre commit documentation surface gate tests
# SCPN Quantum Control -- pre-commit documentation surface gate contract
"""Static contract for the pre-push documentation-surface gate."""

from __future__ import annotations

import re
from pathlib import Path


def test_pre_push_hook_gates_documentation_surface() -> None:
    """The local pre-push hook must mirror the CI documentation-surface gate."""
    config = Path(".pre-commit-config.yaml").read_text(encoding="utf-8")
    preflight = Path("tools/preflight.py").read_text(encoding="utf-8")

    assert "preflight (lint + docs + type-check)" in config
    assert "tools/preflight.py --no-tests" in config
    assert "tools/audit_documentation_surface.py" in preflight
    assert "tools/documentation_surface_allowlist.json" in preflight
    assert "--fail-on-findings" in preflight
    assert "tools/check_differentiable_promotion_language.py" in preflight


def test_pre_commit_mypy_version_matches_developer_lock() -> None:
    """The isolated commit hook must use the same mypy release as CI and developers."""
    config = Path(".pre-commit-config.yaml").read_text(encoding="utf-8")
    requirements = Path("requirements-dev.txt").read_text(encoding="utf-8")
    mirror_match = re.search(
        r"repo: https://github\.com/pre-commit/mirrors-mypy\s+rev: v([0-9.]+)",
        config,
    )
    lock_match = re.search(r"(?m)^mypy==([0-9.]+)$", requirements)

    assert mirror_match is not None, "pre-commit mypy mirror pin is missing"
    assert lock_match is not None, "requirements-dev mypy pin is missing"
    assert mirror_match.group(1) == lock_match.group(1)


def test_pre_push_hook_gates_differentiable_strict_mypy_ratchet() -> None:
    """The local pre-push hook must enforce strict mypy on promoted modules."""
    config = Path("tools/preflight.py").read_text(encoding="utf-8")

    assert '"--strict"' in config
    assert "src/scpn_quantum_control/differentiable.py" in config
    assert "src/scpn_quantum_control/differentiable_claim_ledger.py" in config
    assert "src/scpn_quantum_control/differentiable_architecture_map.py" in config
    assert "src/scpn_quantum_control/differentiable_dependency_environment_map.py" in config
    assert "src/scpn_quantum_control/differentiable_baseline_scorecard.py" in config
    assert "src/scpn_quantum_control/differentiable_api.py" in config
    assert "src/scpn_quantum_control/benchmarks/differentiable_programming.py" in config
    assert "src/scpn_quantum_control/differentiable_external_validation.py" in config
    assert "src/scpn_quantum_control/differentiable_framework_overlay.py" in config
    assert "src/scpn_quantum_control/differentiable_module_hardening_audit.py" in config
    assert "src/scpn_quantum_control/benchmarks/differentiable_isolated_benchmark_plan.py" in (
        config
    )
    assert "src/scpn_quantum_control/benchmarks/differentiable_hardening_gate.py" in config
    assert "src/scpn_quantum_control/benchmarks/differentiable_evidence.py" in config
    assert "src/scpn_quantum_control/phase/differentiable_readiness.py" in config
    assert "src/scpn_quantum_control/phase/differentiable_audit.py" in config
    assert "src/scpn_quantum_control/phase/gradient_support_matrix.py" in config
    assert "src/scpn_quantum_control/phase/provider_gradient.py" in config
    assert "src/scpn_quantum_control/phase/hardware_gradient_policy.py" in config
    assert "src/scpn_quantum_control/phase/provider_gradient_audit.py" in config
    assert "src/scpn_quantum_control/phase/hardware_gradient_publication.py" in config
    assert "src/scpn_quantum_control/phase/provider_hardware_gradient_audit.py" in config
    assert "src/scpn_quantum_control/phase/hardware_gradient_campaign.py" in config
    assert "src/scpn_quantum_control/phase/gradient_backend.py" in config
    assert "src/scpn_quantum_control/phase/gradient_tape.py" in config
    assert "src/scpn_quantum_control/phase/natural_gradient.py" in config
    assert "src/scpn_quantum_control/phase/gradient_descent.py" in config
    assert "src/scpn_quantum_control/phase/qnode_affinity_benchmark.py" in config
    assert "src/scpn_quantum_control/phase/qnode_tape.py" in config
    assert "src/scpn_quantum_control/phase/qnode_provider_transforms.py" in config
    assert "src/scpn_quantum_control/phase/qnode_transforms.py" in config
    assert "src/scpn_quantum_control/phase/qnode_vector_transforms.py" in config
    assert "src/scpn_quantum_control/phase/qnode_framework_parity.py" in config
    assert "src/scpn_quantum_control/phase/qnode_circuit.py" in config
    assert "src/scpn_quantum_control/phase/pennylane_bridge.py" in config
    assert "src/scpn_quantum_control/phase/jax_bridge.py" in config
    assert "src/scpn_quantum_control/phase/torch_bridge.py" in config
    assert "src/scpn_quantum_control/phase/tensorflow_bridge.py" in config
    assert "src/scpn_quantum_control/phase/tensorflow_maintenance.py" in config
    assert "src/scpn_quantum_control/phase/qiskit_bridge.py" in config
    assert "src/scpn_quantum_control/phase/qnn_framework_bridge_matrix.py" in config
    assert "src/scpn_quantum_control/phase/transform_nesting.py" in config
    assert "src/scpn_quantum_control/benchmarks/differentiable_external_comparison.py" in config
    assert "src/scpn_quantum_control/phase/xy_compiler.py" in config


def test_pre_push_hook_gates_differentiable_docstring_ratchet() -> None:
    """The pre-push hook must enforce Ruff D on clean differentiable modules."""
    config = Path("tools/preflight.py").read_text(encoding="utf-8")

    assert '"ruff"' in config
    assert '"--select"' in config
    assert '"D"' in config
    assert "src/scpn_quantum_control/differentiable_external_validation.py" in config
    assert "src/scpn_quantum_control/differentiable_module_hardening_audit.py" in config
    assert "src/scpn_quantum_control/phase/tensorflow_maintenance.py" in config
    assert "src/scpn_quantum_control/benchmarks/differentiable_isolated_benchmark_plan.py" in (
        config
    )
    assert "src/scpn_quantum_control/benchmarks/differentiable_hardening_gate.py" in config
    assert "tests/test_differentiable_external_validation.py" in config
    assert "tests/test_differentiable_module_hardening_audit.py" in config
    assert "tests/test_phase_tensorflow_maintenance.py" in config
    assert "tests/test_differentiable_hardening_gate.py" in config
