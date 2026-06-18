# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- pre-commit documentation surface gate contract
"""Static contract for the pre-push documentation-surface gate."""

from __future__ import annotations

from pathlib import Path


def test_pre_push_hook_gates_documentation_surface() -> None:
    """The local pre-push hook must mirror the CI documentation-surface gate."""
    config = Path(".pre-commit-config.yaml").read_text(encoding="utf-8")

    assert "preflight (lint + docs + type-check)" in config
    assert "python tools/audit_documentation_surface.py" in config
    assert "--allowlist tools/documentation_surface_allowlist.json" in config
    assert "--fail-on-findings" in config


def test_pre_push_hook_gates_differentiable_strict_mypy_ratchet() -> None:
    """The local pre-push hook must enforce strict mypy on promoted modules."""
    config = Path(".pre-commit-config.yaml").read_text(encoding="utf-8")

    assert "mypy --strict" in config
    assert "src/scpn_quantum_control/differentiable.py" in config
    assert "src/scpn_quantum_control/differentiable_claim_ledger.py" in config
    assert "src/scpn_quantum_control/differentiable_api.py" in config
    assert "src/scpn_quantum_control/benchmarks/differentiable_programming.py" in config
    assert "src/scpn_quantum_control/differentiable_external_validation.py" in config
    assert "src/scpn_quantum_control/differentiable_framework_overlay.py" in config
    assert "src/scpn_quantum_control/differentiable_module_hardening_audit.py" in config
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
