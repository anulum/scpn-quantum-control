# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for Differentiable Framework CI Workflow
"""Static tests for differentiable framework CI and benchmark workflow policy."""

from __future__ import annotations

from pathlib import Path


def test_differentiable_framework_workflow_declares_sparse_and_full_matrices() -> None:
    workflow = Path(".github/workflows/differentiable-frameworks.yml")
    text = workflow.read_text(encoding="utf-8")

    assert "3.10" in text
    assert "3.11" in text
    assert "3.12" in text
    assert "3.13" in text
    assert "jax" in text
    assert "torch" in text
    assert "tensorflow" in text
    assert "pennylane" in text
    assert "tests/test_phase_qnode_framework_parity.py" in text
    assert "not hardware" in text
    assert "workflow_dispatch" in text
    assert "schedule" in text
    assert "taskset" in text
    assert "upload-artifact" in text
    assert "actions/checkout@df4cb1c069e1874edd31b4311f1884172cec0e10" in text
    assert "actions/setup-python@a309ff8b426b58ec0e2a45f0f869d46889d02405" in text
    assert "install-differentiable-framework-overlay --overlay-path" in text
    assert "--install" in text
