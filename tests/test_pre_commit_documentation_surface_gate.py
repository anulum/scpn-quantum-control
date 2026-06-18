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
