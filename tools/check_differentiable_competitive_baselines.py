#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — differentiable competitive baseline gate.
"""Fail CI when competitive SOTA baselines go stale or claims outrun evidence."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from scpn_quantum_control.differentiable_competitive_baselines import (  # noqa: E402
    audit_competitive_baseline_promotion_gate,
)


def main() -> int:
    """Run the differentiable competitive-baseline refresh gate."""
    gate = audit_competitive_baseline_promotion_gate(repo_root=ROOT)
    if gate.passed:
        print(
            "differentiable competitive-baseline gate: PASS "
            f"({len(gate.baseline_validation.checked_baselines)} baselines, "
            f"{len(gate.checked_categories)} categories)"
        )
        return 0
    print("differentiable competitive-baseline gate: FAIL")
    for error in gate.errors:
        print(f"  - {error}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
