#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — differentiable transform-algebra gate.
"""Check differentiable transform-algebra metamorphic coverage."""

from __future__ import annotations

from scpn_quantum_control.differentiable_transform_algebra import (
    assert_transform_algebra_audit_passes,
)


def main() -> int:
    """Run the transform-algebra audit CLI."""
    audit = assert_transform_algebra_audit_passes()
    print(
        "differentiable transform-algebra gate: PASS "
        f"({len(audit.passed_cases)} passed, {len(audit.blocked_cases)} blocked, "
        f"{len(audit.support_matrix)} support rows)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
