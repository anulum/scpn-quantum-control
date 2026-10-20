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

import argparse
import json

from scpn_quantum_control.differentiable_transform_algebra import (
    assert_transform_algebra_audit_passes,
    run_native_transform_algebra_audit,
)


def main() -> int:
    """Run the transform-algebra audit CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--framework", choices=("jax", "pytorch", "tensorflow"))
    parser.add_argument("--values", help="JSON vector or matrix; default: [0.25, -0.5, 0.75]")
    parser.add_argument("--batch-axis", type=int, choices=(0, 1))
    args = parser.parse_args()
    candidate = None
    if args.framework is not None:
        candidate = run_native_transform_algebra_audit(
            args.framework,
            [0.25, -0.5, 0.75] if args.values is None else json.loads(args.values),
            batch_axis=0 if args.batch_axis is None else args.batch_axis,
        )
    elif args.batch_axis is not None or args.values is not None:
        parser.error("--values and --batch-axis require --framework")
    audit = assert_transform_algebra_audit_passes(candidate)
    print(
        "differentiable transform-algebra gate: PASS "
        f"({len(audit.passed_cases)} passed, {len(audit.blocked_cases)} blocked, "
        f"{len(audit.support_matrix)} support rows)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
