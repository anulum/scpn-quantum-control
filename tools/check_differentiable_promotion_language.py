# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — differentiable promotion-language gate.
"""Fail CI when public differentiable promotional wording outruns evidence."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from scpn_quantum_control.differentiable_baseline_scorecard import (  # noqa: E402
    audit_differentiable_promotion_language,
)


def main() -> int:
    """Run the public differentiable promotion-language audit."""
    audit = audit_differentiable_promotion_language(repo_root=ROOT)
    if audit.passed:
        print(
            "differentiable promotion-language gate: PASS "
            f"({len(audit.checked_paths)} public surfaces checked)"
        )
        return 0
    print("differentiable promotion-language gate: FAIL")
    for error in audit.errors:
        print(f"  - {error}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
