# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — QFI/FSS Differentiable Evidence Example
"""No-credential QFI/FSS finite-size evidence example."""

from __future__ import annotations

from collections.abc import Mapping
from typing import cast

import numpy as np

from scpn_quantum_control import UnifiedDifferentiableAPIResult, differentiable_qfi_fss_report


def build_report() -> UnifiedDifferentiableAPIResult:
    """Return a small deterministic QFI/FSS differentiable evidence report."""
    return differentiable_qfi_fss_report(
        system_sizes=[2, 3],
        k_range=np.linspace(0.5, 3.0, 6, dtype=np.float64),
    )


def main() -> None:
    """Run the no-credential QFI/FSS report workflow."""
    report = build_report()
    payload = report.payload
    bkt_fit = cast(Mapping[str, object], payload["bkt_fit"])
    power_fit = cast(Mapping[str, object], payload["power_fit"])

    print("qfi/fss differentiable evidence")
    print(f"  operation: {report.operation}")
    print(f"  supported: {report.supported}")
    print(f"  method: {report.method}")
    print(f"  system sizes: {payload['system_sizes']}")
    print(f"  bkt model: {bkt_fit['model']}")
    print(f"  power model: {power_fit['model']}")
    print(f"  hardware: {'blocked' if 'no hardware' in report.claim_boundary else 'unknown'}")
    print(f"  claim boundary: {report.claim_boundary}")


if __name__ == "__main__":
    main()
