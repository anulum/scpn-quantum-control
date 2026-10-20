# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — native transform qualification example
"""Run one required CPU framework through the unified evidence facade.

Example: ``python examples/native_transform_algebra.py jax``.
"""

from __future__ import annotations

import argparse
import json

from scpn_quantum_control.differentiable_api import differentiable_api


def main() -> None:
    """Print native quartic, structured and custom derivative qualification."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("framework", choices=("jax", "pytorch", "tensorflow"))
    args = parser.parse_args()
    result = differentiable_api(
        "transform_algebra_report",
        adapter=args.framework,
        values=[[0.2, -0.4], [0.5, 0.7]],
    )
    print(json.dumps(result.to_dict(), indent=2))
    if not result.supported:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
