#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- synchronisation benchmark comparator
"""Compare regenerated synchronisation benchmark rows against committed rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scpn_quantum_control.benchmark_harness.synchronisation_compare import (
    compare_default_artifacts,
    compare_files,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXPECTED = (
    REPO_ROOT
    / "data"
    / "synchronisation_benchmarks"
    / "kuramoto_ring_n4_linear_omega_reference_rows.json"
)
DEFAULT_ACTUAL = None


def main() -> int:
    """Run the synchronisation benchmark comparator."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected", type=Path, default=None)
    parser.add_argument("--actual", type=Path, default=None)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    if args.expected is None and args.actual is None:
        payload = compare_default_artifacts(REPO_ROOT)
    elif args.expected is not None and args.actual is not None:
        payload = compare_files(args.expected, args.actual)
    else:
        raise SystemExit("--expected and --actual must be provided together")

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"synchronisation benchmark comparison valid: {payload['valid']}")
        for blocker in payload["blockers"]:
            print(f"  blocker: {blocker}")
    return 0 if payload["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
