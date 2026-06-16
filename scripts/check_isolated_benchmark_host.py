# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — isolated-benchmark host readiness checker
"""Check whether this host can produce ``isolated_affinity`` benchmark evidence.

Run on a candidate self-hosted runner before dispatching the isolated benchmark
workflow. Exits non-zero (and lists the blockers) when the host would downgrade
the evidence to ``functional_non_isolated``.

    python scripts/check_isolated_benchmark_host.py --reserved-core 0
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys

from scpn_quantum_control.benchmarks.isolated_host_readiness import capture_host_readiness


def main() -> int:
    """Run the isolated-benchmark host readiness check."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reserved-core",
        type=int,
        default=0,
        help="CPU index reserved for the benchmark (default: 0)",
    )
    parser.add_argument("--json", action="store_true", help="emit a machine-readable report")
    args = parser.parse_args()

    readiness = capture_host_readiness(args.reserved_core)

    if args.json:
        payload = dataclasses.asdict(readiness)
        payload["load_average"] = (
            list(readiness.load_average) if readiness.load_average is not None else None
        )
        print(json.dumps(payload, indent=2))
    else:
        status = "READY" if readiness.ready else "NOT READY"
        print(f"isolated-benchmark host: {status}")
        print(f"  reserved core  : cpu{readiness.reserved_core}")
        print(f"  governor       : {readiness.governor} (stable={readiness.governor_is_stable})")
        print(f"  frequency MHz  : {readiness.frequency_mhz}")
        print(f"  load average   : {readiness.load_average} (low={readiness.load_is_low})")
        for blocker in readiness.blockers:
            print(f"  blocker        : {blocker}")

    return 0 if readiness.ready else 1


if __name__ == "__main__":
    sys.exit(main())
