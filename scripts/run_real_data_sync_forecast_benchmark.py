#!/usr/bin/env python
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Real-Data Forecast Benchmark CLI
"""Run the real-data synchronisation forecasting benchmark suite."""

from __future__ import annotations

import argparse
import json

from scpn_quantum_control.forecasting import run_real_data_sync_forecast_suite


def main() -> int:
    """Run the real-data synchronization forecast benchmark."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hardware-only",
        action="store_true",
        help="Run only the committed hardware synchronisation trace.",
    )
    parser.add_argument(
        "--min-improvement",
        type=float,
        default=0.10,
        help="Minimum held-out MSE reduction required for pass/fail reporting.",
    )
    args = parser.parse_args()
    results = run_real_data_sync_forecast_suite(
        include_topology_replay=not args.hardware_only,
        min_improvement_fraction=args.min_improvement,
    )
    payload = {
        "suite": "real_data_sync_forecasting",
        "passes": all(result.passes for result in results),
        "results": [result.as_dict() for result in results],
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["passes"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
