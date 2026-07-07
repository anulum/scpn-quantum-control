#!/usr/bin/env python
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Psi-field HMC update cost benchmark
"""Measure the wall-clock cost of one U(1) lattice HMC update per graph size.

Emits a JSON artefact so an operator can budget the thermalisation length in
``gauge.crosscheck_confinement_on_lattice``: HMC cost grows with the edge and
plaquette count of the coupling graph. Evidence-only; no latency claim is
published from this script.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path
from typing import cast

import numpy as np
from numpy.typing import NDArray

from scpn_quantum_control.psi_field.lattice import U1LatticGauge, hmc_update

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPO_ROOT / "artifacts" / "benchmarks" / "psi_field_hmc_update.json"


def _ring_with_chords(n: int) -> NDArray[np.float64]:
    """Return a ring topology with next-nearest chords (plaquette-rich)."""
    K = np.zeros((n, n))
    for i in range(n):
        K[i, (i + 1) % n] = K[(i + 1) % n, i] = 0.7
        K[i, (i + 2) % n] = K[(i + 2) % n, i] = 0.3
    return K


def measure(sizes: tuple[int, ...], repeats: int, n_leapfrog: int, seed: int) -> dict[str, object]:
    """Time one HMC update per size, repeated ``repeats`` times after warm-up."""
    rows = []
    for size in sizes:
        gauge = U1LatticGauge(_ring_with_chords(size), beta=2.0, seed=seed)
        hmc_update(gauge, n_leapfrog=n_leapfrog)
        samples = []
        accepted = 0
        for _ in range(repeats):
            start = time.perf_counter()
            step_accepted, _ = hmc_update(gauge, n_leapfrog=n_leapfrog)
            samples.append(time.perf_counter() - start)
            accepted += int(step_accepted)
        rows.append(
            {
                "n_sites": size,
                "n_edges": gauge.n_edges,
                "median_s_per_update": float(np.median(samples)),
                "acceptance_rate": accepted / repeats,
                "samples_s": [float(sample) for sample in samples],
            }
        )
    return {
        "schema_version": "scpn-quantum-control.psi-field-hmc-benchmark.v1",
        "classification": "functional_non_isolated",
        "claim_boundary": (
            "thermalisation-budget evidence only; measured on an unpinned host "
            "and never published as a latency claim"
        ),
        "parameters": {
            "sizes": list(sizes),
            "repeats": repeats,
            "n_leapfrog": n_leapfrog,
            "seed": seed,
            "topology": "ring_with_next_nearest_chords",
            "beta": 2.0,
        },
        "platform": platform.platform(),
        "rows": rows,
    }


def main(argv: list[str] | None = None) -> int:
    """Run the HMC-cost benchmark and write the JSON artefact."""
    parser = argparse.ArgumentParser(description="Psi-field HMC update cost benchmark.")
    parser.add_argument("--sizes", default="8,16,32", help="comma-separated site counts")
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--n-leapfrog", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260708)
    parser.add_argument("--json-out", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)

    sizes = tuple(int(token) for token in args.sizes.split(",") if token.strip())
    if not sizes or any(size < 3 for size in sizes):
        raise SystemExit("--sizes must list integers >= 3")
    if args.repeats < 1:
        raise SystemExit("--repeats must be >= 1")
    if args.n_leapfrog < 1:
        raise SystemExit("--n-leapfrog must be >= 1")

    payload = measure(sizes, args.repeats, args.n_leapfrog, args.seed)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    for row in cast("list[dict[str, object]]", payload["rows"]):
        median = cast(float, row["median_s_per_update"])
        print(f"n={row['n_sites']!s:>4}  median {median:.5f} s/update")
    print(f"wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
