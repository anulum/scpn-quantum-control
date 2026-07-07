#!/usr/bin/env python
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Topology-policy projection cost benchmark
"""Measure the wall-clock cost of one topology-policy projection per network size.

Emits a JSON artefact so an operator can pick a sensible
``topology_policy_interval`` for the QSNN bridge: the policy runs a projected
SPSA optimisation over the recurrent matrix, so its cost grows with both the
network size and ``max_steps``. Evidence-only; no latency claim is published
from this script.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path
from typing import cast

import numpy as np

from scpn_quantum_control.topology_control import (
    CouplingTopologyObjective,
    NetworkCycleBackend,
    ProjectedSPSAOptimizer,
    TopologicalDynamicCouplingPolicy,
    TopologyConstraintLedger,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPO_ROOT / "artifacts" / "benchmarks" / "topology_policy_projection.json"


def _policy(seed: int, max_steps: int) -> TopologicalDynamicCouplingPolicy:
    objective = CouplingTopologyObjective(
        ph_backend=NetworkCycleBackend(threshold=0.02),
        ledger=TopologyConstraintLedger(),
        h1_target=1.0,
        allow_approximate_ph_backend=True,
    )
    return TopologicalDynamicCouplingPolicy(
        objective=objective,
        optimizer=ProjectedSPSAOptimizer(seed=seed, max_steps=max_steps),
    )


def measure(sizes: tuple[int, ...], repeats: int, max_steps: int, seed: int) -> dict[str, object]:
    """Time one policy projection per size, repeated ``repeats`` times."""
    rng = np.random.default_rng(seed)
    rows = []
    for size in sizes:
        weights = rng.uniform(0.0, 0.5, size=(size, size))
        np.fill_diagonal(weights, 0.0)
        samples = []
        for index in range(repeats):
            policy = _policy(seed + index, max_steps)
            start = time.perf_counter()
            policy.apply(weights)
            samples.append(time.perf_counter() - start)
        rows.append(
            {
                "n_neurons": size,
                "median_s_per_projection": float(np.median(samples)),
                "samples_s": [float(sample) for sample in samples],
            }
        )
    return {
        "schema_version": "scpn-quantum-control.topology-policy-projection-benchmark.v1",
        "classification": "functional_non_isolated",
        "claim_boundary": (
            "interval-sizing evidence only; measured on an unpinned host and "
            "never published as a latency claim"
        ),
        "parameters": {
            "sizes": list(sizes),
            "repeats": repeats,
            "spsa_max_steps": max_steps,
            "seed": seed,
        },
        "platform": platform.platform(),
        "rows": rows,
    }


def main(argv: list[str] | None = None) -> int:
    """Run the projection-cost benchmark and write the JSON artefact."""
    parser = argparse.ArgumentParser(description="Topology-policy projection cost benchmark.")
    parser.add_argument("--sizes", default="8,16,32", help="comma-separated neuron counts")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=8, help="SPSA steps per projection")
    parser.add_argument("--seed", type=int, default=20260707)
    parser.add_argument("--json-out", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)

    sizes = tuple(int(token) for token in args.sizes.split(",") if token.strip())
    if not sizes or any(size < 2 for size in sizes):
        raise SystemExit("--sizes must list integers >= 2")
    if args.repeats < 1:
        raise SystemExit("--repeats must be >= 1")

    payload = measure(sizes, args.repeats, args.max_steps, args.seed)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    for row in cast("list[dict[str, object]]", payload["rows"]):
        median = cast(float, row["median_s_per_projection"])
        print(f"n={row['n_neurons']!s:>4}  median {median:.4f} s/projection")
    print(f"wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
