# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Phase QNode Affinity Benchmark CLI
"""Write Phase-QNode affinity benchmark metadata as JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lean_phase_import import load_phase_module


def main() -> None:
    """Run the CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--reserved-cpus", default="")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--require-isolated",
        action="store_true",
        help="Exit non-zero unless the written evidence is classified as isolated_affinity.",
    )
    args = parser.parse_args()
    reserved = tuple(int(item.strip()) for item in args.reserved_cpus.split(",") if item.strip())
    run_phase_qnode_affinity_benchmark = load_phase_module(
        "qnode_affinity_benchmark"
    ).run_phase_qnode_affinity_benchmark
    result = run_phase_qnode_affinity_benchmark(
        repetitions=args.repetitions,
        warmups=args.warmups,
        reserved_cpus=reserved,
        command=(
            f"taskset -c {args.reserved_cpus} python tools/run_phase_qnode_affinity_benchmark.py"
        ),
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    if args.require_isolated and result.evidence_label != "isolated_affinity":
        raise SystemExit(
            "isolated_affinity evidence was required but benchmark classified as "
            f"{result.evidence_label}: {', '.join(result.isolation_failures)}"
        )


if __name__ == "__main__":
    main()
