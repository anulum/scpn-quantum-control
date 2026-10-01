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
import shlex
from pathlib import Path

from scpn_quantum_control.phase.qnode_affinity_benchmark import (
    run_phase_qnode_affinity_benchmark,
)


def _canonical_command(
    *,
    repetitions: int,
    warmups: int,
    reserved_cpus: str,
    output: str,
    require_isolated: bool,
) -> str:
    """Return a shell-escaped command that reproduces the requested run.

    Parameters
    ----------
    repetitions
        Number of measured benchmark repetitions.
    warmups
        Number of unmeasured warmup repetitions.
    reserved_cpus
        Comma-separated requested processor reservation.
    output
        Evidence output path included in the replay command.
    require_isolated
        Whether replay must enforce actual isolated affinity evidence.

    Returns
    -------
    str
        Quoted command preserving the requested benchmark controls.

    """
    command = [] if not reserved_cpus else ["taskset", "-c", reserved_cpus]
    command.extend(
        [
            "python",
            "tools/run_phase_qnode_affinity_benchmark.py",
            "--repetitions",
            str(repetitions),
            "--warmups",
            str(warmups),
            "--reserved-cpus",
            reserved_cpus,
            "--output",
            output,
        ]
    )
    if require_isolated:
        command.append("--require-isolated")
    return shlex.join(command)


def main() -> None:
    """Run the CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--reserved-cpus", default="")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--recorded-command",
        default="",
        help=(
            "Exact outer command to record when an orchestrator adds chrt or other "
            "admitted isolation controls. Defaults to a reproducible taskset command."
        ),
    )
    parser.add_argument(
        "--require-isolated",
        action="store_true",
        help="Exit non-zero unless the written evidence is classified as isolated_affinity.",
    )
    args = parser.parse_args()
    reserved = tuple(int(item.strip()) for item in args.reserved_cpus.split(",") if item.strip())
    command = args.recorded_command or _canonical_command(
        repetitions=args.repetitions,
        warmups=args.warmups,
        reserved_cpus=args.reserved_cpus,
        output=args.output,
        require_isolated=args.require_isolated,
    )
    result = run_phase_qnode_affinity_benchmark(
        repetitions=args.repetitions,
        warmups=args.warmups,
        reserved_cpus=reserved,
        command=command,
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
