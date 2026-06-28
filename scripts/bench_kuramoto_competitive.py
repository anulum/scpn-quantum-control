# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Kuramoto external competitive benchmark runner
"""Run the Kuramoto external competitive comparison and serialise the artifact.

Integrates one deterministic Kuramoto problem through our RK4 and DOPRI5
integrators and through every available external competitor (SciPy
``solve_ivp``, Julia ``DifferentialEquations.jl``), records each solver's final
order parameter, its accuracy error against the high-precision reference, and
its wall-clock time, and writes the full provenance-carrying record as JSON.

Absent competitors (NetworkDynamics.jl, DynamicalSystems.jl, SciMLSensitivity.jl,
jitcdde) are recorded as fail-closed rows with their install commands rather than
fabricated, so the artifact is complete and reproducible on any host. The timings
are functional/reproducibility evidence on the recorded host, not a
production-latency claim — run on a quiesced, core-reserved host for clean
numbers (see ``docs/internal`` isolation guidance).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from scpn_quantum_control.benchmarks import (
    build_default_problem,
    run_kuramoto_competitive_comparison,
)

_DEFAULT_OUTPUT = Path("docs/benchmarks/kuramoto_competitive.json")


def _parse_args(argv: list[str]) -> argparse.Namespace:
    """Parse the runner command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run the Kuramoto external competitive comparison",
    )
    parser.add_argument("--n", type=int, default=12, help="oscillator count (default 12)")
    parser.add_argument("--t-max", type=float, default=6.0, help="integration time (default 6.0)")
    parser.add_argument("--dt", type=float, default=0.01, help="fixed-grid step (default 0.01)")
    parser.add_argument("--seed", type=int, default=20260628, help="problem seed")
    parser.add_argument(
        "--julia-timeout",
        type=float,
        default=600.0,
        help="hard wall-clock limit per Julia subprocess (default 600s for cold start)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_DEFAULT_OUTPUT,
        help=f"artifact path (default {_DEFAULT_OUTPUT})",
    )
    return parser.parse_args(argv)


def _print_summary(record: dict[str, object]) -> None:
    """Print a terse human-readable summary of the comparison."""
    rows = record["rows"]
    assert isinstance(rows, list)
    print(f"reference: {record['reference_method']}  (n={record['n_oscillators']})")
    print(f"{'method':22} {'avail':5} {'r_final':>12} {'err_vs_ref':>12} {'ms':>10}")
    for row in rows:
        assert isinstance(row, dict)
        r_final = row["r_final"]
        err = row["r_error_vs_reference"]
        ms = row["elapsed_ms"]
        print(
            f"{row['method']:22} {str(row['available']):5} "
            f"{('' if r_final is None else f'{r_final:.8f}'):>12} "
            f"{('' if err is None else f'{err:.2e}'):>12} "
            f"{('' if ms is None else f'{ms:.3f}'):>10}"
        )


def main(argv: list[str] | None = None) -> int:
    """Run the competitive comparison and write the JSON artifact."""
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    problem = build_default_problem(
        n_oscillators=args.n, seed=args.seed, t_max=args.t_max, dt=args.dt
    )
    comparison = run_kuramoto_competitive_comparison(problem, julia_timeout=args.julia_timeout)
    record = comparison.to_dict()

    output: Path = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")

    _print_summary(record)
    print(f"\nwrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
