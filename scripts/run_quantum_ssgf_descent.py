#!/usr/bin/env python
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Quantum SSGF descent runner
"""Run the quantum SSGF variational geometry descent and record the artefact.

Production entry point for ``ssgf.quantum_outer_cycle``: optimises a latent
vector ``z`` parameterising the coupling geometry ``W(z)`` against the
quantum synchronisation cost ``1 - R_global`` (statevector Trotter
evolution), and writes a JSON artefact carrying the full convergence trace
plus an explicit claim boundary. Small-system statevector simulation only.
"""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
from typing import cast

from scpn_quantum_control.ssgf.quantum_outer_cycle import OuterCycleResult, quantum_outer_cycle

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPO_ROOT / "artifacts" / "ssgf" / "quantum_ssgf_descent.json"

CLAIM_BOUNDARY = (
    "small-system statevector simulation of the quantum SSGF outer cycle; "
    "not a hardware run, not a scaling claim, and not evidence for the "
    "classical SSGF production stack in SCPN-CODEBASE"
)


def descent_artifact(result: OuterCycleResult, parameters: dict[str, object]) -> dict[str, object]:
    """Assemble the JSON artefact for one descent run."""
    return {
        "schema_version": "scpn-quantum-control.quantum-ssgf-descent.v1",
        "classification": "functional_non_isolated",
        "claim_boundary": CLAIM_BOUNDARY,
        "parameters": parameters,
        "platform": platform.platform(),
        "result": {
            "n_iterations": result.n_iterations,
            "converged": result.converged,
            "final_cost": result.final_cost,
            "final_r_global": result.final_r_global,
            "initial_r_global": result.r_global_history[0],
            "cost_history": [float(value) for value in result.cost_history],
            "r_global_history": [float(value) for value in result.r_global_history],
            "z_optimised": [float(value) for value in result.z_optimised],
            "w_optimised": [[float(v) for v in row] for row in result.W_optimised],
        },
    }


def main(argv: list[str] | None = None) -> int:
    """Run one quantum SSGF descent and write the artefact."""
    parser = argparse.ArgumentParser(description="Quantum SSGF variational geometry descent.")
    parser.add_argument("--n-osc", type=int, default=4, help="oscillator count (qubits)")
    parser.add_argument("--alpha", type=float, default=1.0, help="quantum cost weight in [0, 1]")
    parser.add_argument(
        "--allow-classical-surrogate",
        action="store_true",
        help="opt into the legacy coupling-balance surrogate when alpha < 1",
    )
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--max-iterations", type=int, default=30)
    parser.add_argument(
        "--dt",
        type=float,
        default=1.0,
        help="Trotter evolution time; at dt~0.1 the quantum gradient is weak "
        "(|grad|~2e-3) and descent stalls at the convergence threshold",
    )
    parser.add_argument("--trotter-reps", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260708)
    parser.add_argument("--json-out", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)

    if args.n_osc < 2 or args.n_osc > 10:
        raise SystemExit("--n-osc must be in [2, 10] (statevector simulation)")
    if not 0.0 <= args.alpha <= 1.0:
        raise SystemExit("--alpha must be in [0, 1]")
    if args.max_iterations < 1:
        raise SystemExit("--max-iterations must be >= 1")

    result = quantum_outer_cycle(
        n_osc=args.n_osc,
        alpha=args.alpha,
        allow_classical_surrogate=args.allow_classical_surrogate,
        learning_rate=args.learning_rate,
        max_iterations=args.max_iterations,
        dt=args.dt,
        trotter_reps=args.trotter_reps,
        seed=args.seed,
    )

    parameters: dict[str, object] = {
        "n_osc": args.n_osc,
        "alpha": args.alpha,
        "allow_classical_surrogate": args.allow_classical_surrogate,
        "learning_rate": args.learning_rate,
        "max_iterations": args.max_iterations,
        "dt": args.dt,
        "trotter_reps": args.trotter_reps,
        "seed": args.seed,
    }
    payload = descent_artifact(result, parameters)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    trace = cast("dict[str, object]", payload["result"])
    print(
        f"descent: {result.n_iterations} iterations, converged={result.converged}, "
        f"R_global {trace['initial_r_global']:.4f} -> {result.final_r_global:.4f}"
    )
    print(f"wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
