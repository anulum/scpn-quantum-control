# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Neural-operator surrogate advantage study runner
"""Run the neural-operator surrogate advantage study and serialise the artefact.

Trains the DeepONet surrogate on a concrete Kuramoto network, measures its held-out forecast fidelity
against the persistence baseline, assembles the host-independent operation-count model, records
advisory host-bounded timings, and writes the full provenance-carrying record as JSON. A host-
independent operation-count scaling sweep is appended (per-query FLOP ratio across a grid of network
sizes and horizons) so the crossover can be cited without a training run.

The reproducible quantities are the held-out fidelity and the operation-count model. The wall-clock
milliseconds are advisory host-bounded evidence — run on a quiesced, core-reserved host for clean
numbers; the surrogate's advantage is structural (random access and amortisation), not a single-query
millisecond margin at small ``N``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from scpn_quantum_control.forecasting.neural_operator_advantage import (
    evaluate_neural_operator_advantage,
)
from scpn_quantum_control.forecasting.neural_operator_cost_model import (
    deeponet_forward_flops,
    direct_simulation_flops,
    rk4_right_hand_side_evaluations,
)

_DEFAULT_OUTPUT = Path("docs/benchmarks/neural_operator_advantage.json")
_SCALING_OSCILLATORS = (16, 32, 64, 128)
_SCALING_STEPS = (20, 40, 80, 160)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    """Parse the runner command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run the neural-operator surrogate advantage study",
    )
    parser.add_argument("--n", type=int, default=32, help="oscillator count (default 32)")
    parser.add_argument("--dt", type=float, default=0.05, help="RK4 step (default 0.05)")
    parser.add_argument(
        "--n-steps", type=int, default=20, help="steps to the horizon (default 20)"
    )
    parser.add_argument(
        "--n-trajectories", type=int, default=256, help="training trajectories (default 256)"
    )
    parser.add_argument("--n-eval", type=int, default=40, help="held-out evaluations (default 40)")
    parser.add_argument(
        "--latent-dim", type=int, default=32, help="DeepONet latent width (default 32)"
    )
    parser.add_argument(
        "--hidden-dim", type=int, default=96, help="DeepONet hidden width (default 96)"
    )
    parser.add_argument("--epochs", type=int, default=400, help="training epochs (default 400)")
    parser.add_argument("--sigma", type=float, default=0.5, help="frequency spread (default 0.5)")
    parser.add_argument(
        "--mean-field-k",
        type=float,
        default=2.0,
        help="mean-field coupling strength (default 2.0)",
    )
    parser.add_argument("--seed", type=int, default=11, help="network seed (default 11)")
    parser.add_argument(
        "--no-wall-clock", action="store_true", help="skip the advisory host-bounded timings"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_DEFAULT_OUTPUT,
        help=f"artefact path (default {_DEFAULT_OUTPUT})",
    )
    return parser.parse_args(argv)


def _build_network(
    n: int, *, sigma: float, mean_field_k: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Build a complete-graph Kuramoto network with Gaussian natural frequencies."""
    rng = np.random.default_rng(seed)
    omega = rng.normal(0.0, sigma, size=n)
    coupling = np.full((n, n), mean_field_k / n, dtype=np.float64)
    np.fill_diagonal(coupling, 0.0)
    return omega, coupling


def _operation_count_scaling(latent_dim: int, hidden_dim: int) -> list[dict[str, object]]:
    """Host-independent per-query FLOP ratio across a grid of sizes and horizons (no training)."""
    rows: list[dict[str, object]] = []
    for n in _SCALING_OSCILLATORS:
        for steps in _SCALING_STEPS:
            direct = direct_simulation_flops(n, steps)
            surrogate = deeponet_forward_flops(n, latent_dim, hidden_dim)
            rows.append(
                {
                    "n_oscillators": n,
                    "n_steps": steps,
                    "direct_flops_per_query": direct,
                    "surrogate_flops_per_query": surrogate,
                    "per_query_flop_ratio": direct / surrogate,
                    "rk4_right_hand_side_evaluations": rk4_right_hand_side_evaluations(steps),
                }
            )
    return rows


def _print_summary(record: dict[str, object]) -> None:
    """Print a terse human-readable summary of the study."""
    fidelity = record["fidelity"]
    cost = record["cost_model"]
    assert isinstance(fidelity, dict) and isinstance(cost, dict)
    print(
        f"network: N={record['n_oscillators']}  horizon={record['horizon']}  rk4_tier={record['rk4_tier']}"
    )
    print(f"loss: {record['loss_start']:.5f} -> {record['loss_final']:.5f}")
    print(
        f"held-out mean error: surrogate {fidelity['surrogate_mean_error']:.4f} rad  "
        f"persistence {fidelity['persistence_mean_error']:.4f} rad  "
        f"beats={fidelity['beats_persistence']}"
    )
    print(
        f"operation count: {cost['rk4_right_hand_side_evaluations']} RHS evals eliminated per query; "
        f"per-query FLOP ratio {cost['per_query_flop_ratio']:.2f}x; break-even "
        f"{cost['break_even_queries']} queries"
    )
    wall_clock = record["wall_clock_ms"]
    if isinstance(wall_clock, dict):
        print(
            f"wall-clock (host-bounded): direct {wall_clock['direct_full_trajectory_ms']:.3f}ms  "
            f"surrogate {wall_clock['surrogate_single_query_ms']:.3f}ms"
        )


def main(argv: list[str] | None = None) -> int:
    """Run the advantage study and write the JSON artefact."""
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    omega, coupling = _build_network(
        args.n, sigma=args.sigma, mean_field_k=args.mean_field_k, seed=args.seed
    )
    study = evaluate_neural_operator_advantage(
        omega,
        coupling,
        dt=args.dt,
        n_steps=args.n_steps,
        n_trajectories=args.n_trajectories,
        n_eval=args.n_eval,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        epochs=args.epochs,
        measure_wall_clock=not args.no_wall_clock,
    )
    record = study.to_dict()
    record["operation_count_scaling"] = _operation_count_scaling(args.latent_dim, args.hidden_dim)

    output: Path = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")

    _print_summary(record)
    print(f"\nwrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
