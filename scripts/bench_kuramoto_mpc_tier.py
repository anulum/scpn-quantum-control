# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — JAX differentiable-model MPC tier benchmark runner
"""Measure the JAX differentiable-model receding-horizon MPC tier and serialise it.

Records the reproducible quantities the tier is built on — the control-sequence gradient parity against
the hand-derived discrete adjoint at ``r*=0``, the closed-loop terminal tracking error for a
synchronising and a desynchronising target, and the model/plant-mismatch tracking delta (open-loop
minus receding-horizon terminal error, over several seeds) — plus advisory per-call timings, host and
JAX-device provenance, and writes the record as JSON.

The reproducible quantities are the gradient parity and the tracking errors; the millisecond timings are
advisory host/GPU-bound evidence (``production_claim_allowed: false``). The mismatch delta being positive
is a property of receding-horizon feedback, not a superiority claim. This requires the optional ``[jax]``
extra.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from scpn_quantum_control.accel.jax_kuramoto_mpc import (
    jax_mpc_control_value_and_grad,
    jax_mpc_horizon_control,
    receding_horizon_control,
)
from scpn_quantum_control.accel.kuramoto_network_control import (
    integrate_controlled_network,
    network_control_value_and_grad,
)
from scpn_quantum_control.accel.order_parameter_observables import order_parameter
from scpn_quantum_control.accel.tier_benchmark import capture_provenance, measure
from scpn_quantum_control.hardware.jax_accel import is_jax_gpu_available, jax_device_name

_DEFAULT_OUTPUT = Path("docs/benchmarks/kuramoto_mpc_tier.json")
_SCHEMA = "scpn-quantum-control.kuramoto-mpc-tier.v1"
_CLAIM_BOUNDARY = (
    "The reproducible quantities are the r*=0 control-gradient parity against the hand-derived discrete "
    "adjoint (grad_parity_max_abs_diff) and the closed-loop terminal tracking errors. The per-call "
    "milliseconds are advisory host/GPU-bound evidence captured under the recorded load, governor and "
    "GPU clock, and are excluded from any performance claim (production_claim_allowed is false). The "
    "positive mismatch delta (open-loop minus receding terminal error) is a property of receding-horizon "
    "feedback re-planning on the measured plant state, not a superiority claim."
)


def _network(n: int, *, seed: int, clustered: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    omega = rng.normal(0.0, 0.3, size=n)
    coupling = rng.uniform(0.0, 1.2, size=(n, n))
    np.fill_diagonal(coupling, 0.0)
    phases = rng.normal(0.0, 0.15, size=n) if clustered else rng.uniform(0.0, 2.0 * np.pi, size=n)
    return phases, omega, coupling


def _parse_args(argv: list[str]) -> argparse.Namespace:
    """Parse the runner command-line arguments."""
    parser = argparse.ArgumentParser(description="Benchmark the JAX differentiable-model MPC tier")
    parser.add_argument("--n", type=int, default=12, help="oscillator count (default 12)")
    parser.add_argument("--dt", type=float, default=0.05, help="RK4 step (default 0.05)")
    parser.add_argument("--horizon", type=int, default=15, help="planning horizon (default 15)")
    parser.add_argument("--control-steps", type=int, default=40, help="control steps (default 40)")
    parser.add_argument(
        "--control-weight", type=float, default=1e-3, help="control weight (default 1e-3)"
    )
    parser.add_argument(
        "--inner-iterations", type=int, default=60, help="inner iterations (default 60)"
    )
    parser.add_argument(
        "--inner-step-size", type=float, default=30.0, help="inner step (default 30)"
    )
    parser.add_argument("--seed", type=int, default=20260703, help="network seed")
    parser.add_argument(
        "--warmup", type=int, default=2, help="discarded warm-up calls (default 2)"
    )
    parser.add_argument("--repeats", type=int, default=5, help="timed calls (default 5)")
    parser.add_argument(
        "--output",
        type=Path,
        default=_DEFAULT_OUTPUT,
        help=f"artefact path (default {_DEFAULT_OUTPUT})",
    )
    return parser.parse_args(argv)


def _mismatch_delta(
    n: int,
    dt: float,
    horizon: int,
    steps: int,
    weight: float,
    iters: int,
    step: float,
    seed: int,
) -> tuple[float, float]:
    """Return ``(open_loop_error, receding_error)`` for one mismatched seed at target r*=0.2."""
    phases, omega, coupling = _network(n, seed=seed, clustered=True)
    plant = 1.15 * coupling
    target = 0.2
    receding = receding_horizon_control(
        phases,
        omega,
        coupling,
        dt,
        horizon=horizon,
        n_control_steps=steps,
        target_coherence=target,
        control_weight=weight,
        inner_iterations=iters,
        inner_step_size=step,
        plant_coupling=plant,
    )
    receding_error = abs(receding.terminal_coherence - target)
    open_control, _ = jax_mpc_horizon_control(
        phases,
        omega,
        coupling,
        dt,
        steps,
        target_coherence=target,
        control_weight=weight,
        step_size=step,
        n_iterations=iters,
    )
    open_terminal = integrate_controlled_network(phases, open_control, omega, plant, dt).phases[-1]
    open_error = abs(order_parameter(open_terminal) - target)
    return open_error, receding_error


def main(argv: list[str] | None = None) -> int:
    """Run the JAX MPC-tier benchmark and write the JSON artefact."""
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    dt, horizon, steps, weight = args.dt, args.horizon, args.control_steps, args.control_weight
    iters, step = args.inner_iterations, args.inner_step_size

    # (1) r*=0 gradient parity against the hand-derived discrete adjoint
    phases, omega, coupling = _network(args.n, seed=args.seed, clustered=False)
    control = np.random.default_rng(args.seed + 1).normal(0.0, 0.2, size=(horizon, args.n))
    hand = network_control_value_and_grad(
        phases, control, omega, coupling, dt, control_weight=weight
    )
    jax_grad = jax_mpc_control_value_and_grad(
        phases, control, omega, coupling, dt, target_coherence=0.0, control_weight=weight
    )
    grad_parity = float(np.max(np.abs(jax_grad.control_gradient - hand.control_gradient)))
    cost_parity = float(abs(jax_grad.cost - hand.cost))

    # (2) closed-loop terminal tracking error, synchronise (r*=0.9) and desynchronise (r*=0.1)
    ps_sync, om_sync, k_sync = _network(args.n, seed=args.seed + 2, clustered=False)
    sync = receding_horizon_control(
        ps_sync,
        om_sync,
        k_sync,
        dt,
        horizon=horizon,
        n_control_steps=steps,
        target_coherence=0.9,
        control_weight=weight,
        inner_iterations=iters,
        inner_step_size=step,
    )
    ps_de, om_de, k_de = _network(args.n, seed=args.seed + 3, clustered=True)
    desync = receding_horizon_control(
        ps_de,
        om_de,
        k_de,
        dt,
        horizon=horizon,
        n_control_steps=steps,
        target_coherence=0.1,
        control_weight=weight,
        inner_iterations=iters,
        inner_step_size=step,
    )

    # (3) model/plant-mismatch delta over three seeds
    mismatch = [
        _mismatch_delta(args.n, dt, horizon, steps, weight, iters, step, s) for s in (10, 11, 12)
    ]
    open_errors = [pair[0] for pair in mismatch]
    receding_errors = [pair[1] for pair in mismatch]
    mismatch_delta = [o - r for o, r in mismatch]

    # advisory timing: one full receding-horizon closed-loop run
    timing = measure(
        lambda: receding_horizon_control(
            ps_sync,
            om_sync,
            k_sync,
            dt,
            horizon=horizon,
            n_control_steps=steps,
            target_coherence=0.9,
            control_weight=weight,
            inner_iterations=iters,
            inner_step_size=step,
        ),
        warmup=args.warmup,
        repeats=args.repeats,
    )

    import jax

    record: dict[str, object] = {
        "schema": _SCHEMA,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "n_oscillators": args.n,
        "dt": dt,
        "horizon": horizon,
        "control_steps": steps,
        "control_weight": weight,
        "inner_iterations": iters,
        "inner_step_size": step,
        "jax_version": jax.__version__,
        "jax_device": jax_device_name(),
        "jax_gpu": is_jax_gpu_available(),
        "production_claim_allowed": False,
        "claim_boundary": _CLAIM_BOUNDARY,
        "grad_parity_max_abs_diff": grad_parity,
        "cost_parity_abs_diff": cost_parity,
        "sync_target": 0.9,
        "sync_terminal_coherence": sync.terminal_coherence,
        "sync_terminal_error": abs(sync.terminal_coherence - 0.9),
        "desync_target": 0.1,
        "desync_terminal_coherence": desync.terminal_coherence,
        "desync_terminal_error": abs(desync.terminal_coherence - 0.1),
        "mismatch_seeds": [10, 11, 12],
        "mismatch_open_loop_error": open_errors,
        "mismatch_receding_error": receding_errors,
        "mismatch_delta": mismatch_delta,
        "receding_run_us": timing.to_dict(),
        "provenance": capture_provenance().to_dict(),
    }

    output: Path = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")

    print(f"N={args.n} horizon={horizon} control_steps={steps} jax_device={jax_device_name()}")
    print(f"r*=0 grad parity {grad_parity:.2e}  cost parity {cost_parity:.2e}")
    print(
        f"sync rT={sync.terminal_coherence:.3f} (err {abs(sync.terminal_coherence - 0.9):.3f}); "
        f"desync rT={desync.terminal_coherence:.3f} (err {abs(desync.terminal_coherence - 0.1):.3f})"
    )
    print(
        f"mismatch delta (open-receding): {[round(d, 3) for d in mismatch_delta]}  "
        f"receding p50 {timing.p50_us / 1000.0:.1f} ms (advisory)"
    )
    print(f"\nwrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
