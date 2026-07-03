# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — JAX delayed-Kuramoto tier benchmark runner
"""Measure the JAX autodiff delayed-Kuramoto tier against the NumPy method-of-steps integrator.

Integrates one time-delayed networked-Kuramoto problem through the JAX tier (on whatever accelerator
JAX selected) and through the production ``integrate_delayed_kuramoto`` NumPy integrator, records the
reproducible cross-tier parity (``parity_max_abs_diff``) and advisory per-call timings, captures host +
JAX-device provenance, and writes the record as JSON.

The reproducible quantity is the parity — the JAX tier reproduces the method-of-steps map at 64-bit
precision. The millisecond timings are advisory host/GPU-bound evidence (``production_claim_allowed:
false``); a clean absolute number needs a quiesced, reserved host and a fixed GPU clock. The delayed
integrator has no Rust or Julia tier, so the reference is the NumPy floor. This requires the optional
``[jax]`` extra.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from scpn_quantum_control.accel.jax_kuramoto_delayed import (
    jax_kuramoto_delayed_ensemble,
    jax_kuramoto_delayed_trajectory,
)
from scpn_quantum_control.accel.kuramoto_delayed import (
    delayed_networked_force,
    integrate_delayed_kuramoto,
)
from scpn_quantum_control.accel.tier_benchmark import capture_provenance, measure
from scpn_quantum_control.hardware.jax_accel import is_jax_gpu_available, jax_device_name

_DEFAULT_OUTPUT = Path("docs/benchmarks/kuramoto_jax_delayed_tier.json")
_SCHEMA = "scpn-quantum-control.kuramoto-jax-delayed-tier.v1"
_CLAIM_BOUNDARY = (
    "The reproducible quantity is the cross-tier parity (parity_max_abs_diff): the JAX delayed tier is "
    "faithful to the production NumPy method-of-steps integrator at 64-bit precision. The per-call "
    "milliseconds are advisory host/GPU-bound evidence captured under the recorded load, governor and "
    "GPU clock, and are excluded from any performance claim (production_claim_allowed is false). The "
    "gradient tier matches the hand-derived method-of-steps sensitivity to machine precision (see "
    "tests/test_jax_kuramoto_delayed)."
)


def _build_network(
    n: int, delay_steps: int, *, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    history = rng.uniform(-0.6, 0.6, size=(delay_steps + 1, n))
    omega = rng.normal(0.0, 0.4, size=n)
    coupling = np.full((n, n), 1.8 / n, dtype=np.float64)
    np.fill_diagonal(coupling, 0.0)
    return history, omega, coupling


def _parse_args(argv: list[str]) -> argparse.Namespace:
    """Parse the runner command-line arguments."""
    parser = argparse.ArgumentParser(description="Benchmark the JAX delayed-Kuramoto tier")
    parser.add_argument("--n", type=int, default=256, help="oscillator count (default 256)")
    parser.add_argument("--dt", type=float, default=0.05, help="integration step (default 0.05)")
    parser.add_argument("--n-steps", type=int, default=200, help="steps (default 200)")
    parser.add_argument("--delay-steps", type=int, default=5, help="delay in steps (default 5)")
    parser.add_argument("--seed", type=int, default=20260703, help="network seed")
    parser.add_argument(
        "--warmup", type=int, default=3, help="discarded warm-up calls (default 3)"
    )
    parser.add_argument("--repeats", type=int, default=15, help="timed calls (default 15)")
    parser.add_argument(
        "--batch",
        type=int,
        default=64,
        help="ensemble batch size for the vmap comparison (default 64)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_DEFAULT_OUTPUT,
        help=f"artefact path (default {_DEFAULT_OUTPUT})",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the JAX delayed-tier benchmark and write the JSON artefact."""
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    delay = args.delay_steps * args.dt
    history, omega, coupling = _build_network(args.n, args.delay_steps, seed=args.seed)

    def _numpy_forward() -> np.ndarray:
        return integrate_delayed_kuramoto(
            history,
            omega,
            lambda current, lagged: delayed_networked_force(current, lagged, coupling),
            delay=delay,
            dt=args.dt,
            n_steps=args.n_steps,
        ).phases

    reference = _numpy_forward()
    jax_result = jax_kuramoto_delayed_trajectory(
        history, omega, coupling, delay=delay, dt=args.dt, n_steps=args.n_steps
    )
    parity_max_abs_diff = float(np.max(np.abs(jax_result - reference)))

    jax_stats = measure(
        lambda: jax_kuramoto_delayed_trajectory(
            history, omega, coupling, delay=delay, dt=args.dt, n_steps=args.n_steps
        ),
        warmup=args.warmup,
        repeats=args.repeats,
    )
    reference_stats = measure(_numpy_forward, warmup=args.warmup, repeats=args.repeats)

    # vmap ensemble: solve args.batch initial histories in one accelerator call vs the same count
    # solved one at a time. The batching factor is advisory; the reproducible quantity is that each
    # batched member equals its single-call solve.
    rng = np.random.default_rng(args.seed + 1)
    history_batch = rng.uniform(-0.6, 0.6, size=(args.batch, args.delay_steps + 1, args.n))
    ensemble = jax_kuramoto_delayed_ensemble(
        history_batch, omega, coupling, delay=delay, dt=args.dt, n_steps=args.n_steps
    )
    ensemble_parity = max(
        float(
            np.max(
                np.abs(
                    ensemble[i]
                    - jax_kuramoto_delayed_trajectory(
                        history_batch[i],
                        omega,
                        coupling,
                        delay=delay,
                        dt=args.dt,
                        n_steps=args.n_steps,
                    )
                )
            )
        )
        for i in range(args.batch)
    )
    ensemble_stats = measure(
        lambda: jax_kuramoto_delayed_ensemble(
            history_batch, omega, coupling, delay=delay, dt=args.dt, n_steps=args.n_steps
        ),
        warmup=args.warmup,
        repeats=args.repeats,
    )

    def _sequential() -> None:
        for member in history_batch:
            jax_kuramoto_delayed_trajectory(
                member, omega, coupling, delay=delay, dt=args.dt, n_steps=args.n_steps
            )

    sequential_stats = measure(_sequential, warmup=1, repeats=max(3, args.repeats // 3))

    import jax

    record: dict[str, object] = {
        "schema": _SCHEMA,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "n_oscillators": args.n,
        "dt": args.dt,
        "n_steps": args.n_steps,
        "delay_steps": args.delay_steps,
        "delay": delay,
        "reference_tier": "python",
        "jax_version": jax.__version__,
        "jax_device": jax_device_name(),
        "jax_gpu": is_jax_gpu_available(),
        "production_claim_allowed": False,
        "claim_boundary": _CLAIM_BOUNDARY,
        "parity_max_abs_diff": parity_max_abs_diff,
        "jax_forward_us": jax_stats.to_dict(),
        "reference_forward_us": reference_stats.to_dict(),
        "ensemble_batch": args.batch,
        "ensemble_parity_max_abs_diff": ensemble_parity,
        "ensemble_forward_us": ensemble_stats.to_dict(),
        "sequential_forward_us": sequential_stats.to_dict(),
        "provenance": capture_provenance().to_dict(),
    }

    output: Path = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")

    print(
        f"N={args.n} steps={args.n_steps} delay_steps={args.delay_steps} "
        f"jax_device={jax_device_name()}"
    )
    print(f"parity_max_abs_diff={parity_max_abs_diff:.2e}")
    print(
        f"jax p50 {jax_stats.p50_us:.1f} us  reference p50 {reference_stats.p50_us:.1f} us (advisory)"
    )
    print(
        f"ensemble B={args.batch} parity={ensemble_parity:.2e}  batched p50 {ensemble_stats.p50_us:.1f} us"
        f"  sequential p50 {sequential_stats.p50_us:.1f} us (advisory)"
    )
    print(f"\nwrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
