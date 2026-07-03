# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — JAX Kuramoto tier benchmark runner
"""Measure the JAX autodiff Kuramoto RK4 tier against the production Rust tier and serialise it.

Integrates one networked-Kuramoto problem through the JAX tier (on whatever accelerator JAX selected)
and through the production ``kuramoto_rk4_trajectory`` facade, records the reproducible cross-tier
parity (``parity_max_abs_diff``) and advisory per-call timings, captures host + JAX-device provenance,
and writes the record as JSON.

The reproducible quantity is the parity — the JAX tier is faithful to the production integrator at
64-bit precision. The millisecond timings are advisory host/GPU-bound evidence (``production_claim_
allowed: false``); a clean absolute number needs a quiesced, reserved host and a fixed GPU clock. This
requires the optional ``[jax]`` extra.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from scpn_quantum_control import kuramoto
from scpn_quantum_control.accel.jax_kuramoto import jax_kuramoto_rk4_trajectory
from scpn_quantum_control.accel.tier_benchmark import capture_provenance, measure
from scpn_quantum_control.hardware.jax_accel import is_jax_gpu_available, jax_device_name

_DEFAULT_OUTPUT = Path("docs/benchmarks/kuramoto_jax_tier.json")
_SCHEMA = "scpn-quantum-control.kuramoto-jax-tier.v1"
_CLAIM_BOUNDARY = (
    "The reproducible quantity is the cross-tier parity (parity_max_abs_diff): the JAX tier is "
    "faithful to the production Rust integrator at 64-bit precision. The per-call milliseconds are "
    "advisory host/GPU-bound evidence captured under the recorded load, governor and GPU clock, and "
    "are excluded from any performance claim (production_claim_allowed is false). The gradient tier "
    "matches the hand-derived reverse-mode adjoint to machine precision (see tests/test_jax_kuramoto)."
)


def _build_network(n: int, *, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    omega = rng.normal(0.0, 0.7, size=n)
    coupling = np.full((n, n), 1.8 / n, dtype=np.float64)
    np.fill_diagonal(coupling, 0.0)
    theta0 = rng.uniform(0.0, 2.0 * np.pi, size=n)
    return theta0, omega, coupling


def _parse_args(argv: list[str]) -> argparse.Namespace:
    """Parse the runner command-line arguments."""
    parser = argparse.ArgumentParser(description="Benchmark the JAX Kuramoto RK4 tier")
    parser.add_argument("--n", type=int, default=256, help="oscillator count (default 256)")
    parser.add_argument("--dt", type=float, default=0.05, help="RK4 step (default 0.05)")
    parser.add_argument("--n-steps", type=int, default=200, help="steps (default 200)")
    parser.add_argument("--seed", type=int, default=20260703, help="network seed")
    parser.add_argument(
        "--warmup", type=int, default=3, help="discarded warm-up calls (default 3)"
    )
    parser.add_argument("--repeats", type=int, default=15, help="timed calls (default 15)")
    parser.add_argument(
        "--output",
        type=Path,
        default=_DEFAULT_OUTPUT,
        help=f"artefact path (default {_DEFAULT_OUTPUT})",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the JAX-tier benchmark and write the JSON artefact."""
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    theta0, omega, coupling = _build_network(args.n, seed=args.seed)

    rust = kuramoto.kuramoto_rk4_trajectory(theta0, omega, coupling, args.dt, args.n_steps)
    rust_tier = kuramoto.last_kuramoto_rk4_trajectory_tier_used()
    jax_result = jax_kuramoto_rk4_trajectory(theta0, omega, coupling, args.dt, args.n_steps)
    parity_max_abs_diff = float(np.max(np.abs(jax_result - rust)))

    jax_stats = measure(
        lambda: jax_kuramoto_rk4_trajectory(theta0, omega, coupling, args.dt, args.n_steps),
        warmup=args.warmup,
        repeats=args.repeats,
    )
    rust_stats = measure(
        lambda: kuramoto.kuramoto_rk4_trajectory(theta0, omega, coupling, args.dt, args.n_steps),
        warmup=args.warmup,
        repeats=args.repeats,
    )

    import jax

    record: dict[str, object] = {
        "schema": _SCHEMA,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "n_oscillators": args.n,
        "dt": args.dt,
        "n_steps": args.n_steps,
        "reference_tier": rust_tier,
        "jax_version": jax.__version__,
        "jax_device": jax_device_name(),
        "jax_gpu": is_jax_gpu_available(),
        "production_claim_allowed": False,
        "claim_boundary": _CLAIM_BOUNDARY,
        "parity_max_abs_diff": parity_max_abs_diff,
        "jax_forward_us": jax_stats.to_dict(),
        "reference_forward_us": rust_stats.to_dict(),
        "provenance": capture_provenance().to_dict(),
    }

    output: Path = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")

    print(
        f"N={args.n} steps={args.n_steps} reference_tier={rust_tier} jax_device={jax_device_name()}"
    )
    print(f"parity_max_abs_diff={parity_max_abs_diff:.2e}")
    print(
        f"jax p50 {jax_stats.p50_us:.1f} us  reference p50 {rust_stats.p50_us:.1f} us (advisory)"
    )
    print(f"\nwrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
