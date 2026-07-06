#!/usr/bin/env python
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Consolidated multi-tier Kuramoto benchmark runner
"""Measure every registered Kuramoto compute primitive across every tier.

This is the single source of truth for the multi-language acceleration chain:
it enumerates the dispatcher registry (``accel.dispatcher.registered_dispatchers``),
and for each primitive measures every tier in its chain — Rust, Julia, and the
Python floor — on shared, seeded inputs at a sweep of problem sizes. Each tier's
output is parity-checked against the Python floor so the artefact records
cross-tier agreement alongside the timings.

The run emits two artefacts under ``--output-dir`` (default
``docs/benchmarks/tiers``):

* ``kuramoto_tiers.<environment>.json`` — one ``scpn-quantum-control.tier-benchmark.v1``
  artefact with the full per-(operation, backend) timing rows.
* ``kuramoto_tiers_manifest.<environment>.json`` — the consolidated
  reproducibility manifest (host + toolchain provenance, run parameters, tier
  availability, and the per-primitive fastest-backend / parity index).

Per the unified benchmark standard, the *canonical* number is measured on the
fixed CI runner (``--environment ci``); the workstation figure is committed as
the ``--environment local`` sibling and presented side by side by
``tools/render_tier_benchmarks.py``. Hosted-runner CPU differs from the declared
workstation, so neither artefact is a production claim
(``production_claim_allowed: false``); both are reproducible evidence.

Usage
-----

.. code-block:: shell

    python scripts/bench_kuramoto_tiers.py --environment local
    python scripts/bench_kuramoto_tiers.py --environment ci --tiers rust,julia,python
    python scripts/bench_kuramoto_tiers.py --tiers rust,python --sizes 8,64,512 --pin-core 2
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from scpn_quantum_control.accel import dispatcher as _dispatcher
from scpn_quantum_control.accel.tier_benchmark import (
    BackendRow,
    PrimitiveResult,
    Provenance,
    build_manifest,
    build_primitive_artifact,
    capture_provenance,
    measure,
    measured_row,
    unavailable_row,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "docs" / "benchmarks" / "tiers"
DEFAULT_SIZES = (8, 32, 128, 512, 2048)
DEFAULT_SEED = 20260625

# Fixed secondary parameters held constant across the size sweep so the only
# free variable is N. Chosen to exercise a representative, non-degenerate regime
# of each model rather than an edge case.
_DAIDO_MODE = 2
_SAKAGUCHI_FRUSTRATION = 0.3
_SCALAR_COUPLING = 1.5
_TRAJECTORY_DT = 0.01
_TRAJECTORY_STEPS = 64

# Cost classes select the hot-loop ``inner`` repeat count: cheap scalar kernels
# are measured against a tight loop to clear the timer-resolution floor; the
# quadratic matrix kernels use a smaller loop; the trajectory integrators and
# their adjoints are expensive enough to time one call at a time.
_INNER_BY_COST = {"scalar": 50, "matrix": 10, "trajectory": 1}


# ---------------------------------------------------------------------------
# Input generation — one builder per dispatcher signature class
# ---------------------------------------------------------------------------

ArgsBuilder = Callable[[int, np.random.Generator], tuple[Any, ...]]


def _phases(n: int, rng: np.random.Generator) -> NDArray[np.float64]:
    """Return ``n`` phases drawn uniformly from ``(-π, π]``."""

    return rng.uniform(-np.pi, np.pi, size=n)


def _symmetric_coupling(n: int, rng: np.random.Generator) -> NDArray[np.float64]:
    """Return a symmetric, zero-diagonal ``n × n`` coupling matrix."""

    base = rng.uniform(-1.0, 1.0, size=(n, n))
    coupling = (base + base.T) / 2.0
    np.fill_diagonal(coupling, 0.0)
    return np.ascontiguousarray(coupling, dtype=np.float64)


def _adjacency(n: int, rng: np.random.Generator) -> NDArray[np.float64]:
    """Return a symmetric, zero-diagonal ``0/1`` adjacency matrix (~40 % dense)."""

    upper = (rng.uniform(0.0, 1.0, size=(n, n)) < 0.4).astype(np.float64)
    adjacency = np.triu(upper, k=1)
    adjacency = adjacency + adjacency.T
    return np.ascontiguousarray(adjacency, dtype=np.float64)


def _build_phases(n: int, rng: np.random.Generator) -> tuple[Any, ...]:
    return (_phases(n, rng),)


def _build_phases_mode(n: int, rng: np.random.Generator) -> tuple[Any, ...]:
    return (_phases(n, rng), _DAIDO_MODE)


def _build_phases_coupling(n: int, rng: np.random.Generator) -> tuple[Any, ...]:
    return (_phases(n, rng), _symmetric_coupling(n, rng))


def _build_phases_scalar(n: int, rng: np.random.Generator) -> tuple[Any, ...]:
    return (_phases(n, rng), _SCALAR_COUPLING)


def _build_phases_scalar_mode(n: int, rng: np.random.Generator) -> tuple[Any, ...]:
    return (_phases(n, rng), _SCALAR_COUPLING, _DAIDO_MODE)


def _build_phases_scalar_frustration(n: int, rng: np.random.Generator) -> tuple[Any, ...]:
    return (_phases(n, rng), _SCALAR_COUPLING, _SAKAGUCHI_FRUSTRATION)


def _build_phases_coupling_frustration(n: int, rng: np.random.Generator) -> tuple[Any, ...]:
    return (_phases(n, rng), _symmetric_coupling(n, rng), _SAKAGUCHI_FRUSTRATION)


def _build_phases_adjacency(n: int, rng: np.random.Generator) -> tuple[Any, ...]:
    return (_phases(n, rng), _adjacency(n, rng))


def _build_trajectory_args(n: int, rng: np.random.Generator) -> tuple[Any, ...]:
    return (
        _phases(n, rng),
        rng.uniform(-0.5, 0.5, size=n),
        _symmetric_coupling(n, rng),
        _TRAJECTORY_DT,
        _TRAJECTORY_STEPS,
    )


def _forward_trajectory(operation: str, n: int, rng: np.random.Generator) -> NDArray[np.float64]:
    """Integrate a forward trajectory with the Python floor for a VJP input."""

    theta0 = _phases(n, rng)
    omega = rng.uniform(-0.5, 0.5, size=n)
    coupling = _symmetric_coupling(n, rng)
    forward = "kuramoto_euler_trajectory" if "euler" in operation else "kuramoto_rk4_trajectory"
    chain = dict(_dispatcher.registered_dispatchers()[forward].chain)
    trajectory = chain["python"](theta0, omega, coupling, _TRAJECTORY_DT, _TRAJECTORY_STEPS)
    return np.ascontiguousarray(np.asarray(trajectory, dtype=np.float64))


def _build_euler_vjp_args(n: int, rng: np.random.Generator) -> tuple[Any, ...]:
    trajectory = _forward_trajectory("kuramoto_euler_vjp", n, rng)
    coupling = _symmetric_coupling(n, rng)
    cotangent = rng.standard_normal(size=n)
    return (trajectory, coupling, _TRAJECTORY_DT, cotangent)


def _build_rk4_vjp_args(n: int, rng: np.random.Generator) -> tuple[Any, ...]:
    trajectory = _forward_trajectory("kuramoto_rk4_vjp", n, rng)
    omega = rng.uniform(-0.5, 0.5, size=n)
    coupling = _symmetric_coupling(n, rng)
    cotangent = rng.standard_normal(size=n)
    return (trajectory, omega, coupling, _TRAJECTORY_DT, cotangent)


def _build_dopri_trajectory_args(n: int, rng: np.random.Generator) -> tuple[Any, ...]:
    return (
        _phases(n, rng),
        rng.uniform(-0.5, 0.5, size=n),
        _symmetric_coupling(n, rng),
        _TRAJECTORY_DT * _TRAJECTORY_STEPS,
        1e-6,
        1e-9,
        100_000,
        0.9,
        0.2,
        5.0,
    )


def _build_inertial_trajectory_args(n: int, rng: np.random.Generator) -> tuple[Any, ...]:
    return (
        _phases(n, rng),
        rng.uniform(-0.5, 0.5, size=n),
        rng.uniform(-0.5, 0.5, size=n),
        _symmetric_coupling(n, rng),
        1.0,
        0.5,
        _TRAJECTORY_DT,
        _TRAJECTORY_STEPS,
    )


_DELAY_STEPS = 5


def _build_delayed_trajectory_args(n: int, rng: np.random.Generator) -> tuple[Any, ...]:
    base = _phases(n, rng)
    history = base[np.newaxis, :] + rng.normal(0.0, 0.1, size=(_DELAY_STEPS + 1, n))
    return (
        np.ascontiguousarray(history, dtype=np.float64),
        rng.uniform(-0.5, 0.5, size=n),
        _symmetric_coupling(n, rng),
        _DELAY_STEPS * _TRAJECTORY_DT,
        _TRAJECTORY_DT,
        _TRAJECTORY_STEPS,
    )


@dataclass(frozen=True)
class PrimitiveSpec:
    """How to generate inputs and classify the cost of one compute primitive."""

    operation: str
    build: ArgsBuilder
    cost: str


# The registry-name → (input builder, cost class) table. Every name here must be
# a registered dispatcher; the runner asserts the two sets agree so a newly
# registered primitive cannot silently escape the benchmark.
_SPECS: tuple[PrimitiveSpec, ...] = (
    # Scalar / vector observables over a phase vector.
    PrimitiveSpec("order_parameter", _build_phases, "scalar"),
    PrimitiveSpec("order_parameter_gradient", _build_phases, "scalar"),
    PrimitiveSpec("order_parameter_hessian", _build_phases, "matrix"),
    PrimitiveSpec("mean_phase", _build_phases, "scalar"),
    PrimitiveSpec("mean_phase_gradient", _build_phases, "scalar"),
    PrimitiveSpec("mean_phase_hessian", _build_phases, "matrix"),
    # Daido harmonic observables (mode index m).
    PrimitiveSpec("daido_order_parameter", _build_phases_mode, "scalar"),
    PrimitiveSpec("daido_order_parameter_gradient", _build_phases_mode, "scalar"),
    PrimitiveSpec("daido_order_parameter_hessian", _build_phases_mode, "matrix"),
    PrimitiveSpec("daido_mode_phase", _build_phases_mode, "scalar"),
    PrimitiveSpec("daido_mode_phase_gradient", _build_phases_mode, "scalar"),
    PrimitiveSpec("daido_mode_phase_hessian", _build_phases_mode, "matrix"),
    # All-to-all mean-field forces and Jacobians (scalar coupling strength K).
    PrimitiveSpec("mean_field_force", _build_phases_scalar, "scalar"),
    PrimitiveSpec("mean_field_jacobian", _build_phases_scalar, "matrix"),
    PrimitiveSpec("triadic_mean_field_force", _build_phases_scalar, "scalar"),
    PrimitiveSpec("triadic_mean_field_jacobian", _build_phases_scalar, "matrix"),
    # Networked forces / Jacobians and interaction energy over a coupling matrix.
    PrimitiveSpec("networked_kuramoto_force", _build_phases_coupling, "matrix"),
    PrimitiveSpec("networked_kuramoto_jacobian", _build_phases_coupling, "matrix"),
    PrimitiveSpec("kuramoto_interaction_energy", _build_phases_coupling, "matrix"),
    PrimitiveSpec("kuramoto_interaction_energy_gradient", _build_phases_coupling, "matrix"),
    PrimitiveSpec("kuramoto_interaction_energy_hessian", _build_phases_coupling, "matrix"),
    # Daido mean-field (scalar coupling + mode index).
    PrimitiveSpec("daido_mean_field_force", _build_phases_scalar_mode, "scalar"),
    PrimitiveSpec("daido_mean_field_jacobian", _build_phases_scalar_mode, "matrix"),
    # Sakaguchi frustrated coupling — mean-field (scalar K) and networked (matrix K).
    PrimitiveSpec("sakaguchi_mean_field_force", _build_phases_scalar_frustration, "scalar"),
    PrimitiveSpec("sakaguchi_mean_field_jacobian", _build_phases_scalar_frustration, "matrix"),
    PrimitiveSpec("sakaguchi_force", _build_phases_coupling_frustration, "matrix"),
    PrimitiveSpec("sakaguchi_jacobian", _build_phases_coupling_frustration, "matrix"),
    # Local neighbourhood observables over an adjacency matrix.
    PrimitiveSpec("local_order_parameter", _build_phases_adjacency, "matrix"),
    PrimitiveSpec("local_order_parameter_jacobian", _build_phases_adjacency, "matrix"),
    PrimitiveSpec("local_mean_phase", _build_phases_adjacency, "matrix"),
    PrimitiveSpec("local_mean_phase_jacobian", _build_phases_adjacency, "matrix"),
    # Differentiable integrators and their reverse-mode adjoints.
    PrimitiveSpec("kuramoto_euler_trajectory", _build_trajectory_args, "trajectory"),
    PrimitiveSpec("kuramoto_rk4_trajectory", _build_trajectory_args, "trajectory"),
    PrimitiveSpec("kuramoto_euler_vjp", _build_euler_vjp_args, "trajectory"),
    PrimitiveSpec("kuramoto_rk4_vjp", _build_rk4_vjp_args, "trajectory"),
    # Adaptive Dormand–Prince and inertial (second-order) forward trajectories.
    PrimitiveSpec("kuramoto_dopri_trajectory", _build_dopri_trajectory_args, "trajectory"),
    PrimitiveSpec("networked_inertial_trajectory", _build_inertial_trajectory_args, "trajectory"),
    PrimitiveSpec(
        "networked_symplectic_inertial_trajectory",
        _build_inertial_trajectory_args,
        "trajectory",
    ),
    PrimitiveSpec("networked_delayed_trajectory", _build_delayed_trajectory_args, "trajectory"),
)


# ---------------------------------------------------------------------------
# Parity
# ---------------------------------------------------------------------------


def _flatten_numeric(value: Any) -> list[NDArray[np.float64]]:
    """Flatten a tier return (array, tuple of arrays, or scalar) to arrays."""

    if isinstance(value, (tuple, list)):
        flattened: list[NDArray[np.float64]] = []
        for item in value:
            flattened.extend(_flatten_numeric(item))
        return flattened
    return [np.atleast_1d(np.asarray(value, dtype=np.float64))]


def _max_abs_diff(reference: Any, candidate: Any) -> float | None:
    """Return the largest absolute deviation of ``candidate`` from ``reference``.

    Returns ``None`` when the two cannot be aligned (mismatched structure or a
    non-numeric return) rather than fabricating a parity number.
    """

    ref_parts = _flatten_numeric(reference)
    cand_parts = _flatten_numeric(candidate)
    if len(ref_parts) != len(cand_parts):
        return None
    worst = 0.0
    for ref, cand in zip(ref_parts, cand_parts, strict=True):
        if ref.shape != cand.shape:
            return None
        worst = max(worst, float(np.max(np.abs(ref - cand))))
    return worst


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------


def _selected_tiers(
    chain: Sequence[tuple[str, Callable[..., Any]]], wanted: set[str]
) -> list[str]:
    """Return chain tier names, preserving chain order, filtered to ``wanted``."""

    return [name for name, _ in chain if name in wanted]


def benchmark_primitive(
    spec: PrimitiveSpec,
    size: int,
    *,
    tiers: set[str],
    seed: int,
    warmup: int,
    repeats: int,
) -> PrimitiveResult:
    """Measure every requested tier of one primitive at one problem size.

    Each tier is fed an identical, seeded input. The Python floor result is the
    parity reference; every other measured tier records its largest absolute
    deviation from it. A tier that raises, or whose binding is absent, is
    recorded as an unavailable row with the exception text — never silently
    dropped.
    """

    chain = _dispatcher.registered_dispatchers()[spec.operation].chain
    rng = np.random.default_rng(seed)
    args = spec.build(size, rng)
    inner = _INNER_BY_COST[spec.cost]

    reference = dict(chain)["python"](*args)
    rows: list[BackendRow] = []
    worst_parity: float | None = None

    for tier_name, impl in chain:
        if tier_name not in tiers:
            rows.append(unavailable_row(tier_name, "tier excluded by --tiers"))
            continue
        try:
            output = impl(*args)
        except Exception as exc:  # noqa: BLE001 — a failing tier is recorded, not raised.
            rows.append(unavailable_row(tier_name, f"{type(exc).__name__}: {exc}"))
            continue
        if tier_name != "python":
            diff = _max_abs_diff(reference, output)
            if diff is not None:
                worst_parity = diff if worst_parity is None else max(worst_parity, diff)

        def _call(impl: Callable[..., Any] = impl, args: tuple[Any, ...] = args) -> object:
            return impl(*args)

        stats = measure(_call, warmup=warmup, repeats=repeats, inner=inner)
        rows.append(measured_row(tier_name, stats))

    return PrimitiveResult(
        operation=spec.operation,
        size=size,
        rows=tuple(rows),
        parity_max_abs_diff=worst_parity,
        extra={"cost_class": spec.cost, "inner": inner},
    )


def _tier_availability(tiers: set[str]) -> dict[str, str]:
    """Summarise which requested tiers are usable in this process."""

    summary: dict[str, str] = {}
    available = set(_dispatcher.available_tiers())
    for tier in sorted(tiers):
        if tier == "python" or tier in available:
            summary[tier] = "available"
        else:
            summary[tier] = "unavailable: tier probe returned false"
    return summary


def run_suite(
    *,
    tiers: set[str],
    sizes: Sequence[int],
    seed: int,
    warmup: int,
    repeats: int,
) -> list[PrimitiveResult]:
    """Measure every spec primitive across every requested size."""

    registered = set(_dispatcher.registered_dispatchers())
    spec_names = {spec.operation for spec in _SPECS}
    missing = registered - spec_names
    if missing:
        raise RuntimeError(
            f"registered dispatchers without a benchmark spec: {sorted(missing)}",
        )

    results: list[PrimitiveResult] = []
    for spec in _SPECS:
        for size in sizes:
            result = benchmark_primitive(
                spec, size, tiers=tiers, seed=seed, warmup=warmup, repeats=repeats
            )
            results.append(result)
            fastest = result.fastest_backend()
            print(
                f"  {spec.operation:<38s} N={size:<6d} fastest={fastest or 'n/a':<7s} "
                f"parity={result.parity_max_abs_diff}",
                file=sys.stderr,
                flush=True,
            )
    return results


def _write_artefacts(
    *,
    environment: str,
    output_dir: Path,
    provenance: Provenance,
    parameters: dict[str, Any],
    results: list[PrimitiveResult],
    tiers: set[str],
    generated_utc: str,
) -> tuple[Path, Path]:
    """Write the per-environment artefact and the manifest; return their paths."""

    output_dir.mkdir(parents=True, exist_ok=True)
    artifact = build_primitive_artifact(
        environment=environment,
        generated_utc=generated_utc,
        provenance=provenance,
        parameters=parameters,
        results=results,
    )
    manifest = build_manifest(
        generated_utc=generated_utc,
        provenance=provenance,
        parameters=parameters,
        results=results,
        tier_availability=_tier_availability(tiers),
    )
    artifact_path = output_dir / f"kuramoto_tiers.{environment}.json"
    manifest_path = output_dir / f"kuramoto_tiers_manifest.{environment}.json"
    artifact_path.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return artifact_path, manifest_path


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Consolidated Kuramoto tier benchmark")
    parser.add_argument("--environment", default="local", help="run label (local / ci)")
    parser.add_argument("--tiers", default="rust,julia,python", help="comma-separated tier subset")
    parser.add_argument(
        "--sizes",
        default=",".join(str(size) for size in DEFAULT_SIZES),
        help="comma-separated N values",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--pin-core", type=int, default=None, help="pin to one CPU core")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the consolidated tier benchmark and write the artefact + manifest."""

    args = _parse_args(list(sys.argv[1:] if argv is None else argv))
    tiers = {tier.strip() for tier in args.tiers.split(",") if tier.strip()}
    sizes = [int(size) for size in args.sizes.split(",") if size.strip()]

    if args.pin_core is not None and hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, {args.pin_core})

    generated_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    provenance = capture_provenance()
    parameters: dict[str, Any] = {
        "tiers": sorted(tiers),
        "sizes": sizes,
        "seed": args.seed,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "daido_mode": _DAIDO_MODE,
        "sakaguchi_frustration": _SAKAGUCHI_FRUSTRATION,
        "trajectory_dt": _TRAJECTORY_DT,
        "trajectory_steps": _TRAJECTORY_STEPS,
    }

    print(
        f"[bench] environment={args.environment} tiers={sorted(tiers)} cpu={provenance.cpu_model}",
        file=sys.stderr,
    )
    results = run_suite(
        tiers=tiers, sizes=sizes, seed=args.seed, warmup=args.warmup, repeats=args.repeats
    )
    artifact_path, manifest_path = _write_artefacts(
        environment=args.environment,
        output_dir=args.output_dir,
        provenance=provenance,
        parameters=parameters,
        results=results,
        tiers=tiers,
        generated_utc=generated_utc,
    )
    print(f"[bench] artefact -> {artifact_path}", file=sys.stderr)
    print(f"[bench] manifest -> {manifest_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
