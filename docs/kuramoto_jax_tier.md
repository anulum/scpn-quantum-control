# Kuramoto JAX autodiff tier

The networked-Kuramoto forward integrators dispatch Rust → Julia → NumPy, and each carries a
hand-written reverse-mode adjoint. This page documents a fourth kind of tier — the same RK4 solve
expressed in **JAX**, where the gradient comes from **automatic differentiation** rather than a
hand-derived scheme, and the whole solve runs on whatever accelerator JAX selected (a CUDA GPU when
one is present). It is Slice 1 of the 7.7 backend programme (one facade, two tiers: a JAX autodiff
tier and, later, a hand-tuned GPU-kernel tier).

The tier is **opt-in**: `accel.jax_kuramoto.jax_kuramoto_rk4_trajectory` and
`jax_kuramoto_rk4_gradient` are directly callable accelerated paths, re-exported through the
`kuramoto` facade, but they are **not** members of the default dispatch chain. The default
`kuramoto_rk4_trajectory` still serves the Rust tier, so every existing behaviour is unchanged.

## Two claims, verified

* **Forward faithfulness (reproducible).** With 64-bit precision enabled, the JAX RK4 forward matches
  the production Rust integrator to machine precision. The committed benchmark
  (`docs/benchmarks/kuramoto_jax_tier.json`) records `parity_max_abs_diff = 8.66e-15` for a 256-
  oscillator network over 200 steps on a CUDA GPU. Parity is asserted under a tolerance (not as
  bit-identity) because GPU reduction ordering need not equal NumPy's; here it lands at machine
  precision.
* **Gradient faithfulness (reproducible).** The autodiff gradient — `∂L/∂θ₀`, `∂L/∂ω`, `∂L/∂K` from
  `jax.vjp` of the forward solve — matches the hand-derived `kuramoto_rk4_vjp` to machine precision
  (~1e-15) across networks of 6 to 64 oscillators (`tests/test_jax_kuramoto.py`). The autodiff tier
  therefore both **verifies** the hand-written adjoint and supplies the same gradient for objectives
  whose adjoint would be laborious to derive by hand.

## Wall clock (host- and GPU-dependent, boundary-guarded — not a claim)

The benchmark also records advisory per-call timings. On the recorded host (an 11th Gen Intel Core
i5-11600K with a CUDA GPU) the JAX tier integrated the 256-oscillator, 200-step problem with a median
of about 29 ms against about 1082 ms for the production tier on the same host. These milliseconds are
**excluded from any performance claim** (`production_claim_allowed: false`): they depend on the host,
the governor, the GPU model and its clock, and the JIT warm-up. A clean absolute number needs a
quiesced, reserved host with a fixed GPU clock. The reproducible quantity is the parity, not the
milliseconds.

## Requirements and precision

The tier requires the optional `[jax]` extra (`pip install scpn-quantum-control[jax]`,
`jax[cuda12]` for a GPU). JAX is imported lazily behind a guard, so the pure-NumPy/Rust core stays
importable without it; calling the tier without JAX raises `ImportError` with the install hint — the
exception the dispatcher treats as "fall through", so a later slice may place this tier in a
size-aware accelerated chain. 64-bit precision (`jax_enable_x64`) is a global JAX flag the tier sets
lazily before tracing; it enables float64 support without forcing other JAX code off float32.

## Reproduce

```bash
python scripts/bench_kuramoto_jax_tier.py --n 256 --n-steps 200
```

This writes `docs/benchmarks/kuramoto_jax_tier.json` with the parity, the advisory timings, the JAX
version and device, and the full host provenance. In continuous integration JAX is CPU-only
(`jax[cpu]`), where the same 64-bit parity holds within tolerance; the GPU path is exercised locally.

## Related

* [Kuramoto Tier Benchmark](tier_benchmarks.md) — the Rust / Julia / Python tier provenance for the
  default dispatch.
* [Kuramoto Handbook](kuramoto_handbook.md) — the facade the tier's forward and gradient are
  re-exported through.
