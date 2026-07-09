# Kuramoto JAX tier

The networked-Kuramoto production integrators dispatch Rust → Julia → NumPy. This page documents the
opt-in **JAX** tier: fixed-step Euler, RK4, adaptive Dormand-Prince, networked inertial RK4,
networked symplectic inertial, and seeded noisy Euler-Maruyama trajectories expressed in JAX and run
on whatever accelerator JAX selected (a CUDA GPU when one is present).

The tier is **opt-in**: `accel.jax_kuramoto.jax_kuramoto_rk4_trajectory` /
`jax_kuramoto_rk4_gradient` own the RK4 autodiff surface, while
`accel.jax_kuramoto_integrators` owns `jax_kuramoto_euler_trajectory`,
`jax_kuramoto_dopri_trajectory`, `jax_networked_inertial_trajectory`,
`jax_networked_symplectic_inertial_trajectory`, and `jax_networked_noisy_trajectory`. All are
re-exported through the `kuramoto` facade, but they are **not** members of the default dispatch
chain. The default integrators still serve the Rust → Julia → NumPy chain, so existing behaviour is
unchanged.

## Verified parity

* **RK4 forward faithfulness (reproducible).** With 64-bit precision enabled, the JAX RK4 forward
  matches the production Rust integrator to machine precision. The committed benchmark
  (`docs/benchmarks/kuramoto_jax_tier.json`) records `parity_max_abs_diff = 8.88e-16` for a
  64-oscillator, 50-step run on the selected CUDA device. Parity is asserted under a tolerance because
  GPU reduction ordering need not equal NumPy's.
* **Integrator breadth parity (reproducible).** The same artefact records a 12-oscillator breadth
  cohort for Euler, Dormand-Prince, inertial RK4, symplectic inertial, and noisy trajectories. The
  maximum recorded differences are at machine precision: Euler `5.55e-17`, DOPRI terminal
  `2.78e-17`, inertial velocity `1.11e-16`, symplectic velocity `4.34e-18`, and noisy order parameter
  `5.55e-17`.
* **Gradient faithfulness (reproducible).** The autodiff gradient — `∂L/∂θ₀`, `∂L/∂ω`, `∂L/∂K` from
  `jax.vjp` of the forward solve — matches the hand-derived `kuramoto_rk4_vjp` to machine precision
  (~1e-15) across networks of 6 to 64 oscillators (`tests/test_jax_kuramoto.py`). The autodiff tier
  therefore verifies the hand-written adjoint and supplies the same gradient for objectives whose
  adjoint would be laborious to derive by hand.

## Batched ensembles (vmap)

`jax_kuramoto_rk4_ensemble` and `jax_kuramoto_rk4_ensemble_gradient` solve — and differentiate — a
whole batch of `B` initial conditions in a **single** accelerator call, by `jax.vmap` over the batch
axis. This is a vectorisation of the *entire* solve, not just the inner force evaluation, and the
NumPy and Rust tiers cannot express it: they would loop over the ensemble one member at a time.

The reproducible guarantee is that batching changes nothing but the layout — each batched member is
identical to its single-initial-condition `jax_kuramoto_rk4_trajectory` /
`jax_kuramoto_rk4_gradient`, and the batched gradient matches the per-member single gradient to
machine precision (`tests/test_jax_kuramoto.py`). The committed artefact records the ensemble parity
alongside advisory batched-versus-sequential timings (`ensemble_forward_us` versus
`sequential_forward_us`); the throughput factor is advisory and host/GPU-bound, the parity is the
reproducible quantity. This is the capability that makes the tier useful for Monte-Carlo basin studies
and machine-learning pipelines that evaluate many initial conditions or parameters at once.

## Wall clock (host- and GPU-dependent, boundary-guarded — not a claim)

The benchmark also records advisory per-call timings. On the recorded host (an 11th Gen Intel Core
i5-11600K with a CUDA GPU), the refreshed artefact measured the 64-oscillator, 50-step RK4 problem
with three samples. These milliseconds are **excluded from any performance claim**
(`production_claim_allowed: false`): they depend on the host, the governor, the GPU model and clock,
and JIT warm-up. A clean absolute number needs a quiesced, reserved host with a fixed GPU clock. The
reproducible quantities are the parity rows, not the milliseconds.

## Requirements and precision

The tier requires the optional `[jax]` extra (`pip install scpn-quantum-control[jax]`,
`jax[cuda12]` for a GPU). JAX is imported lazily behind a guard, so the pure-NumPy/Rust core stays
importable without it; calling the tier without JAX raises `ImportError` with the install hint — the
exception the dispatcher treats as "fall through", so a later slice may place this tier in a
size-aware accelerated chain. 64-bit precision (`jax_enable_x64`) is a global JAX flag the tier sets
lazily before tracing; it enables float64 support without forcing other JAX code off float32.

## Reproduce

```bash
python scripts/bench_kuramoto_jax_tier.py --n 64 --n-steps 50 --batch 8 --warmup 1 --repeats 3 --parity-n 12
```

This writes `docs/benchmarks/kuramoto_jax_tier.json` with RK4 parity, breadth parity for the new
integrator surface, advisory timings, the JAX version and device, and full host provenance. In
continuous integration JAX is CPU-only (`jax[cpu]`), where the same 64-bit parity holds within
tolerance; the GPU path is exercised locally.

## Related

* [Kuramoto Tier Benchmark](tier_benchmarks.md) — the Rust / Julia / Python tier provenance for the
  default dispatch.
* [Kuramoto Handbook](kuramoto_handbook.md) — the facade the tier's forward and gradient are
  re-exported through.
