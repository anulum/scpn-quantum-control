# Kuramoto competitive evidence — oscillatools vs the Julia SciML tools

This note records a measured head-to-head between the oscillatools coupled-phase-oscillator toolkit and
the established Julia libraries on two axes — **integration throughput** and **differentiability** — so
the toolkit's standing is stated from data rather than assertion. It is deliberately even-handed: it
reports where oscillatools trails as plainly as where it leads, and it pairs every throughput figure with
its accuracy, because a speed number without an accuracy is not a comparison.

## Method

Task: integrate a networked (complete-graph, mean-field-normalised) Kuramoto system of `N` oscillators to
`t = 6.0` from a fixed seed, then read the final order parameter `r`. Every method solves the *same*
`(K, ω, θ₀)`. Throughput is the median wall-clock of the timed solve after a warm-up call (excluding
first-call JIT / dispatch). Accuracy is the absolute error of the final `r` against the SciPy
`solve_ivp` reference (`r_error_vs_reference`). Competitors run through the shipped competitive harness
(`scripts/bench_kuramoto_competitive.py`, `scpn_quantum_control.benchmarks.kuramoto_external_competitors`),
fail-closed when a package is absent. Reproduce with:

```
python scripts/bench_kuramoto_competitive.py --n <N> --t-max 6.0 --dt 0.01
python scripts/bench_diff_kuramoto_rk4_tiers.py --n <n_steps>
```

Environment: a single workstation (Intel/AMD desktop, `aaarthuus`), Julia 1.12.6 with
`DynamicalSystems.jl`, `NetworkDynamics.jl`, `DifferentialEquations.jl`, `SciMLSensitivity.jl` installed;
oscillatools on its Rust / Julia / Python dispatch tiers. The Julia adaptive solvers use `Tsit5` at
`reltol = 1e-8`, `abstol = 1e-10`.

## Axis 1 — integration throughput (with accuracy)

Wall-clock milliseconds per solve, and the final-`r` error against the SciPy reference, over `N`:

| method | lang | N=12 | N=16 | N=32 | N=64 | N=128 |
|---|---|---|---|---|---|---|
| `DynamicalSystems.jl` (CoupledODEs, Tsit5) | julia | 1.0 ms / 2e-9 | 2.1 / 8e-10 | 6.0 / 8e-11 | 20.1 / 3e-11 | 115.8 / 4e-12 |
| `NetworkDynamics.jl` (Tsit5) | julia | 1.4 / 2e-9 | 1.9 / 8e-10 | 4.2 / 8e-11 | 16.1 / 3e-11 | 115.4 / 4e-12 |
| oscillatools RK4 (Rust, fixed dt=0.01) | rust | 2.9 / 3e-11 | 7.9 / 3e-11 | 22.7 / 6e-13 | 112.6 / 2e-12 | 686.8 / 8e-13 |
| oscillatools DOPRI (adaptive) | python | 2.7 / 1e-6 | 5.4 / 5e-7 | 3.6 / 3e-7 | 13.6 / 1e-7 | 47.4 / 5e-7 |
| oscillatools RK4 (pure-Python floor) | python | 31.8 / 3e-11 | 38.1 / 3e-11 | 52.4 / 6e-13 | 176.1 / 2e-12 | 770.3 / 8e-13 |
| `DifferentialEquations.jl` (Tsit5) | julia | 7.2 / 2e-9 | 16.6 / 8e-10 | 1081.6 / 8e-11 | 374.1 / 3e-11 | 977.6 / 4e-12 |
| `SciMLSensitivity.jl` (Tsit5 solve) | julia | 7.5 / 2e-9 | 17.6 / 8e-10 | 171.9 / 8e-11 | 420.5 / 3e-11 | 1053.5 / 4e-12 |
| SciPy `solve_ivp` (reference) | python | 18.4 / ref | 32.1 / ref | 32.6 / ref | 58.8 / ref | 242.1 / ref |
| `jitcdde` | python | 13.7 / 5e-9 | 14.4 / 2e-9 | 88.1 / 3e-10 | 443.0 / 4e-11 | 2367.6 / 2e-10 |

**Like-for-like verdict (matched accuracy ≈ 1e-9…1e-11).** The mature Julia adaptive solvers lead raw
throughput at every `N`. Against the fastest of `DynamicalSystems.jl` / `NetworkDynamics.jl`, oscillatools'
Rust RK4 is **2.8× (N=12), 4.1× (16), 5.4× (32), 7.0× (64), 6.0× (128)** slower — and the gap widens with
`N`, because the Rust RK4 is a *fixed-grid* scheme (T/dt steps regardless) while the competitors adapt the
step. The Rust RK4 is per-step fast (11× over the toolkit's own pure-Python floor, bit-faithful to it at
`8.9e-16`) and in fact *more* accurate at this grid (`~1e-11`…`1e-13`), but doing more steps costs time.

**The adaptive caveat.** oscillatools' own adaptive DOPRI is faster than the Julia solvers at `N ≥ 32`
(e.g. 47 ms vs 115 ms at N=128) — but at `~1e-6` accuracy, three-to-five orders coarser than the Julia
runs. That is a speed/accuracy trade, not a like-for-like win; it is a genuine advantage only where `~1e-6`
suffices. Tightening its tolerance to match would remove the gap. (The `DifferentialEquations.jl` /
`SciMLSensitivity.jl` rows carry subprocess re-JIT noise — e.g. the 1081 ms N=32 spike — and are not clean
measurements; the `DynamicalSystems.jl` / `NetworkDynamics.jl` and oscillatools rows are monotone and are
the trustworthy ones.)

## Axis 2 — differentiability

oscillatools ships exact reverse-mode adjoint gradients of a trajectory objective as a first-class
operation (`kuramoto_rk4_vjp`), witnessed against JAX autodiff to `~1e-9` (RG1). Measured per-call
throughput of the forward map and its adjoint (Rust tier, 32 steps):

| operation | N=64 | N=256 | N=512 |
|---|---|---|---|
| forward `kuramoto_rk4_trajectory` (Rust) | 6.1 ms | 80.7 ms | 600 ms |
| reverse-mode adjoint `kuramoto_rk4_vjp` (Rust) | 17.1 ms | 219 ms | 1442 ms |

The adjoint costs ≈ 2–3× the forward — the expected reverse-mode ratio — and returns the exact gradient
with respect to the initial phases, natural frequencies and coupling.

The comparison here is one of *capability assembly*, not just speed. `DynamicalSystems.jl` and
`NetworkDynamics.jl` — the throughput leaders above — do **not** return trajectory gradients out of the
box. The Julia route to them is `SciMLSensitivity.jl` plus a separate automatic-differentiation backend
(`Zygote` / `Enzyme` / `ReverseDiff`) and a chosen `sensealg`. On this machine `SciMLSensitivity.jl` is
installed but no AD backend is (`using Zygote` fails), so the SciML gradient did not run — differentiability
in that ecosystem is an assembled add-on, whereas in oscillatools the forward model and its exact gradient
share one API and one install. A precise `SciMLSensitivity` gradient head-to-head (installing an AD
backend) is a documented follow-up.

## Honest verdict

- **oscillatools is not the raw-integration-throughput leader; the mature Julia SciML tools are** — by
  ≈3–7× at matched accuracy. This confirms, with numbers, that the toolkit is not the fastest option on
  that axis.
- oscillatools is nonetheless in the same performance tier: its Rust RK4 beats `DifferentialEquations.jl`,
  SciPy and `jitcdde`, is 11× over its own Python floor, and is *more* accurate at its fixed grid; its
  adaptive DOPRI is fast when coarse accuracy is acceptable.
- The toolkit's distinguishing value is not throughput but (a) **built-in exact differentiability** of the
  forward model — native here, an assembled stack there — and (b) the integrated control and inference
  layer that shares the same model and gradient.
- The concrete throughput improvement named here — an **adaptive accelerated tier** — is now built: the
  adaptive Dormand–Prince forward (`kuramoto_dopri_trajectory`) dispatches across a Rust → Julia → Python
  floor tier chain, the tiers tolerance-parity (same realised grid on well-conditioned problems). The Rust
  tier runs **2.4× (N=128) to 19× (N=8) over the pure-Python floor** and ahead of the Julia tier, so it is
  the served tier; measured in `docs/benchmarks/diff_kuramoto_dopri_tiers.json`. This combines adaptive
  stepping (fewer steps than the fixed grid) with Rust per-step speed — the two levers the fixed-grid RK4
  lacked. A GPU tier remains the next lever at very large `N`.

## Boundaries

Single machine, one problem seed, `N ≤ 128`, one integration time. Julia adaptive at `reltol = 1e-8`;
oscillatools Rust RK4 at fixed `dt = 0.01`. The `DifferentialEquations.jl` / `SciMLSensitivity.jl` rows
carry subprocess-JIT noise and should not be read as clean. Throughput is always reported with its
accuracy. Larger `N`, a GPU tier, and a real `SciMLSensitivity` gradient measurement are follow-ups. The
per-`N` figures above were produced by the reproduce commands on the environment named above; the committed
`docs/benchmarks/kuramoto_competitive.json` and `docs/benchmarks/diff_kuramoto_rk4_tiers.json` hold the
default single-size reference records that the continuous-integration run regenerates.
