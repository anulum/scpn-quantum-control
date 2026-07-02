# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Kuramoto competitive benchmark

# Kuramoto Competitive Benchmark

This page documents the external competitive harness that measures our Kuramoto
toolkit against real third-party solvers on one deterministic Kuramoto forward
problem. It is the cross-package counterpart to the in-repository
[Kuramoto Tier Benchmark](tier_benchmarks.md): the tier benchmark sweeps our own
Rust/Julia/Python backends against each other across problem sizes, while this
harness sets our toolkit against independent external libraries. So that the
comparison against the externals reflects the accelerated kernel rather than the
NumPy floor, the fixed-step RK4 row here also runs each installed tier explicitly
— the Rust `scpn_quantum_engine` kernel and the Python floor — recording the true
executing language and the Rust-over-floor speedup alongside the external solvers.

The harness lives in
`scpn_quantum_control.benchmarks.kuramoto_competitive_benchmark`; the runner is
`scripts/bench_kuramoto_competitive.py`; the committed artefact is
`docs/benchmarks/kuramoto_competitive.json`.

## What is compared

All solvers integrate the identical networked Kuramoto field

$$
\dot\theta_i = \omega_i + \sum_j K_{ij}\sin(\theta_j - \theta_i)
$$

on one deterministic, seed-built problem, and report the final order parameter
`r`, its absolute error against the high-precision reference, and the wall-clock
time. Our fixed-step RK4 appears once per installed tier so the languages are
compared side by side.

| Method | Family | Backend | Status |
|---|---|---|---|
| `ours_rk4_rust` | ours | `kuramoto_rk4_trajectory` Rust `scpn_quantum_engine` kernel | live when the engine tier is built, else fail-closed |
| `ours_rk4_python` | ours | `kuramoto_rk4_trajectory` NumPy Python floor | live |
| `ours_dopri` | ours | `kuramoto_dopri_trajectory` adaptive DOPRI5 (pure Python, no accelerated tier) | live |
| `scipy_solve_ivp` | external | SciPy `solve_ivp(RK45)`, tight tolerances | live (reference) |
| `julia_diffeq` | external | Julia `DifferentialEquations.jl(Tsit5)` | live |
| `networkdynamics_jl` | external | NetworkDynamics.jl | declared target |
| `dynamicalsystems_jl` | external | DynamicalSystems.jl | declared target |
| `scimlsensitivity_jl` | external | SciMLSensitivity.jl (differentiable competitor) | declared target |
| `jitcdde` | external | jitcdde (just-in-time C) | declared target |

The reference is the SciPy high-precision run (`rtol=1e-10`, `atol=1e-12`) when
SciPy is installed, otherwise our adaptive DOPRI5. The `ours_rk4_rust` row fails
closed to its `maturin develop --release` build command when the Rust engine
tier is not built, exactly as the external targets fail closed to their install
commands; the metadata also records which tier the RK4 facade dispatches to by
default (`dispatched_rk4_tier`).

## Fail-closed contract

Every external competitor is probed before it is run. A solver that is not
installed — or whose subprocess errors or times out — yields an
`available=False` row that carries the documented install command and the
reason, never a fabricated number. The four declared targets above are recorded
as such on a host without them, so the artefact is complete and reproducible
everywhere and the rows flip to live once the package is added.

## Measured comparison (committed artefact)

The committed artefact was generated on the development workstation (11th-gen
Intel Core i5-11600K, Rust engine `scpn_quantum_engine` 0.2.0). Correctness
errors are host-independent; the timings were captured on a **non-isolated,
loaded host** (`powersave` governor, load average ≈ 36), so per the claim
boundary below the absolute milliseconds are functional and reproducibility
evidence, **not** an isolated performance claim. Our in-process rows report the
median (P50) of 15 timed repeats after 3 warm-ups; the SciPy row is its single
tight-tolerance solve and the Julia row its second (warm) in-Julia solve. Under
load the absolute times are inflated, so the reproducible timing quantity is the
within-toolkit **Rust-over-Python-floor ratio** (both tiers measured back to
back under the same contention), not the raw milliseconds; for clean absolute
numbers, re-run on a quiesced, core-reserved host.

Problem: `n = 12`, `t_max = 6.0`, `dt = 0.01`, seed `20260628`; reference
`scipy_solve_ivp`.

| Method | Available | Language | `r_final` | Error vs reference | Time (ms) |
|---|---|---|--:|--:|--:|
| `ours_rk4_rust` | yes | rust | 0.77562419 | 2.96e-11 | 7.708 |
| `ours_rk4_python` | yes | python | 0.77562419 | 2.96e-11 | 70.429 |
| `ours_dopri` | yes | python | 0.77562558 | 1.39e-06 | 5.474 |
| `scipy_solve_ivp` | yes | python | 0.77562419 | — (reference) | 35.618 |
| `julia_diffeq` | yes | julia | 0.77562419 | 1.76e-09 | 26.358 |
| `networkdynamics_jl` | no | julia | — | — | — |
| `dynamicalsystems_jl` | no | julia | — | — | — |
| `scimlsensitivity_jl` | no | julia | — | — | — |
| `jitcdde` | no | python | — | — | — |

Reading: the RK4 facade dispatches to the **Rust** tier by default
(`dispatched_rk4_tier = "rust"`), and the two forced RK4 tiers return a bit-for-bit
identical trajectory (`rk4_rust_python_parity_max_abs_diff ≈ 9e-16`), so the Rust
kernel is an exact substitute for the NumPy floor. On this problem the Rust tier
ran the fixed-step RK4 about **9.1×** faster than the Python floor
(`rk4_rust_speedup_vs_python_floor`), which — being a within-toolkit ratio
measured under the same load — is the reproducible timing statement. Our RK4
tiers, our DOPRI5, SciPy `solve_ivp`, and Julia `DifferentialEquations.jl` also
agree on the final order parameter to between `3e-11` and `2e-9`: four
independent implementations corroborate the result, the primary host-independent
claim. Absolute cross-solver ordering must still be re-measured on an isolated
host before it is quoted as a performance result.

## Reproduce

```bash
python scripts/bench_kuramoto_competitive.py            # default n=12 problem
python scripts/bench_kuramoto_competitive.py --n 32 --t-max 10
```

To bring a declared target live, install it (the artefact prints the exact
command, e.g. `julia -e 'using Pkg; Pkg.add("NetworkDynamics")'`) and re-run.

## Claim boundary

- The order-parameter values, their cross-implementation agreement, and the
  bit-for-bit Rust-vs-Python-floor RK4 parity are the reproducible,
  host-independent quantities.
- Timings are functional and reproducibility evidence on the recorded host, not
  a production-latency, SLA, or universal-hardware claim. Competitor package
  versions, the Rust engine build, `rustc`, and numerical tolerances are
  recorded in the artefact. The within-toolkit Rust-over-Python-floor ratio is
  more robust to host load than the absolute milliseconds, which stay inflated
  until the run is repeated on a quiesced host.
- The verdict reports honestly where a competitor is faster than our toolkit;
  absent competitors do not contribute to it.
