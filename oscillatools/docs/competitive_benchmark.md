<!--
SPDX-License-Identifier: AGPL-3.0-or-later
Commercial license available
© Concepts 1996–2026 Miroslav Šotek. All rights reserved.
© Code 2020–2026 Miroslav Šotek. All rights reserved.
ORCID: 0009-0009-3560-0851
Contact: www.anulum.li | protoscience@anulum.li
-->

# Competitive Benchmark

This page documents how `oscillatools` is compared against independent
third-party solvers, and the boundary that comparison keeps. It is the
cross-package counterpart to the in-package
[Multi-language tier benchmark](tier_benchmarks.md): the tier benchmark sweeps the
package's own Rust, Julia, and Python backends against each other across problem
sizes, whereas the competitive comparison sets the toolkit against external
libraries on one deterministic forward problem.

## What is compared

Every solver integrates the identical networked Kuramoto field

$$
\dot\theta_i = \omega_i + \sum_j K_{ij}\sin(\theta_j - \theta_i)
$$

on one deterministic, seed-built problem and reports the final order parameter
`r`, its absolute error against a high-precision reference, and the wall-clock
time. The `oscillatools` fixed-step RK4 is measured once per installed tier — the
Rust `scpn-quantum-engine` kernel and the NumPy floor — so the executing language
and the Rust-over-floor ratio are recorded alongside the external solvers. The
adaptive DOPRI5 path is measured as the pure-Python reference-quality integrator.

Representative external comparators are the SciPy `solve_ivp(RK45)` reference, the
Julia `DifferentialEquations.jl`, `DynamicalSystems.jl`, `NetworkDynamics.jl`, and
`SciMLSensitivity.jl` stacks, and the JIT-compiled-C `jitcdde` integrator.

## Fail-closed contract

Every external comparator is fail-closed: a solver whose toolchain (Julia, a Julia
package, a C compiler, or a Python package) is not installed — or whose subprocess
errors or times out — yields an unavailable row that carries the documented
install command and the reason, never a fabricated number. The record is
therefore complete and reproducible on any host: installed comparators yield real
rows, absent ones honest unavailable rows that flip to live once the package is
added.

## Where the harness runs

The competitive harness that drives this comparison imports `oscillatools` as the
toolkit under test and currently lives with the parent distribution's benchmark
surface (`scpn_quantum_control.benchmarks.kuramoto_competitive_benchmark`), which
is where the external Julia and C toolchains are provisioned. It exercises the
public `oscillatools` facade only — the same RK4, DOPRI5, and order-parameter
entry points documented in the [handbook](handbook.md) — so its results describe
this package. A standalone runner co-located with `oscillatools` is a planned
addition; until then, the parent harness is the reference implementation and this
page states the methodology and claim boundary it keeps.

## Claim boundary

- The order-parameter values, their agreement across independent implementations
  and languages, and the bit-for-bit Rust-versus-Python-floor RK4 parity are the
  reproducible, host-independent quantities. Cross-implementation agreement on the
  final order parameter is the primary claim.
- Timings are functional and reproducibility evidence on the recorded host, not a
  production-latency, SLA, or universal-hardware claim. The within-toolkit
  Rust-over-Python-floor ratio, measured back to back under the same load, is more
  robust to host contention than the absolute milliseconds.
- No external superiority claim is made. A comparison reports honestly where an
  external solver is faster or more accurate than this toolkit; absent comparators
  do not contribute to any verdict.
- Competitor package versions, the Rust engine build, the compiler, and the
  numerical tolerances are recorded in the harness artefact so the comparison can
  be reproduced rather than taken on trust.
