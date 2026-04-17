# Language Policy

This document codifies which programming language this repository uses
for which part of the stack, and under what criteria a module picks a
particular language. It exists so that every new module has a
defensible, written rationale for its language choice instead of
defaulting to "whatever the author reaches for first".

The repository-wide rule from `SHARED_CONTEXT.md` is stricter than the
three-line summary:

> A module is NOT done until (1) wired into pipeline, (2) multi-angle
> tests, (3) **Rust path**, (4) benchmarks measured, (5) performance
> documented, (6) elite docs, (7) all rules followed.

The word "Rust" in rule 3 is shorthand for "the appropriate
non-Python compiled language". An orchestration module with no compute
surface has no meaningful compiled-language path; forcing one is an
anti-pattern. This policy document is the explicit carve-out that rule
3 anticipates.

## Multi-language acceleration chain (CRITICAL)

Per the ecosystem-wide rule codified in
`feedback_multi_language_accel.md` (2026-04-17), every compute function
in every ANULUM project may have **one or more** acceleration backends
drawn from this palette:

| Tier | Language | Niche |
|---|---|---|
| 1 | **Rust** | Systems-tier, predictable performance, zero-cost abstractions, graph traversal, tight integer loops |
| 2 | **Julia** | Numerics-tier, BLAS/LAPACK, autodiff, dense matrix ops at scale, scientific computing |
| 3 | **Go** | Concurrency-tier, goroutines + channels for massively-concurrent I/O glue |
| 4 | **Mojo** | AI-tier, MLIR + Python compat, GPU/AI hot-paths, ML inference paths with autodiff |
| ∞ | **Python** | Correctness floor, reproducibility ground truth — ALWAYS the last fallback |

**Fallback chain ordering rule:** the *measured fastest* path runs
first; next-fastest runs on fallback; Python is last. The dispatcher
must measure (or know from prior measurement) which language wins on
a given module's hot loop and put that one at the top.

**Target directory layout** for future compute modules:

```
src/scpn_quantum_control/accel/
  rust/              # Rust extensions (exists via scpn_quantum_engine)
  julia/             # Julia bindings (PyJulia / JuliaCall)
  go/                # Go cgo or shared-lib FFI
  mojo/              # Mojo callable bindings
  dispatch.py        # central dispatcher, fastest-at-top
```

This repository currently ships only the Rust tier
(`scpn_quantum_engine`, 37 PyO3 functions). Julia / Go / Mojo tiers
are tracked as future work — see the per-module audit table below
for which compute functions are candidates.

## Language matrix

| Language | When | Examples in this repo |
|---|---|---|
| **Rust** | Compute hot paths invoked from Python via PyO3. Default for every new compute function. | `scpn_quantum_engine` — 37 PyO3 functions covering build_knm, order_parameter, commutator_dense, kuramoto_trajectory, Pauli expansion. Measured 4.7× – 69× over NumPy on benchmarked hot paths. |
| **C / C++** | Hardware-SDK interop below the Python wrapper, kernel-level SIMD, or when a vendor only ships a C / C++ client. | Not currently used; IBM Runtime exposes a Python client so C interop is not needed at this layer. Candidate for future pulse-level control latency budget. |
| **Go** | Standalone network services with heavy concurrent I/O, where a sidecar process is a cleaner boundary than in-process asyncio. | Not currently used. Candidate for a future IBM-job-pool daemon that multiplexes across accounts; in-process asyncio (see `hardware/async_runner.py`) covers the single-account case. |
| **Julia** | Scientific cross-validation as a third independent solver, especially when Yao.jl / QuantumOptics.jl have a published reference for the same problem class. | Not yet wired. Planned as a supplement to the QuTiP + Dynamiqs branches in `tests/test_cross_validation_qutip_dynamiqs.py` — tracked as a follow-up under audit item C7. |
| **CUDA / HIP / Metal** | GPU quantum simulators or dense-matrix kernels above 2^14 Hilbert dimension. | `hardware/gpu_accel.py` dispatches to CuPy; `scpn_quantum_engine` exposes Rust-side GPU kernels. |
| **JAX** | Auto-differentiable physics, XLA JIT compilation, batched parameter sweeps. | `hardware/jax_accel.py`. |
| **OpenQASM 3** | Portable circuit serialisation across vendor backends. | Qiskit export pipeline in `hardware/circuit_export.py`. |
| **TypeScript + WebGPU** | Browser-side visualisation and interactive demos. | Lives in a separate repo (`SCPN-CONTROL-infinity`); not present here. |

## Criteria for picking a compiled-language path

A module qualifies for a Rust path (or the appropriate compiled
alternative) if and only if **any** of these is true:

1. It performs numerical computation that dominates wall-clock on a
   realistic input size.
2. It parses structured input at a rate where the Python interpreter
   cost shows up in benchmarks.
3. It drops below a microbenchmark floor defined in
   `docs/pipeline_performance.md` in its Python implementation.

A module is **exempt** from the compiled-language rule when **all**
of these are true:

1. It performs no computation — dict lookups, attribute access,
   callable dispatch, subprocess spawning, asyncio orchestration,
   structured-log assembly, config parsing, docstring strings.
2. The best Python library for the domain is itself already compiled
   (pydantic-core is Rust, orjson is Rust, asyncio / dict / list are
   C-level) — reimplementing a wrapper around those in Rust would add
   PyO3 boundary cost without changing the effective compute path.
3. A compiled rewrite would worsen ergonomics without moving any
   measurable metric.

Each exempt module must state its exemption explicitly in its module
docstring, citing this policy.

## Current-state audit (2026-04-17)

Columns: `Rust` / `Julia` / `Go` / `Mojo` show which accel tier is
currently wired (✓) or tracked as a future tier (TBD); `—` means not
applicable for this module.

| Module | Compute? | Rust | Julia | Go | Mojo | Rationale / niche |
|---|---|---|---|---|---|---|
| `bridge/knm_hamiltonian.py` | Yes | ✓ | TBD | — | TBD | Dense matrix build; Julia LAPACK is a strong candidate for the Julia tier. |
| `hardware/classical.py` | Yes | ✓ (`kuramoto_trajectory`, `kuramoto_euler`) | TBD | — | TBD | Tight integer-scale integrator; Rust wins on small N, Mojo on GPU for large N. |
| `analysis/dynamical_lie_algebra.py` | Yes | ✓ (commutator + DLA closure) | TBD | — | — | Graph / symbolic loop; Rust is the natural primary, Julia secondary for spectral checks. |
| `hardware/backends.py` | **No** | **Exempt** | — | — | — | Plugin registry is a `dict[str, Callable]`. No compute surface. |
| `config.py` | **No** | **Exempt** | — | — | — | pydantic-settings → pydantic-core is already Rust; re-wrapping adds PyO3 cost. |
| `logging_setup.py` | **No** | **Exempt** | — | — | — | structlog composes records from Python primitives. No compute. |
| `hardware/async_runner.py` | **No** | **Exempt** | — | — | Go (future) | asyncio orchestration over IBM Python client. If a fan-out daemon pattern emerges, Go is the candidate tier. |

Every new module added to this repository must appear in this audit
table as either a compiled-path row or an explicit exempt row.

**Follow-up — when a `TBD` cell gets wired**, the commit must attach
a microbenchmark that compares wall-time against the incumbent tier,
and the dispatcher's fallback order must be updated accordingly.

## Cross-validation addenda

Beyond the compiled-path rule, scientific claims benefit from being
reproduced in an independently implemented framework:

* **QuTiP (Python + C)** — condensed-matter canonical reference.
  Already wired in `tests/test_cross_validation_qutip_dynamiqs.py`.
* **Dynamiqs (Python + JAX)** — JAX backend, auto-diff. Already wired.
* **Yao.jl / QuantumOptics.jl (Julia)** — third-party Julia stack,
  different numerics, different compiler. Will additionally double
  as the Julia-tier accel path per the multi-language rule. Tracked
  as a follow-up.
* **QSharp (.NET)** — Microsoft's stack, different LLVM toolchain.
  Low-priority follow-up.

Cross-validation does not replace the Python / Rust pair of this
repo; it triangulates scientific claims so that a bug in any single
stack cannot silently falsify a published number. With the
multi-language rule landed, an in-tree Julia implementation closes
both gaps at once (accel path + cross-validation) for the modules
that get a Julia tier wired in.
