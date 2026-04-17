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

| Module | Compute? | Compiled path | Rationale |
|---|---|---|---|
| `src/scpn_quantum_control/bridge/knm_hamiltonian.py` | Yes | Rust (`build_knm_paper27`, `knm_to_dense_matrix`) | Hot path, on every circuit build. |
| `src/scpn_quantum_control/hardware/classical.py` | Yes | Rust (`kuramoto_trajectory`, `kuramoto_euler`) | Hot path, Euler integrator. |
| `src/scpn_quantum_control/analysis/dynamical_lie_algebra.py` | Yes | Rust (commutator + DLA closure) | Hot path, exponential-dimension loop. |
| `src/scpn_quantum_control/hardware/backends.py` | **No** | **Exempt** | Plugin registry is a `dict[str, Callable]`. Python dict is already C-level; a PyO3 HashMap wrapper would be slower. See module docstring. |
| `src/scpn_quantum_control/config.py` | **No** | **Exempt** | Pydantic-settings parses env / `.env` / kwargs. pydantic-core is itself Rust; re-wrapping in Rust would add PyO3 cost without new compute. |
| `src/scpn_quantum_control/logging_setup.py` | **No** | **Exempt** | structlog assembles records from Python primitives and hands them to the stdlib logger. No numeric compute surface. |
| `src/scpn_quantum_control/hardware/async_runner.py` | **No** | **Exempt** | asyncio orchestration over the IBM Python client. I/O bridge per the `feedback_rustify_all.md` carve-out. |

Every new module added to this repository must appear in this audit
table as either a compiled-path row or an explicit exempt row.

## Cross-validation addenda

Beyond the compiled-path rule, scientific claims benefit from being
reproduced in an independently implemented framework:

* **QuTiP (Python + C)** — condensed-matter canonical reference.
  Already wired in `tests/test_cross_validation_qutip_dynamiqs.py`.
* **Dynamiqs (Python + JAX)** — JAX backend, auto-diff. Already wired.
* **Yao.jl / QuantumOptics.jl (Julia)** — third-party Julia stack,
  different numerics, different compiler. Tracked as a follow-up.
* **QSharp (.NET)** — Microsoft's stack, different LLVM toolchain.
  Low-priority follow-up.

Cross-validation does not replace the Python / Rust pair of this
repo; it triangulates scientific claims so that a bug in any single
stack cannot silently falsify a published number.
