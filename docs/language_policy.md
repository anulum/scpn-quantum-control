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

---

## Decision tree: how to pick a language for a new module

Use this tree every time a new module lands. The goal is to make the
language decision defensible in the audit table above, not to pick
the shiniest option.

```
Is the module performing numerical compute or structured parsing?
├── No → Python stays. Skip the rest of this tree and go to the
│        "Exempt declaration" section below.
└── Yes
    ├── Does it run inside a single Python process?
    │   ├── Yes
    │   │   ├── Is the hot loop iterator-bound (tight Python for-loop)?
    │   │   │   └── Rust via PyO3. Default choice.
    │   │   ├── Is the hot loop BLAS/LAPACK heavy and would benefit
    │   │   │   from auto-diff or symbolic manipulation?
    │   │   │   └── Julia via juliacall. Cross-validation bonus.
    │   │   ├── Is it an ML inference or GPU kernel?
    │   │   │   └── Mojo via its Python compat layer. Tracked.
    │   │   └── Is it CPU-bound NumPy code with no per-element work?
    │   │       └── Usually NumPy already dispatches to BLAS; no
    │   │           new tier needed. Revisit if a benchmark shows
    │   │           otherwise.
    │   └── No — it is a cross-process service
    │       ├── Fan-out over many concurrent I/O streams?
    │       │   └── Go with a CGo shim, or a sidecar Go binary
    │       │       that the Python main talks to over HTTP/gRPC.
    │       └── Single-request latency-sensitive?
    │           └── Rust again; Tokio plus a PyO3 handle into the
    │               in-process variant.
```

### Worked examples from this repo

| Module | Hot-loop shape | Chosen tier | Why |
|---|---|---|---|
| `build_knm_paper27` | N² nested loop over integer indices | Rust | tight integer-loop archetype; no BLAS escape |
| `kuramoto_trajectory` | Euler-step loop, ~10 k steps × N oscillators | Rust | same — PyO3 beats numpy for N ≤ 32 |
| `order_parameter` | sum of cis(θ_k) over N values | Rust (measured), Julia provisional | see §"Multi-language accel chain" in `docs/pipeline_performance.md` |
| `_xy_hamiltonian_pl` (PennyLane) | Pauli string assembly + coefficient lookup | Python only | Pennylane's own kernel is already compiled; re-wrapping is pure overhead |
| DLA commutator closure | dense matrix product over 2^(2N-1) basis | Rust today; Julia + GPU candidates for large N | `feedback_rust_everything.md` covers exactly this |
| `capture_provenance` | dict assembly + git subprocess | Python only | I/O adapter; no compute |

## Tier-specific guidance

### Rust

* Use `maturin` for building. Pin it with a `[build-system]`
  `requires` entry in `pyproject.toml`; otherwise contributors get
  silent version drift that breaks wheel reproducibility.
* PyO3 calls across a Python ↔ Rust boundary cost ~200–500 ns each.
  For loops below that granularity, prefer a single Rust call that
  *contains* the loop over N Python → Rust calls each doing tiny
  work. The benchmark in `scripts/bench_order_parameter_tiers.py`
  makes this visible.
* Feature-gate every optional dependency with a Cargo feature so
  the wheel stays minimal. Example from `scpn_quantum_engine`:
  `parallel` feature gates `rayon`; disabling it produces a
  single-threaded wheel half the size.
* Every public Rust function must have both a Python-side unit test
  and a Rust-side `#[test]` so regressions are caught regardless of
  which side breaks first.

### Julia (juliacall)

* First call incurs a ~20 s JIT cost. Production scripts that call
  a Julia-tier function exactly once pay the tax without amortising
  it; they should either warm Julia explicitly at startup or fall
  through to another tier. The dispatcher's `last_tier_used`
  surfaces this decision to observability.
* `juliacall` re-exports the running Julia session; **never**
  import `juliacall.Main` inside a library module at import time,
  because it boots Julia at library-import time. Use the lazy-load
  pattern from `src/scpn_quantum_control/accel/julia/__init__.py`:
  `_load()` is called at first use only.
* NumPy arrays crossing into Julia are zero-copy when they are
  `float64`, `complex128` and C-contiguous. Always call
  `np.ascontiguousarray(arr, dtype=np.float64)` before the crossing
  to avoid accidental copies that wipe the Julia-tier win.
* Julia is the right home for scientific cross-validation (Yao.jl,
  QuantumOptics.jl, DifferentialEquations.jl). When you add a Julia
  tier, consider whether the corresponding scientific package
  reference (for example Yao's `order_parameter` equivalent) can
  double as a cross-validation oracle. If yes, add that test under
  `tests/test_cross_validation_*` the same day.

### Go

* Integration pattern is a sidecar process plus gRPC or HTTP, not
  cgo. cgo closes the door on goroutines from the Python side and
  defeats most of the reason to pick Go.
* Target modules: the Phase-2 multi-account IBM submission path, a
  webhook receiver for Zenodo notifications, and the commercial
  Polar.sh webhook router. None are yet wired.
* A Go tier implicitly adds a second deployment surface (the
  binary). Every Go module MUST ship with a Dockerfile and a
  `docker-compose` entry for reproducibility.
* Per `feedback_multi_language_accel.md` §"Fallback chain ordering",
  a Go tier is only at the top of the chain for a workload class it
  actually wins; for in-process compute, Rust always wins, so Go
  lives in a different pool (distinct from the `order_parameter`
  dispatcher).

### Mojo

* As of 2026-04-17, Mojo's Python-compat layer is maturing but not
  yet wheel-deployable from PyPI. A Mojo tier that ships in this
  repo requires a Modular SDK install on the reader's side; until
  that changes, Mojo code stays under `accel/mojo/` as opt-in.
* Mojo is particularly strong at MLIR-compiled AI inference and GPU
  kernels. The candidate compute surfaces are the DLA-closure dense
  product at N ≥ 16 and any dense-matrix exponential used in
  `classical_exact_evolution`.
* Benchmark gating is the same as every other tier: a Mojo path
  lands only with a measured wall-time comparison in
  `docs/pipeline_performance.md`.

### Python floor

* `numpy` + `scipy` + plain loops. Never the fastest, always
  available, always correct.
* Used for:
  * Correctness oracles in tests (the `_python_order_parameter`
    implementation is the ground truth all other tiers match
    within `1e-10`).
  * The final fallback in every dispatcher chain.
  * Prototypes that haven't yet earned a compiled-tier port.
* Never use a Python floor as the only path for a module that the
  rule marks as compute. Prototype in Python, then port.

## Exempt declaration

A module without numeric compute — orchestration, config, logging,
registry, I/O adapter — is **exempt** from the compiled-path rule.
The exemption is not automatic; it must be declared in the module
header immediately after the SPDX block:

```python
# Language policy: EXEMPT from the Rust-path rule. <one-sentence reason>
# See docs/language_policy.md §"Current-state audit".
```

Auditors grep for `Language policy:` to reach every exempt module
at once. If that grep turns up a module that has numeric compute,
the exemption is wrong and must be revoked by adding a compiled
tier.

### Common exemption categories

* **Pure dict / list / callable registry** — `hardware/backends.py`.
  Python's dict is already C-level; a PyO3 `HashMap<String, fn>`
  wrapper would be slower because of the per-lookup boundary cost.
* **pydantic-settings config** — `config.py`. pydantic-core is
  already Rust. Wrapping it again adds FFI cost with zero new
  compute.
* **structlog bootstrap** — `logging_setup.py`. structlog composes
  strings and tuples from Python primitives; no numeric surface.
* **asyncio orchestration** — `hardware/async_runner.py`. The
  compute is inside the IBM Runtime client and IBM's cloud; the
  module is a thin fan-out.
* **I/O adapters** — files under `hardware/` that wrap a vendor
  Python client (Qiskit, PennyLane, Braket). These are bridges, not
  computers.

### Common cases that look exempt but are not

* **Regex parsers over gigabytes of log** — feels like I/O, actually
  compute. Needs a Rust tier (`regex` crate).
* **String-mangling** inside a benchmark inner loop — Python's
  string ops are C-level but the interpreter overhead per call
  isn't. If it's on a hot path, Rust wins.
* **dataclass validators** — `bridge/phase_artifact.py`'s
  `__post_init__` checks are called per-record on a hot ingestion
  path. Today they are Python-only because the per-record cost is
  tolerable, but they are on the watchlist for future Rust
  migration if a downstream ingestion workflow saturates on them.

## Dispatcher design rules

Every multi-tier compute function must expose a single dispatcher
that follows these invariants. Failure to satisfy any of them is a
review block.

1. **The chain terminates in a Python floor.** The Python path must
   be the last entry of the ordered list. `MultiLangDispatcher`
   enforces this in its constructor.
2. **Fallthrough only on structural errors.** The tier is allowed to
   raise `ImportError`, `ModuleNotFoundError` or `RuntimeError`;
   every other exception bubbles up. A `ValueError` is a bug in the
   tier, not a reason to silently move to the next one.
3. **No hidden global state between tiers.** A dispatcher call must
   not mutate a module-global that a later call depends on. The
   `last_tier` observation is the explicit exception — it is
   read-only to callers.
4. **Benchmarks measured on the runner class used in production.**
   Not a developer laptop. Not the SSH server. The GitHub Actions
   `ubuntu-latest` runner (or the mining rig when that becomes the
   production target) is the only valid measurement host.
5. **Measurement JSON is committed.** Every reorder of the chain
   requires updating `docs/benchmarks/<fn>.json` so the decision can
   be audited months later.

## Version policy per tier

| Tier | Pinning rule | Update cadence |
|---|---|---|
| Rust | `Cargo.lock` committed; `maturin` pinned via `[build-system]`; MSRV documented | Monthly, alongside Dependabot PRs |
| Julia | `juliacall>=0.9,<2` in `pyproject.toml [julia]`; Julia itself on `julialang-jll` auto-install | Quarterly — juliacall breaks less often than Julia itself |
| Go | Sidecar `go.mod` pinned to minor; go toolchain via `go.mod`'s `go 1.XX` directive | When a Go tier ships |
| Mojo | SDK version pinned once it stabilises | When a Mojo tier ships |
| Python | `pyproject.toml` `dependencies` SemVer-pinned | Monthly, Dependabot |

## Testing requirements per tier

Every compute function with a compiled-tier path must ship three
classes of test:

1. **Floor correctness.** The Python implementation passes a
   Hypothesis-generated grid of inputs within the physical-
   invariant bounds of the function (e.g. `R ∈ [0, 1]` for
   `order_parameter`).
2. **Cross-tier agreement.** Every tier produces the same output as
   the Python floor to within `1e-10` on the same inputs.
   `tests/test_accel_dispatch.py::TestCrossTierAgreement` is the
   template.
3. **Dispatcher mechanics.** The registered chain falls through
   correctly on `ImportError`, preserves `last_tier`, doesn't
   swallow unrelated exceptions. Same test file.

Every **fuzz** test must use `hypothesis` with `max_examples >= 30`
and `deadline=None` (some tiers boot slowly on cold imports);
`HealthCheck.too_slow` is acceptable to suppress with a documented
reason.

## Cross-platform notes

* **macOS arm64** — juliacall ships for arm64; maturin wheels
  need an explicit `[tool.maturin] target = ...` for
  cross-compilation. Rust tier MUST build; Julia tier MAY build.
  The test matrix in `ci.yml` probes both.
* **Windows** — juliacall has known path-separator issues in CI;
  use `Path(...).as_posix()` when passing paths into Julia. Rust
  tier via `maturin` works out of the box.
* **Linux x86_64** — the primary target; every tier must build.
* **Linux aarch64** — Rust builds via cross-compilation; Julia
  builds via juliacall's bundled Julia binary; Go builds natively.

## Security considerations

Adding a new compiled tier is adding a new attack surface:

* Every Rust crate passes through `cargo audit` in the CI
  `security` job. A high/critical advisory on a transitive dep
  blocks the release.
* Julia's `JuliaPkg.toml` (generated by juliacall) must be
  committed. It is the Julia equivalent of a lock file; without it
  two machines could silently install different Julia packages.
* Go binaries must be built reproducibly from a pinned toolchain;
  `go.mod` + `go.sum` both committed.
* Mojo SDK downloads are not yet reproducible — defer a Mojo tier
  until Modular publishes a SemVer-compatible toolchain.

## When this document changes

Update this document (and re-run the audit table) in the same
commit that adds, removes, or re-orders a tier. A language-policy
PR that changes the wording without the accompanying code change is
a documentation-only PR; a code PR that changes a tier without
updating this document is a review block.

---

## Registering a new compute function with the dispatcher

The pattern is specific — follow it exactly. A registration that
deviates breaks the invariants that the tests in
`tests/test_accel_dispatch.py` enforce.

1. **Implement the Python floor first.** The function lives under
   `src/scpn_quantum_control/accel/dispatcher.py` as a module-level
   helper prefixed `_python_<name>`. It must accept the same
   signature every other tier will accept and return a
   reproducibility-checkable scalar or array.

   ```python
   def _python_<name>(theta: np.ndarray) -> float:
       ...
   ```

2. **Add a tier probe if the tier is conditional.** The existing
   probes for Rust and Julia return `True` if the underlying
   library imports cleanly *without* booting a runtime. Expensive
   probes (Julia runtime boot, GPU context creation) go behind
   `is_available()` inside the tier module, not in the probe.

3. **Implement the compiled tier(s).** Each tier gets a helper
   `_<tier>_<name>(theta)` that calls into the compiled
   implementation. Keep these helpers thin — unwrap arguments,
   call the underlying function, re-wrap return. Any business
   logic belongs inside the compiled code.

4. **Measure all tiers on representative N.** Run
   `scripts/bench_<name>_tiers.py` (pattern after
   `bench_order_parameter_tiers.py`) across the production input
   sizes. Commit the JSON artefact to `docs/benchmarks/`.

5. **Assemble the chain in measured order.** Python floor MUST be
   last. Unmeasured tiers go below measured ones.

   ```python
   _<NAME>_CHAIN: list[tuple[str, Callable]] = [
       ("rust", _rust_<name>),
       ("julia", _julia_<name>),
       ("python", _python_<name>),
   ]
   ```

6. **Register the dispatcher.** Add to `_REGISTRY`:

   ```python
   _<name>_dispatcher = MultiLangDispatcher(_<NAME>_CHAIN)
   _REGISTRY["<name>"] = _<name>_dispatcher

   def <name>(theta):
       return _<name>_dispatcher(theta)
   ```

   and add to `__all__` in `accel/__init__.py`.

7. **Test.** `tests/test_accel_<name>.py` copies the three
   test-class pattern from the template:
   `TestDispatcherMechanics`, `TestCrossTierAgreement`,
   `TestPipelineAccel`. A Hypothesis grid of at least 30 examples
   exercises tier agreement to `1e-10`.

8. **Document.** Add the row to this file's "Current-state audit"
   table and to `docs/pipeline_performance.md §"Multi-language
   accel chain"`. The benchmark JSON is the authoritative source
   for both.

9. **Wire.** Replace every call to the legacy Python-only helper
   with a call to the new dispatcher. `_order_param` in
   `hardware/classical.py` is the template for a minimally
   invasive wiring with a bare-import fallback.

## Migration pattern: Python → compiled tier

When a Python function crosses into "hot enough to need a compiled
tier" territory, follow this migration pattern rather than a
one-shot rewrite.

### Stage 1 — benchmark the existing Python

Write the micro-benchmark *before* the port, save the numbers.
This gives you the "before" column for the speed-up claim. Without
this baseline, a future reader cannot verify the port actually
moved a metric.

### Stage 2 — port the function to the chosen tier

Keep the Python version intact. The compiled port lives in the
appropriate `accel/<tier>/` subdirectory or, for Rust, inside
`scpn_quantum_engine`.

### Stage 3 — add a dispatcher

Per the registration pattern above. The dispatcher dispatches
between the Python floor (existing) and the new compiled tier.

### Stage 4 — wire the dispatcher into the original call site

Replace the direct Python call with the dispatcher call. Keep the
Python implementation in the dispatcher chain as the floor; do not
delete it.

### Stage 5 — re-benchmark the wired path

End-to-end run against the "before" number from stage 1. Only now
may the commit message claim a speed-up.

### Stage 6 — keep the Python floor indefinitely

The Python floor is the correctness oracle. Deleting it removes the
tier-agreement test. Even if the Python implementation is 100×
slower, it stays — the 100× is the price of correctness triangulation.

## Debugging multi-tier code

### Symptom — "the fast path disagrees with the Python floor"

1. Is it a dtype issue? Julia sees `float32` when you meant
   `float64`? Check the first line of the tier wrapper.
2. Is it a qubit-ordering issue? Qiskit is big-endian, QuTiP is
   little-endian — the `_H_qutip` helper reverses before
   `tensor()`. Any new tier must pick one convention and stick.
3. Is it a JIT corruption? Restart the Python process; if the
   disagreement goes away, Julia's JIT cached a stale compilation
   — commonly a sign that `juliacall.Main.include` was called twice
   with two versions of the same file.

### Symptom — "Rust tier raises `TypeError: expected ndarray`"

PyO3 enforces exact dtype; `np.ascontiguousarray(arr, dtype=...)`
is cheap and should be called in the `_rust_<name>` wrapper
unconditionally.

### Symptom — "Julia tier times out in CI"

Julia's JIT cold start is ~20 s. CI caches should include the
Julia package cache under `~/.julia/` (or the `juliapkg` path on
macOS); without it, every CI run re-installs JIT dependencies and
the timeout fires.

### Symptom — "dispatcher never picks Julia even though juliacall is installed"

Check the chain order. Julia is second in the `order_parameter`
chain, so it serves requests only when Rust is unavailable. Force
Rust unavailable for a test by replacing its entry with an
ImportError-raising stub — see
`test_julia_full_dispatch_reaches_julia_when_rust_disabled` in
`tests/test_accel_dispatch.py` for the exact pattern.

## Deprecating a tier

When a tier stops being useful — Julia 2.x breaks juliacall, Mojo
is replaced by a native PyTorch 3.x accelerator, Go sidecar
rewritten as Rust — follow this deprecation flow:

1. Mark the tier's chain entry with a comment
   `# DEPRECATED <date>: <reason>` and leave it in place.
2. Announce in `DEPRECATIONS.md` with a target removal minor
   version (SemVer: deprecations remove on the next major bump,
   never within a minor).
3. After the grace window (two minor releases), remove the tier's
   implementation + probe + chain entry + its `[<tier>]`
   pyproject extra. Update the audit table.
4. Keep the benchmark JSON artefact under
   `docs/benchmarks/archived/` for historical comparison.

## Interop contracts across tiers

Every tier must honour these promises. Failure is a bug in the
tier, not a dispatcher problem.

1. **Input invariance.** A tier does not mutate input arrays.
   Rust's `ndarray` views are borrowed; Julia's `@view` is
   read-only. If a future tier needs a copy for correctness, it
   copies internally.
2. **Output shape.** Every tier returns the same dtype, same
   shape, same numeric type as the Python floor. A Julia
   `Complex{Float64}` return must be unboxed to Python `complex`
   before crossing the FFI.
3. **Exception surface.** Tiers raise `RuntimeError` for runtime
   failures (IBM job pool empty, GPU OOM, JIT compilation error)
   and `ValueError` for argument-shape mismatches. These are the
   only two exception classes the dispatcher interprets.
4. **Determinism.** Tiers that use randomness must accept an
   explicit seed argument. The dispatcher does not thread seeds;
   each tier implementation handles its own seeding.
5. **Reentrancy.** Tiers must be safe to call concurrently from
   multiple Python threads if their underlying runtime allows it
   (Julia's GIL-analog, Rust's `Sync` trait). Otherwise they must
   document the restriction in the tier module docstring.

## Quick reference: Do / Don't

**Do.**

* Benchmark every tier before committing the chain order.
* Keep the Python floor forever.
* Use `np.ascontiguousarray(arr, dtype=np.float64)` at FFI
  boundaries.
* Document exempt modules with the `# Language policy:` header
  line.
* Run `pytest --cov` on every new module; 95 % minimum, 100 %
  target.

**Don't.**

* Fabricate "measured fastest" claims. Measure or say "unmeasured".
* Delete the Python floor when a tier is fast enough.
* Chain a tier above Python without measuring it.
* Boot Julia / Mojo / GPU context at library import time.
* Accept a `ValueError` as a reason to fall through to the next
  tier; that is a tier bug.

## Appendix: glossary

* **Tier.** A language-specific implementation of a compute
  function (Rust, Julia, Go, Mojo, Python).
* **Chain.** An ordered list of tier implementations for a single
  compute function, with the Python floor last.
* **Dispatcher.** The runtime object that walks a chain until one
  tier succeeds; records `last_tier` on success.
* **Probe.** A cheap `bool`-returning callable that says whether
  a tier is usable in this process *without* booting its runtime.
* **Floor.** The Python implementation at the bottom of every
  chain; the correctness oracle.
* **Warm-up.** A one-off cost paid the first time a tier's runtime
  initialises (Julia JIT, GPU context, Mojo MLIR compile). Should
  be amortised over the process lifetime or explicitly warmed at
  startup.
* **FFI.** Foreign Function Interface — the crossing between
  Python and a compiled runtime (PyO3 for Rust, juliacall for
  Julia, ctypes/cgo for Go).

