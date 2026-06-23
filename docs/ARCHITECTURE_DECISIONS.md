# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Architecture Decision Records

# Architecture Decision Records

This page records the major design decisions behind `scpn-quantum-control`: for
each one, the context that forced a choice, the decision taken, why it was taken,
the alternatives weighed, and the trade-offs accepted. It complements
[Architecture](architecture.md), which documents the resulting structure (module
boundaries, dependency graph, pipeline); this page documents the *reasoning*.

Each record is intentionally short and standalone. Records are append-only: when a
decision is superseded, a new record is added and the old one is marked superseded
rather than rewritten.

---

## ADR-0001 — Module boundaries by single responsibility, not line count

**Status:** Active.

**Context.** The package is large (see the generated capability inventory in the
README) and grew quickly. A line-count rule (for example "split at 300 lines")
is easy to enforce but produces artificial seams: a single connected,
mutually-coupled responsibility gets cut into fragments joined by import cycles,
while two unrelated responsibilities can hide under the threshold in one file.

**Decision.** A module is split when, and only when, it holds two or more
*independent* responsibility clusters. The test is structural: decompose the
module's internal call graph into connected components, then remove the
most-referenced shared symbols and confirm the remainder is still one cluster
rather than independent groups joined by a single helper. Line-count thresholds
are a prompt to ask "is this still one thing?", never an automatic limit. See
[Architecture → Module size and single-responsibility policy](architecture.md).

**Rationale.** Cohesion and the absence of import cycles are the properties that
actually keep a large codebase reviewable; line count is only a weak proxy. The
`compiler` package was decomposed on exactly this basis (a former `mlir` module
that mixed seven independent concerns became a thin facade plus cohesive leaves),
which is the worked precedent.

**Alternatives considered.** (a) Hard line-count caps — rejected: they force
artificial seams and import cycles. (b) No policy — rejected: multi-responsibility
files accrete silently.

**Trade-offs.** Some single-responsibility modules are deliberately large; a
reader must trust the policy rather than a number. The policy is enforced by
review judgement, not a lint rule, so it depends on discipline.

---

## ADR-0002 — Multi-language acceleration, measured-fastest-first, Python floor

**Status:** Active.

**Context.** Several compute kernels (Hamiltonian construction, order-parameter
and Daido observables and their derivatives, classical Kuramoto integration) are
hot enough to benefit from a compiled backend, but the package must remain
installable and correct with pure Python only.

**Decision.** A compute function may carry one or more acceleration backends
(Rust, Julia, Go, Mojo) and dispatches to the *measured fastest* available one,
with pure Python as the always-present final fallback. The rule is codified in
`agentic-shared/feedback_multi_language_accel.md` and implemented in the
`accel` package (`accel/__init__.py`, `accel/dispatcher.py`). Today Rust (the
`scpn_quantum_engine` PyO3 wheel at the repo root) and Julia (`accel/julia`, via
`juliacall`) are shipped; Go and Mojo are tracked as future tiers and not shipped
until a kernel needs them.

**Rationale.** Ordering by *measured* speed (not by assumed language ranking)
keeps the fast path honest: a backend that benchmarks slower than NumPy is a
signal the algorithm is wrong, not that the language is slow (this is the same
principle as the native-speedup benchmark, see
[Native Speedup Benchmark](native_speedup_benchmark.md)). The Python floor keeps
the package usable with no optional toolchains.

**Alternatives considered.** (a) Single compiled backend (Rust only) — rejected:
forecloses Julia/Go/Mojo where they measure faster for a given kernel. (b)
Fixed language priority order — rejected: contradicts measured reality and hides
slow compiled paths. (c) Pure Python — rejected for hot kernels on performance.

**Trade-offs.** A polyglot fallback chain is more surface to test and benchmark
(each backend needs parity and measured benchmarks); the first Julia call carries
a one-off JIT cost (~20 s) that amortises over the process. Bit-exact parity
across backends is required and must be maintained per kernel.

---

## ADR-0003 — Kuramoto→XY gate-model mapping and publication-faithful solvers

**Status:** Active.

**Context.** The repository turns coupled-oscillator models (a coupling matrix
`K_nm` and natural frequencies `omega`) into quantum-control experiments. The
mapping and every evolution algorithm must be reproducible against the cited
physics, not an approximation chosen for convenience.

**Decision.** The standard mapping is Kuramoto→XY:
`K[i,j]·sin(θ_j − θ_i)` ↔ `−J_ij·(X_iX_j + Y_iY_j)` and `omega_i` ↔ `−h_i·Z_i`
(`bridge/knm_hamiltonian.py`), with an XXZ extension carrying the `ZZ` anisotropy
for the full-Heisenberg formulation (Kouchekian & Teodorescu 2025,
arXiv:2601.00113). Time evolution uses Qiskit `LieTrotter`/`SuzukiTrotter`
synthesis (`phase/xy_kuramoto.py`). Every model matches its cited publication
exactly — the "no simplifications" rule: if the paper uses a specific driving
force or current set, that exact form is implemented.

**Rationale.** Simplified models cannot reproduce published results and carry no
scientific value; faithful mappings are what make hardware/simulator evidence
defensible. The `K_nm` parameterisation follows Paper 27 (Šotek), with that
provenance disclosed in the README limitations.

**Alternatives considered.** (a) Linearised driving force `(V − E)` instead of
the published nonlinearity — rejected: changes the mathematical formulation. (b)
A single hard-coded solver — rejected: different campaigns need exact vs Trotter
vs variational paths.

**Trade-offs.** Faithfulness costs implementation effort and more code than a
toy model. Solver ownership is split across the SCPN ecosystem: the canonical
physics-solver laboratory is `scpn-fusion-core` and the control-grade integration
laboratory is `scpn-control`; this repository is the quantum-native path (see
[Architecture → Cross-Repository Integration](architecture.md) and
`agentic-shared/BROADCAST_2026-05-31_scpn_cross_repo_solver_ownership.md`).

---

## ADR-0004 — Evidence governance: a claim taxonomy with falsification culture

**Status:** Active.

**Context.** Quantum hardware results are noisy and easy to over-claim. In a
hype-prone field, an experiment workbench that cannot distinguish a theorem from
a simulator run from a mitigated hardware run from a falsification is not
trustworthy.

**Decision.** Every promoted result is classified into one of five claim classes
— Theory, Simulator, Hardware-unmitigated, Hardware-mitigated, and
Falsification-or-noise-limited — each requiring a named evidence form (derivation,
script/seed/versions, raw counts/job IDs, mitigation provenance). The public
index is `docs/hardware_status_ledger.md`; raw counts, job IDs, reproduction
scripts, manifests, and a verifier (`scripts/verify_hardware_result_packs.py`)
back it. Negative results are promoted, not buried — the SCPN/FIM digital
`lambda=4` protection hypothesis is published as a falsification
(`docs/campaigns/scpn_fim_claim_boundary_2026-05-05.md`).

**Rationale.** A claim taxonomy plus published falsifications is the project's
strongest differentiator and the precondition for any external citation or
hardware claim. It also constrains benchmark wording: environment-dependent
numbers are marked `production_claim_allowed: false` (see ADR-0002 and the native
benchmark page) rather than published as performance claims.

**Alternatives considered.** (a) Headline metrics without provenance — rejected:
unverifiable and corrosive to trust. (b) Suppressing negative results — rejected:
falsifications constrain claims and are first-class evidence.

**Trade-offs.** Every promotion carries an evidence and bookkeeping cost, and the
ledger must be maintained as campaigns evolve. The discipline slows
announcements; that is the intended cost.

---

## ADR-0005 — A small stable core contract versus churning internals

**Status:** Active.

**Context.** Research internals (the `phase`, `analysis`, `compiler`,
differentiable-programming surfaces) are under active refactor and must be free
to move, but compilers, backend adapters, benchmark harnesses, and hardware
result-pack replay need a durable shape to depend on.

**Decision.** The durable contracts live in a small, intentionally minimal
`stable_core` module that defines the shapes higher-level workflows share without
depending on low-level module layout; volatile research internals stay private,
experimental, or workbench-only. A v1.0 API-stability gate (tracked in the
internal backlog) inventories the public facades and requires a
compatibility/deprecation plan before broad refactors touch public imports.
Internal-only material (TODO, audits, evidence detail) lives under
`docs/internal/` and is excluded from the public surface.

**Rationale.** Decoupling the durable contract from the churning layout lets the
research surface evolve without breaking external users, and concentrates
stability guarantees in one reviewable place.

**Alternatives considered.** (a) Treat the whole package as public/stable —
rejected: would freeze active research surfaces prematurely. (b) No stability
contract — rejected: external integrations would break on every refactor.

**Trade-offs.** Contributors must know which surface is durable; the boundary is
maintained by convention plus the v1.0 gate rather than enforced automatically
today. Until v1.0 the package advertises `Development Status :: 4 - Beta` and
APIs may evolve.

---

## ADR-0006 — Fail-closed resource budgets before allocation

**Status:** Active.

**Context.** The compiler accepts arbitrary `K_nm`/`omega`. A pathological qubit
count `n` can exhaust memory before any quantum work runs: a dense Hilbert
operator is `2^n × 2^n`, a sparse Pauli operator builds `O(n^2)` terms, and a
statevector simulation stores `2^n` amplitudes.

**Decision.** Each path that can blow up estimates its resource demand and fails
closed with a clear, typed error before allocating, against a configurable budget
(host-memory fraction with a cap, overridable by environment variable or call
argument). The dense path uses `dense_budget` (`require_dense_allocation`,
`SCPN_MAX_DENSE_GIB`); the sparse Pauli-operator path uses `compile_budget`
(`require_pauli_operator_budget`, `SCPN_MAX_PAULI_GIB`); the local statevector
path uses `max_statevector_gib`. Input validation (square/finite/symmetric/
real-dtype `K_nm`, matching `omega`) runs at problem construction
(`kuramoto_core.KuramotoProblem`).

**Rationale.** Arbitrary-input robustness is a denial-of-service concern, not just
ergonomics; a bounded, typed failure is far better than an out-of-memory kill mid
run. Estimating from `n` alone (worst case) keeps the guard deterministic and
independent of the coupling values.

**Alternatives considered.** (a) No guards, rely on the OS — rejected: an OOM
kill loses the process and gives no actionable error. (b) Hard fixed caps —
rejected: legitimate large runs on big hosts would be blocked; budgets are
host-relative and overridable.

**Trade-offs.** Guards add a small estimate step and a configuration surface
(environment variables), and the budget defaults must be generous enough not to
reject legitimate work while still catching pathological `n`.
