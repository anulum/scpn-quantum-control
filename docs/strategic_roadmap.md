# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Strategic Roadmap (post-v1.0 differentiation)

# Strategic Roadmap — Post-v1.0 Differentiation

Seven differentiation items scoped beyond the feature list in
`ROADMAP.md`. Each item is a multi-week-to-multi-month track with
explicit prerequisites, deliverables, and risks. None is scheduled
yet; this document is the bill of work so future sessions can
prioritise, scope, and sequence without re-deriving the case.

**Status:** all seven are **DEFERRED / CEO-gated**. No execution
until each item is individually activated.

**Relationship to `ROADMAP.md`:** that file tracks release-scoped
work (v0.9.x → v1.0). This file tracks post-v1.0 differentiation
work that pushes the project into novel territory rather than
closing gaps.

## Priority matrix

Ordering below is by expected research/adoption impact per effort
unit. Ordering is not fixed — re-evaluate each quarter as the
scientific landscape moves.

| # | Track | Impact | Effort | Dependencies |
|---|-------|--------|--------|--------------|
| S1 | Hybrid classical–quantum feedback loop | high | 6–10 weeks | runner refactor, latency budget, IBM Dynamic Circuits |
| S2 | Quantum advantage benchmarks at scale | medium-high | 4–6 weeks | IBM credits, MPS/GPU baselines |
| S3 | ML-augmented pulse / ansatz design | medium-high | 8–12 weeks | JAX tier, DLA-closure data |
| S4 | Multi-hardware + pulse-level control | medium | 6–8 weeks | PennyLane adapter, OpenPulse docs |
| S5 | Open-data + classical validation harness | medium | 3–5 weeks | Zenodo DOI, QuTiP/Dynamiqs already wired |
| S6 | Decoupled `quantum-kuramoto` subpackage | medium | 2–3 weeks | import-graph audit |
| S7 | Fault-tolerant / logical-level extension | strategic | 12+ weeks | surface code + DLA theory pass |

---

## S1 — Hybrid classical–quantum feedback loop

### Motivation

Current pipeline is **batch** (submit → wait → analyse). A closed
loop where a classical sync observer feeds back into pulse shaping
on the next shot / circuit unlocks a class of adaptive experiments
that are not currently covered by any polished open-source
quantum-Kuramoto tool. Adaptive measurement protocols (Bayesian
phase estimation, reinforcement-learning pulse shaping, active
decoherence tracking) all require sub-second feedback.

### Deliverables

1. `hardware/feedback_loop.py` — classical observer API plus a
   `FeedbackRunner(scheduler, observer)` orchestrator that
   submits one circuit, receives counts, feeds them into the
   observer callback, updates the circuit parameters, and submits
   the next shot.
2. IBM Dynamic Circuits integration — leverage mid-circuit
   measurement + conditional gates available on Heron r2+ for
   intra-shot feedback within a single circuit, not only
   across-shot.
3. Reference classical observer: Kalman-filtered Kuramoto
   sync-R estimator written in Rust (hot-loop code) and exposed
   through the multi-language dispatcher.
4. Two end-to-end demos:
   - Cross-shot closed-loop: 20-shot Bayesian refinement of the
     K_ij coupling estimate on a small network.
   - Intra-shot Dynamic-Circuits demo: mid-circuit parity
     measurement gates the subsequent RZZ angle.
5. Benchmark: wall-time budget per feedback cycle, amortised over
   1 000 shots. Document in `docs/pipeline_performance.md`.

### Risks

* IBM Runtime session-mode latency is bounded below by ~200 ms
  round-trip even on Session. Real-time sub-kHz feedback is
  infeasible without hardware-side primitives.
* Dynamic Circuits API surface is still evolving; pin
  `qiskit-ibm-runtime` version in the feedback module's own
  `importlib.metadata` check.

### Prerequisites

* Runner refactor to support Session + callback protocol.
* Explicit latency budget ≤ 1 s per feedback cycle documented and
  benchmarked.
* IBM credits allocation or paid Runtime window (this burns
  session minutes fast).

### Acceptance

A 20-step Bayesian-refinement demo completes end-to-end in ≤ 5
minutes on `ibm_kingston` (or equivalent Heron r2+) with the
feedback loop demonstrably converging to a ground-truth K_ij
within 10 %. Notebook + raw data archived to Zenodo.

---

## S2 — Quantum advantage benchmarks at scale

### Motivation

The project currently demonstrates scientific fidelity (DLA parity,
BKT, CHSH) but does not publish an explicit scaling curve showing
where classical ODE / MPS / sparse methods break down and the
quantum approach wins. That scaling study is the single most
compelling evidence for a post-NISQ audience and is also the
foundation for the Phase 2 IBM credits justification.

### Deliverables

1. Sweep script `scripts/bench_advantage_scaling.py` that builds
   the same K_nm physics at N = 4, 6, 8, 10, 12, 14, 16, 18, 20
   and measures wall-time + memory for each of:
   - Classical exact diagonalisation (numpy `eigh`).
   - Classical Trotter via dense matrix exponential.
   - Classical sparse MPS (quimb already wired via `[tensor]`).
   - GPU-accelerated paths where available.
   - Quantum Trotter + ZNE on `ibm_kingston` (noisy) and
     `AerSimulator` (ideal).
2. Explicit crossover table: memory breakpoint, wall-time
   breakpoint, noisy vs. ideal gap as a function of N.
3. Figure set: log-log scaling plots plus a combined panel
   figure for publication.
4. Short paper section + preprint integration — the scaling
   curve is the headline plot for the expanded `paper/` draft.
5. `tests/test_advantage_scaling_regression.py` — a tight
   regression test that catches unintended classical-speedup
   regressions in future refactors.

### Risks

* Phase 2 IBM credits are pending; without them, the hardware
  column caps at ≤ 4 qubits on the free tier.
* Some classical baselines (quimb, JAX) have distinct memory
  models that make "memory footprint at N" comparisons unfair
  without careful instrumentation.

### Prerequisites

* IBM Credits decision (currently pending since 2026-03-29).
* `quimb` + `dynamiqs` + `jax` already wired (done).
* MPS memory profiler + GPU memory profiler harness.

### Acceptance

Published scaling figure with every column populated up to N = 20
(with the hardware column graceful-degrading to its accessible
subset). Raw data archived to Zenodo DOI. Figure + table embedded
in the paper.

---

## S3 — ML-augmented pulse / ansatz design

### Motivation

Hand-crafted Kuramoto-XY ansätze (`structured_ansatz.py`) and
hand-chosen pulse shapes (`pulse_shaping.py`: PMP-ICI, hypergeometric)
leave performance on the table for networks with non-trivial
topology. Reinforcement learning + differentiable programming can
discover better mappings for large networks, exploiting the DLA
closure structure as an implicit symmetry prior. The project
already has the JAX tier wired — the differentiable-programming
substrate is in place.

### Deliverables

1. `ml/ansatz_rl.py` — PPO-style RL agent that takes a
   K_nm coupling matrix as state and proposes ansatz parameter
   schedules. Reward: VQE ground-state fidelity on the classical
   reference.
2. `ml/pulse_diff.py` — JAX-based differentiable pulse shaper
   that trains a parameterised pulse envelope (hypergeometric
   family is the natural candidate; JAX autodiff through the
   Rust hypergeometric path via a custom VJP).
3. DLA-symmetry prior — bake the `build_xy_generators` output
   into the RL policy as a mask over allowed operator classes;
   agent chooses coefficients, not operators.
4. Benchmark vs. hand-designed baselines on the N=4 DLA-parity
   task and on a new 6-oscillator "mixed-parity" benchmark.
5. Reproducibility harness: saved weights + training logs +
   one-command replay script.
6. `docs/ml_ansatz_design.md` — theory note explaining the
   RL formulation, the DLA mask, and the differentiable-pulse
   backward pass.

### Risks

* RL on quantum circuits has a well-known sample-efficiency
  problem. If classical-reference reward takes > 1 s per
  candidate, training on realistic N quickly becomes
  compute-bound.
* "Learned ansatz beats hand-crafted" is a non-trivial claim that
  needs falsification criteria (track under `docs/falsification.md`).

### Prerequisites

* JAX tier already installed (`[jax]` extra exists).
* `stable-baselines3` or `cleanrl` added as a new
  `[rl]` extra.
* GPU access (mining rig or ML350; not laptop CPU).

### Acceptance

On the N=6 mixed-parity benchmark, the learned ansatz achieves
≥ 5 % lower VQE energy error than `structured_ansatz` on ≥ 80 %
of 50 randomised K_nm instances. Signed reproducibility run on
the mining rig, logs + weights archived.

---

## S4 — Multi-hardware backend + pulse-level control

### Motivation

The PennyLane adapter advertises IBM / IonQ / Rigetti / Quantinuum /
Braket / Cirq vendor strings and a 67-test cross-vendor mock suite
exists (commit `c3dcbf6`), but no real hardware run has been made
against a non-IBM backend. Pulse-level control on IBM is
similarly under-exploited: the Rust hypergeometric pulse engine
would compose cleanly with OpenPulse for custom DRAG / Gaussian
envelopes.

### Deliverables

1. Two non-IBM hardware runs, mock → real promotion:
   - IonQ Aria (trapped-ion, PennyLane `ionq.qpu` route).
   - Rigetti Aspen-M-3 or Quantinuum H1-1 (superconducting or
     trapped-ion, TBD by quota).
2. OpenPulse integration on IBM — plug the Rust
   `ici_three_level_evolution_batch` output into a
   `qiskit.pulse.Schedule`; measure a Rabi-equivalent
   on-hardware calibration of a custom hypergeometric pulse.
3. `hardware/pulse_compiler.py` — thin compiler from the
   existing `pulse_shaping.py` numeric envelope to a vendor's
   pulse format (OpenPulse for IBM, native sequence for IonQ
   via PennyLane).
4. Cross-vendor replication of the Phase 1 DLA-parity result at
   N = 4 on at least one non-IBM backend. Fidelity degradation
   attribution separated per vendor.
5. `docs/multi_vendor_hardware.md` — operator manual: how to
   acquire credits, how to transpile, per-vendor gotchas
   (IonQ's all-to-all connectivity, Quantinuum's mid-circuit
   measurement, Braket's cost accounting).

### Risks

* Vendor costs are not free. IonQ QPU time is expensive;
  Quantinuum requires a commercial relationship. Without a
  research credit programme, hardware costs cap what runs.
* PennyLane adapter wraps Qiskit + vendor plugins — plugin
  version drift is a chronic risk (cf. the Pennylane adapter
  fix in v0.9.4).

### Prerequisites

* IonQ research credits application (currently no relationship).
* Quantinuum research credits application.
* The Rust hypergeometric → OpenPulse converter pass needs a
  local calibration against a known `ibm_kingston` backend
  before external submission.

### Acceptance

DLA parity asymmetry measured on two distinct hardware backends
at N = 4, qualitative direction agrees (both positive, both in
the 5–20 % band). Raw data archived to Zenodo.

---

## S5 — Open-data + classical validation harness

### Motivation

Phase 1 raw data lives in `data/phase1_dla_parity/*.json` and has
a Zenodo DOI (10.5281/zenodo.18821929). A community-usable
benchmark would ship: the raw data, the classical reference
solvers, a reproducer script, and a diff-check harness against the
published statistics — all in one `pip install`-able subpackage.
This positions the repo as a reference benchmark for "quantum
synchronisation dynamics on NISQ" the way
[QED-C](https://github.com/SRI-International/QC-App-Oriented-Benchmarks)
positions itself for application-oriented QC benchmarks.

### Deliverables

1. `scpn_quantum_control.benchmark_harness` subpackage:
   - `load_phase1_dataset()` — loads raw JSON, returns a typed
     pandas DataFrame plus metadata.
   - `reproduce_phase1_statistics()` — runs Welch + Fisher +
     readout-baseline correction, asserts published values within
     tolerance. Already partially in
     `tests/test_phase1_dla_parity_reproduces.py`; promote to
     public API.
   - `classical_baselines` — thin wrappers around QuTiP,
     Dynamiqs, MPS (quimb), and our own Python-floor
     dispatcher for apples-to-apples comparison.
2. `scripts/run_benchmark_suite.py` — one-command reproduction
   of every published claim (DLA parity, CHSH, BKT, OTOC, DLA
   dim).
3. `docs/benchmark_harness.md` — operator manual: how to install,
   how to run, what each output means, how to contribute new
   benchmarks.
4. Submission to [Papers With Code](https://paperswithcode.com/)
   and [Qiskit Ecosystem](https://www.qiskit.org/ecosystem/)
   after the harness lands.
5. Zenodo deposit refreshed with tagged benchmark snapshot.

### Risks

* "Community resource" only works if the community adopts it.
  Visibility campaign (launch copy drafts already in
  `.coordination/launch_copy/`) is a prerequisite.
* Reproducibility-at-a-distance is brittle — CI must exercise
  the harness on every release.

### Prerequisites

* B9 (Phase 1 reproducer) — done, commit `16b3f8e`.
* C7 (cross-validation vs QuTiP / Dynamiqs) — done, commit
  `82b51d9`.
* Zenodo GitHub integration re-enabled by CEO.

### Acceptance

A clean-room user with only `pip install
scpn-quantum-control[benchmark]` can reproduce every published
number within documented tolerance on the first run. CI gate
runs the harness on every release tag.

---

## S6 — Decoupled `quantum-kuramoto` subpackage

### Motivation

`scpn_quantum_control` carries a mix of core XY / Kuramoto
infrastructure and SCPN-specific research code (K_nm from Paper 27,
the 16-layer UPDE, SCPN mapping modules). A user who only wants
"quantum Kuramoto on superconducting hardware" has to take the
whole stack, which increases the adoption barrier. A lightweight
`quantum-kuramoto` subpackage — just `phase/`, `bridge/` minus
SCPN-specific parts, `hardware/` core, and the `accel` dispatcher —
would let that user opt in without the SCPN research baggage.

### Deliverables

1. `src/quantum_kuramoto/` sub-distribution — separate
   `pyproject.toml` inside the same monorepo publishing as a
   second PyPI package that depends only on the reusable parts
   of `scpn_quantum_control`.
2. Import-graph audit — measure exactly which modules depend
   on SCPN-specific symbols (`K_nm`, `OMEGA_N_16`,
   `build_knm_paper27`, SSGF, FIM) and mark them out of scope
   for the subpackage.
3. Re-export surface in `scpn_quantum_control` — keep existing
   users unchanged; `scpn_quantum_control.phase` continues to
   work.
4. Separate README for `quantum-kuramoto` that targets the
   general quantum-simulation audience (not SCPN-specific).
5. Dual-publish pipeline — the existing publish workflow
   forks to produce two wheels from the same source.

### Risks

* Package boundary discipline is a maintenance cost — once there
  are two published packages, API breakage audits have to run on
  both.
* "Adoption explosion" is hypothetical; reconfirm with user
  research (Qiskit Ecosystem post-submission feedback, downloads
  telemetry) before committing to dual-publish.

### Prerequisites

* Import-graph audit (2 days of work) — needs to be the first
  step to confirm the split is feasible.
* Version-sync policy decision — the two packages share a
  SemVer or diverge?

### Acceptance

`pip install quantum-kuramoto` installs without pulling any
SCPN-specific module. `import quantum_kuramoto; print(dir(...))`
surfaces the documented public API (Kuramoto solver, VQE, Trotter,
HardwareRunner, accel dispatcher). The existing
`scpn_quantum_control` package continues to work unchanged for
existing users.

---

## S7 — Fault-tolerant / logical-level extension roadmap

### Motivation

The DLA parity asymmetry is a physical-qubit phenomenon; its
survival under error correction is not guaranteed and is not yet
theoretically analysed. Positioning the project for the post-NISQ
era requires (i) an explicit theory of which DLA-parity features
survive at the logical level and (ii) a resource-estimation model
for the surface-code implementation. The project already has
`qec/` scaffolding (repetition code UPDE, surface code UPDE,
multi-scale QEC) — the logical-level theory is the missing piece.

### Deliverables

1. `docs/logical_dla_parity.md` — theory note on whether /
   under what conditions DLA parity survives logical-qubit
   encoding. Follow-up to Sec. 4.2 of the Phase 1 short paper.
2. `qec/dla_parity_logical.py` — simulation of DLA parity on
   logical qubits under a configurable Pauli noise model.
3. Resource-estimation table:
   - Physical qubits needed for N = 16 oscillators at code
     distance d = {3, 5, 7}.
   - Wall-clock for a single Trotter step at each distance.
   - Expected fidelity under the Pauli noise model.
4. Explicit cross-check against the `multiscale_qec` hierarchy
   already in the repo — does hierarchical encoding buy us a
   lower overhead than flat surface code?
5. `paper/logical_dla_parity.md` — short-paper continuation
   for the post-v1 release cycle.

### Risks

* Full-scale logical simulation at N = 16 is compute-bound even
  on the mining rig; requires careful truncation.
* The theory step may conclude that DLA parity does NOT survive
  a given code — in which case the result is still publishable
  (a negative result on a prominent conjecture), but needs
  careful framing.

### Prerequisites

* A theorist's pass on the representation theory of the XY
  Hamiltonian under the surface-code stabiliser group (this is
  a multi-week analytical task, not engineering).
* `qec/` subpackage is already wired but untested at logical
  scale; the multi-scale QEC modules need benchmarking first.

### Acceptance

Publishable theory note with a clean positive or negative result
on "does DLA parity survive surface-code encoding", accompanied
by reproducible simulation code. Resource-estimation table cited
in the v1.0 paper as an explicit "post-NISQ outlook" section.

---

## Cross-cutting dependencies

Several tracks share prerequisites. If any of these prerequisites
slip, the dependent tracks slip with them.

| Prerequisite | Blocks |
|---|---|
| Phase 2 IBM credits | S1 (feedback loop on real HW), S2 (hardware column at N>4), S4 (cross-vendor real runs) |
| JAX tier (`[jax]` extra) | S3 (diff-pulse) — DONE |
| QuTiP + Dynamiqs wired | S5 (classical baselines) — DONE |
| `qec/` benchmarks | S7 (logical-level resources) |
| Visibility campaign (launch copy drafts) | S5 (community adoption of harness) |

## Funding / credit considerations

S1, S2, S4 are credit-intensive. A realistic cost estimate
(Phase 2 scope) for a single hardware backend across six quarterly
runs is ≳ 100 IBM Runtime hours at current pricing. Before
activating S1 or S2 execution:

1. Confirm the 5-hour IBM Credits grant status (applied
   2026-03-29).
2. Decide whether to pursue a paid Runtime window for the gap.
3. Consider a Google / Microsoft / QuEra research-credits
   application to diversify hardware access (supports S4
   directly).

## Relationship to `docs/falsification.md`

Three of the seven tracks produce claims that need falsification
criteria:

* **S3** — "learned ansatz beats hand-crafted at N ≥ 6". Falsifier:
  on a pre-registered benchmark set, hand-crafted baseline wins
  ≥ 60 % of trials. Add to `docs/falsification.md` §C as `C9`
  when S3 activates.
* **S4** — "DLA parity asymmetry reproduces on non-IBM hardware".
  Falsifier: asymmetry direction flips or disappears on the
  second backend. Add as `C10`.
* **S7** — "DLA parity survives surface-code encoding at d ≥ 3".
  Falsifier: logical-level simulation shows the asymmetry washes
  out at any finite distance. Add as `C11`.

## Activation checklist (for the future session that picks one up)

Before starting execution on any of S1–S7:

1. Re-read this document top to bottom.
2. Confirm the CEO has activated the specific track (none is
   auto-active).
3. Re-check the "Risks" and "Prerequisites" sections — the
   landscape will have moved.
4. Create a dedicated audit file under
   `docs/internal/audit_<date>_<track>.md` with a new gap
   list for the specific deliverables.
5. Schedule a session log per `CLAUDE_RULES.md` / `SHARED_CONTEXT.md`
   protocol.
6. Update `ROADMAP.md` "In progress" section with the track
   identifier.
7. Add falsification criterion to `docs/falsification.md` if the
   track produces a scientific claim.

## Cadence

This strategic roadmap is reviewed **quarterly** (January, April,
July, October). The review:

* Re-orders the priority matrix based on the last quarter's
  landscape.
* Closes items that have been executed or superseded.
* Adds new differentiation tracks as the field moves.

Each review produces a timestamped entry in `ROADMAP.md` §"Future".
