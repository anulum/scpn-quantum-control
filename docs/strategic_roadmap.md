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
| S8 | Mid-circuit adaptive branching (Dynamic Circuits) | high | 4–6 weeks (after S1) | S1 feedback runner, Dynamic Circuits API |
| S9 | Quantum thermodynamics of sync transitions | medium-high | 6–8 weeks | LindbladSyncEngine ✓, GUESS ✓, OTOC ✓ |
| S10 | Analog-native backends (Rydberg / neutral-atom / CV photonic) | high (post-NISQ positioning) | 8–12 weeks per backend | vendor SDK access, analog mapper |
| S11 | DLA-driven quantum sensing (sync-as-sensor) | medium | 3–4 weeks | qfi_criticality.py ✓, witnesses ✓ |
| S12 | Automated phase-diagram exploration via Bayesian optimisation | medium-high | 6–8 weeks | persistent-homology gem ✓, Krylov complexity ✓, Rust Hamiltonian builder ✓ |
| S13 | Bosonic / continuous-variable quantum Kuramoto | medium-high | 8–10 weeks | photonic SDK access, Rust hypergeometric pulse ✓ |
| S14 | Hybrid quantum-classical forecasting engine | medium-high | 6–8 weeks | OTOC ✓, DLA invariants ✓, sc-neurocore / SSGF / SPO bridges ✓ |

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

## S8 — Mid-circuit adaptive branching for real-time sync stabilisation

### Motivation

Follow-up to **S1**. S1 establishes a cross-shot feedback loop
(measure → observer → next shot). S8 collapses the loop into a
single circuit: partial order-parameter measurements mid-Trotter
feed a classical branch condition evaluated on the IBM classical
register, and conditional gates apply corrective drives when a
local desync metric crosses a threshold. IBM Dynamic Circuits on
Heron r2+ support the required mid-circuit measurement + conditional
gate primitives. A hardware demo of this kind of intra-circuit
adaptive branching on a Kuramoto / XY Hamiltonian has not surfaced
in the open literature we have surveyed (2026-04; the field moves,
re-check before activation).

### Deliverables

1. `control/adaptive_trotter.py` — `AdaptiveTrotterRunner` that
   composes a branched Qiskit circuit: per-slice mid-circuit
   measurement on a chosen sub-register → classical-register
   condition → conditional RZZ / RX correction → next Trotter
   slice.
2. Partial order-parameter measurement primitive (library):
   k-qubit projective X/Y readout that reconstructs $R_{local}$
   for a chosen subset of oscillators without destroying the
   rest of the state.
3. Branch-condition library — three policies committed out of the
   box:
   - Threshold on local R (corrective kick when $R < R_*$).
   - Threshold on DLA-parity leakage between sectors.
   - Chimera-state detector (clustered desync; triggers a
     topology-aware pulse from `pulse_shaping.py`).
4. Rust pre-compute helper: `scpn_quantum_engine` function that,
   given a K_nm matrix and a target $R_*$, emits the mid-circuit
   condition table (look-up for the classical branch logic).
5. Hardware demo on `ibm_kingston` at N = 4: show that the
   adaptive path achieves a higher final R than the open-loop
   Trotter baseline at equal circuit depth.
6. `docs/adaptive_branching.md` — operator manual explaining the
   API, the Dynamic Circuits constraints on the target backend,
   and the decision trade-off between "more branches = more
   reactive" vs. "more branches = more classical stalls".

### Risks

* Dynamic Circuits latency per branch is backend-dependent; the
  feature is most capable on Heron r2+. Older systems reject the
  conditional pattern entirely.
* Mid-circuit measurement collapses the measured register;
  accidentally measuring the wrong sub-register destroys the
  Trotter state. The partial-order-parameter primitive needs a
  formal proof that only the targeted sub-register is projected.
* Circuit compilation overhead grows with branch count; an eight-
  branch Trotter may exceed the 5-minute IBM Runtime soft-cap.

### Prerequisites

* **S1 (hybrid feedback loop) must land first** — S8 reuses the
  runner-level observer plumbing that S1 introduces. Activating
  S8 before S1 is feasible but duplicates work.
* Dynamic Circuits support confirmed on the target backend.
* Rust-side branch-table generator — 2 weeks on top of the
  existing `scpn_quantum_engine` build.

### Acceptance

On a pre-registered N = 4 benchmark with controlled frequency
spread, the adaptive branched circuit outperforms the open-loop
Trotter baseline on final R at equal circuit depth in ≥ 70 % of
50 random $\omega$ realisations. Raw data archived to Zenodo.
Applications crossover panel — if the same adaptive pattern
improves fidelity on a disruption-proxy ITER workload or a
power-grid cascade toy model, it is evidence that the primitive
generalises beyond Kuramoto.

### Falsification

- New entry for `docs/falsification.md` as **C12** when S8
  activates: "Adaptive branching improves final R over open-loop
  Trotter at equal depth." Falsifier — win-rate ≤ 50 % on the
  pre-registered benchmark.

---

## S9 — Quantum thermodynamics of synchronisation transitions

### Motivation

The repo already ships the machinery a quantum-thermodynamics
study needs: `LindbladSyncEngine` (v0.9.5) for open-system
dynamics, `mitigation/symmetry_decay` (GUESS) for a
magnetisation-based ZNE that cleanly isolates a symmetry-protected
sector, and `analysis/otoc.py` for information-scrambling probes.
What is missing is the physical thermodynamic framing:
entropy-production rates, irreversibility measures, heat
dissipation signatures across the quantum Kuramoto transition —
and how the DLA-parity constraint shapes them.

Standalone quantum-thermodynamics literature is mature on harmonic
oscillators and simple qubit systems; heterogeneous Kuramoto order
parameters on real hardware is a gap we are positioned to close.

### Deliverables

1. `thermodynamics/` subpackage (new, per monolith rule — no
   single-file dump):
   - `entropy_production.py` — Landi-Paternostro formalism for
     bipartite open-system entropy-production rate, applied to
     the two DLA parity sectors as the bipartition.
   - `irreversibility.py` — Jarzynski / Crooks work-fluctuation
     estimators specialised for Kuramoto-XY quenches.
   - `heat_dissipation.py` — scalar observable based on the
     Lindblad jump statistics already exposed by
     `LindbladSyncEngine`.
   - Tests: multi-angle (deterministic reference at $T = 0$,
     fluctuation identities at finite $T$, DLA-sector
     invariance).
2. Hardware demo at N = 4 on `ibm_kingston`: measure entropy
   production across the sync transition ($K$-sweep), using GUESS
   + readout-baseline correction. Companion classical baseline
   from QuTiP Lindblad.
3. Short paper: "Quantum thermodynamic signatures of
   synchronisation transitions on superconducting hardware".
   Target venue: `PRX Quantum` or `npj Quantum Information`.
4. `docs/quantum_thermo.md` — theory note: derivation of the
   bipartite entropy-production rate on the heterogeneous
   Kuramoto-XY model, DLA-sector decomposition, experimental
   protocol.

### Risks

* Thermodynamic observables are notoriously noise-sensitive;
  hardware readout bias can mimic entropy-production signatures.
  Mitigation: GUESS plus a noise-free AerSimulator control run.
* "Preprint would dominate citations" is a marketing claim,
  not a scientific one — the realistic framing is "fills a
  specific gap; expected to land on the lower rungs of the
  quantum-thermo citation tree".

### Prerequisites

* `LindbladSyncEngine` (already in repo).
* GUESS ZNE (already in repo).
* OTOC (already in repo).
* Dedicated CEO / theorist time for the formalism derivation
  (the engineering port is trivial; the theory is not).

### Acceptance

A reproducible hardware + classical pipeline that computes the
entropy-production rate at five $K$ values straddling the Kuramoto
transition, with a classical reference within 2 $\sigma$ of the
hardware point. Paper draft circulated for internal review.

### Falsification

- New entry **C13** in `docs/falsification.md`: "Entropy
  production rate peaks at the Kuramoto transition." Falsifier —
  no statistically significant peak above the classical baseline
  across the $K$-sweep.

---

## S10 — Analog-native Kuramoto backends (Rydberg / neutral-atom / CV photonic)

### Motivation

Every Kuramoto / XY implementation in this repo today is digital-
circuit based: K_nm and ω compile into sequences of RZZ / RX / CX
gates, Trotterised. Analog-native platforms — Rydberg arrays,
neutral-atom tweezers, continuous-variable photonic modes — are
*literally* built out of interacting oscillators. A direct map
from K_nm + ω to native Rydberg blockade / optical lattice /
photonic coupling bypasses Trotter entirely and inherits
hardware-native physics.

The post-NISQ simulation landscape (~2027–2030) is moving toward
analog and hybrid-analog machines. Owning the software layer that
compiles a Kuramoto coupling matrix directly onto a Rydberg /
photonic simulator is a long-horizon positioning move.

### Deliverables

1. `hardware/analog/` subpackage (new):
   - `rydberg.py` — compile K_nm → Rydberg array geometry +
     detuning schedule. Integrate via QuEra Bloqade or
     `pennylane-rydberg` plugin.
   - `photonic.py` — compile K_nm → CV-mode coupling schedule.
     Integrate via Xanadu Strawberry Fields / Bosonic-Qiskit /
     PennyLane `default.gaussian`.
   - `mapping.py` — unit-conversion and topology-fitting helpers
     (not every K_nm is physically realisable on a fixed lattice;
     document which subset is).
2. Three end-to-end demos:
   - Rydberg simulator (Bloqade or equivalent): DLA parity
     asymmetry on a 6-atom chain at N = 6.
   - CV photonic simulator: Kuramoto synchronisation with three
     modes, heterogeneous frequencies.
   - Hybrid digital-analog: portion of the K_nm in Rydberg, the
     sync observables read via a digital Pauli readout.
3. Pulse shaping via the existing `scpn_quantum_engine`
   hypergeometric pulse envelope — native to analog platforms.
4. `docs/analog_backends.md` — operator manual: vendor SDK
   install instructions, per-platform caveats, the mapping
   trade-offs.

### Risks

* Vendor SDK access is non-trivial. QuEra Bloqade is open but
  requires a Julia runtime (our Julia tier helps). Xanadu
  Strawberry Fields is maintenance-mode since 2024; a more
  actively maintained photonic alternative may need
  identification.
* "Analog is 10–100× more natural than anyone else's" is
  unfalsifiable as a marketing claim. The falsifiable version:
  on a fixed Kuramoto benchmark, the analog compilation uses
  fewer primitives (detunings / beam-splitters) than the digital
  Trotter at matched fidelity.
* Post-NISQ adoption timeline for analog machines is a bet.

### Prerequisites

* Julia tier landed — helps with Bloqade integration.
* PennyLane adapter (67-test cross-vendor suite in commit
  `c3dcbf6`) is the structural starting point.
* Rust hypergeometric pulse engine (already shipped in v0.9.5).

### Acceptance

DLA parity asymmetry reproduced on at least one analog
platform at N ≥ 4 with a direction agreeing with the IBM
digital-circuit baseline. Photonic CV demo shows order-parameter
synchronisation with three heterogeneous-frequency modes.
`docs/analog_backends.md` published.

### Falsification

- New entry **C14** in `docs/falsification.md`: "Analog
  compilation of K_nm uses fewer primitive operations than the
  Trotter digital compilation at matched fidelity." Falsifier —
  digital Trotter hits a lower gate count at the same fidelity on
  the benchmark.

---

## S11 — DLA-driven quantum sensing via sync order parameter

### Motivation

The sync order parameter R (global) and its DLA-protected
fluctuations are a natural quantum sensor for external perturbations
— noise, applied fields, parameter drifts. The project already has
`analysis/qfi_criticality.py` (quantum Fisher information at the
Kuramoto transition) plus the witness machinery in
`analysis/sync_witness.py`. A small addition — quantify the
metrological gain of the sync observable as a function of K near
the transition, then demonstrate it on hardware — converts the
synchronisation demo into a quantum-enhanced sensor.

Applied targets: EEG connectivity drift, power-grid spectral
perturbations, tokamak plasma-mode drift.

### Deliverables

1. `analysis/sensing.py` (new):
   - `metrological_gain_vs_K(K_array, omega, perturbation)` —
     returns $F_Q(K)$ of R under a parametric perturbation,
     using the QFI tooling already in `qfi_criticality.py`.
   - `optimal_sensing_K(K_grid, omega, target)` — finds the
     K value that maximises the QFI-per-shot for a given
     sensing target.
2. Hardware demo at N = 4 on `ibm_kingston`: measure
   $F_Q(K)$ at five $K$ values; confirm the expected peak near
   the critical coupling.
3. Application cross-overs (one each):
   - Injected classical perturbation on an EEG PLV matrix —
     demonstrate detection above classical baseline.
   - Injected topology perturbation on the Josephson array
     benchmark.
4. `docs/quantum_sensing.md` — theory note tying QFI $\to$
   sync observable $\to$ real-world sensing use case.

### Risks

* QFI estimation on hardware needs careful shot-budget
  management; classical shadows (shadow_tomography already in
  repo) may be required for statistical sensitivity.
* "Quantum-enhanced for real-world oscillator networks" must not
  be over-claimed; the rigorous claim is "QFI gain above
  classical Fisher information on a pre-registered benchmark".

### Prerequisites

* `analysis/qfi_criticality.py` — already in repo.
* `analysis/sync_witness.py` — already in repo.
* `analysis/shadow_tomography.py` — already in repo.

### Acceptance

Published figure: hardware-measured $F_Q(K)$ vs. K with a
peak-detection test that isolates the transition region to within
$\Delta K = 0.1$. Cross-over demo on at least one applied target
(EEG or Josephson). Paper note added to the v1.0 release cycle.

### Falsification

- New entry **C15** in `docs/falsification.md`: "QFI-based
  sync-order-parameter sensing beats classical Fisher
  information on a pre-registered perturbation benchmark."
  Falsifier — ratio of Fisher informations below 1 on the
  benchmark mean.

---

## S12 — Automated quantum exploration of the full synchronisation phase diagram

### Motivation

The sync phase diagram (K, $\omega$ distribution, coupling
topology) is vast. Classical explorations are either exhaustive on
tiny N or mean-field on large N. A QPU-in-the-loop Bayesian
optimiser, driven by signal-rich observables already in the repo
(persistent homology $p_{h1}$, Krylov complexity, OTOC scrambling
speed), can prioritise "interesting" regions — chimera states,
explosive-sync precursors, metastable basins — and only spend
hardware shots where the classical reference has lost confidence.

This is the step from "simulator" to "discovery engine".

### Deliverables

1. `discovery/` subpackage (new):
   - `bayes_explorer.py` — Gaussian-process / scikit-optimize
     Bayesian optimiser over (K_scale, $\omega$-spread,
     topology-parameter) space.
   - `phase_scan.py` — orchestrator that calls the Rust
     Hamiltonian builder for classical pre-screening, promotes
     high-interest points to the IBM queue, and streams results
     back to update the GP surrogate.
   - `interest_metrics.py` — weighted combination of $p_{h1}$,
     Krylov complexity, OTOC, spectral form factor, chimera
     index. Each contributor already exists.
2. Hardware campaign: a 100-point phase-diagram scan on
   `ibm_kingston` (N = 4) prioritised by the Bayesian loop.
   Phase 2 IBM credits scope.
3. Scientific deliverable: a published phase-diagram figure
   with at least one labelled "discovered" feature (a chimera,
   a metastable basin, an explosive-sync edge) not previously
   characterised on hardware for this coupling class.
4. `docs/discovery_engine.md` — operator manual + reproducibility
   hooks.

### Risks

* GP surrogates scale poorly with dimensionality. Keep the
  search space low-D (≤ 4 free parameters) or use sparse GP
  approximations.
* The Bayesian optimiser can collapse into an exploitation mode
  that never visits genuinely new regions; explicit diversity
  constraints needed.
* IBM credits for a 100-point scan are substantial; budget
  confirmed before activation.

### Prerequisites

* Phase 2 IBM credits.
* Persistent-homology module, Krylov complexity, OTOC, spectral
  form factor — all already in `analysis/` (v0.9.1).
* `[rl]` or `[bayes]` new extra with `scikit-optimize` or
  `botorch`.

### Acceptance

At least one novel feature of the hardware-measured phase diagram
(confirmed against the classical reference as a "QPU-first
detection") is documented in the paper. Reproducer script runs
end-to-end under ≤ 30 minutes of QPU time plus a classical budget
ceiling.

### Falsification

- New entry **C16** in `docs/falsification.md`: "The Bayesian
  discovery loop finds a feature not visible in the classical
  pre-screen at the same compute budget." Falsifier — every
  hardware-flagged feature is already present in the classical
  pre-screen.

---

## S13 — Bosonic / continuous-variable quantum Kuramoto

### Motivation

Companion to S10 but on the software-layer side. Every physical
oscillator is natively a harmonic (or anharmonic) CV mode.
Mapping Kuramoto directly to qumodes — without the qubit-encoding
overhead — is the natural compilation for CV hardware (Xanadu
photonic, IonQ's planned CV, trapped-ion motional modes). This is
the digital/analog-bridge complement to S10: instead of picking a
specific vendor, pick a mode-centric abstraction and compile into
whichever CV backend is available.

### Deliverables

1. `phase/cv_kuramoto.py` — qumode-level solver that accepts the
   same K_nm + $\omega$ API as `QuantumKuramotoSolver` but
   compiles to a CV-gate sequence (beam-splitter + squeezing +
   displacement) rather than RZZ / RX / CX.
2. `hardware/photonic.py` (extends S10 if S10 lands first) —
   bridge to Strawberry Fields / Bosonic Qiskit for simulator
   execution.
3. Rust-side CV pulse shaper: extend the hypergeometric engine
   to parametric-squeezing envelopes appropriate for CV drives.
4. Conversion layer: map a measured qumode state back to a
   "standard" sync order parameter R so CV and qubit results
   can be compared apples-to-apples.
5. Demo: sync transition at three heterogeneous-frequency
   qumodes, reproduced on a CV simulator with a fidelity target
   documented in the acceptance criterion.
6. `docs/cv_kuramoto.md` — theory note explaining the qumode
   encoding, the mode-to-sync-observable conversion, and the
   trade-offs vs. digital qubit encoding.

### Risks

* CV hardware access is harder than superconducting QPU access;
  the demo may live on a simulator for the first release.
* The CV state space is infinite-dimensional; truncating cleanly
  (Fock cutoff) without distorting the physics is a non-trivial
  parameter choice.
* Bosonic Qiskit is a third-party fork with slower release
  cadence; upstream dependency on a potentially unmaintained
  package.

### Prerequisites

* S10 (analog backends) is a natural parent — do S10 first or in
  parallel.
* Rust hypergeometric pulse engine — already shipped in v0.9.5.

### Acceptance

A reproducible CV-simulator demo at N = 3 qumodes where the
CV-computed order parameter agrees with the qubit-computed
reference within 5 % over a K-sweep. Theory note published.

### Falsification

- New entry **C17** in `docs/falsification.md`: "CV-encoded
  Kuramoto reproduces the qubit-encoded sync transition to
  within 5 %." Falsifier — mean absolute deviation > 5 % on the
  pre-registered K-sweep.

---

## S14 — Hybrid quantum-classical forecasting engine

### Motivation

The chronic pain point of classical Kuramoto is exponential cost
in chaotic or high-dimensional regimes (brain connectomes, grid
cascades, tokamak disruption dynamics). QPU-computed signals —
OTOC scrambling rate, DLA invariants, partial sync snapshots —
are *exactly* the quantities classical solvers cannot cheaply
access. Feeding those as correction terms into a large-N classical
Kuramoto solver (`scpn-fusion-core`, `sc-neurocore`) closes the
prediction loop in the regime where pure classical solvers
saturate.

The SNN / SSGF / ITER / EEG cross-repo bridges already exist in
this codebase; S14 closes the prediction loop that those bridges
have been waiting for.

### Deliverables

1. `forecasting/` subpackage (new):
   - `quantum_corrections.py` — extract OTOC + DLA invariants +
     partial R snapshots from a QPU run, packed into a
     `CorrectionBundle` dataclass.
   - `hybrid_solver.py` — wrapper around the existing classical
     Kuramoto solver (`scpn_quantum_control.hardware.classical.
     classical_kuramoto_reference`) that injects the
     `CorrectionBundle` as additive forcing on the right-hand
     side. Coupling strength of the injection is a hyperparameter.
   - `forecast_validator.py` — compares a hybrid trajectory
     against a held-out "ground truth" (a long classical run or a
     held-out measurement).
2. Three applied demos:
   - Brain-connectome: use the EEG PLV adapter already in
     `applications/eeg_benchmark.py`; show the hybrid forecast
     beats pure classical on a held-out trajectory from the
     available EEG dataset.
   - ITER: use the `applications/iter_benchmark.py` stub;
     compare hybrid forecast on an 8-mode tokamak proxy.
   - Power grid: `applications/power_grid.py` IEEE-5-bus with a
     contrived chaotic regime.
3. `docs/hybrid_forecasting.md` — theory note on where quantum
   corrections help (chaotic / high-dim) vs. where they do not
   (near-integrable / small-N).

### Risks

* "Hybrid beats classical" is a non-trivial claim and needs
  honest falsification: classical can be made to win by tuning
  the correction weight to zero. The benchmark must fix the
  hyperparameter across all runs and not post-hoc optimise.
* The three applied targets each carry their own data quality
  risk (EEG ground-truth is noisy; IEEE-5-bus is a toy; ITER data
  is partially synthetic).

### Prerequisites

* OTOC + DLA modules already in `analysis/`.
* `hardware/classical.py` Kuramoto reference — already in repo.
* Cross-repo bridges (`sc-neurocore`, `scpn-fusion-core`,
  `scpn-phase-orchestrator`) already exist in
  `src/scpn_quantum_control/bridge/`.

### Acceptance

On a pre-registered benchmark set of chaotic Kuramoto
trajectories, the hybrid forecast achieves ≥ 15 % lower
mean-squared-error over a held-out window than the pure classical
forecast, at matched compute budget, on ≥ 2 of the 3 applied
targets. Zenodo deposit of the benchmark.

### Falsification

- New entry **C18** in `docs/falsification.md`: "Hybrid
  quantum-classical forecast beats pure classical on chaotic
  Kuramoto trajectories at matched compute budget." Falsifier —
  hybrid underperforms pure classical on ≥ 2 of the 3 pre-
  registered benchmarks.

---

## Cross-cutting dependencies

Several tracks share prerequisites. If any of these prerequisites
slip, the dependent tracks slip with them.

| Prerequisite | Blocks |
|---|---|
| Phase 2 IBM credits | S1 (feedback loop on real HW), S2 (hardware column at N > 4), S4 (cross-vendor real runs), S8 (adaptive-branching hardware demo), S9 (thermo hardware at N = 4), S11 (sensing hardware demo), S12 (phase-diagram scan) |
| JAX tier (`[jax]` extra) | S3 (diff-pulse) — DONE |
| QuTiP + Dynamiqs wired | S5 (classical baselines) — DONE |
| `qec/` benchmarks | S7 (logical-level resources) |
| Visibility campaign (launch copy drafts) | S5 (community adoption of harness) |
| **S1 (hybrid feedback loop)** | **S8 (adaptive branching is a follow-up of S1)** |
| `LindbladSyncEngine` + GUESS + OTOC | S9 — DONE |
| Julia tier + Rust hypergeometric pulse | S10 (Rydberg SDK bridge); S13 (CV pulse shaping) — DONE |
| `analysis/qfi_criticality.py` + shadow tomography | S11 — DONE |
| Persistent homology + Krylov complexity + SFF | S12 — DONE |
| Cross-repo bridges (sc-neurocore / fusion-core / phase-orchestrator) | S14 — DONE |
| Vendor SDK access (QuEra Bloqade, Xanadu Strawberry Fields, Bosonic Qiskit) | S10 / S13 |
| `[bayes]` or `[rl]` extra (scikit-optimize / botorch) | S12 |

**Dependency graph (forward direction only):**

```
S1 → S8 (adaptive branching reuses the S1 observer plumbing)
S5 ⟂ S2 (independent, but S5's harness benefits from S2's scaling data)
S10 ⟂ S13 (parallel work on analog / CV; share SDK work but independent deliverables)
S7 ⟂ every other track (theory-bound, can run in parallel)
```

## Funding / credit considerations

S1, S2, S4, S8, S9, S11, S12 are credit-intensive. A realistic
cost estimate (Phase 2 scope) for a single hardware backend
across six quarterly runs is ≳ 100 IBM Runtime hours at current
pricing. S12 (phase-diagram scan) alone is ≳ 30 hours. Before
activating any credit-intensive track:

1. Confirm the 5-hour IBM Credits grant status (applied
   2026-03-29).
2. Decide whether to pursue a paid Runtime window for the gap.
3. Consider a Google / Microsoft / QuEra research-credits
   application to diversify hardware access (supports S4, S10
   directly).
4. For analog / CV tracks (S10, S13), application windows to
   QuEra, Xanadu, PsiQuantum research programmes are the
   practical path.

## Relationship to `docs/falsification.md`

Every differentiation track that produces a scientific claim
needs a falsifier pre-registered in `docs/falsification.md`.
Summary of claim IDs to be added on track activation:

| Track | Falsifier ID | Claim |
|---|---|---|
| S3 | C9 | Learned ansatz beats hand-crafted at N ≥ 6 |
| S4 | C10 | DLA parity asymmetry reproduces on non-IBM hardware |
| S7 | C11 | DLA parity survives surface-code encoding at d ≥ 3 |
| S8 | C12 | Adaptive branching improves final R over open-loop Trotter at equal depth |
| S9 | C13 | Entropy-production rate peaks at the Kuramoto transition |
| S10 | C14 | Analog compilation uses fewer primitives than Trotter at matched fidelity |
| S11 | C15 | QFI-based sensing beats classical Fisher information on a pre-registered benchmark |
| S12 | C16 | Bayesian discovery loop finds a feature invisible in the classical pre-screen |
| S13 | C17 | CV-encoded Kuramoto reproduces qubit-encoded sync transition to ≤ 5 % |
| S14 | C18 | Hybrid quantum-classical forecast beats pure classical on chaotic Kuramoto |

S1, S2, S5, S6 are infrastructure / engineering tracks; they
have internal acceptance criteria but no scientific claim that
needs falsification.

## Activation checklist (for the future session that picks one up)

Before starting execution on any of S1–S14:

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
