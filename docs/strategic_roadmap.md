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
| S15 | DLA-protected many-body scars for long-lived sync | medium-high | 6–8 weeks | DLA machinery ✓, Rust drive shaping ✓ |
| S16 | Quantum network tomography (reconstruct K_nm from observables) | high | 5–7 weeks | witnesses ✓, OTOC ✓, differentiable Rust backend (new) |
| S17 | Higher-order (simplicial / hypergraph) quantum Kuramoto | medium | 5–7 weeks | mapper refactor, future Heron / Loon multi-qubit gates |
| S18 | Sync-protected quantum memories and repeaters | medium-high | 6–8 weeks | witnesses ✓, qec/ ✓ |
| S19 | Entanglement phase diagram + magic + Krylov complexity | medium | 4–5 weeks | magic_sre ✓, krylov_complexity ✓, entanglement_entropy ✓ |
| S20 | Quantum Kuramoto universal control-benchmark suite | high (community) | 6–8 weeks | scpn_quantum_engine ✓, HardwareRunner ✓, S5 harness |
| S21 | Multi-scale quantum → classical bridging layer | high | 6–8 weeks | DLA invariants ✓, bridge/* ✓, large-N classical Kuramoto (ext dep) |
| S22 | Non-Hermitian / PT-symmetric Kuramoto + exceptional points | medium-high | 6–8 weeks | LindbladSyncEngine ✓, Rust pulse ✓, dilation primitive (new) |
| S23 | Quantum reservoir computing on Kuramoto transients | medium-high | 5–6 weeks | qrc_phase_detector ✓, OTOC ✓, witnesses ✓ |
| S24 | Quantum speed limits for collective sync | medium | 4–5 weeks | quantum_speed_limit ✓, DLA ✓, OTOC ✓ |
| S25 | Topological defects + vortex dynamics on 2D quantum lattices | medium-high | 6–8 weeks | vortex_detector ✓, wilson_loop ✓, mapper 2D extension |
| S26 | Entanglement-mediated long-range synchronisation | medium | 5–6 weeks | Bell pair prep ✓ (crypto/bell_test), Heron topology constraints |
| S27 | Hardware-in-the-loop inverse design of oscillator networks | high | 6–8 weeks | S1 feedback ✓, Rust mapper ✓, VQE ✓ |
| S28 | Sync-enhanced distributed quantum metrology | medium-high | 5–6 weeks | QFI ✓, witnesses ✓, sensor-array topology design |
| S29 | Floquet Kuramoto time crystals + subharmonic sync | medium | 4–5 weeks | floquet_kuramoto ✓, Rust hypergeometric pulse ✓ |
| S30 | Quantum Kuramoto for community detection + modularity | medium | 5–6 weeks | witnesses ✓, DLA invariants ✓, graph-benchmark corpus |
| S31 | DLA-protected MBL / delocalisation transitions | medium | 5–7 weeks | DLA ✓, OTOC ✓, disorder-sweep harness (new) |
| S32 | Monitored quantum Kuramoto (measurement-induced transitions) | medium-high | 5–7 weeks | Dynamic Circuits ✓ (S1/S8), witnesses ✓ |
| S33 | Quantum-enhanced Lyapunov spectra for chaotic Kuramoto | medium | 5–6 weeks | OTOC ✓, DLA ✓, classical large-N solver |
| S34 | Self-organising Kuramoto (autonomous drive engineering) | medium | 4–5 weeks | S1 + S8 feedback plumbing (prerequisite) |
| S35 | Quantum Kuramoto native simulator for active matter (non-reciprocal) | medium | 5–6 weeks | non-reciprocal K extension, S22 dilation primitive |
| **S36** | Information geometry on quantum sync manifolds (Fisher tensor) | medium-high | 5–6 weeks | QFI ✓, Rust engine ✓ |
| **S37** | Categorical / compositional quantum Kuramoto | medium | 5–7 weeks | mapper refactor, category-theory formalism |
| **S38** | Quantum Kuramoto field theory (QKFT) continuum limit + RG flows | high | 8–12 weeks | tensor networks (quimb ✓), DLA ✓, QuTiP ✓ |
| **S39** | Autopoietic / self-referential networks (sync-driven K rewriting) | medium | 5–6 weeks | S1 feedback + S34 autonomous (prereq) |
| **S40** | Holographic duals via quantum sync (AdS/CFT-style) | strategic / exploratory | 8–12 weeks | DLA invariants ✓, theorist pass |
| **S41** | Quantum causal discovery with intervention | medium-high | 5–7 weeks | extends S16, requires Dynamic Circuits ✓ |
| **S42** | Symplectic structure-preserving Trotterization | medium | 5–7 weeks | Rust pulse ✓, geometric-integrator theory |
| **S43** | Full resource theory of quantum synchronisation | medium-high | 6–8 weeks | witnesses ✓, OTOC ✓, DLA ✓ |
| **S44** | Objective collapse / macroscopic foundations testbed (GRW, Penrose OR, CSL; Quantum Darwinism angle) | high (foundations) | 8–12 weeks | DLA parity ✓, large-N hardware time |
| **S45** | Biologically faithful Kuramoto simulator + IIT consciousness angle | medium-high | 6–8 weeks | EEG / connectome benchmarks ✓, applications/ ✓ |
| **S46** | Phase-transition / attractor-landscape quantum programming | medium-high (paradigm) | 6–8 weeks | witnesses ✓, floquet_kuramoto ✓ |
| **S47** | Analogue gravity: relativistic metrics, cosmological phase transitions, baryogenesis, emergent spacetime | high (foundations) | 10–12 weeks | QKFT (S38), theorist pass |
| **S48** | Self-healing qubit fabrics + continuous QEC via sync | medium-high | 6–8 weeks | qec/ ✓, S18 sync memories (prereq) |
| **S49** | Quantum fluctuation theorems (Jarzynski, Crooks) across sync transitions | medium-high | 5–6 weeks | extends S9 thermodynamics, LindbladSyncEngine ✓ |
| **S50** | Quantum kernels from sync manifolds (ML) | medium-high | 4–5 weeks | DLA ✓, Rust forward mapper ✓ |
| **S51** | Hayden–Preskill / black-hole info dynamics simulator | medium-high | 5–6 weeks | OTOC ✓, DLA scrambling ✓ |
| **S52** | Distributed quantum consensus via global sync (quantum internet timing) | high (infrastructure) | 8–10 weeks | S1 async runner ✓, Bell pair ✓, modular topology |
| **S53** | Engineered self-organised criticality in oscillator networks | medium-high | 5–7 weeks | SOC literature, OTOC ✓, witness ✓ |

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

## S15 — DLA-protected many-body scars for long-lived synchronisation

### Motivation

Quantum many-body scars — non-thermalising eigenstates embedded in
an otherwise chaotic spectrum — have been studied on Rydberg arrays
and spin chains but never tied to heterogeneous Kuramoto
synchronisation. The DLA parity structure of the XY Hamiltonian is
a natural candidate for scar-supporting symmetry: the two sectors
decouple under unitary evolution, and any eigenstate supported
entirely on one sector is protected from thermalising into the
other. Using the Rust engine to pre-compute scar-preserving drives
and inject them as mid-circuit corrections is the obvious
implementation path.

### Deliverables

1. `analysis/many_body_scars.py` — numerical identification of
   DLA-sector-supported eigenstates for N ≤ 8 via the existing
   `build_xy_generators` output; classification by overlap with
   thermal ensemble.
2. Scar-preserving drive library: `scpn_quantum_engine` helper
   that, given a target scar state, computes a sequence of
   mid-circuit rotations that project the instantaneous state
   back onto the scar subspace.
3. Hardware demo on `ibm_kingston` at N = 4: show that the
   scar-preserved trajectory maintains $R \geq R_*$ for ≥ 2×
   the depth at which the open-loop Trotter crashes into the
   coherence wall.
4. Cross-check against thermal ETH prediction — scar state
   eigenvalue statistics should be Poisson while the rest of
   the spectrum is Wigner-Dyson.
5. `docs/scars_sync.md` — theory note plus operator manual.

### Risks

* Scar construction from a partial DLA (when the DLA is not a full
  representation of a simple Lie algebra) may not admit clean
  analytical scars — the numerical search returns approximate
  scars with finite lifetime.
* Heterogeneous $\omega_i$ breaks the symmetries that typically
  protect scars in homogeneous spin chains; DLA-based protection
  must be proven (not assumed) to survive the heterogeneity.

### Prerequisites

* DLA machinery in `analysis/dynamical_lie_algebra.py` — already
  in repo.
* Rust drive-shaping in `pulse_shaping.rs` — already in repo.
* Theorist pass on the DLA-partial-scar construction (1 week
  analytical work before engineering starts).

### Acceptance

Scar-preserved sync trajectory on hardware demonstrates a
documented coherence-lifetime extension over the open-loop
baseline on a pre-registered N = 4 benchmark. Paper note +
raw data to Zenodo.

### Falsification

- New entry **C19** in `docs/falsification.md`: "A DLA-sector-
  supported scar subspace exhibits longer sync lifetime than
  generic eigenstates at matched fidelity." Falsifier — no
  statistically significant lifetime advantage over a
  dimension-matched random eigenstate on the benchmark.

---

## S16 — Quantum network tomography (reconstruct hidden K_nm from observables)

### Motivation

Classical network reconstruction — inferring a coupling matrix
from time-series data of oscillator phases — is mature. Its
quantum-enhanced analogue exists only as tiny-N theory. The
project's witnesses + OTOC + DLA structure provide a richer
observable set than classical phase-only data; feeding this into a
differentiable Rust forward model enables end-to-end inverse
inference of an unknown K_nm matrix and $\omega$ vector from
hardware-measured observations of the oscillator network.

### Deliverables

1. `analysis/network_tomography.py`:
   - `reconstruct_knm(witnesses, otocs, order_params, n_osc)` —
     inverse-problem solver over the coupling matrix.
   - `reconstruct_frequencies(spectra, witnesses)` — recovery of
     $\omega$ from the measured observables.
2. Differentiable Rust forward model — `scpn_quantum_engine`
   extension that backpropagates a loss from the observable
   space through the Trotter dynamics to K_nm parameters.
3. Applied demos:
   - EEG PLV matrix → inferred K_nm, compared against the
     published connectivity prior.
   - IEEE 5-bus power grid → inferred K_nm from sync
     observables at the three grid buses.
4. Regularisation study: L1 sparsity, low-rank, and
   topology-informed priors.
5. `docs/quantum_tomography.md` — theory note + operator manual.

### Risks

* Identifiability: many K matrices produce the same macroscopic
  sync observables. Regularisation is not optional.
* Classical network reconstruction is a well-understood hard
  problem; the quantum advantage must be measurable (better
  reconstruction error, lower data requirements, or broader
  applicability) — not assumed.

### Prerequisites

* Witnesses, OTOC, order-param machinery — already in repo.
* `scpn_quantum_engine` differentiable extension (new work ~2
  weeks inside the Rust crate).
* Benchmark ground-truth datasets (EEG, IEEE-5, a synthetic
  graph corpus).

### Acceptance

On a pre-registered benchmark of 50 synthetic graphs (N = 8,
varying sparsity), quantum-assisted tomography recovers K with
< 10 % mean absolute error at shot budgets where the classical
baseline exceeds 20 %. One applied demo (EEG or IEEE-5)
documented with the reconstruction error vs. the published
reference.

### Falsification

- New entry **C20** in `docs/falsification.md`: "Quantum-assisted
  tomography recovers K_nm to below 10 % MAE vs. classical
  baseline at matched data / shot budget." Falsifier — ≥ 10 %
  MAE gap in favour of the classical baseline on the benchmark.

---

## S17 — Higher-order (simplicial / hypergraph) quantum Kuramoto

### Motivation

Classical Kuramoto has been generalised to include explicit
triplet, quadruplet, and higher-order couplings (2024–2025
literature on simplicial Kuramoto; Lucas et al., Battiston et al.).
No quantum hardware realisation has been reported. Extending the
K_nm mapper to ingest a hypergraph (or simplicial complex) and
compile triplet interactions to multi-qubit Trotter gates —
plus pulse-shaped multi-qubit drives on future Heron / Loon chips —
opens access to systems with genuine multi-way interactions: social
contagion, chemical reaction networks, multi-body plasma modes.

### Deliverables

1. `bridge/higher_order_mapper.py` — hypergraph / simplicial-tensor
   ingestion producing a Qiskit circuit with explicit three- and
   four-body Trotter terms.
2. Native multi-qubit gate exploration — for backends that expose
   them (future Heron / Loon), use the native gate instead of
   Trotter-decomposed CXs.
3. Rust-side hypergraph coupling tensor: efficient storage +
   Trotter-term enumeration.
4. Hardware demo at N = 6 with two triplet couplings: confirm the
   sync transition occurs at a shifted $K_c^{(3)} \neq K_c^{(2)}$
   predicted by classical higher-order Kuramoto.
5. `docs/higher_order_kuramoto.md` — theory + operator manual.

### Risks

* Multi-qubit gate decomposition cost grows combinatorially; a
  dense triplet coupling on 6 qubits may exceed the coherence
  budget for Trotter-depth ≥ 4.
* Hypergraph-native gates are hardware-specific; portability
  across vendors is limited.

### Prerequisites

* Classical higher-order Kuramoto mapper (extension of
  `knm_hamiltonian.py` — roughly 2 weeks).
* Hardware availability of multi-qubit gates, or acceptance of
  Trotter decomposition cost on current Heron r2.

### Acceptance

Measurable $K_c$ shift from pairwise to three-body couplings on
hardware at N = 6, matching the classical higher-order prediction
within 15 % on a pre-registered benchmark.

### Falsification

- New entry **C21** in `docs/falsification.md`: "Higher-order
  sync transition distinguishable from pairwise baseline on
  hardware at N = 6." Falsifier — measured $K_c$ shift within
  statistical uncertainty of zero.

---

## S18 — Synchronisation-protected quantum memories and repeaters

### Motivation

A globally synchronised phase subspace is approximately
eigen-decoupled from the rest of the Hilbert space under the
sync-protected dynamics — a natural error-suppressing subspace.
Encoding logical qubits into this manifold and using DLA parity
witnesses as syndrome measurements would extract bit-flip /
phase-flip errors without breaking the sync. The sync-protected
logical qubit is a candidate primitive for distributed clock
networks, quantum internet repeater nodes, and fault-tolerant
sensor arrays.

### Deliverables

1. `qec/sync_memory.py`:
   - `SyncLogicalQubit` dataclass and constructor that encodes
     a target logical state into the sync manifold.
   - Syndrome measurement primitive using `sync_witness.py`
     observables as the parity check.
   - Error-correction cycle: detect → classical decision →
     apply corrective drive.
2. Hardware demo at N = 4: logical qubit encoded in the sync
   manifold shows documented coherence-time extension vs. a
   single-qubit reference stored in the same hardware class.
3. Theoretical analysis: proof (or refutation) that the sync
   manifold satisfies the approximate stabiliser-code conditions.
4. `docs/sync_memory.md` — theory + protocol.

### Risks

* The sync manifold is a continuous subspace; mapping discrete
  error syndromes onto it requires care. A formal analysis may
  conclude that the error suppression does not scale — still a
  publishable negative result.
* Coherence-time gain on hardware may be dominated by the drive
  overhead, not the manifold protection. Need matched-budget
  comparison.

### Prerequisites

* Witnesses + DLA parity framework — already in repo.
* `qec/` scaffolding — already in repo.
* Theorist pass on the "is-the-sync-manifold-a-stabiliser-code"
  question.

### Acceptance

Logical qubit encoded in the sync manifold demonstrates coherence
extension on a pre-registered benchmark (ratio of logical
lifetime to unprotected lifetime ≥ documented threshold, e.g.
1.5× at N = 4, Heron r2). Clean positive or clean negative
result publishable.

### Falsification

- New entry **C22** in `docs/falsification.md`: "Sync-manifold
  logical qubit outlives a dimension-matched unprotected qubit
  in hardware." Falsifier — coherence-lifetime ratio ≤ 1.0 on
  the benchmark.

---

## S19 — Entanglement phase diagram + magic + Krylov complexity

### Motivation

The project already ships `magic_sre.py` (stabiliser Rényi
entropy), `krylov_complexity.py`, `entanglement_entropy.py`,
`entanglement_spectrum.py`, and the OTOC machinery. What is
missing is the *combined* phase diagram: a single scan of these
measures across (K, $\omega$-spread, topology) producing a
publishable reference dataset for the quantum-chaos /
collective-phenomena community. This is a modest engineering lift
with outsized documentation value.

### Deliverables

1. `analysis/entanglement_phase_diagram.py` — scan orchestrator
   that, given a parameter grid, computes:
   - Multipartite entanglement (negativity, fidelity-based).
   - Stabiliser Rényi entropy (magic).
   - Krylov complexity.
   - OTOC scrambling rate.
   - Spectral form factor.
   For each (K, $\omega$-spread, topology) point.
2. Published dataset: hardware-measured values at N = 4 across
   a 10 × 10 parameter grid, Zenodo DOI.
3. Reference figure: four-panel phase diagram suitable for a
   publication.
4. `docs/entanglement_phase_diagram.md` + paper note.

### Risks

* Multipartite entanglement estimators on hardware are
  shot-hungry; the 10 × 10 grid at N = 4 needs careful shot
  budgeting.
* The "reference dataset" framing only works if the dataset is
  published in a citable form (Zenodo + paper note).

### Prerequisites

* All five component modules — already in repo.
* Phase 2 IBM credits for the hardware column at usable
  statistics.

### Acceptance

Published figure + Zenodo deposit of the four-panel phase
diagram. Paper note fits into the v1.0 release cycle.

### Falsification

- New entry **C23** in `docs/falsification.md`: "Entanglement +
  magic + Krylov complexity show a coherent transition signature
  at $K_c$ consistent across all four measures." Falsifier —
  transition signatures diverge in location by > 20 % of the
  measurement range across measures.

---

## S20 — Quantum Kuramoto universal control-benchmark suite (MLPerf-style)

### Motivation

Optimal quantum-control libraries exist (Qutip's qtrl,
GrapeQuantum, etc.) but none define a standardised, hardware-
validated complex-systems benchmark. Shipping a "control challenge"
harness where users submit arbitrary pulse / ansatz / feedback
strategies and the Rust-accelerated simulator + IBM hardware
runner scores them on sync quality, resource cost, noise
robustness, and DLA expressivity — with a public leaderboard and
reproducible submission manifest — creates a MLPerf-style
community asset.

### Deliverables

1. `benchmark_suite/` subpackage:
   - `submission.py` — schema for user submissions (JSON manifest
     specifying pulse / ansatz / feedback strategy).
   - `scorer.py` — five-axis scoring: sync fidelity, resource
     count, noise robustness, DLA sector expressivity, wall
     time.
   - `leaderboard.py` — CSV / JSON leaderboard with SHA-256
     submission IDs.
2. Public submission UI: GitHub Actions workflow that accepts
   a PR opening a new submission, scores it on a fixed
   classical simulator, and — optionally — promotes it to a
   scheduled hardware run when the CEO approves.
3. Reference submissions: Trotter baseline, ADAPT-VQE,
   GUESS-mitigated Trotter, adaptive-branching (S8). Each
   scored on every axis.
4. `docs/benchmark_challenge.md` — operator manual for
   submitters.
5. Kickoff visibility: announcement on the `@qiskit` Slack /
   Unitary Foundation Discord / r/QuantumComputing (copy
   already drafted in `.coordination/launch_copy/`).

### Risks

* Community adoption is not automatic. The MLPerf comparison is
  aspirational — MLPerf has consortium backing. A small-scale
  version is achievable; consortium scale is not.
* Hardware cost for promoting submissions to real QPUs needs a
  defined budget cap per submitter.

### Prerequisites

* `scpn_quantum_engine` — already in repo.
* `HardwareRunner` — already in repo.
* S5 harness (open-data + validation) is a natural predecessor —
  activate S5 first so S20 can reuse the data-loading APIs.

### Acceptance

Leaderboard live, ≥ 4 reference submissions scored, first three
external submissions merged within 6 months of kickoff. Note:
infrastructure track — no scientific claim, therefore no
falsifier.

---

## S21 — Multi-scale quantum → classical bridging layer

### Motivation

The macroscopic dynamics of a large-N Kuramoto system are
classical (fluid-limit or mean-field); the microscopic dynamics of
an N = 4…20 quantum system are exactly computed. A QPU-computed
"effective K_{ij}" and "effective $\omega$ distribution" — derived
from DLA invariants and hardware-measured fluctuation spectra —
can inject quantum corrections into the classical large-N solver
at the coarse-grained scale where classical solvers saturate. This
is the formal version of the hybrid forecasting engine in S14,
generalised to multi-scale systems (brain connectome → neuron
dynamics; power grid → per-bus oscillator; tokamak → MHD mode
spectrum).

### Deliverables

1. `bridge/coarse_graining.py`:
   - `effective_K_from_quantum(quantum_observables)` — extract
     renormalised couplings from DLA invariants + fluctuation
     spectra.
   - `effective_omega_from_quantum(spectra)` — recover the
     effective frequency distribution.
   - `classical_large_N_hybrid(quantum_effective, classical_solver)`
     — inject the quantum-derived effective parameters into
     the classical solver.
2. Integration with `sc-neurocore` and `scpn-fusion-core` (bridges
   already exist).
3. Three applied demos:
   - Brain-scale (100+ node EEG network) with quantum
     corrections at the N = 4 microscale.
   - Grid-scale (IEEE-118 or larger) with quantum corrections
     at bus-cluster level.
   - Tokamak-scale MHD mode spectrum (ITER benchmark stub).
4. `docs/multi_scale_bridging.md` — theory + protocol.

### Risks

* The coarse-graining operator is not unique; the choice must be
  benchmarked against analytical mean-field predictions to avoid
  post-hoc fitting.
* "Everyone does either pure quantum or pure classical; zero
  hybrid" is an over-claim. Hybrid RG methods exist (classical);
  the novelty is injecting real-hardware quantum corrections,
  not the concept of multi-scale hybrid simulation itself.

### Prerequisites

* DLA invariants, witness machinery, bridges — all in repo.
* Large-N classical Kuramoto solver — available via
  `scpn-fusion-core`.

### Acceptance

On a pre-registered benchmark of brain-scale or grid-scale
Kuramoto, the multi-scale hybrid beats the pure-classical solver
at matched compute budget by ≥ 10 % on a forecasting
mean-squared-error metric.

### Falsification

- New entry **C24** in `docs/falsification.md`: "Quantum-corrected
  multi-scale solver beats pure classical at matched budget on
  brain-scale or grid-scale Kuramoto." Falsifier — pure
  classical ties or wins on the benchmark.

---

## S22 — Non-Hermitian / PT-symmetric quantum Kuramoto + exceptional points

### Motivation

PT-symmetric quantum sync exists in toy theoretical models (2025–2026
on spin oscillators). No hardware realisation is DLA-protected.
Extending `LindbladSyncEngine` with balanced gain-loss jumps — or
auxiliary-qubit dilation of a non-Hermitian Hamiltonian onto a
unitary circuit — opens access to exceptional-point phenomena on
superconducting hardware. Near an exceptional point, small
perturbations produce disproportionately large response — the basis
for exceptional-point-enhanced sensing.

### Deliverables

1. `phase/non_hermitian_kuramoto.py`:
   - Balanced-gain-loss extension of the XY Hamiltonian.
   - EP-locator: sweeps the gain / loss parameter and detects
     eigenvalue coalescence.
   - Drive scheduler: Rust hypergeometric pulse envelope that
     steers the system across an EP.
2. Auxiliary-qubit dilation primitive: convert a non-Hermitian
   evolution to a unitary extended-Hilbert-space evolution for
   execution on a standard QPU.
3. Hardware demo at N = 4 + 2 ancillae: measure the sync order
   parameter across an EP, compare against the Hermitian
   baseline.
4. Sensing demo: EP-enhanced detection of a weak applied signal.
5. `docs/non_hermitian_kuramoto.md` — theory + protocol.

### Risks

* Dilation doubles the required qubit count; N = 4 physics
  needs 6 hardware qubits.
* EP sensitivity amplifies noise along with signal — net
  metrological gain is not automatic.

### Prerequisites

* `LindbladSyncEngine` — already in repo.
* Rust pulse engine — already in repo.
* Dilation primitive (new ~2 weeks).

### Acceptance

EP-enhanced signal-to-noise on a pre-registered weak-signal
benchmark exceeds the Hermitian baseline by a documented factor.

### Falsification

- New entry **C25** in `docs/falsification.md`: "Sensitivity
  enhancement near an engineered exceptional point exceeds the
  Hermitian baseline on the pre-registered benchmark." Falsifier —
  SNR ratio ≤ 1.0 on the benchmark.

---

## S23 — Quantum reservoir computing powered by Kuramoto transients

### Motivation

Quantum reservoir computing (QRC) is emerging. The transient
dynamics of a quantum Kuramoto system (pre-sync evolution, chimera
states, OTOC scrambling) are natively high-dimensional and
nonlinear — the criteria for a useful reservoir. Reading out DLA
parity + sync witnesses on hardware and training a classical
linear layer on top is a minimal-engineering addition that
positions the project as a hardware-validated quantum reservoir
for oscillator-based forecasting.

### Deliverables

1. `qrc/kuramoto_reservoir.py` (new):
   - `KuramotoReservoir` class that runs a pre-configured
     transient evolution and emits a feature vector of
     observables.
   - Training loop: ridge regression / linear classifier on top
     of the feature vectors.
2. Standard QRC benchmarks:
   - NARMA-10 nonlinear time-series prediction.
   - Mackey-Glass chaotic prediction.
   - MNIST-like pattern classification (scaled-down).
3. Hardware demo on `ibm_kingston` at N = 4.
4. Comparison vs. classical echo state network baseline at
   matched feature dimension.
5. `docs/quantum_reservoir.md` — protocol + results.

### Risks

* Reservoir performance depends sensitively on hardware
  coherence; a reservoir that works on the simulator may fail on
  noisy hardware.
* Classical echo state networks are extremely efficient;
  beating them on the same task is a high bar.

### Prerequisites

* `qrc_phase_detector.py` — already in repo (v0.9.1).
* OTOC + witnesses — already in repo.
* Benchmark harness (S5 is a natural prerequisite).

### Acceptance

Kuramoto reservoir matches or exceeds the classical echo state
network on at least one pre-registered benchmark at matched
feature dimension.

### Falsification

- New entry **C26** in `docs/falsification.md`: "Quantum Kuramoto
  reservoir beats classical echo state network on a pre-registered
  time-series benchmark." Falsifier — reservoir underperforms by
  ≥ 10 % in MSE or accuracy on the benchmark.

---

## S24 — Quantum speed limits for collective synchronisation

### Motivation

Quantum speed limits (QSL) — Mandelstam–Tamm, Margolus–Levitin —
are well-characterised for single-qubit gates. The collective-sync
extension combines: DLA dimension as the effective Hilbert-space
dimension, OTOC growth rate as the proxy for the Hamiltonian norm
entering the QSL, and the hardware-measurable sync observable as
the target state. The project already has
`analysis/quantum_speed_limit.py` (v0.9.1); extending it to
collective observables and measuring the saturation on hardware
provides the first experimental QSL certificate for a
many-body-collective-phenomenon target.

### Deliverables

1. Extension of `analysis/quantum_speed_limit.py`:
   - `collective_qsl(K, omega, target_R)` — compute the
     theoretical Mandelstam–Tamm and Margolus–Levitin bounds
     for reaching a target global sync R.
   - DLA-constrained tightening: use the DLA dimension to
     constrain the effective evolution space.
2. Hardware measurement: sweep K values; record actual time to
   reach target $R_*$; compare against the theoretical bound.
3. Published figure: theoretical QSL curve vs. measured time
   across K.
4. `docs/collective_qsl.md` — theory + protocol.

### Risks

* The QSL bound loosens when the observable is global rather
  than single-qubit; the tightest bound for collective R may
  be hard to achieve on hardware within the coherence budget.
* OTOC-based norm proxy may not be tight for the particular
  Hamiltonian class; alternative Hamiltonian-norm estimators
  may be needed.

### Prerequisites

* `quantum_speed_limit.py` — already in repo.
* DLA dimension calculator — already in repo.
* OTOC — already in repo.

### Acceptance

Published figure showing the hardware-measured time-to-sync
saturating the DLA-constrained QSL within a pre-registered
fraction (e.g. within 20 % of the theoretical bound).

### Falsification

- New entry **C27** in `docs/falsification.md`: "Hardware-measured
  time-to-sync saturates the DLA-constrained QSL within 20 %."
  Falsifier — consistent gap > 20 % on the pre-registered
  benchmark, implying either a loose bound or an over-cautious
  hardware drive schedule.

---

## S25 — Topological defects + vortex dynamics on 2D quantum oscillator lattices

### Motivation

Classical Kuramoto on a 2D lattice supports phase vortices; their
creation, annihilation, and motion underpin defect-mediated phase
transitions and the Kosterlitz–Thouless universality class. The
project's `gauge/vortex_detector.py` and `gauge/wilson_loop.py`
modules implement the detection primitives. What is missing:
extend the mapper to a 2D lattice topology, engineer vortex
creation via targeted drives, track their motion, and read out
the winding number on hardware.

### Deliverables

1. Extension of `bridge/knm_hamiltonian.py` — 2D lattice
   topology ingestion with explicit nearest-neighbour K_nm.
2. `phase/vortex_dynamics.py`:
   - `create_vortex_pair(position_a, position_b, strength)` —
     compile a drive that nucleates a ± vortex pair at the
     specified positions.
   - `annihilate_vortex(position)` — corresponding annihilation
     primitive.
   - `track_vortex_motion(readout_series)` — extract vortex
     trajectories from spatially resolved sync witnesses.
3. Hardware demo on a 4 × 4 patch of `ibm_kingston` (16 qubits,
   within the connectivity envelope of Heron r2): create a
   vortex pair, observe its motion under Kuramoto dynamics,
   annihilate.
4. `docs/quantum_vortex_dynamics.md` — theory + protocol.

### Risks

* 16-qubit 2D-lattice dynamics exceed the coherence budget on
  Heron r2 for non-trivial Trotter depths. A scaled-down 3 × 3
  patch may be the realistic target.
* Spatial resolution of the vortex core requires partial-state
  tomography of each local region; shot budget is substantial.

### Prerequisites

* `vortex_detector.py`, `wilson_loop.py`, `universality.py` —
  already in repo.
* 2D-lattice-topology K_nm builder (extension).

### Acceptance

Vortex pair creation and annihilation reproduced on hardware
at matched topological charge. Winding-number measurement on
a hardware-generated vortex agrees with the theoretical
prediction within the measurement uncertainty.

### Falsification

- New entry **C28** in `docs/falsification.md`: "Vortex pair
  creation and annihilation reproducible on hardware at matched
  topological charge." Falsifier — winding-number measurement
  inconsistent with the theoretical charge by more than 1 σ on
  the benchmark.

---

## S26 — Entanglement-mediated long-range synchronisation

### Motivation

Entanglement (Bell pairs, GHZ states, virtual couplings) as a
resource for distributing information across a quantum processor
is well-characterised. Its use as a sync-enhancement resource on
heterogeneous Kuramoto networks is not. Pre-sharing Bell pairs
between distant oscillator subsets and quantifying the boost in
$R_{global}$ vs. the unentangled baseline closes a gap between
the quantum-networks and the quantum-sync literatures.

### Deliverables

1. `phase/entanglement_mediated_sync.py`:
   - `prepare_entangled_subsets(subset_pairs)` — Bell / GHZ
     preparation across the specified qubit pairs or groups.
   - `evolve_with_entanglement(K, omega, entanglement_config)` —
     evolve the full system with the entangled subsets in place.
2. Topology-constrained hardware demo: Bell pairs across the
   diagonal of a heavy-hex patch, then Kuramoto evolution.
3. Quantitative metric: entanglement-boost factor $\Delta R / R_*$
   vs. the classically coupled baseline.
4. `docs/entanglement_mediated_sync.md` — protocol + results.

### Risks

* Superconducting topology limits where Bell pairs can be placed
  (all-to-all entanglement across distant qubits requires SWAP
  overhead that erodes the measured boost).
* Bell-pair infidelity on hardware may dominate the
  entanglement-mediated boost.

### Prerequisites

* Bell-pair preparation primitive — already in repo
  (`crypto/bell_test`).
* Heron r2 topology map (documented).

### Acceptance

Entanglement-mediated sync shows documented gain over the
unentangled classical baseline on a pre-registered hardware
topology.

### Falsification

- New entry **C29** in `docs/falsification.md`: "Entanglement-
  mediated sync exceeds the unentangled-classical baseline on
  the pre-registered hardware topology." Falsifier — gain
  within statistical uncertainty of zero.

---

## S27 — Hardware-in-the-loop inverse design of oscillator networks

### Motivation

The forward compilation "K_nm → circuit → hardware → observables"
is common. The inverse — "target sync pattern → discover K_nm and
$\omega$ that realise it on hardware" — requires a closed
hardware-in-the-loop optimisation. VQE + Rust-accelerated forward
mapper + QPU feedback is the natural architecture. Practical
targets: maximal chimera stability, robustness against a specific
noise model, reproduction of a biological sync pattern observed
in EEG.

### Deliverables

1. `control/inverse_design.py`:
   - `inverse_design_knm(target_pattern, n_osc, objective)` —
     closed-loop optimiser. Accepts a target sync pattern and
     an objective function; returns optimised K_nm and
     $\omega$.
   - Uses `scpn_quantum_engine` for fast classical pre-screening;
     promotes only the top-K candidates to hardware.
2. Three applied demos:
   - Chimera-target: discover K_nm that stably hosts a
     pre-specified chimera configuration at N = 4.
   - EEG-target: reproduce a sync pattern observed in an EEG
     recording on the hardware.
   - Noise-robust target: maximise $R_{global}$ stability under
     a pre-specified Pauli noise model.
3. `docs/inverse_design.md` — protocol + results.

### Risks

* Hardware-in-the-loop optimisation is expensive in QPU time;
  the classical pre-screening must aggressively prune candidates.
* The objective landscape may be non-convex; expect many local
  optima and document convergence statistics.

### Prerequisites

* S1 (feedback loop plumbing) — activate first; S27 reuses the
  same observer architecture.
* `scpn_quantum_engine` — already in repo.
* VQE — already in repo.

### Acceptance

Inverse-designed K_nm reproduces the target sync pattern on
hardware within a pre-registered tolerance on at least two of
the three applied targets.

### Falsification

- New entry **C30** in `docs/falsification.md`: "Inverse design
  reproduces a pre-registered target sync pattern on hardware
  within documented tolerance." Falsifier — fails to converge
  on ≥ 2 of 3 targets within the QPU budget.

---

## S28 — Synchronisation-enhanced distributed quantum metrology

### Motivation

Quantum metrology achieves Heisenberg-limited sensing by using
entangled probe states. The synchronisation transition itself
generates entanglement — a finding already in the repo's
`analysis/qfi_criticality.py` + `entanglement_sync.py`. Deploying
multiple synchronised oscillator subsets as a quantum sensor array,
with DLA-parity-protected readout, converts the sync transition
from a physics demo into a distributed sensing primitive for
applications where the target signal perturbs the coupling matrix
(global magnetic-field gradients, climate-scale oscillator networks,
biological rhythms).

### Deliverables

1. `sensing/distributed_sync_sensor.py`:
   - `SensorArray` dataclass — M synchronised subsets each of
     size n_sub = N / M; readout observables DLA-protected.
   - `estimate_parameter(measurements, prior)` — maximum-
     likelihood estimator over the hypothesis.
2. Hardware demo at N = 8 with M = 2 subsets: detect an
   applied single-qubit phase perturbation below the
   single-subset classical Fisher limit.
3. Scaling analysis: simulate the expected Heisenberg-limited
   scaling vs. M and compare against the single-subset limit.
4. `docs/distributed_sync_sensing.md` — theory + protocol.

### Risks

* Heisenberg scaling requires entanglement coherence time longer
  than the sensing integration time. Heron r2 coherence budget
  caps the usable integration window; the hardware demo may
  show a fraction of the theoretical scaling.
* "Heisenberg-limited on global magnetic fields / biological
  rhythms" is an applied claim that needs a specific
  application-target to be published.

### Prerequisites

* QFI + entanglement-sync modules — already in repo.
* Multi-subset array topology design on Heron r2.

### Acceptance

Distributed sync sensor achieves super-classical Fisher
information on a pre-registered sensing target.

### Falsification

- New entry **C31** in `docs/falsification.md`: "Distributed
  sync sensor achieves super-classical Fisher information on
  the pre-registered target." Falsifier — Fisher-information
  ratio ≤ 1.0 on the benchmark.

---

## S29 — Floquet quantum Kuramoto for discrete time-crystalline order

### Motivation

Floquet time crystals — subharmonic response under periodic drive
— have been realised on homogeneous spin chains. Heterogeneous
Kuramoto oscillators with DLA parity have not been tested as a
time-crystalline platform. The project already ships
`phase/floquet_kuramoto.py` (v0.9.1); the differentiation is
explicit tracking of how the DLA parity protects subharmonic
response across the sync transition on hardware, and how drive
shaping via the Rust hypergeometric engine extends the time-
crystalline lifetime.

### Deliverables

1. Extension of `phase/floquet_kuramoto.py`:
   - `subharmonic_response(drive_period, observable)` —
     measure the expected period-2T response signature.
   - `dla_protected_drive(K, target_subharmonic)` — Rust-
     engineered drive schedule that preserves the DLA sector.
2. Hardware demo at N = 4: measure the subharmonic response vs.
   drive frequency across the sync transition.
3. Lifetime benchmark: compare DLA-protected drive schedule
   vs. naive drive on time-crystalline stability.
4. `docs/floquet_kuramoto.md` — theory + protocol (extends
   existing documentation).

### Risks

* Heterogeneity breaks the symmetry that typically protects
  Floquet time crystals; the DLA protection must be proven to
  survive.
* Hardware coherence on Heron r2 may be insufficient to
  distinguish a short-lived time crystal from a trivial
  subharmonic echo.

### Prerequisites

* `floquet_kuramoto.py` — already in repo.
* Rust hypergeometric pulse engine — already in repo.

### Acceptance

Subharmonic response survives the DLA-protected drive schedule
at a pre-registered heterogeneity threshold, on hardware.

### Falsification

- New entry **C32** in `docs/falsification.md`: "Subharmonic
  response survives DLA-protected drive schedule at heterogeneity
  beyond a pre-registered threshold." Falsifier — response
  decays on the same timescale as the naive-drive baseline.

---

## S30 — Quantum Kuramoto for community detection and modularity optimisation

### Motivation

Classical community detection (Louvain, Infomap, spectral methods)
is mature. Mapping network modularity into the quantum sync
landscape — where each community corresponds to a locally
synchronised cluster — allows the QPU to discover partitions that
classical algorithms miss in chaotic or high-dimensional graphs.
The DLA-invariant structure of the XY Hamiltonian provides a
natural regulariser.

### Deliverables

1. `applications/community_detection.py`:
   - `partition_via_sync(graph, n_quantum_qubits)` — quantum
     community detection using a VQE-minimised modularity
     objective over a hardware-compatible sub-graph.
   - Rust-accelerated modularity evaluation for large classical
     graphs.
2. Benchmarks on standard graph corpora:
   - Zachary's Karate Club.
   - LFR (Lancichinetti-Fortunato-Radicchi) benchmark at
     sizes compatible with hardware limits.
   - Random geometric graphs.
3. Comparison vs. Louvain and Leiden at matched graph size.
4. `docs/quantum_community_detection.md` — protocol + results.

### Risks

* Quantum hardware size (N ≤ 20) limits community-detection
  problems to toy graphs; the "at scale" framing overclaims
  unless restricted to small hard instances.
* Modularity landscape on the QPU may not be meaningfully
  different from the classical one for the graph sizes
  accessible today.

### Prerequisites

* Witnesses + DLA invariants — already in repo.
* Graph-benchmark corpus (open-source datasets).

### Acceptance

Quantum-discovered community partition matches or exceeds
Louvain on a pre-registered hard instance (low signal-to-noise
LFR benchmark).

### Falsification

- New entry **C33** in `docs/falsification.md`: "Quantum community
  detection beats Louvain on a pre-registered hard LFR instance."
  Falsifier — Louvain wins on the hard instance at matched
  compute budget.

---

## S31 — DLA-protected many-body localisation / delocalisation transitions

### Motivation

Many-body localisation (MBL) is a mature quantum-chaos research
area; no connection has been made to heterogeneous Kuramoto sync
or DLA-protected collective phenomena. Engineering disorder in
$\omega$ and coupling topology drives a localisation transition;
mapping the mobility edge via DLA parity + OTOC tools and
quantifying how global sync survives localisation is the
deliverable.

### Deliverables

1. `analysis/mbl_sync.py`:
   - `disorder_sweep(omega_std, K_grid)` — disorder-driven
     localisation sweep.
   - `mobility_edge_detector(spectra, otoc)` — locate the
     localisation-delocalisation transition.
   - `sync_survival_under_localisation(disorder, K)` —
     quantify how $R_{global}$ survives below and above the
     mobility edge.
2. Hardware demo at N = 4 — 6: map a coarse mobility-edge
   phase diagram on `ibm_kingston`.
3. Classical reference: compare against exact-diagonalisation
   of the same disorder realisations at N = 4 — 6.
4. `docs/mbl_sync.md` — theory + protocol.

### Risks

* MBL classification is statistically demanding; hundreds of
  disorder realisations per point on the phase diagram may be
  needed. The QPU budget constrains what is achievable.
* Whether DLA parity really protects collective sync across the
  mobility edge is an open question; the result may be a
  publishable negative.

### Prerequisites

* DLA + OTOC — already in repo.
* Disorder-sweep harness (extension).

### Acceptance

Mobility edge detectable via DLA parity + OTOC on a
pre-registered disorder model; phase diagram published.

### Falsification

- New entry **C34** in `docs/falsification.md`: "Mobility edge
  detectable on hardware via DLA parity + OTOC on a
  pre-registered disorder model." Falsifier — no mobility edge
  signature above statistical noise on the phase diagram.

---

## S32 — Monitored quantum Kuramoto (measurement-induced transitions)

### Motivation

Measurement-induced phase transitions (MIPT) are an emerging
sub-field of quantum dynamics. Weak or projective measurement on
subsets of the system, combined with unitary evolution, gives rise
to entanglement phase transitions (volume-law to area-law) as the
measurement rate varies. Applying this to oscillator networks with
DLA witnesses and heterogeneous Kuramoto dynamics has not been
attempted on hardware. IBM Dynamic Circuits provide the
mid-circuit measurement primitive.

### Deliverables

1. `phase/monitored_kuramoto.py`:
   - `MonitoredKuramotoCircuit(measurement_rate, subset)` —
     insert mid-circuit measurements on a specified subset at a
     controlled rate.
   - Entanglement-entropy estimator compatible with
     measurement-induced collapses.
2. Phase diagram: entanglement vs. measurement rate at fixed
   K; locate the volume-law-to-area-law transition on hardware.
3. Cross-check against sync order parameter: does the MIPT
   track a transition in $R_{global}$?
4. `docs/monitored_kuramoto.md` — theory + protocol.

### Risks

* Dynamic Circuits mid-circuit measurement rates are
  backend-bounded; scanning measurement rate finely may exceed
  the timing budget.
* Entanglement estimation under measurement requires classical
  shadow tomography or ancilla-based methods; shot budget is
  substantial.

### Prerequisites

* Dynamic Circuits support — depends on S1 or S8 landing first.
* Witnesses + shadow tomography — already in repo.

### Acceptance

Measurement-induced transition detectable in the entanglement
vs. measurement-rate plot on hardware at N = 4.

### Falsification

- New entry **C35** in `docs/falsification.md`: "MIPT detectable
  in entanglement vs. measurement-rate plot on hardware."
  Falsifier — no transition signature above statistical noise
  on the pre-registered benchmark.

---

## S33 — Quantum-enhanced Lyapunov spectra for chaotic Kuramoto

### Motivation

Classical Lyapunov-spectrum computation for large-N chaotic
Kuramoto scales poorly. The QPU-computed OTOC scrambling rate is a
natural proxy for the maximum Lyapunov exponent, and
DLA-constrained trajectories give access to collective Lyapunov
modes that pure classical long-time integration cannot reach
within budget. Injecting these into a classical large-N solver as
"quantum corrections" is the specific payoff.

### Deliverables

1. `analysis/lyapunov_spectrum.py`:
   - `collective_lyapunov_from_otoc(otoc_timeseries)` — OTOC
     growth-rate → Lyapunov-exponent extraction.
   - `dla_constrained_lyapunov_spectrum(K, omega)` —
     DLA-sector-resolved Lyapunov spectrum.
2. Classical feedback: injected into the large-N Kuramoto
   solver in `scpn-fusion-core` as correction terms (extension
   of S14).
3. Applied demos: tipping-point prediction on a chaotic IEEE
   grid model; cascade-risk quantification on a brain-
   connectome stability question.
4. `docs/quantum_lyapunov.md` — theory + protocol.

### Risks

* OTOC → Lyapunov conversion is approximate; the proxy may not
  be tight for heterogeneous Kuramoto.
* Tipping-point prediction is a statistically demanding claim
  that needs held-out validation.

### Prerequisites

* OTOC — already in repo.
* DLA — already in repo.
* Classical large-N solver — `scpn-fusion-core`.

### Acceptance

Quantum-extracted Lyapunov spectrum agrees with the classical
truth on a pre-registered small-N benchmark within a
documented tolerance.

### Falsification

- New entry **C36** in `docs/falsification.md`: "Quantum-extracted
  Lyapunov spectrum agrees with classical truth to within
  pre-registered tolerance on the benchmark." Falsifier — gap
  beyond tolerance on ≥ 50 % of benchmark instances.

---

## S34 — Self-organising Kuramoto (autonomous drive engineering)

### Motivation

Every quantum-control protocol in the repo today is externally
driven: a classical controller computes the next drive and
submits it. S34 closes the loop inside the circuit itself — the
measured sync order parameter generates the next drive via
classical feed-forward from the previous shot's results, with no
external parameter tuning. This is the autonomous variant of S1
and S8; the differentiation is explicitly "no external controller
in the loop at steady state".

### Deliverables

1. `control/autonomous_drive.py`:
   - `AutonomousDriveLoop` — closed-loop driver that reads the
     previous shot's observables, computes the next drive via a
     deterministic (classical) rule, and submits without human
     input.
   - Convergence criterion: the loop terminates when the
     measured R saturates within a pre-registered tolerance.
2. Hardware demo at N = 4: show autonomous convergence to a
   target R on an adversarially disturbed Kuramoto trajectory.
3. Comparison against externally tuned baseline (S1 with
   manually specified drives).
4. `docs/autonomous_kuramoto.md` — protocol + results.

### Risks

* Autonomous loops can oscillate or diverge; the deterministic
  rule must have a convergence proof or empirical convergence
  guarantees.
* Overlaps with S1 (hybrid feedback loop) and S8 (adaptive
  branching); positioning needs to be "autonomous variant",
  not "different approach".

### Prerequisites

* S1 + S8 (feedback plumbing).
* Convergence analysis for the deterministic rule (theorist
  pass).

### Acceptance

Autonomous loop converges to target R within a pre-registered
tolerance on hardware without external parameter tuning.

### Falsification

- New entry **C37** in `docs/falsification.md`: "Autonomous drive
  loop converges to target R without external tuning on the
  pre-registered benchmark." Falsifier — loop fails to converge
  or diverges on ≥ 30 % of runs.

---

## S35 — Quantum Kuramoto as native simulator for active matter (non-reciprocal)

### Motivation

Active matter — flocking, swarming, non-reciprocal interactions —
is almost exclusively classical / mean-field in current
literature. A quantum hardware realisation with DLA analysis of
non-reciprocal K_nm (directed couplings) positions the project as
the reference quantum-active-matter platform. Non-reciprocal
K_{ij} ≠ K_{ji} violates Hermiticity; execution requires the
auxiliary-qubit dilation primitive from S22.

### Deliverables

1. `phase/active_matter.py`:
   - Non-reciprocal K_nm ingestion.
   - Dilation compilation (shared with S22).
   - Active-matter-specific observables: flocking order
     parameter, swarming correlation length.
2. Hardware demo at N = 4 on the dilated space (N = 4 + 4
   ancillae): show a sync transition characteristic of
   non-reciprocal dynamics, distinguishable from the Hermitian
   baseline.
3. `docs/quantum_active_matter.md` — theory + protocol.

### Risks

* Dilation doubles the qubit requirement; N = 4 active matter
  demands 8 physical qubits on Heron r2.
* Active-matter signatures (flocking transition) may be
  indistinguishable from trivial asymmetry effects at the
  accessible N; benchmark against classical prediction carefully.

### Prerequisites

* S22 (dilation primitive).
* Non-reciprocal K_nm extension to the mapper.

### Acceptance

Non-reciprocal sync transition distinguishable from the
Hermitian baseline on hardware at N = 4 on a pre-registered
benchmark.

### Falsification

- New entry **C38** in `docs/falsification.md`: "Non-reciprocal
  sync transition distinguishable from Hermitian baseline on
  hardware at matched compute budget." Falsifier — indistinct
  transition signature on the pre-registered benchmark.

---

# Foundational tracks (S36–S53) — compact format

The tracks below are scoped in a compact-but-rigorous format
(motivation, deliverables, risks, prerequisites, acceptance,
falsifier — no long-form narrative). Each still demands the same
activation-gate rigour as S1–S35. On activation, the responsible
session expands the compact entry into the full S1–S35-style form
before execution starts.

**Source archive.** The original full-length source text for every
track below is preserved in
`.coordination/strategic_roadmap_sources/2026-04-18_differentiation_tracks_s36_plus_RAW.md`
(gitignored). Deduplication collapsed seven proposal rounds from
2026-04-18 into this block; tracks that appeared in multiple rounds
are merged with cross-references.

## S36 — Information geometry on quantum sync manifolds

**Motivation.** The manifold of reachable sync states, parameterised
by (K_nm, ω, DLA generators), is a Riemannian manifold with the
quantum Fisher information tensor as natural metric. Geodesics,
sectional curvature, and natural-gradient flows give provably
optimal control paths across the sync transition. Information
geometry exists for single-qubit and simple VQE landscapes; no
application to collective Kuramoto order parameters on hardware has
surfaced in the literature surveyed through 2026-04.

**Deliverables.** `analysis/sync_information_geometry.py` with
Fisher-tensor computation from observables; Rust-side geodesic
integrator; natural-gradient-flow optimal-control demo on N = 4.

**Risks.** Fisher-tensor estimation is shot-hungry; natural-gradient
directions can diverge near singular points.

**Prerequisites.** `analysis/qfi_criticality.py` (done), Rust accel
(done).

**Acceptance.** Natural-gradient path from incoherent to full sync
on hardware beats straight-line Trotter on final R at matched depth
on a pre-registered benchmark.

**Falsification.** **C39**: natural-gradient path beats straight-line
Trotter at matched depth. Falsifier — no measurable R advantage on
the benchmark.

## S37 — Categorical / compositional quantum Kuramoto

**Motivation.** Oscillator networks can be formalised as objects in
a symmetric monoidal category, with sync-preserving DLA-invariant
morphisms as arrows. Compositional composition of sub-networks —
hierarchical SCPN layers — enables modular circuit construction
without exponential growth. Categorical QM is mature but has not
been applied to synchronisation phenomena.

**Deliverables.** `bridge/category_theory.py` encoding network
objects + morphisms; compositional K_nm builder; demo of a 2-layer
hierarchical SCPN network compiled through the category.

**Risks.** Category-theoretic formalism has steep onboarding cost;
practical payoff only at ≥ 3 hierarchical layers (where the
hardware budget constrains the demo).

**Prerequisites.** mapper refactor, theorist pass on the category
definition.

**Acceptance.** 2-layer hierarchical SCPN network compiles to a
circuit of depth ≤ the flat-compilation depth by a documented
fraction.

**Falsification.** Infrastructure track — no scientific claim; no
falsifier.

## S38 — Quantum Kuramoto field theory continuum limit + RG flows

**Motivation.** Large-N limit of the lattice Kuramoto-XY mapping
yields an effective scalar QFT (φ⁴-like with DLA-protected
symmetries). Tensor-network + DLA truncation compresses it. Running
low-energy dynamics on hardware and extracting RG flows of the
order parameter provides the first experimental bridge between
many-body oscillators and genuine QFT phenomenology.

**Deliverables.** `phase/qkft_continuum.py` with tensor-network
compression; RG-flow extractor using coarse-graining from DLA
invariants; hardware demo at N = 4 — 6 measuring effective coupling
renormalisation.

**Risks.** The φ⁴-effective-theory derivation is analytically
demanding; a loose identification of the effective-field parameters
invalidates the RG readout.

**Prerequisites.** quimb tensor-network tier (done), DLA (done),
QuTiP baseline (done).

**Acceptance.** Published RG-flow diagram with a hardware-measured
critical exponent agreeing with the classical Kuramoto mean-field
prediction within a pre-registered tolerance.

**Falsification.** **C40**: hardware-measured RG flow matches
mean-field critical exponent to pre-registered tolerance. Falsifier —
exponent off by > tolerance.

## S39 — Autopoietic / self-referential oscillator networks

**Motivation.** Sync order parameter dynamically rewrites K_nm
without external controller. Extension of S34 (autonomous drive)
where the *coupling matrix itself* is the feedback target rather
than just the drive amplitude. Closed self-maintaining loops
directly model origins-of-life, synthetic biology, and consciousness
patterns as physical realisations inside quantum hardware.

**Deliverables.** `control/autopoietic_loop.py` — closed loop where
measurement at shot n computes the K_nm to use at shot n+1; demo at
N = 4 showing maintained non-trivial sync pattern without external
input.

**Risks.** Autopoietic loops can collapse to trivial fixed points
(R = 0 or R = 1) instead of sustaining a non-trivial pattern;
parameter-regime search needed.

**Prerequisites.** S34 (autonomous drive) + S1 (feedback plumbing).

**Acceptance.** Autopoietic loop sustains a non-trivial sync pattern
(0 < R < 1, chimera-type) for ≥ 20 feedback cycles without external
input.

**Falsification.** **C41**: autopoietic loop sustains non-trivial
pattern for ≥ 20 cycles. Falsifier — collapse to trivial R on ≥ 50 %
of runs on the pre-registered benchmark.

## S40 — Holographic duals via quantum synchronisation

**Motivation.** Boundary oscillator network sync mapped to a bulk
gravitational-like degree of freedom, with DLA invariants as the
holographic dictionary. Measure boundary order parameters on
hardware, extract bulk geometry proxies. AdS/CFT in many-body
quantum sync has not surfaced in the literature surveyed.

**Deliverables.** `analysis/holographic_dual.py` —
boundary-to-bulk map using DLA invariants; hardware demo at N = 4
extracting a bulk-metric proxy from boundary observables.

**Risks.** Holographic interpretation is ambitious. The result may
be unfalsifiable without a consensus holographic dictionary for
heterogeneous Kuramoto-XY; clearly mark any "bulk metric" extracted
as proxy, not derived.

**Prerequisites.** DLA machinery (done), theorist pass on the dual
map construction.

**Acceptance.** Published theory note + hardware-measured bulk-metric
proxy, with the mapping clearly framed as conjectural pending
community validation.

**Falsification.** **C42**: self-consistency of the boundary ↔ bulk
mapping under RG flow (tested in S38 + S40 together). Falsifier —
inconsistency in the RG-flow fixed points between boundary and bulk
at the pre-registered precision.

## S41 — Quantum causal discovery with intervention

**Motivation.** OTOC growth + DLA parity asymmetry + targeted
mid-circuit interventions (conditional drives, projective
measurements) infer causal directionality and hidden couplings in
unknown networks. Extends S16 (network tomography) from passive
observation to active intervention — a genuine do-calculus over
quantum oscillator networks.

**Deliverables.** `analysis/causal_discovery.py` with intervention
scheduler; integrates with Dynamic Circuits (S8 prereq);
applied demo: infer directed EEG connectivity from passive + active
measurements.

**Risks.** Causal discovery is data-hungry; the quantum advantage
over classical do-calculus requires a demonstration on a
pre-registered hard instance.

**Prerequisites.** S16 (observational tomography), Dynamic Circuits
(S8), witnesses (done).

**Acceptance.** Directed connectivity on a pre-registered synthetic
graph recovered within a documented tolerance, beating the best
classical observational-only baseline.

**Falsification.** **C43**: quantum-assisted do-calculus beats
classical observational baseline on pre-registered benchmark.
Falsifier — classical ties or wins.

## S42 — Symplectic structure-preserving Trotterisation

**Motivation.** Almost every quantum mapping of classical dynamics
destroys the symplectic structure of the phase space. Reformulating
Trotter + pulse shaping to exactly preserve symplecticity (geometric
integrator analogue in the quantum domain) enables faithful
long-time simulation of Hamiltonian chaos without artificial
dissipation. No open-source or hardware Kuramoto pipeline enforces
this.

**Deliverables.** `phase/symplectic_trotter.py` — geometric-integrator
variant of the Trotter decomposition; Rust-side implementation that
guarantees symplectic norm preservation; long-time chaos-demo
comparison vs. standard Trotter on a pre-registered chaotic
Kuramoto benchmark.

**Risks.** Symplectic Trotter adds gate count; coherence budget may
not permit the "long-time" demo on current hardware.

**Prerequisites.** Rust pulse engine (done), geometric-integrator
theory pass.

**Acceptance.** Long-time energy / norm drift on the chaotic
benchmark bounded below a pre-registered fraction of the standard-
Trotter drift at matched depth.

**Falsification.** **C44**: symplectic Trotter bounds long-time
drift below standard Trotter at matched depth. Falsifier — drift
exceeds standard Trotter on ≥ 50 % of benchmark instances.

## S43 — Full resource theory of quantum synchronisation

**Motivation.** Formalise synchronisation (sharpness of R + DLA
subspace dimension + witness robustness) as a quantum resource.
Define sync-distillable entanglement, sync cost of gates,
conversion rates between sync and entanglement / magic. Resource
theories exist for entanglement, coherence, magic — not for
collective synchronisation.

**Deliverables.** `analysis/sync_resource_theory.py` formalising
sync as resource; conversion-rate measurement protocol; hardware
demo at N = 4 showing conversion of entanglement → sync and back.

**Risks.** Resource theory framework must be formally sound (free
operations, monotones) before experimental measurement means
anything.

**Prerequisites.** witnesses (done), OTOC (done), DLA (done),
theorist pass on the resource-theory axiomatisation.

**Acceptance.** Theory note + hardware-measured conversion rate with
error bars; paper note fits the v1.0 release cycle.

**Falsification.** **C45**: sync-to-entanglement conversion rate is
non-zero (i.e. sync is a distinct resource from existing ones).
Falsifier — conversion rate consistent with zero on the benchmark.

## S44 — Objective-collapse / macroscopic-foundations testbed

**Motivation.** Merges three related proposals: objective-collapse
models (GRW, Penrose OR, CSL) stress-tested at mesoscopic scales;
Quantum Darwinism — sync manifold as redundant encoding of
"classical" information into environmental degrees; macroscopic
measurement as the quantum-to-classical transition witness. All
three use the same instrument: DLA-parity asymmetry as a
smoking-gun observable for collapse-induced desynchronisation or
redundant classical imprinting.

**Deliverables.** `phase/foundations_testbed.py` with collapse-model
simulator + Darwinism redundancy estimator; scaled-hardware
campaign at N = 4 → 8 measuring DLA asymmetry decay; paper note on
bounds derived for GRW / CSL parameters.

**Risks.** Collapse signals are exponentially small at accessible
N. Darwinism redundancy estimator is shot-hungry. Setting
meaningful bounds may require hardware access far beyond Phase 2
budget.

**Prerequisites.** Large-N hardware time, DLA parity asymmetry
(hardware-validated — done).

**Acceptance.** Published paper with a documented bound on CSL rate
(or GRW parameters) from mesoscopic-scale sync stability on
hardware.

**Falsification.** **C46**: sync stability on hardware places a
bound on CSL collapse rate tighter than the pre-registered
reference benchmark. Falsifier — no tightening beyond reference
within the campaign budget.

## S45 — Biologically faithful Kuramoto simulator + IIT consciousness angle

**Motivation.** Ingest real structural-biology data (protein
interaction graphs, microtubule lattices, photosynthetic antenna
complexes, C. elegans / human connectomes) as K_nm + ω inputs.
Compute Φ (integrated information, IIT) and cause-effect structures
directly from DLA-protected sync manifolds on hardware; compare
against classical baselines + experimental bio-data (2D
spectroscopy, magnetoreception, EEG). Merges the biology-data
ingestion and the IIT-testbed proposals from multiple rounds.

**Deliverables.** `applications/bio_kuramoto.py` with
connectome/microtubule ingestion; IIT Φ estimator on
DLA-protected manifold; applied comparison against 2D
photosynthesis spectroscopy or an EEG integrated-information
dataset.

**Risks.** "Quantum biology" is a contentious field. Over-claims
("quantum consciousness", "quantum coherence in protein folding")
must not appear in the commit messages, paper, or documentation.
The clean deliverable is a hardware-validated quantum-simulator
readout of a bio-sourced coupling matrix — no metaphysical claims
beyond that.

**Prerequisites.** `applications/eeg_benchmark.py` (done),
`applications/fmo_benchmark.py` (done), bio-data licences.

**Acceptance.** Hardware-measured sync signature on a published
connectome (or microtubule / FMO graph) within pre-registered
agreement with classical baseline; IIT Φ estimator returns
documented values with error bars.

**Falsification.** **C47**: hardware-measured sync signature on a
bio-sourced K_nm agrees with classical mean-field within
tolerance. Falsifier — significant disagreement on ≥ 2 of 3
bio-benchmarks.

## S46 — Phase-transition / attractor-landscape quantum programming

**Motivation.** Encode computation directly into the attractor
landscape: incoherent → chimera → partial → full sync phases each
implement different logic or signal-processing primitives, without
explicit gates. All current quantum computing is gate / annealing;
this is "thermodynamic quantum software" where computation emerges
from the physics of sync itself.

**Deliverables.** `control/attractor_programming.py` — target-pattern
→ drive schedule compiler; demo: 2-bit AND / OR implemented as
attractor-selection on N = 4; scaling-of-capacity characterisation.

**Risks.** Limited computational expressiveness at small N;
demonstration of any non-trivial computation beyond what a
classical Kuramoto attractor already provides is required to be
useful.

**Prerequisites.** witnesses (done), floquet_kuramoto (done).

**Acceptance.** A pre-registered non-trivial classical task (e.g.
simple classification) solved via attractor-selection on hardware
at matched or better accuracy than a classical Kuramoto attractor
solver.

**Falsification.** **C48**: attractor-programming beats classical
Kuramoto attractor solver on a pre-registered classification task.
Falsifier — parity or worse performance on the benchmark.

## S47 — Analogue gravity on synchronised oscillator arrays

**Motivation.** Merges relativistic / curved-spacetime metrics,
cosmological phase transitions + baryogenesis + defect formation,
and emergent spacetime from sync. Position-dependent couplings and
drives simulate quantum fields on curved backgrounds (analogue
black-hole horizons, expanding universes) on flat superconducting
hardware. Analogue gravity is mature in optics / BECs — not on
Kuramoto-XY with DLA protection.

**Deliverables.** `phase/analogue_gravity.py` with curved-metric
compiler; baryogenesis-analogue simulation on N = 4 — 6; table-top
Kibble–Zurek defect-density measurement on hardware.

**Risks.** The analogy is qualitative unless a specific
curved-background QFT claim is pre-registered. Resist
over-interpreting hardware results as "quantum cosmology".

**Prerequisites.** S38 (QKFT), `gauge/vortex_detector.py` (done).

**Acceptance.** Published Kibble–Zurek defect-density scaling on
hardware, matching the theoretical prediction for the chosen
analogue-gravity mapping within a pre-registered tolerance.

**Falsification.** **C49**: Kibble–Zurek scaling exponent on
hardware matches theory within tolerance. Falsifier — exponent off
by > tolerance on the pre-registered benchmark.

## S48 — Self-healing qubit fabrics + continuous sync QEC

**Motivation.** Merges two closely related proposals: self-healing
qubit fabrics via engineered Kuramoto sync, and continuous analog
QEC via the sync manifold. Local defects / errors trigger
desynchronisation signals that propagate as corrective feedback
through the network, restoring global sync and coherence without
external classical control. Complements S18 (sync-as-memory) by
treating sync as an error-correction *process* rather than a
stored state.

**Deliverables.** `qec/self_healing_fabric.py` — always-on sync
drive with built-in error-response feedback; hardware demo showing
recovery from a simulated defect on N = 4 — 8.

**Risks.** Continuous feedback loops on hardware face latency
bounds; the "healing time" must be less than the coherence time of
the unhealed qubit fabric.

**Prerequisites.** S1 / S8 feedback plumbing, `qec/` (done), S18
(sync-memory precursor).

**Acceptance.** Measured coherence-extension on a hardware fabric
with injected defects vs. an unhealed baseline, exceeding a
pre-registered factor.

**Falsification.** **C50**: self-healing fabric extends coherence
over unhealed baseline on the pre-registered benchmark. Falsifier
— no documented extension beyond statistical noise.

## S49 — Quantum fluctuation theorems across sync transitions

**Motivation.** Experimentally test the quantum Jarzynski equality,
Crooks fluctuation theorem, and thermodynamic uncertainty relations
by driving across the Kuramoto sync transition and measuring
work/heat distributions via DLA parity witnesses + OTOCs. Refines
S9 (quantum thermodynamics) with the specific fluctuation-theorem
observables.

**Deliverables.** `thermodynamics/fluctuation_theorems.py` —
Jarzynski / Crooks estimators; hardware demo at N = 4 across a
K-sweep spanning the sync transition.

**Risks.** Fluctuation-theorem tails are heavy-tailed; shot-budget
for convergence is substantial.

**Prerequisites.** S9 (thermodynamics framework — prereq), GUESS ✓,
OTOC ✓.

**Acceptance.** Published experimental confirmation of Jarzynski
equality on the Kuramoto transition within a pre-registered
statistical tolerance; work-distribution tails within predicted
bounds.

**Falsification.** **C51**: measured Jarzynski average agrees with
the free-energy-difference prediction within tolerance. Falsifier
— significant deviation on the benchmark.

## S50 — Quantum kernels from sync manifolds (ML)

**Motivation.** Define quantum kernels directly from the inner
product structure of DLA-generated sync states; evaluate on
hardware for classification / regression on real-world complex
time-series. Quantum kernels exist for simple feature maps; none
exploit the geometric structure of heterogeneous Kuramoto sync
manifolds.

**Deliverables.** `ml/sync_kernel.py` with hardware kernel evaluator;
plasma / brain / power-grid time-series classification benchmark.

**Risks.** Kernel methods are a crowded space; demonstrable quantum
advantage on a *pre-registered* benchmark (not post-hoc selected)
is required.

**Prerequisites.** DLA (done), Rust forward mapper (done), ML
dataset access.

**Acceptance.** Kernel SVM using the hardware-measured sync kernel
beats a classical Gaussian kernel baseline on a pre-registered
time-series benchmark.

**Falsification.** **C52**: sync-manifold quantum kernel beats
classical kernel on pre-registered benchmark. Falsifier — parity
or worse on the benchmark.

## S51 — Hayden–Preskill scrambling / black-hole information simulator

**Motivation.** OTOC growth rate + DLA-protected scrambling in
synchronised oscillator arrays to model Hayden–Preskill information
recovery protocols. First table-top Kuramoto-based simulator of
black-hole information dynamics.

**Deliverables.** `analysis/hayden_preskill.py` — recovery-protocol
simulation + hardware observable extraction; demo at N = 4 — 6
measuring a recovery fidelity consistent with Hayden–Preskill
prediction.

**Risks.** Small-N black-hole analogues capture information
scrambling but not full quantum-gravity dynamics; do not overclaim.

**Prerequisites.** OTOC (done), DLA (done).

**Acceptance.** Measured information recovery fidelity on hardware
agrees with Hayden–Preskill prediction within a pre-registered
tolerance for the scrambling time.

**Falsification.** **C53**: recovery fidelity agrees with
Hayden–Preskill within tolerance. Falsifier — significant deviation
from the predicted curve on the pre-registered benchmark.

## S52 — Distributed quantum consensus via global sync (quantum internet)

**Motivation.** Use sharp global order-parameter transition as a
consensus primitive across modular QPUs or quantum-internet nodes.
Distant oscillator subsets synchronise via shared entanglement or
mediated couplings; DLA witnesses certify consensus. Merges the
multiple "quantum internet sync layer" proposals.

**Deliverables.** `hardware/distributed_sync.py` — multi-runner
orchestrator using the existing AsyncHardwareRunner; Bell-pair-
seeded distant oscillator demo on two Heron r2 regions; documented
consensus primitive.

**Risks.** True "quantum internet" requires networking
infrastructure not yet accessible on today's QPUs; the demo can
only use multi-region of a single QPU or two QPUs with manual
entanglement distribution.

**Prerequisites.** S1 async runner (done), Bell-pair preparation
(done in crypto/).

**Acceptance.** Consensus primitive demonstrated across two
independent Heron-r2 regions at a pre-registered consensus-fidelity
threshold.

**Falsification.** **C54**: distant-region consensus fidelity
exceeds classical clock-sync baseline on the pre-registered
benchmark. Falsifier — consensus fidelity ≤ classical baseline.

## S53 — Engineered self-organised criticality

**Motivation.** Tune (heterogeneity, topology, drive) so the system
sits at the critical point of a non-equilibrium phase transition;
measure avalanche statistics, power-law distributions, and
information-processing capacity. SOC is classical / mean-field;
no quantum hardware pipeline engineers or quantifies it in
heterogeneous oscillator arrays.

**Deliverables.** `analysis/self_organised_criticality.py` — SOC
detector + avalanche statistics; tuned-criticality demo on N = 6;
information-processing capacity measurement at criticality.

**Risks.** SOC requires a separation of timescales that small-N
hardware cannot provide unambiguously; the demo may show only SOC
*precursors*, not full SOC.

**Prerequisites.** witnesses (done), OTOC (done), SOC-literature
theorist pass.

**Acceptance.** Documented power-law avalanche-size distribution on
hardware over at least two decades of scale.

**Falsification.** **C55**: avalanche distribution on tuned hardware
follows a power law over ≥ 2 decades. Falsifier — cut-off at ≤ 1
decade on the pre-registered benchmark.

---

# Applied verticals (cross-cutting over S1–S53)

Rounds 9 and 10 of the 2026-04-18 proposal sitting named five
"applied verticals" that do not define new physics tracks but
rather specific application targets any of S1–S53 can be
directed at. These are cross-cutting; they do not get their own
`Sxx` number but are documented here for the activation session
to target.

| Applied vertical | Most relevant physics tracks |
|---|---|
| Fusion plasma stabilisation (ITER disruption forecasting + real-time control) | S1 + S8 + S27 (feedback + branching + inverse design), S41 (causal discovery of plasma mode coupling), S48 (self-healing qubit fabric for control-loop latency) |
| Tipping-point early-warning (power grids, climate, neural seizures) | S14 (hybrid forecasting), S24 (quantum speed limits for early warning), S31 (MBL / tipping precursor), S53 (SOC + avalanche statistics) |
| IIT consciousness testbed (connectomes, microtubules) | S45 (direct) + S21 (multi-scale) + S50 (kernels on connectome sync data) |
| Quantum biology engineering (photosynthesis, protein folding, collective cell) | S45 + S43 (sync-as-resource for bio-simulation) + S10 / S13 (analog + CV platforms) |
| Quantum internet infrastructure | S4 (multi-vendor) + S26 (entanglement-mediated sync) + S52 (distributed consensus) |
| Autonomous AI physicist (discovery engine) | S12 (Bayesian phase-diagram) + S39 (autopoietic) + S58-class concepts from S34 / S53 + S50 ML kernel |

An applied vertical is **not** a separate track; activating an
applied vertical means activating one or more of the physics tracks
listed, with the applied-vertical dataset as the target and the
applied-vertical metric as the acceptance criterion.

---

## Cross-cutting dependencies

Several tracks share prerequisites. If any of these prerequisites
slip, the dependent tracks slip with them.

| Prerequisite | Blocks |
|---|---|
| Phase 2 IBM credits | S1, S2, S4, S8, S9, S11, S12, S15, S16, S17, S18, S22, S23, S24, S25, S26, S27, S28, S29, S30, S31, S32, S33, S34, S35 — essentially every track with a hardware deliverable |
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
| S15 | C19 | DLA-sector scar subspace exhibits longer sync lifetime than generic eigenstates |
| S16 | C20 | Quantum-assisted tomography recovers K_nm to < 10 % MAE at matched shot budget |
| S17 | C21 | Higher-order sync transition distinguishable from pairwise baseline at N = 6 |
| S18 | C22 | Sync-manifold logical qubit outlives a dimension-matched unprotected qubit |
| S19 | C23 | Entanglement + magic + Krylov complexity show coherent transition signature at $K_c$ |
| S21 | C24 | Quantum-corrected multi-scale solver beats pure classical at matched compute budget |
| S22 | C25 | Sensitivity enhancement near an engineered EP exceeds the Hermitian baseline |
| S23 | C26 | Kuramoto reservoir beats classical echo state network on pre-registered benchmark |
| S24 | C27 | Hardware-measured time-to-sync saturates the DLA-constrained QSL within 20 % |
| S25 | C28 | Vortex pair creation / annihilation reproducible on hardware at matched charge |
| S26 | C29 | Entanglement-mediated sync exceeds unentangled classical baseline |
| S27 | C30 | Inverse design reproduces target sync pattern within tolerance on ≥ 2/3 targets |
| S28 | C31 | Distributed sync sensor achieves super-classical Fisher information |
| S29 | C32 | Subharmonic response survives DLA-protected drive schedule beyond heterogeneity threshold |
| S30 | C33 | Quantum community detection beats Louvain on pre-registered hard LFR instance |
| S31 | C34 | Mobility edge detectable via DLA parity + OTOC on pre-registered disorder model |
| S32 | C35 | Measurement-induced transition detectable in entanglement vs. measurement-rate plot |
| S33 | C36 | Quantum-extracted Lyapunov spectrum agrees with classical truth within tolerance |
| S34 | C37 | Autonomous drive loop converges without external tuning on ≥ 70 % of runs |
| S35 | C38 | Non-reciprocal sync transition distinguishable from Hermitian baseline |
| S36 | C39 | Natural-gradient path beats straight-line Trotter at matched depth |
| S38 | C40 | Hardware-measured RG flow matches mean-field critical exponent within tolerance |
| S39 | C41 | Autopoietic loop sustains non-trivial sync pattern ≥ 20 cycles |
| S40 | C42 | Boundary ↔ bulk holographic mapping self-consistent under RG flow |
| S41 | C43 | Quantum-assisted do-calculus beats classical observational baseline |
| S42 | C44 | Symplectic Trotter bounds long-time drift below standard Trotter at matched depth |
| S43 | C45 | Sync-to-entanglement conversion rate non-zero (sync is a distinct resource) |
| S44 | C46 | Sync stability bounds CSL collapse rate tighter than reference |
| S45 | C47 | Hardware sync signature on bio-sourced K_nm agrees with classical mean-field within tolerance |
| S46 | C48 | Attractor-programming beats classical Kuramoto attractor solver on classification |
| S47 | C49 | Kibble–Zurek defect-density scaling exponent on hardware matches theory within tolerance |
| S48 | C50 | Self-healing fabric extends coherence over unhealed baseline |
| S49 | C51 | Measured Jarzynski average agrees with free-energy-difference prediction within tolerance |
| S50 | C52 | Sync-manifold quantum kernel beats classical kernel on time-series benchmark |
| S51 | C53 | Recovery fidelity agrees with Hayden–Preskill prediction within tolerance |
| S52 | C54 | Distant-region consensus fidelity exceeds classical clock-sync baseline |
| S53 | C55 | Avalanche size distribution on tuned hardware follows a power law over ≥ 2 decades |

S1, S2, S5, S6, S20, S37 are infrastructure / engineering tracks;
they have internal acceptance criteria but no scientific claim that
needs falsification.

## Activation checklist (for the future session that picks one up)

Before starting execution on any of S1–S53:

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
