# KT-4 — Continuous-Relaxation Layout Search: Research Design (RESEARCH LABEL)

**Status: RESEARCH — ANSWERED (2026-07-16). The preregistered experiment ran
(§6): the null hypothesis stands — the relaxation shows no gain over the KT-3
discrete baseline at matched true-cost budget. The research label stays; the
discrete optimiser remains the production recommendation; no KT-5 promotion
case exists on this evidence.**

Date: 2026-07-16. Author seat: SCPN-QUANTUM-CONTROL/claude-7f6b.

> **Location note.** Originally committed (2026-07-16, commit `9f7407ff`)
> under `docs/internal/research_synthesis/`; relocated here unchanged because
> the repository policy keeps the `docs/internal` tree untracked. The commit
> history carries the preregistration timestamp.

## 1. The open question (falsifiable)

Does a Sinkhorn/Gumbel-softmax continuous relaxation over qubit-placement
logits — annealed to a discrete layout with coupling-map feasibility
projection — find layouts with a **lower true KT-2 cost** (and hence a higher
calibration-priced R proxy) than KT-3's multi-restart best-improvement hill
climbing, **at a matched budget of true-cost evaluations**?

Null hypothesis (what falsifies the value of KT-4): at equal true-cost
evaluation budget across a preregistered seed set, the relaxed-then-rounded
search does not beat the discrete baseline's mean best cost.

## 2. Literature grounding (verified at source, 2026-07-16)

Foundations of the relaxation (all arXiv listings fetched this session):

- **Gumbel-Sinkhorn networks** — Mena, Belanger, Linderman, Snoek, *Learning
  Latent Permutations with Gumbel-Sinkhorn Networks*, arXiv:1802.08665
  (ICLR 2018). The Sinkhorn operator as a differentiable analogue of the
  matching/argmax over permutations; the direct template for relaxing a
  placement (an injective assignment) into a doubly-stochastic matrix.
- **Gumbel-softmax** — Jang, Gu, Poole, *Categorical Reparameterization with
  Gumbel-Softmax*, arXiv:1611.01144 (ICLR 2017); and the simultaneous
  **Concrete distribution** — Maddison, Mnih, Teh, arXiv:1611.00712
  (ICLR 2017). Temperature-annealed continuous relaxations of categorical
  sampling; the annealing schedule template.
- **SABRE baseline** — Li, Ding, Xie, *Tackling the Qubit Mapping Problem for
  NISQ-Era Quantum Devices*, arXiv:1809.02573 (ASPLOS 2019). The
  depth-oriented heuristic KT-3 already benchmarks against.

Adjacent discrete-optimisation formulations of placement (no gradient flow):
QUBO qubit allocation (arXiv:2009.00140), MaxSAT mapping-and-routing
(arXiv:2208.13679), Ising-machine compilation ISAAQ (arXiv:2303.02830).

**Novelty statement (bounded).** The searches run this session found
Gumbel-softmax used for differentiable quantum *architecture* search
(QuantumDARTS, OpenReview jGYxcXSg8C) and discrete/annealer formulations of
placement, but **no published Sinkhorn-relaxed qubit-placement optimiser**
against a fidelity-aware cost. The question appears open; absence of prior
art in these searches is evidence of absence in the searched venues only.

## 3. Method sketch

1. **Relax** the placement of `n` logical onto `m` candidate physical qubits
   as a doubly-stochastic matrix `P = Sinkhorn(logits / τ)` (square via
   padding when `m > n`).
2. **Differentiable cost surrogate.** The true KT-2 cost's depth term is
   routing-derived (non-differentiable: `transpile`). Surrogate: expected
   SWAP-distance load `E_P[Σ_{i<j} K_ij · d(p_i, p_j)]` with `d` the
   coupling-map graph distance — continuous in `P`, correlates with routed
   SWAP overhead; Trotter-error and infidelity terms enter unchanged
   (layout-independent at fixed problem/region). Gradient flow via
   `diff.value_and_grad` over the logits.
3. **Anneal** τ down a fixed schedule; **round** with Hungarian assignment;
   **project** to coupling-map feasibility (candidate-set membership,
   injectivity).
4. **Honest evaluation**: every candidate the relaxation proposes is scored
   with the **true seeded KT-2 cost** (`kuramoto_layout_cost` with
   `seed_transpiler` bound — the KT-3 reproducible landscape). The comparison
   metric is true cost, never the surrogate.

## 4. Preregistered comparison protocol

- Baseline: `optimise_kuramoto_layout` (KT-3), same seeds, same candidate
  regions, same weights, same `t/reps/order`.
- Budget match: the relaxation may call the true cost at most as many times
  as the baseline's `n_evaluations` on the same instance.
- Instances: the committed two-cluster topology (KT-3 artifact reference:
  dynq+kuramoto_opt depth 98, success proxy 0.8063, 22 evaluations,
  converged) plus preregistered seed sweep (seeds 0..9) and at least one
  larger region (m ≥ 2n) where relocations dominate.
- Decision: report mean±spread of best true cost per budget; the research
  label stays unless the relaxation wins consistently AND KT-5 (isolated
  host) confirms; "modest/no gain" is a publishable, honest outcome and gets
  recorded in the docs as such.

## 5. Implementation plan (next session)

- New module `hardware/kuramoto_layout_relaxation.py` (RESEARCH label in the
  module docstring), numpy-only Sinkhorn (no torch dependency), reusing
  `kuramoto_layout_cost` + `optimise_kuramoto_layout` for the baseline arm.
- New benchmark surface extending `benchmarks/layout_method_comparison.py`
  with a `sinkhorn_relaxation` method row (same honest labels).
- Tests to 100% line+branch (pure numpy — tracer-safe); strict mypy; docs
  `dynq_qubit_mapping.md` §7.6 + §8.5; the three src-file gates.

*(Delivered as preregistered: the optimiser in
`hardware/kuramoto_layout_relaxation.py`, the budget-matched research row in
`benchmarks/layout_method_comparison.py`, and the sweep in
`benchmarks/layout_relaxation_experiment.py` +
`scripts/run_layout_relaxation_experiment.py`.)*

## 6. Outcome (2026-07-16, measured)

The preregistered protocol of §4 ran in full — seeds 0..9 on the two-cluster
topology (both arms in the DynQ region) plus one full-device instance
(`m = 8 = 2n`), the relaxation's true-cost budget bound per instance to the
discrete baseline's `n_evaluations`:

- **wins/ties/losses = 0/5/6**; baseline mean best cost 97.098 ± 0.514,
  relaxation 99.734 ± 4.245 (population std).
- **The null hypothesis stands: no gain.** The relaxation never produced a
  lower true cost than the discrete optimiser; on the full-device instance it
  lost by +15 despite a 208-evaluation budget.
- Consequences (as preregistered): the research label stays, the discrete
  optimiser (`dynq_qubit_mapping.md` §7.5) remains the production
  recommendation, and no KT-5 promotion case exists on this evidence.

Measured table and honest reading: `dynq_qubit_mapping.md` §8.5; artifact:
`data/layout_relaxation_experiment/layout_relaxation_experiment_n4_seeds0-9.json`.

Authored by Anulum Fortis & Arcane Sapience (protoscience@anulum.li)
Seat: 7f6b
