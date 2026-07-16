# KT-4 — Continuous-Relaxation Layout Search: Research Design (RESEARCH LABEL)

**Status: RESEARCH — open falsifiable question. Not a promised feature. The
honest expected outcome may be "modest or no gain" over the KT-3 discrete
baseline. No claim promotion without KT-5 (isolated reserved-host benchmark,
owner-gated).**

Date: 2026-07-16. Author seat: SCPN-QUANTUM-CONTROL/claude-7f6b.
Plan lane: `docs/internal/strategy/2026-07-15T2223_frontier_hardware_programme_plan.md`, KT-4.

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

Authored by Anulum Fortis & Arcane Sapience (protoscience@anulum.li)
Seat: 7f6b
