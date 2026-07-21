# KYMA v2 composition probe — corrected design, PASS — 2026-07-21

**Status:** run complete. **Verdict: PASS** (pre-committed contract, 5 seeds). Addresses the two
defects diagnosed by the v1 NEGATIVE. **Code:** `src/scpn_quantum_control/benchmarks/kyma_v2/`,
tests `tests/test_kyma_v2_*.py`, runner `scripts/run_kyma_v2_composition_probe.py`, artifact
`data/kyma_v2_composition_probe/kyma_v2_composition_probe.json`. **Pre-registration (frozen before
the run):** `.coordination/planning/CEO/KYMA_V2_PROBE_PREREGISTRATION_7f6b_2026-07-21.md`. 0 QPU
(classical differentiable-Kuramoto probe).

## What v1 proved and why v2 exists

v1 (`KYMA_TOY_PROBE_PREREGISTRATION_7f6b_2026-07-18.md`, commit `2f67de12`) returned an honest
NEGATIVE and named two design defects that made the probe unable to test the "motifs compose"
claim, independent of training quality:

1. **The substrate could not realise both relations at once** — a single shared symmetric coupling
   `K`, conditioned only through an additive frequency drive, cannot be both attractive (in-phase)
   and frustrated (anti-phase); it plateaued at single-motif `R ≈ 0.5/0.4`.
2. **The target was a linear lookup from the code** — a parameter-matched MLP composed it at 100 %
   with no dynamics, so the `≥ pp-over-MLP` bar was structurally unreachable.

The deeper reading (load-bearing for v2): **a param-matched MLP composes *any separable* target.**
If the two relations evolve dynamically independently, the MLP learns each relation's state from its
single-relation trials and the combining function from the *other* training conjunctions, then
applies both to the held-out conjunction — perfect compositional generalisation with no dynamics.
The only way a physical substrate can beat it is if the answer is a genuinely non-factorisable
function of the two relations — they must *interact* before readout.

## The two fixes

* **Fix 1 — per-relation coupling gating.** The code gates the coupling itself,
  `K_eff(code) = K_base + Σ_{r,p} code[r,p]·ΔK[r,p]`, each `ΔK[r,p]` masked to cluster-pair `p`.
  One substrate now realises in-phase on one pair *and* anti-phase on a disjoint pair simultaneously.
* **Fix 2 — non-separable, data-dependent readout.** A **passive readout node** (no motif) is
  bridged to one oscillator of a held-out-R1 cluster and one held-out-R2 cluster; its final phase is
  a non-factorisable, `θ0`-dependent function of *both* relations, quantised into a 4-way label
  (chance 25 %). The substrate composes because the physics composes; a param-matched MLP must
  extrapolate a non-separable map to an unseen combination.

## Mechanism-only design check (§5, teacher dynamics only — no model, no test-accuracy peeking)

The pre-run sanity check fixed `g_sync = 0.5, dt = 0.1, steps = 40, k_bridge = 0.8, n_bins = 4` and
**rejected two drafted mechanisms before any training** (recorded in the pre-registration):

- a *uniform ambient* coupling strong enough to be non-separable **destroys the anti-phase motif**
  (attraction fights frustration) → replaced by a sparse readout bridge touching no motif edge;
- a readout *inside* a locked cluster is pinned, and one coupled to *both* clusters of an anti-phase
  pair sees the two π-apart branches **cancel** (anti-phase is mean-field-invisible) → replaced by a
  passive node reading a single coherent branch of each relation.

Frozen diagnostics: realisability R1 = R2 = **1.00**, non-separability **0.403** (min over relations;
drop-R1 0.403, drop-R2 0.457), class balance max **0.27** — all §5 gates met **without lowering any
target**.

## Result (5 seeds, frozen contract)

| model | held-out-conjunction accuracy | params |
|---|---|---|
| **gated student substrate** | **80.1 % ± 3.0 %** | 336 |
| parameter-matched non-motif MLP | 36.9 % ± 1.5 % | 361 (+7.4 %) |
| chance (most-frequent training class) | 24.2 % ± 2.0 % | — |

Per-seed substrate: 0.80, 0.75, 0.81, 0.84, 0.80. **Margin over MLP +43.1 pp** (contract ≥ 20 pp).
PASS on all three conditions (≥ 60 % AND ≥ 20 pp over MLP AND above chance), all five seeds.

**Selection is task-validity-only (unbiased comparison).** The frozen constants
(`g_sync 0.5, dt 0.1, steps 40, k_bridge 0.8, n_bins 4`) were selected on seed 0's **teacher-only**
realisability / non-separability / balance (§5) — using **no student or MLP performance** — so the
substrate-vs-MLP comparison is not biased by the selection: both models train on the same fixed task.
The result is **stable including vs excluding the selection seed**: excluding seed 0, substrate
80.1 % ± 3.4 % (vs 80.1 % ± 3.0 % including it), MLP 37.3 %, margin +42.8 pp (vs +43.1 pp) — the
PASS does not depend on the seed the mechanism was tuned on.

**J/task (measured, CPU simulation):** substrate 4.6 J (high sd — JIT-recompile wall-time noise on
the memory-constrained host), MLP 1.3 × 10⁻³ J. The substrate is *costlier* on CPU simulation; this
is reported honestly and is **not** evidence of frugality — KYMA's frugality claim concerns
neuromorphic oscillator hardware, not a CPU Kuramoto simulation, and this number must not be cited
for it.

## Interpretation (bounded claim — carry verbatim into Part B §1.2.2a)

*When the ground truth is compositional phase-locking, the gated oscillator substrate generalises
from single-relation trials to an unseen conjunction (80 % over 5 seeds) where a parameter-matched
non-motif MLP does not (37 %, a +43 pp gap).* Because the ground truth is generated by an oscillator
teacher, the task lives in the substrate's hypothesis class, so this is **not** a claim that
oscillators beat MLPs on arbitrary tasks — it is a fair test of the KYMA inductive-bias claim, and
the fair-test guarantee rests entirely on the parameter-matched MLP control, so the +43 pp margin
**is** the claim. Nothing forces gradient descent to find a compositional gated solution rather than
overfitting the seen conjunctions; that it generalises is the measured evidence. v1 (`2f67de12`,
NEGATIVE) stays the honest baseline — its two defects are exactly what v2's two fixes remove.

Authored by Anulum Fortis & Arcane Sapience (protoscience@anulum.li)
Seat: 7f6b
