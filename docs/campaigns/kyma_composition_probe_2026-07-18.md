# KYMA toy compositional-generalisation probe — 2026-07-18

**Status:** completed, honest NEGATIVE result with a mechanistic diagnosis.
**Pre-registration (frozen, no metric shopping):**
`.coordination/planning/CEO/KYMA_TOY_PROBE_PREREGISTRATION_7f6b_2026-07-18.md`.
**Raw artifact:** `data/kyma_composition_probe/kyma_composition_probe_result.json`.
**Code:** `src/scpn_quantum_control/benchmarks/kyma/` (`dynamics`, `task`, `models`,
`probe`); runner `scripts/run_kyma_composition_probe.py`; tests
`tests/test_kyma_{dynamics,task,models}.py`.

## Question

Do reusable Kuramoto **motifs** — a learned in-phase relation (R1) and a learned
anti-phase relation (R2) on cluster pairs — **compose** on a held-out conjunction
never seen jointly in training, and does a Kuramoto-dynamics substrate compose
better than a parameter-matched non-motif MLP baseline?

## Design (as pre-registered)

- **Substrate:** 16 oscillators (4 clusters of 4). A single trainable symmetric
  coupling `K` (16×16, zero diagonal) plus an input-conditioned natural-frequency
  drive `ω(input)` — an additive per-`(relation, pair)` embedding. Fixed RK4,
  horizon `T = 6`. The readout is the **achieved** order parameter after
  integrating the dynamics.
- **Relations:** R1 drives a cluster pair to in-phase (`R → 1`); R2 drives a
  cluster pair to anti-phase (`R → 0`).
- **Compositional split:** train on single-relation trials + all disjoint
  conjunctions except one held-out `(R1-on-AB, R2-on-CD)`; test on that held-out
  conjunction only. Its constituent single relations *are* trained.
- **Success (frozen):** `|1 − R_AB| ≤ 0.15` AND `R_CD ≤ 0.15`.
- **Baselines:** a parameter-matched (±10 %) MLP reading raw initial phases +
  input code and predicting the readout directly (no dynamics); and a measured
  structure-blind random chance floor.
- **PASS iff:** held-out accuracy ≥ 70 % AND ≥ 25 pp above the MLP AND above
  chance, over 5 seeds (mean ± sd).

## Result — NEGATIVE

The motif substrate does **not** clear the pre-registered bar. Exact
mean ± sd over the 5 seeds, the parameter-matched MLP and chance-floor numbers,
and J/task for both models are in the raw artifact. The qualitative outcome was
stable across x64/float32 and initialisation/epoch variations:

- **Substrate held-out-conjunction accuracy ≈ 0 %**, and it does not cleanly
  realise even the *single* motifs on the training set (in-phase order parameter
  plateaus around 0.5, anti-phase around 0.4 — neither near its target).
- **Parameter-matched MLP ≈ 100 %.**
- **Chance floor ≈ 0.5 %.**

## Diagnosis (the load-bearing content for WP1)

1. **Architectural conflict, not undertraining.** A single shared symmetric
   coupling driven only by per-relation *frequency* offsets cannot simultaneously
   support in-phase locking (needs attraction) and anti-phase locking (needs the
   two clusters held π apart) on the *same* cluster pairs. The substrate settles
   on a compromise that realises neither relation to tolerance, so it cannot
   compose them.
2. **The task, as encoded, does not isolate composition.** The additive input
   code is linearly decodable, so a parameter-matched MLP predicts the target
   readout at ~100 % *without any dynamics*. This makes the pre-registered
   "≥ 25 pp above the MLP" criterion mathematically unreachable (the maximum is
   100 %) — the comparison cannot distinguish motif-composition from
   target-prediction.

## Concrete design fixes (feed-forward to WP1)

- Let the drive **gate the coupling** (per-relation modulation of `K`), not only
  the natural frequencies, so one substrate can realise opposing relations.
- Use a **non-linearly-decodable** readout/encoding so a generic MLP cannot
  shortcut the compositional structure.

Both an honest positive and this honest negative were pre-committed as usable;
this negative gives the bid a concrete, mechanistic requirement rather than a
demonstration.

Authored by Anulum Fortis & Arcane Sapience (protoscience@anulum.li)
Seat: 7f6b
