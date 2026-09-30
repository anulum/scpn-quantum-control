<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SCPN Quantum Control — KYMA v3 Symbolic Composition Preregistration -->

# KYMA v3 Symbolic Composition Probe — Preregistration

Date: 2026-09-29

Status: **frozen before any model is trained.** The freeze is the commit that adds
this file and pushes it to the public remote; the code it names
(`src/scpn_quantum_control/benchmarks/kyma_v3/`, `scripts/run_kyma_v3_probe.py`)
is in the same commit. No substrate or baseline has been trained on this task.
The result will be appended below a `RESULT` heading; nothing above it changes
after training.

0 QPU. This is a classical oscillator-substrate probe.

## Scientific question

Does the gated-coupling oscillator substrate still generalise to a held-out
combination of learned operations when the ground truth is produced by a
symbolic program rather than by oscillator dynamics? (KYMA Part B assumption A2,
MS1 criterion family.)

v2 (`docs/campaigns/kyma_v2_composition_probe_2026-07-21.md`, PASS) used an
oscillator teacher, so its task lived inside the substrate's hypothesis class.
v3 removes that objection: no oscillator, integrator or model is in the label
path.

Authority: the KYMA v3 probe specification (seven requirements);
owner decision A9 (probe GO; a negative is reported with its diagnosis) and
owner decision B3 of 2026-09-29 (option (a): evaluate the held-out pair on
query `a` only and state that restriction here).

## Ground truth and split

Three registers `(a, b, c)` with values in `Z4`. Operations:

- `R0`: `(a, b, c) → (b, c, a)`;
- `R1`: `a ← (a + b) mod 4`;
- `R2`: `b ← (b + 1) mod 4`.

A configuration is one operation or an ordered pair applied left to right
(3 singles + 9 ordered pairs). An item is (initial state, configuration,
queried register); its label is the final value of the queried register.

- **Training:** every configuration except the held-out pair, all 64 states,
  all three queries: 11 × 64 × 3 = **2,112 items**.
- **Held-out test:** the ordered pair **`(R0, R1)`**, all 64 states, **query `a`
  only: 64 items per seed.** The pair's queries `b` and `c` are neither trained
  nor evaluated.
- **Restriction (owner decision B3):** query `a` is the only query of the pair
  whose answer is not a function of the single-operation answers. For queries
  `b` and `c` the held-out answer equals the `R0`-alone answer, so they cannot
  test composition and are excluded.

Every query register appears in every trained configuration.

## Teacher-free design checks (recorded before training)

Computed by `kyma_v3.task.design_report()` and `kyma_v3.probe.realisability_accuracy()`
from the symbolic program and hand-set gates only:

| check | value |
|---|---|
| every configuration is a bijection on `Z4^3` | yes |
| label counts per (configuration, query) | exactly 16 per class (all 36) |
| held-out states whose answer the single-op answers do not fix: query a / b / c | **1.00** / 0.00 / 0.00 |
| minimum Hamming distance of the held-out answer vector (`b + c`) to any trained (configuration, query) answer vector | **48 of 64** |
| measured training-marginal chance floor on the test items | **0.25** |
| hand-set substrate, all 12 configurations × 64 states × 3 queries | **2,304 / 2,304 exact**; worst phase error 0.026 rad against a 0.785 rad margin |

The last row shows the task is realisable inside the substrate class. It is a
validity check only: training never sees the hand-set gates.

## Substrate (frozen)

Each register is one oscillator; its phase relative to a fixed reference
(phase 0) encodes the value, `v ↦ v·π/2`. Two banks of three oscillators. One
operation is one **write stage**: the destination bank is reset to the
off-lattice phase `π/4`, the source bank receives no coupling (it is held), and
the destination integrates

    dθ_i/dt = Σ_j K[o,i,j] sin(x_j − θ_i + α[o,i,j])
            + Σ_p T[o,i,p] sin(x_{p1} + x_{p2} − θ_i + β[o,i,p])

with directed pairwise couplings `K` and lags `α` (source register `j` → target
`i`), and triadic couplings `T` and lags `β` over the three source-register
pairs `p ∈ {(a,b), (a,c), (b,c)}`. The operation code `o` gates which couplings
act. The banks then swap roles; an ordered pair is two write stages with the
gates of each operation in program order. Readout: the queried register's final
phase, rounded to the nearest lattice value.

- Integrator: fixed-step RK4, `dt = 0.05`, **60 steps per stage** (`T = 3`).
- Trainable parameters: `K, α, T, β` for 3 operations × 3 targets × 3 sources or
  pairs = **108**.
- Initialisation: every parameter drawn from `N(0, 0.3²)` with the run seed.
- Loss: mean `1 − cos(φ_q − label·π/2)` of the queried register's final phase.
  Only the final queried value is supervised.

**Stated limitation.** Two architectural choices were made for realisability, not
from any model's performance: the staged schedule (one write stage per
operation) and the triadic term (pairwise phase coupling cannot add phases,
which `R1` requires). Because the stages are applied in program order, a
substrate that learns each operation exactly composes the held-out pair by
construction. The probe therefore tests whether gradient descent learns
reusable operation gates from end-of-program labels alone, and the diagnostic
baselines below test whether staging alone, without oscillator dynamics, does
as well.

## Baselines (frozen)

Contract baselines, parameter count within ±10 % of the substrate (108):

| baseline | architecture | width | parameters |
|---|---|---|---|
| MLP | one tanh hidden layer over `sin/cos` of the three register phases, one-hot first op, one-hot second op (with "none") and one-hot query (16 inputs) | 5 | 109 (+0.9 %) |
| staged GNN | register nodes; per-operation learned 3×3 adjacency and node bias; embedding, then one message-passing round per operation in program order; read at the queried node | 4 | 103 (−4.6 %) |
| transformer | one single-head attention layer with residual over tokens `[first op, second op, query, a, b, c]`; learned embeddings and positions; read at the query token | 3 | 100 (−7.4 %) |

Diagnostic baselines (reported; they never change the verdict):

| diagnostic | purpose | width | parameters |
|---|---|---|---|
| sequential MLP | one small MLP per operation maps `sin/cos` phases to `sin/cos` phases, applied in program order; label = queried angle rounded to the lattice; loss as the substrate's | 2 | 96 (−11.1 %, outside ±10 %; diagnostic only) |
| MLP, large | capacity control | 64 | 1,348 |
| staged GNN, large | capacity control | 16 | 703 |
| transformer, large | capacity control | 16 | 1,348 |

Chance floor: always predicting the most frequent training label (ties to the
smallest), measured on the test items (0.25 by the design check).

No baseline receives intermediate states or any privileged information.

## Training (frozen)

Full-batch Adam on the 2,112 training items for every model; **3,000 epochs** for
every model; learning rates fixed in advance without search: substrate 0.05, MLP
0.02, staged GNN 0.01, transformer 0.01, sequential MLP 0.02. Seeds **0, 1, 2, 3,
4**. No early stopping, no model selection and no tuning on the held-out items.
Training accuracy is reported for every model and seed.

## Decision procedure (the contract)

Primary statistic: held-out accuracy (fraction of the 64 test items classified
correctly), per model and seed, reported as mean ± population standard deviation
over the five seeds.

**PASS** iff both hold:

1. substrate mean ≥ (best contract-baseline mean) + **10 percentage points**; and
2. substrate mean − substrate sd > the measured chance floor.

Otherwise **NEGATIVE**, reported with a diagnosis (training accuracy of each
operation's single-op items, per-seed results, where composition failed).

Design-selection seed: the design checks use no random draws, so no seed was
used for design; all five seeds count and the "excluding the design seed"
robustness check is vacuous. It is stated here so it is not read as omitted.

**Pre-declared attribution rule.** If the staged GNN or the sequential MLP reaches
within 10 percentage points of the substrate mean, any Part B wording must
attribute the generalisation to staged operator application, which
non-oscillator staged models share, and not to oscillator dynamics
specifically. If a large-capacity diagnostic reaches within 10 points, the
matched-budget margin is stated as budget-dependent. These rules apply whatever
the verdict.

Exploratory (labelled as such if reported): anything beyond the per-seed
accuracies, training accuracies and the attribution flags above.

## Energy

For every model: J/task = declared nominal package power of the host CPU × the
measured wall-clock seconds per test item of a warm inference pass. This is an
**energy proxy**, not a measurement; no power meter is read, and it is not
evidence for oscillator-hardware frugality. The declared power and the host are
recorded in the artefact.

## Compute and environment

Host: ML350 (approved compute host). Before the run: health check, `uptime`,
other projects' running units; at most 12 threads; a `systemd-run --user` unit;
outputs under `~/`, then copied into the repository. Environment: the pinned
`requirements-ci-py312-linux.txt` plus `requirements-ci-jax-py312-linux.txt`
(jax 0.10.1, CPU).

## Analysis and data

- Analysis: `scripts/run_kyma_v3_probe.py`, which calls
  `scpn_quantum_control.benchmarks.kyma_v3.probe.run_probe` and applies the
  contract in `probe.verdict`, frozen at the freeze commit. The artefact records
  the source commit.
- Data: `data/kyma_v3_symbolic_composition/kyma_v3_symbolic_composition.json`;
  archived with the next Zenodo release.

## Amendments

Any change before training is a new commit labelled as an amendment in this
file, pushed before the run. No change after training starts.

## Reporting

The measured result is reported whichever way it falls, with the attribution
flags. The CEO receives this file's path and SHA-256 at freeze, and the result
or truthful "in progress" wording by 12 October 2026. Part B quotes no partial
or assumed number.

Seat: 90ad

## RESULT

Appended 2026-09-30 after the run; nothing above this heading has changed since the freeze commit `ddae107dc`
(file SHA-256 before this section: `9248cf8a4fe9bfaf4d4807133f86dd9da51fac821fb71f70a3e8932eb232d1a5`).

**Verdict under the frozen contract: PASS.**

- Run: ML350 (hostname `god-of-the-math`), `systemd-run --user` unit pinned to cores 0–11, JAX CPU; source commit
  `dbcb0b0778221f85723ab7f60118c27e6369260f`, a descendant of the freeze commit with **zero** changes under
  `src/scpn_quantum_control/benchmarks/kyma_v3/`, `benchmarks/kyma_v2/` and `scripts/run_kyma_v3_probe.py` (it adds
  only CI registrations and evidence digests). Started 2026-09-29T19:01:33Z (freeze pushed 17:11Z), finished
  2026-09-30T01:25:17Z; 9 h 08 min CPU time. Artefact
  `data/kyma_v3_symbolic_composition/kyma_v3_symbolic_composition.json`, SHA-256
  `dd27ae8953d660626abbd23f23a7ea9c5c13123fa93280bef092982104f90cb8` (identical on the host and in the repository).
- Design checks in the artefact match the table above: 2,112 training items, 64 test items, ambiguity by query
  1.00 / 0.00 / 0.00, minimum distance 48, uniform label counts, hand-set realisability 1.000.

Held-out accuracy on the pair `(R0, R1)`, query `a`, mean ± population SD over seeds 0–4 (training accuracy in
brackets):

| model | params | held-out | training |
|---|---:|---:|---:|
| **substrate** | 108 | **1.000 ± 0.000** | 0.9995 ± 0.0009 |
| MLP (contract) | 109 | 0.256 ± 0.041 | 0.677 ± 0.012 |
| staged GNN (contract) | 103 | 0.441 ± 0.099 | 0.943 ± 0.027 |
| transformer (contract) | 100 | 0.253 ± 0.015 | 0.504 ± 0.002 |
| sequential MLP (diagnostic) | 96 | 0.247 ± 0.025 | 0.495 ± 0.025 |
| MLP, large (diagnostic) | 1,348 | 0.234 ± 0.043 | 1.000 ± 0.000 |
| staged GNN, large (diagnostic) | 703 | 0.306 ± 0.021 | 1.000 ± 0.000 |
| transformer, large (diagnostic) | 1,348 | 0.250 ± 0.000 | 0.496 ± 0.008 |

Every seed of the substrate classified all 64 held-out items correctly (seed 2 missed 5 of 2,112 training items).

**Contract.** (1) 1.000 ≥ 0.441 + 0.10: margin over the best contract baseline (staged GNN) **+55.9 percentage
points**. (2) 1.000 − 0.000 > chance floor 0.25. Both hold → PASS.

**Attribution rule (pre-declared).** Staged GNN within 10 points of the substrate: **no** (0.441). Sequential MLP:
**no** (0.247). Large-capacity diagnostics within 10 points: **no** (best `gnn_large` 0.306). The generalisation is
therefore not attributed to staged operator application alone, and the matched-budget margin is not stated as
budget-dependent within the budgets tested.

**What the result supports and what it does not.** As stated in the frozen limitation, the staged schedule makes a
substrate that learns each operation exactly compose the held-out pair by construction. The result shows that
gradient descent learned the three operation gates from end-of-program labels alone, and that parameter-matched and
larger non-oscillator models, including two with the same staging, did not generalise to the pair. It does not
separate the oscillator dynamics from the other architectural priors chosen for realisability (phase-lattice value
encoding, reset phase, triadic coupling that can add phases); that separation would need an ablation not
pre-registered here.

**Energy proxy (not a measurement).** Declared nominal package power 190 W × measured warm wall time per test item,
on cores shared with other projects' jobs on the host: substrate 1.68 J/item (8.8 ms/item, RK4 simulation
on CPU); contract baselines 0.003–0.043 J/item; all baselines ≤ 0.050 J/item. The simulated substrate costs 33–536
times more per item than the baselines on this CPU. This says nothing about oscillator hardware.

**Exploratory (labelled; not part of the contract).** The matched-budget contract baselines under-fit the training
set (MLP 0.68, transformer 0.50), while the large MLP and large staged GNN fit it exactly and still stay near chance
on the held-out pair (0.23, 0.31): the baselines' held-out failure is not explained by capacity alone.
