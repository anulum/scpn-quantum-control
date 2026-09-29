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
