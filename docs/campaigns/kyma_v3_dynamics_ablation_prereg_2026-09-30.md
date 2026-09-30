<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SCPN Quantum Control — KYMA v3 Dynamics Ablation Preregistration -->

# KYMA v3 Dynamics Ablation — Preregistration

Date: 2026-09-30

Status: **frozen before any ablation variant is trained.** The freeze is the commit
that adds this file and `scripts/run_kyma_v3_ablation.py`, once it is on the public
remote; the run starts only after that push. The result will be appended below a
`RESULT` heading; nothing above it changes after training.

0 QPU. Classical oscillator-substrate ablation.

## Scientific question

KYMA v3 (`docs/campaigns/kyma_v3_symbolic_composition_prereg_2026-09-29.md`, PASS:
substrate 1.000 ± 0.000 on the held-out pair against 0.441 ± 0.099 for the best
matched baseline) could not separate the oscillator **dynamics** from the other
architectural priors chosen for realisability: the phase-lattice value encoding, the
gated write stage with reset, and the triadic coupling that can add phases. Which of
them carries the generalisation?

Authority: owner decision 2026-09-30 ("priprav to a spustíme to").

## Unchanged from v3

Task, split (2,112 training items; held-out pair `(R0, R1)`, query `a`, 64 items),
labels, loss (`1 − cos`), initialisation (`N(0, 0.3²)` per gate, seeded), optimiser
(full-batch Adam, learning rate 0.05, **3,000 epochs**), seeds **0–4**, readout
(nearest lattice value), chance floor 0.25. The frozen v3 modules are imported, not
modified; this commit adds only the runner, its tests and this file.

## Variants (frozen)

| variant | what changes | trainable parameters | hand-set realisability (all 2,304 items) |
|---|---|---:|---:|
| `full` | nothing: the frozen v3 write stage (control) | 108 | 1.000 |
| `phasor` | **no dynamics**: each write stage sets the destination phase to the stable fixed point of the frozen stage dynamics, `θ_i = arg(Σ_j K e^{i(x_j+α)} + Σ_p T e^{i(x_{p1}+x_{p2}+β)})`; no integration, no reset | 108 | 1.000 |
| `pairwise` | frozen dynamics **without the triadic term** (triadic gates fixed at 0, not trained) | 54 | 0.875 (pairwise coupling cannot add phases) |
| `short` | frozen dynamics with **5 RK4 steps** per stage instead of 60 (`T = 0.25`) | 108 | 0.458 at the hand coupling 2.0; 0.995 at 10 and 1.000 at 20 |

Basis of `phasor`: the frozen right-hand side equals `|Z| sin(arg Z − θ)` with `Z` as
above, whose stable fixed point is `arg Z`; the runner's test integrates the frozen
dynamics for 4,000 steps and matches the phasor phase to 1e-3 rad. `full` training in
the runner is bit-identical to the frozen `substrate.train` (tested).

## Measurements

Per variant and seed: held-out accuracy on the 64 test items (primary), training
accuracy, training wall time. Reported as mean ± population SD over seeds 0–4. No
energy figure is computed.

## Interpretation rules (the contract; no PASS/FAIL)

- **R0 — validity.** `full` must reproduce the v3 per-seed held-out accuracies
  (all five seeds 1.000). If it does not, every comparison below is made against this
  run's `full` and the mismatch is reported first.
- **R1 — dynamics.** If `phasor` mean is within **10 percentage points** of `full`
  mean, the v3 generalisation is attributed to the phase-arithmetic structure that the
  stage's fixed point already contains (phase encoding, operation gating, triadic phase
  sum), **not** to oscillator dynamics; KYMA Part B must not name the dynamics as the
  mechanism. If `phasor` is more than 10 points below `full`, the integration dynamics
  contribute beyond their fixed-point map within this task's scope, and that may be
  stated with this probe's limits.
- **R2 — triadic prior.** If `pairwise` is within 10 points of `full`, the triadic
  term is not necessary (contradicting its realisability bound); otherwise the
  phase-sum prior is necessary for this task.
- **R3 — settling.** If `short` is within 10 points of `full`, the long integration is
  not needed; otherwise it is reported as a limitation of short integration.

The rules apply whatever the numbers are. v3 baselines are quoted for context only
and are not re-run.

## Compute and environment

Host: ML350 (approved compute host); one `systemd-run --user` unit pinned to at most 12
cores; outputs under `~/`, then copied into
`data/kyma_v3_symbolic_composition/kyma_v3_dynamics_ablation.json`. Environment: the
pinned `requirements-ci-py312-linux.txt` plus `requirements-ci-jax-py312-linux.txt`
(jax 0.10.1, CPU), as for v3. Command:
`python scripts/run_kyma_v3_ablation.py --commit <freeze commit>` (all four variants,
seeds 0–4, 3,000 epochs). Expected run time about 2–3 hours.

## Amendments

Any change before training is a new commit labelled as an amendment in this file,
pushed before the run. No change after training starts.

## Reporting

The result is reported whichever way it falls, with R0–R3 applied, to the owner and
the CEO, and appended below. Part B quotes no partial or assumed number.

Seat: 90ad

## RESULT

Appended 2026-10-01 after training. Nothing above this heading changed (SHA-256 of the file before this section:
`360c6fbcb3c356ff4b2fc9b35ffa91d1d8351a8e4d258c0a5eb4ccacb9f9443b`).

**Run.** Source commit `d739e6b01b7d51ad35227e1ea77361a194b10005` (this freeze; runner SHA-256
`d9cfb54f93424401b9139c70f92d7be3f462e986d4e8c9f6195f4b6a3f9bab0e`), host ML350 (`god-of-the-math`), one
`systemd-run --user` unit pinned to cores 0–11, Python 3.12.3, jax/jaxlib 0.10.1 (CPU), NumPy 2.4.6. Started
2026-09-30T13:27:18Z, finished 2026-09-30T23:23:44Z (9 h 56 min wall, 13 h 20 min CPU). Artefact
`data/kyma_v3_symbolic_composition/kyma_v3_dynamics_ablation.json`, SHA-256
`b9f56cc10b125eb348efedeb833d6ccdcab1a068d6feadf73c790f80c715b529` (identical on the host and in the repository).

**Interrupted first attempt.** An identical run started 2026-09-30T09:57:20Z was stopped on the owner's instruction at
12:32:45Z for an electrical inspection of the host, after 3 h 40 min CPU time. It wrote no output and nothing from it
was observed; the run above is a fresh start of the same command on the same frozen tree.

**Run time.** The estimate of 2–3 hours above was wrong: `full` needed 73 min per seed, not about 12.

| variant | held-out accuracy (mean ± SD, seeds 0–4) | per seed | training accuracy (mean) | parameters | mean training wall time per seed |
|---|---|---|---:|---:|---:|
| `full` | **1.000 ± 0.000** | 1.000, 1.000, 1.000, 1.000, 1.000 | 0.9995 | 108 | 4,390 s |
| `phasor` | **1.000 ± 0.000** | 1.000, 1.000, 1.000, 1.000, 1.000 | 1.0000 | 108 | 29 s |
| `pairwise` | **0.253 ± 0.006** | 0.250, 0.250, 0.250, 0.266, 0.250 | 0.8883 | 54 | 2,350 s |
| `short` | **0.984 ± 0.031** | 1.000, 1.000, 1.000, 0.922, 1.000 | 0.9898 | 108 | 388 s |

Chance is 0.25; 64 held-out items per seed.

**R0 — validity: holds.** `full` reproduces the v3 per-seed held-out accuracies (all five seeds 1.000; no mismatched
seed). All comparisons below are against this run's `full`.

**R1 — dynamics: the phasor variant is within 10 points (gap 0.0).** By the frozen rule, the v3 generalisation is
attributed to the phase-arithmetic structure that the stage's fixed point already contains (phase encoding, operation
gating, triadic phase sum), **not** to oscillator dynamics. KYMA Part B must not name the dynamics as the mechanism.
The fixed-point map trains about 150 times faster than the integrated stage on this task.

**R2 — triadic prior: necessary.** `pairwise` is 74.7 points below `full`; its held-out accuracy is at chance while its
training accuracy (0.888) sits at its hand-set realisability bound (0.875 on all items). Without the triadic phase sum
the model fits what pairwise coupling can express and does not generalise to the held-out pair.

**R3 — settling: long integration not needed.** `short` (5 RK4 steps per stage) is 1.6 points below `full`, within the
10-point margin. One seed (3) reached 0.922. The hand-set realisability of `short` at the hand coupling 2.0 was 0.458;
training reached 0.984, so the trained gates evidently found couplings under which five steps settle closely enough.
The trained coupling values were not inspected in this probe.

**Limits.** One task, one held-out pair, five seeds, classical simulation on CPU; no energy figure. The result
separates the stage's fixed-point map from its integration dynamics within this task only; it makes no statement
about other tasks, hardware oscillators or quantum execution.
