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
