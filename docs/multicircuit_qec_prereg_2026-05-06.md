<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SCPN Quantum Control — Multi-Circuit QEC Preregistration -->

# Multi-Circuit QEC Demonstration Preregistration

Date: 2026-05-06

This preregistration defines the logical-error metrics required before any
physics-aware multi-circuit QEC demonstration is promoted. It does not submit IBM
jobs, reserve backend time, or authorise QPU spend.

## Scientific Question

Can physics-aware decoding or circuit grouping reduce a preregistered logical
failure metric relative to an unencoded or conventionally decoded baseline for
the same small Kuramoto-XY observable?

## Claim Boundary

Supported after successful execution and analysis:

- logical-error-rate comparison for a specified toy code/circuit family;
- decoder-ablation evidence for a physics-aware feature;
- negative result showing that the QEC overhead dominates for current NISQ
  circuits.

Blocked even after a positive result:

- fault tolerance;
- scalable QEC;
- quantum advantage;
- universal logical protection;
- biological or consciousness-adjacent QEC interpretations;
- use of a postselected metric as a logical-error-rate reduction unless
  discarded-shot cost is reported.

## Required Baselines

Every demonstration must include:

- unencoded physical circuit baseline;
- encoded circuit with standard decoder or majority-vote baseline;
- encoded circuit with physics-aware decoder or grouping rule;
- optional postselected variant, reported separately from deterministic
  decoding.

No baseline may be removed after seeing outcomes.

## Logical Metrics

Primary metric:

```text
logical_failure_rate =
    P(decoded_logical_observable differs from exact target by more than tolerance)
```

Default observable tolerances:

| Observable | Tolerance |
|------------|-----------|
| parity survival | `0.02` absolute |
| exact-state retention | `0.02` absolute |
| magnetisation-sector survival | `0.03` absolute |
| selected Pauli correlator | `0.03` absolute |

Secondary metrics:

- logical observable bias;
- logical observable variance;
- syndrome rate;
- decoder abstention or postselection discard rate;
- physical-to-logical circuit overhead;
- calibration-weighted two-qubit error load.

Postselection rule:

- report postselected logical failure and retained-shot fraction together;
- do not compare postselected logical failure against deterministic baselines
  without a cost-adjusted summary.

## Offline Readiness Matrix

Default no-QPU readiness scope:

| Field | Value |
|-------|-------|
| Codes | repetition code or smallest available surface-code toy instance |
| Families | DLA parity pair; optional FIM sector-survival pair |
| Noise models | ideal, depolarizing, amplitude damping/readout-biased model where available |
| Decoders | baseline majority/MWPM where applicable, physics-aware variant |
| Seeds | at least `20` fixed Monte Carlo seeds |
| Target observables | parity survival, retention, or sector survival |

Readiness outputs:

- encoded and unencoded circuit depth;
- physical qubit count;
- two-qubit gate count;
- syndrome count and measurement count;
- logical failure estimate with confidence interval;
- decoder runtime;
- postselection retained fraction if used.

## Promotion Gates

A hardware QEC demonstration may be considered only if:

- offline logical failure is lower than unencoded and standard-decoder baselines
  under at least one realistic noise model;
- encoded circuit overhead does not exceed the hardware depth ceiling;
- the physics-aware feature beats an ablated decoder by a statistically
  reportable margin;
- the target observable remains the same across all baselines;
- postselection, if used, retains at least 70 % of shots or is clearly labelled
  as a diagnostic only.

## Optional Hardware Scope

If offline gates pass and QPU execution is separately approved, use a minimal
demonstration:

| Field | Value |
|-------|-------|
| Logical circuit families | one DLA parity pair only |
| Baselines | unencoded, encoded standard decoder, encoded physics-aware decoder |
| Repetitions | `6` per baseline/state |
| Shots | `4096` |
| Readout/syndrome calibration | minimum required by the selected code |

Circuit ceiling: `<= 180` circuits.

IBM-reported QPU-time ceiling: `15` minutes.

Do not run FIM or tomography extensions in the first QEC block.

## Live Readiness Gates

Before any hardware submission:

- regenerate all circuits and decoder configuration from committed artefacts
  only;
- live-transpile all circuits on the selected backend/layout;
- reject if encoded circuit depth exceeds unencoded depth by more than the
  preregistered overhead ceiling;
- reject if physical qubit count or routing makes the logical comparison
  backend-infeasible;
- record backend, calibration timestamp, code, decoder, syndrome circuits,
  circuit count, shot count, depth summary, two-qubit gate summary, and
  estimated QPU minutes;
- get explicit approval immediately before submission.

## Analysis Plan

Primary analysis:

- compute logical failure rate with binomial or bootstrap confidence interval;
- compare unencoded, standard-decoder, and physics-aware-decoder baselines;
- report physical overhead and postselection cost;
- report whether any logical gain survives cost normalization.

Required ablations:

- physics-aware decoder with feature disabled;
- deterministic decoder result if postselection is used;
- same observable and same target state across baselines.

## Falsification Rules

The QEC demonstration claim is rejected or downgraded if:

- encoded overhead dominates and worsens logical failure;
- physics-aware decoder does not beat standard decoder or its own ablation;
- improvement appears only under unrealistic noise;
- postselection explains the entire gain;
- logical metrics are not tied to a clear promoted observable.

Negative results are valid and should be reported as NISQ QEC-overhead boundary
evidence.

## Output Artefacts

Expected paths after offline readiness:

- `data/phase3_multicircuit_qec/qec_readiness_<date>.json`;
- `data/phase3_multicircuit_qec/qec_decoder_rows_<date>.csv`;
- `data/phase3_multicircuit_qec/qec_resource_rows_<date>.csv`;
- `docs/phase3_multicircuit_qec_readiness_<date>.md`.

Expected paths after approved hardware execution:

- `data/phase3_multicircuit_qec/qec_counts_<backend>_<timestamp>.json`;
- `data/phase3_multicircuit_qec/qec_summary_<date>.json`;
- `docs/phase3_multicircuit_qec_manifest_<date>.md`.

Every artefact must include code identity, decoder settings, target observable,
noise model or backend target, circuit metadata, syndrome metadata, confidence
intervals, raw counts where applicable, SHA256 hashes, and reproduction
commands.

## Submission Boundary

This preregistration is complete. Hardware execution remains blocked until
offline logical-metric readiness, backend selection, budget confirmation, and
explicit approval are completed in a separate task.
