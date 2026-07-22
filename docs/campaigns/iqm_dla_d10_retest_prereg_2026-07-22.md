<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SCPN Quantum Control — IQM DLA d10 Sign-Replication Preregistration -->

# IQM Garnet Depth-10 Sign-Replication Preregistration

Date: 2026-07-22

Committed BEFORE any of its own hardware data exist. Execution requires the
readiness gates below plus an explicit owner per-submit GO (granted in
session for this campaign, recorded here for the audit trail: owner
directive "priprav a zadokumentuj všetky dokonale a následne spusti FU-1").

## Motivating (already-collected) evidence

Across the executed powered block and depth-profile campaigns the pooled
relative parity-leakage asymmetry on layout `[2, 7, 12, 13]` is positive at
depths 4, 6, 8, and 12 but **negative at depth 10** (−0.049; absolute
even-minus-odd difference −0.0237 at 4,096 shots per arm). The sign pattern
(+, +, +, −, +) places the negative point between positive neighbours,
which suggests an execution-window fluctuation — but the depth-profile
campaign's committed per-depth power (±0.076 relative MDE) cannot decide
this: |−0.049| is below that MDE.

> Does the depth-10 negative sign replicate in a fresh execution window at
> a power matched to the observed effect size?

Either answer is publishable: replication revives the zero-crossing
reading of the depth profile; failure to replicate (bounded null)
strengthens — without proving — the window-fluctuation reading.

## Circuit Matrix

| Field | Value |
|-------|-------|
| Device | IQM Garnet (20 qubits, square lattice) via Resonance |
| `n` | `4` |
| Circuit family | committed `iqm_dla_pinned_n4_d10_{even,odd}` builders (bit-identical to the powered block) |
| States | `0011` even, `0001` odd |
| Depth | `10` only |
| Repetitions | `8` per state at `1024` shots → `8,192` shots per arm |
| Readout states | `0011`, `0001`, `0000`, `1111` at `2048` shots (fresh window) |
| Layout | `[2, 7, 12, 13]` pinned; fallback `[9, 4, 3, 8]` recorded before submission if unavailable |

Circuit count: `16` main + `4` readout = `20` circuit executions.
Shot count: `16,384` main + `8,192` readout = `24,576` shots.

**Batching disclosure (frozen):** all 16 main executions are submitted as
ONE batched job and the 4 readout circuits as a second job (cost
discipline: ~2 jobs). The 8 repetitions are therefore execution-order
replicates within one queue slot, NOT independent windows; the
cross-window comparison is provided by the 2026-07-21 d10 data themselves.
The per-repetition drift table is still reported.

## Endpoints and Decision Rule (frozen)

Power (committed before execution): at 8,192 shots per arm the one-sided
minimum detectable absolute even-minus-odd difference at α = 0.05 and 90 %
power is 0.0228 — matched to the observed −0.0237 (the depth-profile
campaign's 4,096-shot arms could not resolve this).

- **Primary (sign replication):** pooled across the 8 repetitions,
  one-sided two-proportion z-test of `leak_even < leak_odd` at depth 10
  (i.e. the NEGATIVE asymmetry replicates), α = 0.05.
  - Rejects → the depth-10 negative sign is real in this window too; the
    zero-crossing reading of the depth profile revives.
  - Fails to reject → bounded null at 90 % power for the observed effect:
    no evidence of a negative asymmetry at depth 10 in a fresh window.
    Combined with the positive signs at depths 4/6/8/12 this STRENGTHENS
    the window-fluctuation reading, but does not prove it; the report says
    exactly that.
- **Secondary S1:** two-sided test at depth 10 (direction-agnostic) with
  Wilson 95 % intervals per sector.
- **Secondary S2 (cross-window):** comparison of this window's depth-10
  even-minus-odd difference with the 2026-07-21 value (two-proportion
  z-test on the difference of differences), cross-window caveat attached.
- **Secondary S3:** mean total leakage versus the 2026-07-21 value 0.469
  (equilibration consistency).
- Per-repetition drift table (execution-order replicates, as disclosed).
- Raw-count treatment identical to the executed powered-block primary;
  readout calibration circuits committed alongside.
- No coherent-dynamics claim in any branch (exact statevector baseline
  pins noiseless leakage at zero; AUD-6).

## Live Readiness Gates (block submission until all pass)

- `IQMFakeGarnet` dry run of all 20 circuit executions from committed code;
- transpiled depth envelope: the May reference (d10 → 159) + 25 % = 198;
- pinned-layout calibration check on the day's live snapshot; fallback
  substitution recorded before submission if required;
- `garnet:mock` end-to-end submit/retrieve integration check (zero spend).

## Budget and Stop Rules

- Expected cost by the observed batching pattern: ~2 jobs ≈ 2 credits of
  the ~13 visible; single-block campaign — no further submissions under
  this preregistration.
- Abort before retrieval only if submission errors leave the matrix
  incomplete; a partial matrix is reported as such, never silently topped
  up.

## Claim Boundary

Blocked regardless of outcome: quantum advantage; coherence protection;
coherent-dynamics parity claims; extrapolation beyond the sampled device,
calibration window, layout, and depth; any modification of frozen
submissions under `paper/submissions/`. Results extend the
`submission_006` draft (or its successor) only. IQM and IQM Resonance are
credited in every resulting output.
