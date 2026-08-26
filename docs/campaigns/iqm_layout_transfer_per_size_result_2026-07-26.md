<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SCPN Quantum Control — IQM Garnet Per-Size Layout-Transfer Result -->

# IQM Garnet Per-Size Layout-Transfer Result

Execution date: 2026-07-26

This is the result record for the design and decision rules frozen in
`iqm_layout_transfer_per_size_prereg_2026-07-22.md`. The run used calibration
set `c2097be4-1e23-49bc-adaa-8e8c01df6223` and the committed submission
package at `692c915a6ecb272f2083d4007c5d9ec350dc456e`.

## Execution evidence

- IQM Garnet mains job: `019f9c79-1d4b-79e3-a590-aa180819d930`
  (36 circuits at 2,048 shots).
- IQM Garnet readout job: `019f9c79-21e0-7eb2-86e1-69c47fd82715`
  (6 circuits at 1,024 shots).
- Retrieved matrix: 42/42 labels and 79,872/79,872 shots.
- Provider-transpiled two-qubit depth: 40 for every one of the 36 main
  circuits; the frozen per-size depth-parity ratio is therefore 1.0.
- Analysis: frozen seed `20260722`, 10,000 bootstrap resamples.

The zero-spend `garnet:mock` provider integration completed the same two-job
split and retrieval before the hardware call. Its synthetic readout counts
produced a non-positive correction denominator, so mock analysis failed
closed as intended; mock counts are not scientific evidence. QPY generated
under Qiskit 2.4.1 emitted a compatibility warning in the isolated Qiskit
2.1.2 environment, but loading, mock execution, live submission, and live
retrieval all completed successfully.

## Frozen primary endpoint

The corrected-error difference is `default − optimised`, so a positive value
favours the calibration-aware optimiser and a negative value favours the
default placement.

| Size | Difference | Bootstrap CI95 | Holm-adjusted p | Direction |
|------|-----------:|----------------|----------------:|-----------|
| 8 | +0.01868 | [+0.00755, +0.02968] | 0.001200 | optimiser advantage |
| 12 | −0.08728 | [−0.10090, −0.07459] | 0.000600 | optimiser disadvantage |
| 16 | +0.04964 | [+0.04109, +0.05823] | 0.000600 | optimiser advantage |

All three per-size differences reject zero after the frozen Holm correction.
This does **not** mean that the optimiser wins at all sizes: the n=12 effect
is significant in the opposite direction and reproduces the motivating
n=12 reversal at the powered shot budget.

## Secondary endpoints and sensitivity

- Pooled default-minus-optimised difference: −0.00632, bootstrap CI90
  [−0.01187, −0.00099]. The pooled sign is dominated by the n=12
  disadvantage and must not replace the per-size result.
- Cross-size heterogeneity: Cochran Q=292.430 on 2 df,
  p=`3.16e-64`; the homogeneous-effect model is rejected.
- The raw-count sensitivity analysis preserves every primary direction:
  n=8 +0.01642, n=12 −0.02260, n=16 +0.04378, with all three CI95 intervals
  excluding zero.
- Default-minus-naive corrected differences are negative at every size
  (n=8 −0.25877, n=12 −0.20541, n=16 −0.02934), so default placement has
  lower sampled error than the preregistered naive chain in this window.

## Interpretation boundary

The powered per-size run resolves a strongly heterogeneous, size-dependent
layout effect on one IQM Garnet calibration window. It supports neither a
quantum-advantage claim nor a cross-device generalisation. The result
specifically rejects a simple
"calibration-aware placement always helps" account: it helps at n=8 and n=16
but is materially worse than default placement at n=12 under the frozen
observable and correction model.

Canonical artefacts are under
`data/iqm_layout_transfer_per_size/live_2026-07-26/`:

- `live_submission_2026-07-26.json`;
- `live_counts_2026-07-26.json`;
- `live_analysis_2026-07-26.json`.
