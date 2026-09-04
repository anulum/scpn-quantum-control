# IQM DLA window variability: calibration-epoch amendment

**Status:** prospective amendment after observation window 7

**Original preregistration:** `iqm_dla_window_variability_prereg_2026-07-22.md`

**Effective for provider decisions:** before any observation-window 8 submission

## Why this amendment exists

The frozen protocol used a wall-clock gap of at least 12 hours to define a new
window. That gap is an operational spacing rule, not evidence that two hardware
observations are statistically independent. The retained calibration evidence
makes the issue concrete: observation windows 2 and 3 were approximately 12 hours apart but both
used calibration set `c2097be4-1e23-49bc-adaa-8e8c01df6223`.

This amendment does not rewrite the preregistration or its completed observation windows 1 through 7
primary result. It adds a mandatory, separately labelled sensitivity analysis
and changes how subsequent observation windows 8 through 10 are counted for inference.

## Units and counting rules

1. A **nominal window** remains one complete, crash-safe 36-circuit execution.
   It remains the unit of provider custody, budgeting, and the original frozen
   analysis.
2. A **calibration epoch** is the set of nominal windows carrying the same exact
   provider `calibration_set_id`. Windows with an identical ID MUST belong to
   the same epoch regardless of elapsed wall-clock time.
3. A changed calibration-set ID establishes a distinct operational epoch. It is
   not claimed to prove complete physical independence; slow drift can cross
   calibration boundaries. The epoch analysis is therefore a sensitivity
   analysis, not a retroactive replacement primary endpoint.
4. Multiple nominal windows inside one epoch are **technical replicates**. Pool
   their raw leaked and total counts by arm before estimating each depth's
   even-minus-odd difference and binomial variance. They contribute one epoch,
   hence one inverse-variance observation and no extra heterogeneity degree of
   freedom.
5. Report the original window-level result and the epoch-pooled result together.
   Neither may be selected or suppressed based on which p-value is preferable.

## Prospective observation-window 8 through 10 scheduling

The fixed 12-hour delay is removed as a sufficiency claim. A fully prepared
window may run whenever its simulator, custody, calibration, layout, budget,
claim, and explicit owner-GO gates pass. If its calibration-set ID matches an
existing epoch, it is recorded as a technical replicate and does not increase
the inferential epoch count. If the scientific purpose is to add a between-epoch
observation, wait for evidence of an actual calibration-set change rather than
waiting an arbitrary number of hours.

The original cap of ten nominal windows remains. Do not spend credits merely to
make an epoch count equal ten, and do not exceed ten nominal windows or the
35-credit cap without a new prospective protocol. If observation window 10 closes with fewer
than ten distinct epochs, report that fact directly.

## Epoch-pooled sensitivity analysis

At each depth, technical replicates in epoch `e` pool to arm proportions

`p_even(d,e) = sum(leaked_even) / sum(shots_even)` and
`p_odd(d,e) = sum(leaked_odd) / sum(shots_odd)`.

The epoch effect is `Delta(d,e) = p_even(d,e) - p_odd(d,e)`, with the same
plug-in binomial variance formula as the frozen analysis using the pooled arm
totals. The companion analyser then applies:

- d10 Cochran Q across calibration epochs at alpha 0.05;
- d4, d8, and d12 epoch-level Q tests with Holm correction;
- descriptive DerSimonian-Laird tau estimates by depth; and
- descriptive d4 sign stability across epochs.

The epoch analysis becomes interpretable at six distinct epochs, matching the
original minimum-unit threshold. A more elaborate nested random-effects fit is
not promoted at the current data shape: through observation window 7 only one epoch has more than
one nominal window, so a separate within-epoch random-effect variance is weakly
identified. Raw-count pooling is the prespecified conservative treatment until
enough repeated epochs exist to support a new prospective hierarchical model.

## Current frozen mapping through observation window 7

The seven nominal windows form six calibration epochs. Observation windows 2
and 3 are the only technical-replicate pair; observation window 1 and
observation windows 4 through 7 each have distinct calibration-set IDs.
This mapping and both analysis tracks are materialised by
`scripts/analyse_iqm_dla_window_variability_epochs.py` in
`data/iqm_paper_replication/iqm_dla_window_variability_calibration_epoch_sensitivity_through_observation_window_07_2026-09-04.json`.

All interpretation remains confined to device-noise variability. This amendment
does not create a coherent-dynamics, causal-calibration, hardware-performance,
or quantum-advantage claim.
