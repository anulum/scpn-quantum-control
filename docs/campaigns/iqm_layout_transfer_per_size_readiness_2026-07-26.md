<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->

# IQM Garnet Per-Size Layout-Transfer Readiness

This package implements the provider-free gates frozen in
`iqm_layout_transfer_per_size_prereg_2026-07-22.md`. It does not authorize a
provider call or hardware execution.

## Frozen matrix

- sizes: 8, 12, 16;
- arms: optimised, default, naive;
- four execution-order repetitions at 2,048 shots per arm and size;
- one 1,024-shot all-zero/all-one readout pair per size;
- 42 circuits and 79,872 total shots;
- layouts computed once from one calibration snapshot before circuit assembly;
- QPY format version 15 for the isolated IQM environment.

## Local readiness sequence

Use the existing fake-Garnet calibration extractor, then the per-size runner:

```bash
.venv-iqm/bin/python scripts/iqm_layout_transfer_fake_garnet.py dump-calibration \
  --date YYYY-MM-DD --out /tmp/iqm_layout_calibration.json
.venv/bin/python scripts/iqm_layout_transfer_per_size_harness.py prepare \
  --calibration /tmp/iqm_layout_calibration.json --date YYYY-MM-DD \
  --out-dir data/iqm_layout_transfer_per_size
.venv-iqm/bin/python scripts/iqm_layout_transfer_per_size_fake_garnet.py \
  --circuits data/iqm_layout_transfer_per_size/iqm_layout_transfer_per_size_circuits_YYYY-MM-DD.qpy \
  --labels data/iqm_layout_transfer_per_size/iqm_layout_transfer_per_size_labels_YYYY-MM-DD.json \
  --plan data/iqm_layout_transfer_per_size/iqm_layout_transfer_per_size_YYYY-MM-DD_plan.json \
  --date YYYY-MM-DD \
  --out data/iqm_layout_transfer_per_size/fake_counts_YYYY-MM-DD.json
.venv/bin/python scripts/iqm_layout_transfer_per_size_harness.py analyse \
  --plan data/iqm_layout_transfer_per_size/iqm_layout_transfer_per_size_YYYY-MM-DD_plan.json \
  --counts data/iqm_layout_transfer_per_size/fake_counts_YYYY-MM-DD.json \
  --out data/iqm_layout_transfer_per_size/fake_analysis_YYYY-MM-DD.json
```

The reviewed QPY loader intentionally accepts artefacts only beneath the
repository's governed `data/` tree. Keeping QPY and derived evidence there is a
readiness invariant; do not move the circuit bundle to `/tmp` or bypass the
loader. The per-size fake runner uses the frozen two-batch split and atomically
checkpoints each completed batch; rerunning the same command resumes a matching
checkpoint instead of discarding completed simulation work.

The committed 2026-07-26 fake-Garnet evidence used the full frozen budget. Its
10,000-resample result is a readiness check, not hardware evidence: n=8 had a
positive simulated default-minus-optimised difference, n=12 and n=16 included
zero after Holm adjustment, and the all-sizes primary was false.

The analysis enforces a complete label matrix and a green depth-parity plan,
pools the four repetitions per arm and size, propagates main/readout shot noise
with 10,000 multinomial resamples at seed 20260722, and reports the frozen
Holm-adjusted per-size tests, n=12 direction, pooled CI90, Cochran's Q,
default-versus-naive CI95, and raw/corrected sensitivity.

## Remaining live gates

After this local sequence is green, prepare the final matrix from the same-day
live calibration snapshot and exercise the exact live submission surface first
against `garnet:mock`:

```bash
.venv-iqm/bin/python scripts/iqm_layout_transfer_resonance.py dump-calibration \
  --quantum-computer garnet --date YYYY-MM-DD \
  --out data/iqm_layout_transfer_per_size/live_calibration_YYYY-MM-DD.json
.venv/bin/python scripts/iqm_layout_transfer_per_size_harness.py prepare \
  --calibration data/iqm_layout_transfer_per_size/live_calibration_YYYY-MM-DD.json \
  --date YYYY-MM-DD --out-dir data/iqm_layout_transfer_per_size
.venv-iqm/bin/python scripts/iqm_layout_transfer_resonance.py submit \
  --quantum-computer garnet:mock --all-sizes \
  --circuits data/iqm_layout_transfer_per_size/iqm_layout_transfer_per_size_circuits_YYYY-MM-DD.qpy \
  --labels data/iqm_layout_transfer_per_size/iqm_layout_transfer_per_size_labels_YYYY-MM-DD.json \
  --plan data/iqm_layout_transfer_per_size/iqm_layout_transfer_per_size_YYYY-MM-DD_plan.json \
  --date YYYY-MM-DD --out data/iqm_layout_transfer_per_size/mock_submission_YYYY-MM-DD.json \
  --i-have-owner-go
```

The `--all-sizes` route is restricted to the frozen per-size campaign, requires
exactly 36 mains plus six readouts, rechecks depth parity after provider
transpilation, and submits the frozen single-pass split as two jobs. Replace
only `garnet:mock` with `garnet` after the mock record, same-day calibration,
and explicit owner GO are all present. No credential or provider access is
needed for the provider-free local sequence above this section.
