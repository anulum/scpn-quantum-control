<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SCPN Quantum Control — GUESS Symmetry-Decay Preregistration -->

# GUESS / Symmetry-Decay Calibration Preregistration

Date: 2026-05-06

This preregistration prepares a hardware GUESS / symmetry-decay validation
using parity leakage as a witness. It does not submit IBM jobs, reserve backend
time, or authorise QPU spend.

## Readiness Basis

The no-QPU readiness analysis is:

```text
data/phase2_guess_calibration/phase2_guess_calibration_summary_2026-05-05.json
```

It fits log parity survival versus circuit depth as a cumulative-noise proxy.
It is not zero-noise extrapolation, because the promoted Phase 2 datasets do not
contain explicit folded-noise scale factors or another controlled noise knob.

Readiness signal:

- Phase 2 A+G even, odd, and sector-mean series have `R2 >= 0.9989`.
- Popcount-control state fits have `R2` from about `0.958` to `0.992`.
- These fits justify preparing a folded-noise protocol, not claiming GUESS
  mitigation from existing data.

## Scientific Question

Can parity leakage serve as a stable hardware symmetry-decay witness for
symmetry-guided zero-noise extrapolation on the same heterogeneous Kuramoto-XY
family?

## Claim Boundary

Supported after successful execution and analysis:

- calibrated relationship between explicit noise scale and parity survival;
- comparison between standard extrapolation and symmetry-decay guided
  extrapolation for the selected observable;
- statement that the witness is or is not stable enough for the tested
  backend/circuit family.

Blocked even after a positive result:

- broad quantum advantage;
- universal error mitigation;
- backend-general GUESS performance;
- DLA-parity-only protection;
- transfer of fitted coefficients to a different backend, layout, or
  calibration window without new validation.

## Circuit Matrix

Default scope:

| Field | Value |
|-------|-------|
| `n` | `4` |
| States | `0011` even reference, `0001` odd reference |
| Depths | `6, 8, 10, 14` |
| Noise scales | `1, 3, 5` via explicit unitary folding or approved equivalent |
| Repetitions | `8` per state/depth/noise-scale |
| Main shots | `4096` per circuit |
| Readout states | `0011`, `0001`, `0000`, `1111` |
| Readout shots | `8192` per circuit |

Circuit count:

- main circuits: `2 states x 4 depths x 3 noise scales x 8 reps = 192`;
- readout circuits: `4`;
- total circuits: `196`.

Optional extension after the default block:

- add popcount-control states `0101`, `0010`, and `0111` at depths `8` and
  `14` only if the default block passes the live depth and cost gates.

## QPU-Time Estimate

Expected IBM-reported QPU time for the default block: `5-12` minutes,
depending on folded circuit depths.

Budget ceiling for default block: `15` IBM-reported QPU minutes.

Optional extension ceiling: additional `15` IBM-reported QPU minutes.

Abort before submission if folded circuits exceed the ceiling or if the
remaining allocation cannot cover the block plus a 25 % safety margin.

## Live Readiness Gates

Before submission:

- confirm selected backend is Heron-class, account-visible, and operational;
- generate unfolded and folded circuits from committed code only;
- live-transpile every folded circuit;
- reject if max depth at noise scale `5` exceeds the unscaled depth by more
  than the preregistered folding expectation plus 25 % overhead;
- reject if live two-qubit gate count makes the QPU-time estimate exceed the
  budget ceiling;
- record backend name, calibration timestamp, folding rule, depth summary,
  two-qubit gate summary, circuit count, shot count, and estimated QPU minutes;
- get explicit approval immediately before submission.

## Analysis Plan

Primary witness:

- parity survival and parity leakage versus explicit noise scale.

Primary extrapolation:

- fit log survival or leakage trend versus noise scale for each state/depth;
- compare ordinary extrapolation against symmetry-decay guided extrapolation;
- report calibration coefficients only for the sampled backend/layout/window.

Promoted summaries:

- per-depth witness stability;
- fit quality by state and sector;
- extrapolated observable shift with confidence interval;
- failure modes where folding destroys signal or fit quality.

Readout handling:

- apply exact-state parity readout correction where the four readout states
  support it;
- do not claim full confusion-matrix mitigation from four-state readout
  calibration.

## Falsification Rules

The GUESS witness claim is rejected or downgraded if:

- parity survival is not monotone or fit quality is poor across noise scales;
- folded circuits introduce layout/transpilation artefacts larger than the
  witness trend;
- even and odd sectors require incompatible witness models;
- readout correction changes the sign or removes the extrapolation signal;
- the fitted coefficients are unstable across depths.

If the result is negative, report it as a boundary on parity leakage as a
GUESS witness for this circuit/backend family.

## Output Artefacts

Expected paths after approved execution:

- `data/phase3_guess_dla/phase3_guess_<backend>_<timestamp>.json`;
- `data/phase3_guess_dla/phase3_guess_summary_<date>.json`;
- `data/phase3_guess_dla/phase3_guess_fit_rows_<date>.csv`;
- `data/phase3_guess_dla/phase3_guess_extrapolation_rows_<date>.csv`;
- `docs/campaigns/phase3_guess_dla_manifest_<date>.md`.

Each artefact must include backend, folding rule, circuit metadata, raw counts,
SHA256 hashes, depth/gate summaries, fit diagnostics, and reproduction
commands.

## Submission Boundary

This preregistration is complete. QPU execution remains blocked until backend
selection, folded-circuit readiness artefacts, budget confirmation, and
explicit approval are completed in a separate task.
