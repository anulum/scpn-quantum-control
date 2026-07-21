<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SCPN Quantum Control — IQM DLA Backend-Sensitivity Powered Preregistration -->

# IQM DLA Backend-Sensitivity Preregistration (Properly Powered)

Date: 2026-07-21

This preregistration commits the design, decision rule, and budget for a
properly powered continuation of the 2026-05-13 IQM Garnet DLA-parity
replication campaign. It does not submit an IQM job and does not authorise
credit spend; execution requires a separate explicit per-submit approval.

Context: IQM granted extended Resonance access on 2026-07-20 (ticket
IQMCS-1691) for exactly this continuation, after we disclosed the honest
mixed-sign 2026-05-13 outcome. Every publication arising from these runs
credits IQM and IQM Resonance for hardware access.

## Scientific Question (reframed after the exact classical baseline)

The 2026-07-21 exact statevector baseline
(`docs/campaigns/dla_parity_exact_baseline_2026-07-21.md`) proved that the
noiseless XY-Trotter parity leakage is exactly zero at every depth: the
promoted `ibm_kingston` even/odd asymmetry (+10.8 % ± 1.1, Fisher
p ≪ 10⁻¹⁶) is structure inside the device-noise floor, not coherent
parity-selective dynamics. The live question is therefore:

> Is parity-sector-correlated decoherence backend-sensitive? Does IQM Garnet
> exhibit a stable positive even-over-odd leakage asymmetry of the kind the
> IBM Heron devices show, or is that noise structure specific to the IBM
> stack?

The 2026-05-13 Garnet campaign (six pinned layouts, 12,288 shots total,
combined Fisher p between 0.05 and 0.57, sign matches 1/3 to 3/3) was too
low-powered to answer this. This block concentrates the statistics on one
preregistered layout at a power level fixed in advance.

This is the preregistered-power continuation promised in our 2026-07-16
research-credits request to IQM, and a direct extension of the published
backend-sensitivity observation (DOI 10.5281/zenodo.20382000).

## Hypotheses and Primary Endpoint

- `H1` (backend-universal noise structure): Garnet shows a positive pooled
  even-over-odd relative leakage asymmetry, in the direction of the IBM
  Phase 2 A+G reference.
- `H0` (backend-sensitive noise structure): no stable positive asymmetry on
  Garnet. The 2026-05-13 data weakly favour `H0`.

Primary endpoint: pooled across the three depths, one-sided two-proportion
test of `leak_even > leak_odd` at `alpha = 0.05`. Relative asymmetry metric
is the committed `(even - odd) / odd` from `scripts/analyse_iqm_dla_parity.py`.

Power (committed before execution): with 12,288 shots per arm pooled and a
baseline odd-arm leak near the 2026-05-13 level (~0.42), the minimum
detectable relative asymmetry at 90 % power is +0.044 — the magnitude of the
weakest IBM Phase 2 A+G per-depth reference effect (d6 +0.0435, d10 +0.0447).
Per-depth secondary tests at 4,096 shots per arm resolve d4-like effects
(+0.0865 reference against a +0.076 minimum detectable effect).

## Layout Rule (selection bias disclosed)

Physical layout `[2, 7, 12, 13]` — the only 2026-05-13 layout with 3/3
depth-wise sign agreement with IBM. This layout is chosen using the May data,
so the May result cannot be reused as evidence; this block is a confirmatory
test on entirely new data with the decision rule above. If calibration makes
`[2, 7, 12, 13]` unavailable, fall back to `[9, 4, 3, 8]` and record the
substitution before submission.

## Circuit Matrix

| Field | Value |
|-------|-------|
| Device | IQM Garnet (20 qubits, square lattice) via Resonance |
| `n` | `4` |
| Circuit family | committed `iqm_dla_pinned_n4_d{4,6,10}_{even,odd}` builders |
| States | `0011` even reference, `0001` odd reference |
| Depths | `4, 6, 10` |
| Repetitions | `4` per state/depth (calibration-drift resolution) |
| Main shots | `1024` per repetition → `4096` per state/depth |
| Readout states | `0011`, `0001`, `0000`, `1111` |
| Readout shots | `2048` per state |

Circuit count: `2 × 3 × 4 = 24` main + `4` readout = `28` circuits.
Shot count: `24,576` main + `8,192` readout = `32,768` shots.

## Budget and Stop Rules

- Submit the first repetition block (6 main circuits + 4 readout circuits)
  alone, then read the actual credit burn from the Resonance dashboard before
  continuing.
- Abort the campaign if the first block consumes more than one third of the
  currently visible credit allowance.
- No further IQM submissions beyond this matrix without a fresh
  preregistration.

## Live Readiness Gates

Before any submission:

- dry-run the full matrix against `IQMFakeGarnet` from committed code only;
- confirm the pinned layout qubits are calibrated and operational on the
  Resonance dashboard;
- record device name, timestamp, calibration snapshot metadata, transpiled
  depth and two-qubit gate summaries, shot and circuit counts;
- reject any circuit whose transpiled depth exceeds the 2026-05-13 envelope
  (d10 transpiled depth 159) by more than 25 %;
- obtain explicit owner approval immediately before submission (per-submit
  GO).

## Analysis Plan

- Primary: pooled one-sided two-proportion test as committed above.
- Secondary: per-depth Wilson intervals, per-depth one-sided tests,
  per-repetition drift table (the 2026-05-13 partial repeat showed
  calibration drift matters), sign agreement with the IBM Phase 2 A+G
  reference, descriptive combined Fisher statistic with dependence caveat.
- Readout handling: four-state exact-state parity-confusion correction; no
  full confusion-matrix mitigation claim from a four-state calibration.
- All analyses run by committed scripts extending
  `scripts/analyse_iqm_dla_parity.py`; raw counts, job identifiers, and
  SHA-256 hashes published in full.

## Decision Rule (frozen)

- Primary test rejects `H0` (p < 0.05, positive direction): report Garnet
  parity-sector noise asymmetry as replicating the IBM direction at
  preregistered power; publish as backend-shared noise structure.
- Primary test does not reject: report a bounded null — Garnet shows no
  IBM-magnitude positive asymmetry at 90 % power — and publish it as
  backend-sensitivity evidence with the same prominence a positive result
  would receive.
- Either way, no coherent-dynamics claim is available: the exact baseline
  fixes noiseless leakage at zero, so every observed asymmetry statement is
  a statement about device noise.

## Claim Boundary

Blocked regardless of outcome: quantum advantage; DLA-parity-only causality;
monotone scaling; backend-universal protection; any modification of the
frozen submissions under `paper/submissions/` (results feed a new manuscript
only); any claim beyond the sampled device, calibration window, circuit
family, layout, and state pair.

## Output Artefacts

- `data/iqm_backend_sensitivity/iqm_dla_powered_<timestamp>_{plan,executed}.json`;
- `data/iqm_backend_sensitivity/iqm_dla_powered_analysis_<date>.{json,md}`;
- raw job identifiers public; access token never committed;
- campaign manifest `docs/campaigns/iqm_dla_backend_sensitivity_manifest_<date>.md`.

## Submission Boundary

This preregistration is complete once committed. QPU execution remains
blocked until the live readiness gates pass and the owner grants a separate
explicit per-submit GO.
