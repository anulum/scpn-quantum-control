<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SCPN Quantum Control — Depth-Optimal Native Decomposition Preregistration -->

# Depth-Optimal Native Decomposition Preregistration

Date: 2026-05-06

This preregistration defines an offline-first benchmark for reducing compiled
Kuramoto-XY circuit depth by targeting backend-native Heron gate structure. It
does not submit IBM jobs, reserve backend time, or authorise QPU spend.

## Scientific Question

Can an XY-aware native decomposition reduce depth, two-qubit gate count, or
routing overhead relative to generic Pauli-evolution and current XY compiler
baselines without changing the promoted physical observable?

## Claim Boundary

Supported after successful execution and analysis:

- resource comparison between generic Trotter, current XY compiler, and
  native-targeted decomposition;
- circuit-equivalence evidence from exact small-system unitary or observable
  checks;
- backend/layout-specific resource reductions for the sampled transpiler and
  calibration window.

Blocked even after a positive result:

- hardware coherence improvement without a separately approved QPU run;
- backend-general optimality;
- quantum advantage;
- correctness for all Hamiltonians outside the preregistered Kuramoto-XY/FIM
  families;
- substitution of the promoted physics circuit if unitary or observable
  equivalence fails.

## Comparator Circuits

Every readiness run must compare:

- generic Qiskit Pauli-evolution/Trotter circuit;
- current `scpn_quantum_control.phase.xy_compiler` output;
- native-targeted candidate decomposition;
- optionally, provider-transpiler optimized versions of each comparator with
  identical optimization level and seed list.

No comparator may be removed after seeing resource results.

## Native-Target Candidate Rules

The candidate decomposition must:

- preserve the intended `XX + YY` interaction angle convention;
- record any conversion to `rxx`, `ryy`, `rzx`, `ecr`, `cx`, `sx`, `rz`, or
  backend-supported native operations;
- avoid hidden changes to Trotter order, evolution time, or coupling threshold;
- expose a deterministic construction path from committed code;
- emit metadata identifying all approximations and basis-gate assumptions.

If a backend-native operation is unavailable in the selected backend target,
the candidate must fall back to a documented basis-gate decomposition and record
that fallback.

## Offline Readiness Matrix

Default no-QPU readiness scope:

| Field | Value |
|-------|-------|
| `n` | `4, 6, 8` |
| Families | DLA parity, popcount controls, FIM `lambda=0` and `lambda=4` |
| Depths/reps | representative shallow, promoted, and stress depths |
| Layout methods | default plus layer-selective where available |
| Transpiler seeds | `0, 1, 2, 3, 4` |
| Optimization level | fixed and recorded |

Readiness outputs:

- circuit depth;
- two-qubit gate count;
- basis-gate histogram;
- SWAP or routing-overhead count where available;
- transpilation wall time;
- unitary distance for `n=4` where tractable;
- observable-level agreement for `n=6,8` where full unitary comparison is not
  practical.

## Equivalence Gates

Before promoting a resource reduction:

- for `n=4`, compare candidate and baseline unitary or statevector evolution up
  to global phase;
- for `n=6,8`, compare preregistered observables from exact or high-confidence
  classical simulation where tractable;
- reject if parity survival, magnetisation-sector survival, or selected
  correlators differ beyond numerical tolerance before noise is applied;
- record the tolerance and simulator used.

Default tolerance:

- unitary/process comparison for `n=4`: `1e-8` absolute Frobenius-normalized
  distance where feasible;
- observable comparison for larger `n`: `1e-6` absolute difference for promoted
  scalar observables.

## Optional Hardware Scope

If the offline gates pass and QPU execution is later approved, use a minimal
resource-to-noise validation block:

| Field | Value |
|-------|-------|
| `n` | `4` |
| Families | one DLA parity pair and one FIM pair if still needed |
| Depths | one promoted signal depth and one stress depth |
| Decompositions | current XY compiler, native-targeted candidate |
| Repetitions | `6` per circuit |
| Shots | `4096` |
| Readout states | prepared states plus `0000` and `1111` |
| Readout shots | `8192` |

Circuit ceiling: `<= 160` circuits.

IBM-reported QPU-time ceiling: `12` minutes.

## Live Readiness Gates

Before any hardware submission:

- regenerate all comparator circuits from committed code only;
- live-transpile all comparator circuits on the selected backend/layout;
- reject if native-targeted decomposition fails equivalence gates;
- reject if the candidate increases depth or two-qubit count versus the current
  XY compiler for the promoted circuit family;
- reject if the candidate only improves one seed and worsens the median across
  fixed seeds;
- record backend, calibration timestamp, target basis, optimization level,
  transpiler seeds, circuit count, shot count, and estimated QPU minutes;
- get explicit approval immediately before submission.

## Analysis Plan

Primary offline endpoints:

- median and worst-case depth delta versus current XY compiler;
- median and worst-case two-qubit gate-count delta;
- routing overhead delta;
- equivalence-check pass/fail;
- seed sensitivity.

Secondary endpoints:

- estimated error-load proxy from gate counts and calibration data where
  available;
- decomposition-specific basis-gate histogram;
- interaction-layer parallelism achieved.

Hardware endpoints after separate approval:

- parity leakage and exact-state retention by decomposition;
- depth-normalized leakage comparison;
- readout-corrected comparison only where exact-state or full-basis calibration
  exists.

## Falsification Rules

The native-decomposition claim is rejected or downgraded if:

- equivalence checks fail;
- resource improvements are not robust across seeds;
- gains disappear after live backend transpilation;
- the candidate reduces depth but increases two-qubit error load enough to
  worsen the error proxy;
- optional hardware counts show no retention/leakage benefit despite resource
  reduction.

Negative results are publishable as compiler-boundary evidence for this circuit
family.

## Output Artefacts

Expected paths after offline readiness:

- `data/phase3_native_decomposition/native_decomposition_readiness_<date>.json`;
- `data/phase3_native_decomposition/native_decomposition_transpile_rows_<date>.csv`;
- `data/phase3_native_decomposition/native_decomposition_equivalence_rows_<date>.csv`;
- `docs/phase3_native_decomposition_readiness_<date>.md`.

Expected paths after approved hardware execution:

- `data/phase3_native_decomposition/native_decomposition_counts_<backend>_<timestamp>.json`;
- `data/phase3_native_decomposition/native_decomposition_summary_<date>.json`;
- `docs/phase3_native_decomposition_manifest_<date>.md`.

Each artefact must include comparator identity, backend target, basis gates,
transpiler seeds, depth and gate summaries, equivalence diagnostics, raw counts
where applicable, SHA256 hashes, and reproduction commands.

## Submission Boundary

This preregistration is complete. Hardware execution remains blocked until
offline readiness artefacts, backend selection, budget confirmation, and
explicit approval are completed in a separate task.
