# Next validation protocols -- no-submit preparation

Date: 2026-05-05

This document prepares the next validation steps without submitting new IBM jobs.
The objective is to spend zero additional QPU time until offline checks justify
which hardware run is worth the remaining allocation.

## 1. Readout-mitigation cross-check

Status: implemented as `scripts/analyse_phase2_readout_mitigation.py`.

The existing Phase 2 data do not contain a complete computational-basis
confusion matrix. Therefore a literal full `2^n x 2^n` confusion-matrix inversion
is not available from current raw counts. The implemented cross-check performs
state-specific parity-confusion inversion for rows whose initial state has an
exact readout-only calibration.

Decision rule:

- If sign and Fisher significance survive parity-readout correction, keep the
  manuscript claim unchanged and add a short robustness sentence.
- If correction changes the sign of a promoted claim, downgrade the claim before
  any submission.
- A true full confusion-matrix inversion requires a new calibration block with
  all basis states for the measured qubit subset.

## 2. Offline GUESS / symmetry-decay calibration

Status: implemented as `scripts/analyse_phase2_guess_calibration.py`.

The current datasets do not contain explicit noise-scale folding, so the analysis
is not a GUESS zero-noise extrapolation result. It fits parity survival versus
circuit depth only as a readiness check for whether parity leakage is smooth
enough to serve as a future symmetry-decay witness.

Decision rule:

- If several state/sector series have high log-linear fit quality, prepare a
  minimal folded-noise GUESS validation run.
- If fit quality is mixed, keep GUESS as future work only.

## 3. State/layout-randomization QPU protocol

Goal: separate symmetry sector, excitation count, and physical layout effects.

Prepared design:

- Use `n=4` only.
- Depths: `d in {6, 8, 10, 14}`.
- States: `0011`, `0101`, `0001`, `0010`, `0111`.
- Layouts: three connected four-qubit windows selected by live backend coupling
  map and lowest recent readout error.
- Repetitions: 8 per state/depth/layout.
- Shots: 4096.
- Readout calibration: all five prepared states per layout, 8192 shots.

Estimated size:

- Main circuits: `5 states x 4 depths x 3 layouts x 8 reps = 480`.
- Readout circuits: `5 states x 3 layouts = 15`.
- Total: 495 circuits.

Estimated QPU time:

- Based on the completed popcount-control run, this is expected to cost roughly
  8--15 QPU minutes, depending on selected backend queue and transpiled depths.

Submission rule:

- Do not submit until live transpilation confirms max depth below the existing
  popcount-control depth envelope and the user explicitly approves QPU spend.

## 4. Minimal multi-device replication protocol

Goal: determine whether the n=4 asymmetry is specific to `ibm_kingston` or
stable across another Heron backend/calibration.

Prepared design:

- Backend: second Heron r2 backend only.
- Scope: reduced A block, no scaling.
- Depths: `d in {4, 6, 8, 10, 14, 20}`.
- States: original `0011` even and `0001` odd.
- Repetitions: 12 per state/depth.
- Shots: 4096.
- Readout calibration: `0011`, `0001`, plus `0000`, `1111`, 8192 shots.

Estimated size:

- Main circuits: `2 states x 6 depths x 12 reps = 144`.
- Readout circuits: 4.
- Total: 148 circuits.

Estimated QPU time:

- Using observed Heron rates from Phase 2, expected cost is roughly 3--6 QPU
  minutes if live transpilation depths remain comparable.

Submission rule:

- This is the preferred next hardware run only if the publication strategy
  requires backend-transfer evidence before submission. Otherwise keep it as a
  post-preprint replication.

