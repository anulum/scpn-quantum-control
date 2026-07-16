<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SCPN Quantum Control — Mitigated Bell Re-Run Preregistration -->

# Mitigated Bell Re-Run Preregistration (KIMI-9)

Date: 2026-07-16

This preregistration prepares a re-run of the March 2026 Bell test with
per-setting transparency and readout mitigation. It does not submit an IBM
job, reserve backend time, or authorise QPU spend — submission additionally
requires an owner GO with the exact shot plan, and executes only after the
2026-07-16 public-claim corrections are green on `main`.

## Scientific question

The committed `results/ibm_hardware_2026-03-28/bell_test_4q.json` artefact
carries an anomalous second analyser setting: E ≈ +0.286/+0.334 on the two
pairs against ≈ 0.80–0.86 for the other three settings (confirmed by
recomputation, `scripts/recompute_chsh_bell_test.py`; stated publicly in
`docs/results.md`). Is that anomaly a readout/transpilation artefact, or a
real asymmetry of the executed setting?

## Claim boundary

Supported after successful execution and analysis:

- per-setting correlators E with committed per-setting transpiled circuits;
- CHSH S ± σ per pair, reported with readout mitigation on and off;
- a dated live-surface note resolving or documenting the anomaly.

Blocked even after a positive result:

- loophole-free or device-independent claims of any kind;
- QKD viability claims (KIMI-8 tracks the QBER derivation separately);
- any edit of the published March 2026 record — the original artefact and
  preprint stay as they are; this re-run is a NEW dated record.

Both outcomes are publishable: "anomaly was a correctable artefact" and
"anomaly is real and the published S values stand with a documented setting
asymmetry" have equal standing.

## Circuit matrix

| Field | Value |
|-------|-------|
| Pairs | 2 independent Bell pairs (4 qubits total), fresh layout from calibration |
| Settings | 4 CHSH analyser settings, each a separate committed transpiled circuit |
| Main shots | 4096 per setting |
| Readout calibration | per-qubit readout-matrix circuits, 8192 shots |
| Mitigation | readout-matrix correction on E and S; reported on/off |
| Analysis | `scripts/recompute_chsh_bell_test.py` conventions (little-endian pairs, minus sign on setting 1, multinomial σ) applied to the new counts by a committed reproducer |

## Decision rules (preregistered)

- If the mitigated setting-1 correlators rise to within 2σ of the other
  settings' band, the anomaly is explained as a readout/transpilation
  artefact and the live surfaces gain a dated resolution note.
- Otherwise the asymmetry is documented as a real property of the executed
  setting, with the per-setting circuits published for inspection.
- S is stated per pair with exact σ; no blanket significance statements.

## Budget and abort criteria

- Estimated total QPU wall: 2–3 minutes of the remaining ~18-minute budget
  (owner decision 2026-07-16: shared with WIDTH-1 and RC-1).
- Abort before submission if fresh calibration readout error on any chosen
  qubit exceeds the mitigation circuit's correctable range.
- Retrieval failing strict count coercion quarantines the run — partial
  evidence is recorded as partial.

## Evidence protocol

Submission record, retrieval record, hash-bound result pack with raw counts
and verifier, ledger row before public quotation, per the standing rules.

## Status

- [ ] NOW-block corrections green on `main` (prerequisite)
- [ ] Owner GO with exact shot plan (per submission)
- [ ] Executed / analysed / packed / ledgered
