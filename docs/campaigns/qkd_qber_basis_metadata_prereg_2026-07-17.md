<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SCPN Quantum Control — QBER Re-Run With Basis Metadata Preregistration -->

# QBER Re-Run With Committed Basis Metadata (KIMI-8)

Date: 2026-07-17

This preregistration prepares a re-run of the March 2026 QBER measurement
with per-pub basis metadata committed alongside the counts. It does not
submit an IBM job, reserve backend time, or authorise QPU spend —
submission additionally requires an owner GO with the exact shot plan.

## Scientific question

The published QKD bit-error rate of 5.5% (ZZ) / 5.8% (XX) cannot be
re-derived from the committed
`results/ibm_hardware_2026-03-28/qkd_qber_4q.json` artefact because it
carries no basis metadata (dated caveat live on `docs/results.md` since
2026-07-16); a naive matched-basis sift of the committed counts gives
2.0–3.7%. What matched-basis error rate does the same entangled-pair
protocol give when every pub carries committed basis metadata, so the
derivation is reproducible from the pack alone?

## Honesty boundary stated up front

This is a matched-basis mismatch measurement on entangled pairs
(BBM92-style with fixed per-pub bases), NOT a QKD protocol run: there is
no per-shot random basis choice, no sifting in the protocol sense, no
authentication, and no privacy amplification. "QBER" here means the
mismatch rate of the entangled source under matched measurement bases —
a hardware-quality statement only.

## Claim boundary

Supported after successful execution and analysis:

- matched-basis error rate per pair per basis (ZZ and XX), with binomial
  σ, readout mitigation on and off, and the exact basis metadata of every
  pub committed in the pack;
- a committed pure-arithmetic derivation of the March artefact's naive
  matched-basis sift (assumed pub bases stated explicitly) next to the
  new values;
- a dated live-surface note making the new record the citable error rate.

Blocked even after a positive result:

- any QKD security, key-rate, or viability claim;
- any claim that re-derives or repairs the published 5.5%/5.8% — the
  non-derivability caveat on the March record stands regardless of what
  the re-run measures;
- any edit of the published March 2026 record.

## Circuit matrix

| Field | Value |
|-------|-------|
| Pairs | 2 independent Bell pairs (4 qubits), fresh layout from calibration (same selector as the KIMI-9 Bell re-run) |
| Main pubs | 2: matched ZZ (direct measurement) and matched XX (H on all four qubits before measurement), each a separate committed transpiled circuit with explicit `alice_basis`/`bob_basis` metadata |
| Main shots | 4096 per pub |
| Readout calibration | 16 full-basis circuits, 8192 shots (exact 2^n correction, `mitigation/readout_matrix.py`) |
| Analysis | pair marginal mismatch rate q = P(bits differ), binomial σ = sqrt(q(1−q)/N) at the raw shot count; mitigated rate from the exact readout inversion (quasi-probabilities enter unclipped, labelled) |

## Decision rules (preregistered)

- The new matched-basis error rates are reported per pair per basis with
  exact σ, mitigation on and off. No blanket averages.
- If the new rates fall within the March naive-sift band (recomputed by
  the committed script from the committed counts) to within 2σ, the
  reading is: the published 5.5%/5.8% overstate what the March artefact
  supports, and the new dated record becomes the citable error rate.
- If the new rates are materially higher and near the published values,
  the reading is: the March analysis plausibly included effects the
  artefact does not record; the non-derivability caveat still stands, and
  the new dated record is still the citable error rate.
- Either way the caveat on the live surfaces stays; only its follow-up
  note changes.

## Budget and abort criteria

- 18 circuits (2 main + 16 calibration) ≈ 20 estimated QPU seconds at the
  1.1 s/circuit anchor; hard cap 60 s. Spend comes from the remaining
  ~1011 s of the approved budget and needs a per-submit owner GO.
- Abort before submission if fresh calibration readout error on any
  chosen qubit exceeds the abort threshold, if any transpiled depth
  exceeds the ceiling, or if the estimate exceeds the cap.
- Retrieval failing strict count coercion quarantines the run.

## Evidence protocol

Submission record, retrieval record, hash-bound result pack with raw
counts, basis metadata, and verifier, ledger row before public quotation,
per the standing rules.

## Status

- [ ] Owner GO with exact shot plan (per submission)
- [ ] Executed / analysed / packed / ledgered
