<!-- SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->

# IBM v2 raw-count recovery and full disclosure (2026-07-18)

## Summary

The "IBM v2" fair-experiment campaign (2026-03-29, `ibm_fez`) was originally
committed **aggregate-only** — fidelity/survival means plus HMAC-blinded run
labels, with no raw counts and no raw job identifiers. Because it could not be
independently checked, it was quarantined (see
[Count-Integrity Incident, April 2026](../count_integrity_incident_2026-04.md)).

On 2026-07-18 the nine jobs were **re-retrieved read-only from IBM (0 QPU
seconds)** and published in full. This document records the recovery method, the
reproduction result, and the claim boundary.

## What is now public

`data/ibm_hardware_v2_recovered_2026-07-18/`:

- `recovered_raw_counts.json` — for each of the nine experiments: the **real IBM
  job identifier**, the per-pub raw counts (10 pubs × 8192 shots = 81 920 shots
  each, 4 measured qubits), and the dated `ibm_fez` calibration snapshot
  (`last_update 2026-03-29`).
- `manifest.json` — SHA-256 of the pack.
- `reproduction_analysis.json` — the reproducer output.

Full-disclosure policy (owner directive, 2026-07-18): raw counts, real job
identifiers, and calibration are public research data. The only value withheld
is the IBM API token; the recovery tool runs a fail-closed leak check against it
before writing anything.

## Method (reproducible)

1. `scripts/recover_ibm_v2_raw_counts.py` authenticates read-only, enumerates the
   `ibm_fez` job cluster submitted 2026-03-29 ~12:20 UTC (nine jobs in ~10 s),
   pulls per-pub raw counts and the dated calibration, pins each job to its
   experiment by submission order **cross-checked against the committed
   aggregate**, and writes the pack.
2. `scripts/analyse_ibm_v2_recovered.py` recomputes each experiment's survival
   observable from the raw counts (all-zero survival for the ground-state
   preparations `A_even`, `C_xy`, `C_fim`, `B_M+4`; the sector target state for
   `B_M±`; the odd-parity subspace for `A_odd`) and compares to the committed
   value.

## Reproduction result

Eight of nine experiments reproduce to **|Δmean| < 1×10⁻⁴** directly from the raw
counts. `A_odd` reproduces to ~3.7 % under the natural odd-parity-subspace
definition — its committed aggregate applied an original readout mitigation whose
exact form is not in the pack; the residual is reported, not fitted away.

| Experiment | Committed mean | Recovered mean | Observable |
|---|---|---|---|
| A_even | 0.9185 | 0.9185 | P(0000) |
| A_odd | 0.8526 | 0.8900 | odd-parity subspace (±3.7 %) |
| C_xy | 0.8484 | 0.8484 | P(0000) |
| C_fim | 0.9158 | 0.9158 | P(0000) |
| B_M+4 | 0.9936 | 0.9936 | P(0000) |
| B_M+2 / M0 / M−2 | 0.0000 | 0.0000 | sector target |
| B_M−4 | 0.9595 | 0.9595 | P(1111) |

The committed aggregates are therefore **genuine, not fabricated** — the April
2026 count-integrity incident concerned a separate large-N *frontier* automation
lane, not these nine jobs.

## Claim boundary

Reproducing `F_FIM = 0.9158 > F_XY = 0.8484` from real hardware counts establishes
only that the **observation** is genuine. It is **not** evidence of a
coherence-protection ("DUAL PROTECTION") mechanism: the FIM and XY circuits differ
in depth and structure, so a higher all-zero survival for the FIM circuit is
consistent with a shallower/less-decohering circuit rather than any protection
effect. The coherence-protection hypothesis was tested properly on the promoted
`ibm_kingston` SCPN/FIM campaign and committed as a **negative/falsification**
result (digital `λ=4` increases leakage / decreases retention;
[claim boundary](scpn_fim_claim_boundary_2026-05-05.md)). The "DUAL PROTECTION on
IBM hardware" wording is therefore retired as a claim in
`data/retired_claims.json` (`fim-dual-protection-hardware`) while the data stands.
