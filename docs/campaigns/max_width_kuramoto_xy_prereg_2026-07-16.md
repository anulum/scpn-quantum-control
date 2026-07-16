<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SCPN Quantum Control — Maximum-Width Kuramoto-XY Preregistration -->

# Maximum-Width Kuramoto-XY Campaign Preregistration (WIDTH-1)

Date: 2026-07-16

This preregistration prepares the maximum-workload-width campaign on an IBM
Heron r2 backend. It does not submit an IBM job, reserve backend time, or
authorise QPU spend — every submission additionally requires an owner GO with
the exact shot plan and a seconds-level cost estimate, and executes only after
the 2026-07-16 public-claim corrections are green on `main`.

## Motivation and honesty context

Every executed hardware workload in this repository to date spans 2–16 qubits
(verified 2026-07-16: maximum bitstring width across all committed result
artefacts is 16), while the deployment chip is 156-qubit. The honest response
to that gap runs in both directions: the public record now states the
workload width explicitly, and this campaign measures how far the width can
actually be pushed. Both directions are the same discipline.

## Scientific question

How does the measured Kuramoto-XY phase order parameter `R` degrade with
workload width `n` for shallow synchronisation circuits on a Heron r2
processor, from `n = 32` to the maximum usable width of the device, and how
does the degradation compare with an exact classical tensor-network baseline
of the same circuits?

## Claim boundary

Supported after successful execution and analysis:

- workload-width engineering evidence: the widest executed Kuramoto-XY
  circuit with committed raw counts and a verifier;
- an `R(n)` hardware-vs-baseline degradation curve with readout mitigation
  reported on/off;
- per-qubit diagnostic structure at each width.

Blocked even after a positive result:

- ANY quantum-advantage claim — the preregistered circuits are shallow and
  one-dimensional, hence efficiently classically simulable by construction
  (the committed MPS/TN baseline is exact for them);
- extrapolation beyond the sampled backend, calibration window, and depths;
- synchronisation-dynamics claims beyond the measured observable.

A degraded or unusable `R` at large `n` is a publishable outcome of equal
standing.

## Circuit matrix

| Field | Value |
|-------|-------|
| Topology | 1-D nearest-neighbour chain selected by DynQ over the full chip from fresh calibration data |
| Widths `n` | `32, 64, 104`, then device-max usable chain length |
| Coupling model | Kuramoto-XY chain, exponential-decay `K_nm` profile as in the repository's standard workload family |
| Evolution | first-order Trotter, `reps ∈ {1, 2}` |
| Measurement settings | 2 per width: all-qubit X basis, all-qubit Y basis |
| Observable | `R = n⁻¹ · sqrt[(Σ_k ⟨X_k⟩)² + (Σ_k ⟨Y_k⟩)²]` from the two settings |
| Main shots | ≤ 4096 per setting per width point |
| Readout mitigation | per-qubit readout-matrix calibration circuits; `R` reported mitigated and unmitigated |
| Classical baseline | exact MPS/TN simulation of the identical circuits, committed with the pack |

## Budget and abort criteria

- Estimated total QPU wall: 2–4 minutes within the remaining ~18-minute IBM
  Open-plan budget (owner decision 2026-07-16: shared with the Bell
  re-run and RC-1 lanes).
- Abort before submission if: fresh calibration shows median two-qubit error
  above the DynQ region-selection threshold for the target width, or the
  backend queue rejects the width.
- Abort after any width point whose retrieval fails integrity coercion
  (`hardware/_count_integrity.py`) — partial campaigns are recorded as
  partial, never padded.

## Evidence protocol

Submission record, retrieval record, hash-bound result pack with raw counts
and verifier, and a hardware-status-ledger row precede any public quotation
of a number, per the standing evidence rules. Fake or placeholder counts are
structurally excluded (fail-closed coercion; see the April 2026
count-integrity incident note).

## Status

- [ ] NOW-block corrections green on `main` (prerequisite)
- [ ] Owner GO with exact shot plan (per submission)
- [ ] Executed / analysed / packed / ledgered
