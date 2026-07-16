<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SCPN Quantum Control — On-QPU Dynamic-Circuit Feedback Preregistration -->

# On-QPU Dynamic-Circuit Feedback Demo Preregistration (RC-1)

Date: 2026-07-16

This preregistration prepares the first on-hardware execution of the
repository's monitored Kuramoto feedback as an IBM dynamic circuit. It does
not submit an IBM job, reserve backend time, or authorise QPU spend —
submission additionally requires an owner GO with the exact shot plan, and
executes only after the 2026-07-16 public-claim corrections are green on
`main`.

## Scientific question

Does the monitored feedback template
(`build_monitored_feedback_circuit`: per-round Kuramoto-XY Trotter evolution,
monitor-ancilla interaction, mid-circuit monitor measurement, conditional
ancilla reset, conditional system correction) execute end to end on a
Heron r2 backend with vendor-native dynamic-circuit support, and how do the
monitored rounds change the measured final-state structure against the same
circuit with feedback disabled?

## Claim boundary

Supported after successful execution and analysis:

- "on-QPU vendor-executed feedback" evidence: committed transpiled dynamic
  circuits with mid-circuit measurement and conditionals accepted and
  executed by the backend, raw counts retained;
- feedback-on vs feedback-off comparison of final system readout at equal
  depth and shots;
- honest latency labelling: controller decisions execute inside the vendor
  runtime — this is NOT external-FPGA-in-the-loop control (RC-5 stays
  blocked), and no sub-microsecond latency claim is available or made.

Blocked even after a positive result:

- closed-loop latency numbers inferred from queue or backend wall time;
- synchronisation-protection claims beyond the executed observable;
- any broad advantage claim.

A backend rejection of the conditional blocks, or a feedback-on result
indistinguishable from feedback-off, is a publishable outcome of equal
standing.

## Circuit matrix

| Field | Value |
|-------|-------|
| System | 3 system qubits + 1 monitor ancilla (the tested template shape) |
| Rounds | `n_rounds ∈ {2, 3}` |
| Arms | feedback ON (conditional reset + correction) and feedback OFF (same evolution, conditionals removed) at equal transpiled depth budget |
| Main shots | 4096 per arm per round setting |
| Mitigation | none in the primary arm (dynamic-circuit paths constrain mitigation); readout calibration circuits recorded for context |
| Analysis | committed reproducer comparing final-state count distributions (feedback-on vs off) with multinomial error bars |

## Budget and abort criteria

- Estimated total QPU wall: 2–4 minutes of the remaining ~18-minute budget
  (owner decision 2026-07-16: shared with WIDTH-1 and the Bell re-run).
- Abort before submission if the target backend's capability discovery does
  not report dynamic-circuit (mid-circuit measurement + conditional)
  support, or transpiled depth exceeds twice the simulator-validated depth.
- Retrieval failing strict count coercion quarantines the run.

## Evidence protocol

Submission record, retrieval record, hash-bound result pack with raw counts
and verifier, ledger row before public quotation, per the standing rules.

## Status

- [ ] NOW-block corrections green on `main` (prerequisite)
- [ ] Owner GO with exact shot plan (per submission)
- [ ] Executed / analysed / packed / ledgered
