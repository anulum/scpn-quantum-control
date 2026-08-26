# BL-80 adaptive FIM evidence

Schema: `adaptive_fim_evidence.v1`

## Frozen results

| Lane | Actions | Interpretation |
| --- | --- | --- |
| Synthetic calibration | decrease -> hold -> hold | High-signal decrease; boundary and underpowered controls hold |
| Historical offline replay | decrease -> decrease -> decrease | Replays already committed adverse lambda=4 witnesses; no efficacy test |
| BL-47 over-budget request | refused | No schedule or observer emitted |
| Hardware request | refused | No provider submission or execution |

## Custody

- Source: `data/scpn_fim_hamiltonian/fim_ibm_repeated_followup_raw_counts_2026-05-05_ibm-run-cf4835290f607387.json`
- SHA256: `13948b12223dbc64f659cb26de393bd9894dba37c2a3787ce15d3b6aad4089d2`
- Job ID: `ibm-run-cf4835290f607387`
- Use: offline proposal replay only.

## Claim boundary

uncertainty-aware batch-level next-experiment proposals under BL-47 dry-run budgets; offline replay is not closed-loop validation; no provider submission, live QPU feedback, FIM protection, optimal-policy, hardware-efficacy, realtime control, or quantum-advantage claim

The replay shows that the rule is deterministic and conservative for the
selected committed witnesses. It does not test whether later proposed
batches improve leakage or retention, and therefore does not validate a
closed-loop controller, FIM protection, an optimal policy, or advantage.

Content digest: `7aeeb99d94ca3979eed288d7c3d533ec56145ba58596f88933d2691f09107460`
