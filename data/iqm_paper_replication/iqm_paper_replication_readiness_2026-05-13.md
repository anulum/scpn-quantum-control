# IQM Paper Replication Readiness

No IQM service was contacted. No QPU job was submitted. No credits were spent.

- JSON: `data/iqm_paper_replication/iqm_paper_replication_readiness_2026-05-13.json`
- CSV: `data/iqm_paper_replication/iqm_paper_replication_readiness_2026-05-13.csv`
- Provider: `iqm`
- Fake backend: `garnet`
- Transpile status: `iqm_fake_transpile_passed`
- Total circuits across all prepared tiers: `52`
- Total planned shots across all prepared tiers: `13184`

## Tier Plan

| Priority | Tier | Circuits | Shots | Status | Max fake depth |
|---:|---|---:|---:|---|---:|
| 0 | `smoke_account_probe` | 1 | 128 | `passed` | 4 |
| 1 | `dla_parity_minimal` | 6 | 1536 | `passed` | 159 |
| 2 | `dla_parity_paper_core` | 16 | 4096 | `passed` | 459 |
| 3 | `fim_negative_control_minimal` | 12 | 3072 | `passed` | 241 |
| 4 | `readout_full_basis_optional` | 16 | 4096 | `passed` | 2 |
| 5 | `dla_readout_baseline_optional` | 1 | 256 | `passed` | 2 |

## Spend Rule

Thirty IQM credits are treated as a micro-replication budget. Run smoke first, then DLA minimal, then DLA paper core, then FIM minimal, and spend on full-basis readout only if measured credit burn is acceptable.

First real run:

- Tier: `smoke_account_probe`
- Circuits: `1`
- Shots: `128`
- Stop rule: Submit no further IQM jobs until the dashboard/resource calculator confirms actual credits consumed by this smoke job.

## Claim Boundary

This readiness artefact does not contact IQM, does not spend credits, and does not create cross-provider hardware evidence. It prepares the exact paper-critical circuit families for later approved IQM runs.
