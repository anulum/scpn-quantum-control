# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — IBM Hardware Complete Audit

# IBM Quantum Hardware — Complete Audit

> **2026-05-05 claim-hygiene note:** this audit is retained as historical
> evidence, but `docs/hardware_status_ledger.md` is now the canonical source
> for public promotion status. Campaign 4 / V2 aggregate outputs are not
> promoted until raw counts, private retrieval map, and reproduction analysis are
> reviewed. The March 29 DLA parity attempt is superseded by the April
> `data/phase1_dla_parity/` raw-count dataset.

> **2026-07-16 correction note:** the CHSH rows below originally stated
> ">8σ" against S = 2.165. Recomputation from the committed counts
> (`scripts/recompute_chsh_bell_test.py`) attributes 7.54σ to the S = 2.165
> pair (q0–q1) and 8.94σ to the S = 2.188 pair (q2–q3); the rows are amended
> in place with the correct per-pair significances.

**Date:** 2026-03-30
**Backend:** ibm_fez (Heron r2, 156 qubits)
**Plan:** Open (10 min / 28-day cycle)
**Historical reported jobs:** 33
**Historical reported shots:** ~176,000+

---

## Campaign 1 — Initial Batch (submitted 2026-03-18)

9 jobs submitted to ibm_fez via `scripts/submit_march_campaign.py`.
All 7 queued jobs completed between submission and 2026-03-28.
**Retrieved 2026-03-30** via `scripts/retrieve_completed_jobs.py` (had been QUEUED since 18.3.).

| Job ID | Experiment | PUBs | Shots | Status |
|--------|-----------|:----:|------:|--------|
| ibm-run-9317279194d1c740 | baseline_pair_a | 1 | 500 | DONE |
| ibm-run-93b07b15459915d2 | baseline_pair_b | 3 | 1,500 | DONE |
| ibm-run-3821495c7a7a1e0f | noise_baseline | 3 | 12,000 | DONE |
| ibm-run-2ddff7bbc36988b7 | kuramoto_4osc_s1 | 3 | 12,000 | DONE |
| ibm-run-dba7b17d1f4089cd | kuramoto_4osc_s3_proxy | 3 | 12,000 | DONE |
| ibm-run-5f238ed35d404e61 | kuramoto_8osc | 3 | 12,000 | DONE |
| ibm-run-b6d84688f60da3ca | kuramoto_4osc_trotter2 | 3 | 12,000 | DONE |
| ibm-run-245f36b7a6aa4b1d | bell_test_4q | 4 | 16,000 | DONE |
| ibm-run-ed48720009580850 | correlator_4q | 4 | 16,000 | DONE |

**Results:** `results/march_2026/job_*.json` (9 files, freshly retrieved)
**Also in:** `results/ibm_hardware_2026-03-18/` (same job IDs, older format, retrieved 28.3.)

### Noise baseline analysis
- Z-basis: top bitstring `0010` at 25.3% (expected: state preparation fidelity)
- X-basis: top `0000` at 62.5% (Hadamard coherence preserved)
- Y-basis: distributed (max 7.4%) — Y-basis most sensitive to decoherence

### Bell test (Campaign 1)
- S = 0.98 (below classical limit) — initial VQE ansatz with sub-optimal basis rotation
- This was superseded by Campaign 2 which used proper CHSH angles

---

## Campaign 2 — Full Hardware Campaign (2026-03-28)

Historical audit row for 20 baseline experiments on ibm_fez. The ledger now
requires artifact-level citation for any public value derived from this set.
Submitted and retrieved during the March 28 session.

| Experiment | File | Key Result |
|-----------|------|------------|
| noise_baseline_mar28 | `noise_baseline_mar28.json` | Calibration reference |
| kuramoto_4osc_zne | `kuramoto_4osc_zne.json` | ZNE scale [1,3,5] stable |
| kuramoto_8osc_zne | `kuramoto_8osc_zne.json` | 8q ZNE scaling |
| upde_16_dd | `upde_16_dd.json` (144K) | 16q UPDE with dynamical decoupling |
| decoherence_scaling | `decoherence_scaling.json` (7.1K) | R = R_exact × exp(-γ×depth) |
| kuramoto_4osc_trotter_comparison | `kuramoto_4osc_trotter_comparison.json` | Order-1 vs order-2 |
| vqe_8q_hardware | `vqe_8q_hardware.json` (13K) | 8q VQE optimisation |
| sync_threshold | `sync_threshold.json` | Bifurcation R vs K |
| bell_test_4q | `bell_test_4q.json` | **CHSH S = 2.165 ± 0.022 (7.5σ, q0–q1); S = 2.188 ± 0.021 (8.9σ, q2–q3)** \[corrected 2026-07-16\] |
| qkd_qber_4q | `qkd_qber_4q.json` | **QBER 5.5% (< BB84 11%)** |
| ansatz_comparison | `ansatz_comparison.json` | Knm outperforms TwoLocal by 32% |
| zne_higher_order | `zne_higher_order.json` | ZNE stable fold 1-9 |
| correlator_4q | `correlator_4q.json` | ZZ anti-correlation matches K_ij |

**Results:** `results/ibm_hardware_2026-03-28/` (13 JSON files)

### Headline results
- **CHSH S = 2.165 ± 0.022** (pair 0-1, 7.5σ), **S = 2.188 ± 0.021** (pair 2-3, 8.9σ) — Bell inequality violated; only the higher pair clears 8σ \[corrected 2026-07-16: σ was previously misattached to the lower pair\]
- **QBER 5.5%** — below BB84 threshold of 11% → QKD viable
- **State preparation fidelity 94.6%**
- **ZNE <2% variation** across fold levels 1-9
- **Knm ansatz 42% top-bitstring** vs TwoLocal 20%
- **16q UPDE**: 13/16 qubits with |⟨Z⟩| > 0.3

---

## Campaign 3 — DLA Parity (2026-03-29)

2 jobs testing even/odd DLA parity sectors.

| Job ID | Experiment | Fidelity |
|--------|-----------|----------|
| ibm-run-d0ff68f141e586b2 | even parity | **0.994** |
| ibm-run-e013d50173ffe3c7 | odd parity | **0.932** |

**Results:** `results/ibm_hardware_2026-03-29/dla_parity_results.json`

**Issue identified:** Circuit depth artefact — even circuits depth=3, odd depth=66. The fidelity difference reflects circuit depth, not parity physics. Fixed in Campaign 4.

---

## Campaign 4 — V2 Fair Experiments (2026-03-29)

9 equal-depth jobs testing three hypotheses. All circuits matched in depth.

| Job ID | Experiment | Mean Fidelity | Std |
|--------|-----------|:------------:|:---:|
| ibm-run-f303eb8f4cae6b90 | A_even | 0.9185 | 0.0023 |
| ibm-run-1db7222826cba2cc | A_odd | 0.8526 | 0.0037 |
| ibm-run-e161b54f52c388c6 | C_xy | 0.8484 | 0.0032 |
| ibm-run-721a50394119e791 | C_fim | **0.9158** | 0.0023 |
| ibm-run-e25ce044a9384cf8 | B_M+4 | **0.9936** | 0.0000 |
| ibm-run-25507365c5e83e6f | B_M+2 | 0.0000 | 0.0000 |
| ibm-run-0fff063bb180f5ea | B_M0 | 0.0000 | 0.0000 |
| ibm-run-1e872de0fb7926a1 | B_M-2 | 0.0000 | 0.0000 |
| ibm-run-cff643fd2ec8df51 | B_M-4 | **0.9595** | 0.0000 |

**Results:** `results/ibm_hardware_v2_2026-03-29/full_results.json`

### Key findings
- **Exp A (DLA parity):** Even F=0.919 > Odd F=0.853 — aggregate-only parity difference observed at equal depth; unpromoted pending raw-count review
- **Exp B (Sector decoherence):** Aligned sectors (M=±4) survive 96-99%, mixed sectors (M=0,±2) collapse to 0% — **ABSOLUTE separation**
- **Exp C (FIM protection):** F_FIM=0.916 > F_XY=0.849 — **DUAL PROTECTION CONFIRMED ON HARDWARE**

---

## QPU Budget Status

| Cycle | Used | Remaining | Notes |
|-------|------|-----------|-------|
| Feb 2026 | 10 min | 0 | Fully exhausted |
| Mar 2026 (cycle 1) | ~8 min | ~2 min | 33 historical job records |
| Apr 2026+ | 0 | 10 min | Pending IBM Credits (5h applied 29.3.) |

---

## File Inventory

```
results/
├── march_2026/                          # Campaign 1 (retrieved 2026-03-30)
│   ├── job_ibm-run-9317279194d1c740.json    # baseline_pair_a
│   ├── job_ibm-run-93b07b15459915d2.json    # baseline_pair_b
│   ├── job_ibm-run-3821495c7a7a1e0f.json    # noise_baseline
│   ├── job_ibm-run-2ddff7bbc36988b7.json    # kuramoto_4osc_s1
│   ├── job_ibm-run-dba7b17d1f4089cd.json    # kuramoto_4osc_s3_proxy
│   ├── job_ibm-run-5f238ed35d404e61.json    # kuramoto_8osc
│   ├── job_ibm-run-b6d84688f60da3ca.json    # kuramoto_4osc_trotter2
│   ├── job_ibm-run-245f36b7a6aa4b1d.json    # bell_test_4q
│   ├── job_ibm-run-ed48720009580850.json    # correlator_4q
│   ├── campaign_manifest.json
│   └── retrieval_summary.json
├── ibm_hardware_2026-03-18/             # Campaign 1 (old format, retrieved 28.3.)
│   └── *.json                           # 9 files, same job IDs
├── ibm_hardware_2026-03-28/             # Campaign 2
│   └── *.json                           # 13 experiment files
├── ibm_hardware_2026-03-29/             # Campaign 3 (DLA parity)
│   ├── dla_parity_jobs.json
│   └── dla_parity_results.json
├── ibm_hardware_v2_2026-03-29/          # Campaign 4 (fair experiments)
│   ├── full_results.json
│   └── job_ids.json
├── HARDWARE_RESULTS.md                  # Feb 2026 results (12-point decoherence)
├── IBM_HARDWARE_V2_ANALYSIS.md          # Deep analysis of Campaign 4
└── IBM_HARDWARE_COMPLETE_AUDIT_2026-03-30.md  # This file
```

---

## Discrepancy Notes

1. **Campaign 1 Bell test S=0.98** vs **Campaign 2 Bell test S=2.165**: Campaign 1 used a VQE ground state with standard ZZ/ZX/XZ/XX basis measurement. Campaign 2 used optimised CHSH angles (π/8 rotations) computed in `experiments.py` — proper CHSH protocol. S=2.165 is the correct value.

2. **Campaign 1 duplicate data**: `results/ibm_hardware_2026-03-18/` and `results/march_2026/` contain the same 9 jobs in different formats. The `march_2026/` format is richer (includes experiment name, group, retrieval timestamp).

3. **Campaign 3 DLA parity F=0.000**: The `mean_fidelity_even` and `mean_fidelity_odd` fields may use a different key structure. The fidelity arrays are present in the file. The actual values from the private handoff record are F_even=0.994, F_odd=0.932 (but with circuit depth confound).

4. **Campaign 4 B_M+2/M0/M-2 = 0.000**: These mixed magnetisation sectors completely decohere on hardware — the 0.000 is the actual measurement, not a missing value. Only aligned sectors (M=±4) survive.

---

## Prepared but Not Submitted

`scripts/march_2026_hardware_campaign.py` — 8 experiment orchestrator (~294s QPU), overlaps with Campaign 2. Most experiments already executed. Remaining unique: higher-order ZNE and bifurcation diagram refinement.

## IBM Quantum Credits Application

- **Submitted:** 2026-03-29
- **Requested:** 5h QPU (01/04/2026 – 31/08/2026)
- **Status:** Pending review
- **Risk:** Independent researcher, not tenure-track
