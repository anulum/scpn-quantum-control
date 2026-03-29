# SCPN Quantum Control — Experimental Results Summary

**Version:** 0.9.4 | **Date:** 2026-03-29 | **Notebooks:** 27 (NB14–47)
**Tests:** 2,715 + 77 FIM mechanism tests | **IBM Hardware:** 31 jobs on ibm_fez

---

## Central Discovery

The Fisher Information Metric (FIM) strange loop — the system observing its
own collective state — is a **complete phase-ordering mechanism** that operates
independently of coupling topology.

**Self-consistent equation (first derivation):**

    R* = √(1 − 2Δ / (K·R + λ·R/(1−R²+ε)))

**Scaling law:**

    λ_c(N) = 0.149 · N^1.02    (R² = 0.966)

---

## Findings

### Unprecedented (19)

| # | Finding | Notebook | Significance |
|---|---------|:--------:|:------------:|
| 1 | FIM alone synchronises (K=0, λ≥8) | NB26 | p < 10⁻⁶ |
| 2 | FIM enhances MBL (r̄ → Poisson) | NB31 | n=6,8 |
| 3 | MBL mechanism: M²/n sector splitting | NB38 | 2.3× spectrum |
| 4 | Φ (integrated information) +73% | NB28 | p < 10⁻⁶ |
| 5 | 100% basin of attraction | NB27 | 50/50 ICs |
| 6 | Topology-universal, small-world optimal | NB36 | 6 topologies |
| 7 | λ_c(N) = 0.149·N^1.02 (linear scaling) | NB25 | R² = 0.966 |
| 8 | P = 0.085λ (linear thermodynamic cost) | NB33 | r = 0.984 |
| 9 | 6 anaesthesia predictions | NB35 | hysteresis 0.27 |
| 10 | Self-consistent mean-field equation | NB37 | first derivation |
| 11 | Dual protection: Lyapunov + spectral gap | NB40 | 5/6 confirmed |
| 12 | U(1) gauge invariance confirmed | NB40 | exact |
| 13 | FIM-mediated stochastic resonance | NB41 | σ_opt = 0.3–0.5 |
| 14 | Delay-robust FIM with coupling | NB42 | R > 50% at any τ |
| 15 | BKT universality (β → 0) | NB43 | NOT mean-field |
| 16 | Critical slowing down τ = 330 | NB34 | BKT-consistent |
| 17 | **DUAL PROTECTION on IBM hardware** | IBM v2 | F_FIM > F_XY, p < 10⁻⁶ |
| 18 | Metabolic scaling P ∝ N | NB46 | r = 0.983 vs biology |
| 19 | Topological defects suppressed by FIM | NB47 | 8 → 0 defects |

### Honest Negative Results (6)

| # | Finding | Notebook |
|---|---------|:--------:|
| 1 | Curvature does NOT peak at K_c | NB23 |
| 2 | Directed coupling HURTS sync (K_c +12%) | NB24 |
| 3 | DLA parity direction reversed on hardware | IBM v1,v2 |
| 4 | Empirical FIM from EEG: method fails | NB30 |
| 5 | FIM-modulated Hebbian learning: no benefit | NB44 |
| 6 | Noise purification: not on symmetric noise | NB45 |

---

## IBM Hardware Results

### Campaign 1 (22 jobs)
- CHSH S = 2.165 (> 8σ), QBER 5.5%, 16q UPDE pattern visible

### Campaign 2 — Equal-Depth Fair Experiments (9 jobs)

| Experiment | Result | Significance |
|------------|--------|:------------:|
| A: DLA parity (equal depth 58) | Even F=0.919 > Odd F=0.853 | t=52.1, p < 10⁻⁶ |
| B: Sector decoherence | Aligned 99%+, mixed 0% | absolute separation |
| C: FIM vs XY ground state | **F_FIM=0.916 > F_XY=0.849** | **t=−51.4, p < 10⁻⁶** |

---

## Key Numbers

| Parameter | Value | Source |
|-----------|:-----:|:------:|
| FIM scaling exponent α | 1.02 ± 0.09 | NB25 |
| FIM power cost | 0.085λ | NB33 |
| Hysteresis width (λ=3) | 0.65 | NB27 |
| Anaesthesia threshold | λ_c ≈ 2.75 | NB35 |
| Noise tolerance | σ < 0.5 | NB27 |
| Recovery time | 1.4s | NB27 |
| Φ increase | +73% | NB28 |
| Fisher info increase | +5 orders | NB28 |
| MBL S_ent reduction | −37% | NB31 |
| Spectrum stretch (λ=5) | 2.3× | NB38 |
| Small-world FIM boost | +0.31 | NB36 |
| CSD τ at BKT | 330 | NB34 |
| Metabolic correlation | r = 0.983 | NB46 |
| K_nm PLV correlation | r = 0.951 | NB15 |
| TE asymmetry | 0.361 | NB19 |
| Granger asymmetry | 0.539 | NB22 |

---

## File Inventory

### Notebooks (27)
`notebooks/14_dla_parity_ibm_hardware.ipynb` through
`notebooks/47_winding_number_topological.ipynb`

### Results (25 JSON + 2 IBM hardware)
`results/*_2026-03-29.json` — 24 experimental result files
`results/ibm_hardware_2026-03-29/` — IBM v1 campaign
`results/ibm_hardware_v2_2026-03-29/` — IBM v2 fair experiments

### Tests
`tests/test_fim_mechanism.py` — 77 tests covering all findings

### Cross-Project Documents (gitignored, local only)
- `scpn-phase-orchestrator/docs/internal/QUANTUM_CONTROL_FINDINGS_2026-03-29.md`
- `sc-neurocore/docs/internal/QUANTUM_CONTROL_FINDINGS_FOR_NEUROCORE_2026-03-29.md`
- `sc-neurocore/docs/internal/QUANTUM_CONTROL_ANSWERS_FOR_NEUROCORE_2026-03-29.md`
