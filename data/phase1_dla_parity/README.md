# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Phase 1 DLA parity dataset

# Phase 1 DLA Parity Campaign — Raw Dataset

This directory holds the complete raw dataset from the Phase 1 campaign
that first observed the dynamical Lie algebra parity asymmetry on IBM
Heron r2 hardware (ibm_kingston), April 2026. All files were written
verbatim by the `HardwareRunner` during live IBM Quantum Platform
execution; nothing here has been post-processed.

## Citation

If you use this dataset, please cite:

```bibtex
@misc{sotek2026dlaparityibm,
  author       = {Šotek, Miroslav},
  title        = {Hardware observation of dynamical Lie-algebra parity
                  asymmetry in the XY Hamiltonian on IBM Heron r2},
  year         = {2026},
  month        = {apr},
  howpublished = {\url{https://anulum.li/scpn-quantum-control/phase1-results.html}},
  note         = {Phase 1 campaign on ibm\_kingston, 342 circuits,
                  git commit 1b60f7b},
  orcid        = {0009-0009-3560-0851},
}
```

## Files

Each JSON file stores a single sub-phase of the campaign. All four
together constitute the full 342-circuit Phase 1 dataset at $n = 4$.

| File | Circuits | Reps/point | QPU wall | Purpose |
|------|---------:|-----------:|---------:|---------|
| `phase1_bench_2026-04-10T183728Z.json` | 42 | 2 | 44.11 s | Baseline bench across all 8 depths |
| `phase1_5_reinforce_2026-04-10T184909Z.json` | 72 | 4 more → 6 | 56.70 s | First reinforcement |
| `phase2_exhaust_2026-04-10T185634Z.json` | 138 | 6 more → 12 | 97.46 s | Cycle-exhausting bulk sweep |
| `phase2_5_final_burn_2026-04-10T190136Z.json` | 90 | 9 more → 21 (at d ∈ {4,6,8,10,14}) | 65.06 s | Final reinforcement at the middle depths |
| **Total** | **342** | **12–21** | **~263.3 s** | — |

**Backend:** `ibm_kingston` (IBM Heron r2, 156 qubits)
**Trotter step:** $t_\text{step} = 0.3$
**Coupling matrix:** $K_{ij} = 0.45\, e^{-0.3|i-j|}$ (all sub-phases)
**Depths swept:** $\{2, 4, 6, 8, 10, 14, 20, 30\}$
**Shots per circuit:** 2048

## IBM Quantum job IDs

```
d7ck79m5nvhs73a4nr10   phase1_bench
d7ck7hb0g7hs73dqvbg0   phase1_bench
d7ckcrh5a5qc73dosbmg   phase1_5_reinforce
d7ckft95a5qc73doseu0   phase2_exhaust
d7ckide5nvhs73a4o780   phase2_5_final_burn
```

All five jobs are retained in the IBM Quantum Platform job log.

## JSON schema

Each sub-phase JSON has the following top-level keys:

- `experiment` (str) — sub-phase name (e.g. `phase1_dla_parity_mini_bench`)
- `timestamp_utc` (str) — ISO 8601 timestamp of run start
- `backend` (str) — `ibm_kingston`
- `job_ids` (list of str) — IBM Quantum Platform job IDs submitted from this sub-phase
- `wall_time_s` (float) — total wall-clock time for the submission
- `n_circuits` (int) — number of circuits in this sub-phase
- `t_step` (float) — Trotter step size (all sub-phases use 0.3)
- `circuits` (list) — per-circuit metadata, with each entry containing:
  - `depth` (int) — number of Trotter steps
  - `sector` (str) — `"even"` or `"odd"`
  - `initial_state` (str) — bitstring (e.g. `"0011"` for popcount 2)
  - `counts` (dict) — raw Qiskit bitstring → count dictionary
- `aggregated_*` (dict, optional) — running aggregate across all
  preceding sub-phases, included for convenience so each file is
  self-contained without needing to replay earlier sub-phases

## Readout baseline

Experiment C (the readout calibration) is *not* in this directory; it
was a single independent circuit set whose results are summarised in
the `readout_baseline` key of
`figures/phase1/phase1_dla_parity_summary.json`:

- Mean retention across $\{|0000\rangle, |1111\rangle, |0101\rangle, |1010\rangle\}$: 98.3 %
- Per-state range: 97.6 % — 99.0 %
- Mean readout error estimate: 1.67 % per qubit

An order of magnitude too small to explain the 10 – 17 % middle-depth
asymmetry, ruling it out as a systematic source of the observed effect.

## Reproduce the full analysis

```bash
git clone https://github.com/anulum/scpn-quantum-control
cd scpn-quantum-control
git checkout 1b60f7b
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
python scripts/analyse_phase1_dla_parity.py
```

Expected output: `figures/phase1/phase1_dla_parity_summary.json`,
`leakage_vs_depth.png`, `asymmetry_vs_depth.png`, plus a console table
matching Table 1 in the short paper. Total wall time below 10 seconds
on a modern laptop; no QPU access required.

## Related files

- **Analysis script:** [`../../scripts/analyse_phase1_dla_parity.py`](../../scripts/analyse_phase1_dla_parity.py)
- **Short paper draft:** [`../../paper/phase1_dla_parity_short_paper.md`](../../paper/phase1_dla_parity_short_paper.md)
- **Aggregated summary JSON:** [`../../figures/phase1/phase1_dla_parity_summary.json`](../../figures/phase1/phase1_dla_parity_summary.json)
- **Figures:** [`../../figures/phase1/leakage_vs_depth.png`](../../figures/phase1/leakage_vs_depth.png), [`../../figures/phase1/asymmetry_vs_depth.png`](../../figures/phase1/asymmetry_vs_depth.png)
- **Website presentation:** https://anulum.li/scpn-quantum-control/phase1-results.html
- **Reproducibility manifest:** https://anulum.li/scpn-quantum-control/reproducibility.html

## Licence

Dataset released under AGPL-3.0-or-later, consistent with the parent
repository. Commercial licence available via protoscience@anulum.li.
