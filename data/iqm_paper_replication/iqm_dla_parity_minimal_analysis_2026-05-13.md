# IQM DLA Parity Minimal Analysis

This report reuses the IBM DLA/parity leakage metric: leakage is the fraction of shots whose measured bitstring has parity opposite to the prepared initial parity.

- IQM input: `data/iqm_paper_replication/iqm_dla_parity_minimal_2026-05-13_sanitized.json`
- IBM reference: `data/phase2_dla_parity/phase2_reduced_ag_summary_2026-05-05.json`
- Depths tested: `3`
- IQM Fisher combined p-value: `0.295392`
- Depths with IQM Fisher p < 0.05: `0`
- Sign matches vs IBM Phase 2: `1 / 3`

| Depth | IQM even leak | IQM odd leak | IQM asymmetry | IQM p | IBM Phase 2 asymmetry | Sign match |
|---:|---:|---:|---:|---:|---:|---|
| 4 | 0.457031 | 0.500000 | -0.085938 | 0.376336 | +0.086512 | no |
| 6 | 0.488281 | 0.531250 | -0.080882 | 0.376686 | +0.043469 | no |
| 10 | 0.527344 | 0.464844 | +0.134454 | 0.184831 | +0.044680 | yes |

## Interpretation

The IQM minimal tier does not reproduce the positive IBM Phase 2 DLA/parity asymmetry sign at the tested depths. Each IQM depth has only one even/odd pair at 256 shots, so this is a low-statistics cross-provider diagnostic rather than a manuscript-grade replication.

A second technical boundary applies: the first IQM minimal run used automatic transpiler layout. Follow-up inspection showed that IQM can choose different physical qubits for the even and odd circuits at the same depth, which confounds sector with layout/calibration. Repeated statistics should therefore pin the same physical layout for paired even/odd circuits or use an explicitly randomised layout block.

## Claim Boundary

IQM minimal tier is suitable for cross-provider sanity evidence and protocol debugging, but has one 256-shot replicate per sector and is not sufficient to upgrade manuscript claims. The next IQM run should use fixed paired layouts or an explicit randomised layout block, not blind automatic layout, before paper-core replication or repeated statistics are interpreted.
