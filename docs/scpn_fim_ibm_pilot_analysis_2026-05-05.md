# SCPN/FIM IBM pilot analysis

Date: 2026-05-05

Backend: `ibm_kingston`

Job ID: `d7t53ofljm6s73bc6bj0`

## Artefacts

Raw counts:

- `data/scpn_fim_hamiltonian/fim_ibm_pilot_raw_counts_2026-05-05_d7t53ofljm6s73bc6bj0.json`
- SHA256: `be284b9b2f71dfecd978703d979a8893e79b35dcc4537d7a372b83ba48305790`

Analysis:

- `scripts/analyse_fim_ibm_pilot.py`
- `data/scpn_fim_hamiltonian/fim_ibm_pilot_analysis_2026-05-05_d7t53ofljm6s73bc6bj0.json`
- SHA256: `90ca87916a0ae92bbf2d7212ee7def265f76b59e73925963eaedca3955c8d98b`

Row metrics:

- `data/scpn_fim_hamiltonian/fim_ibm_pilot_row_metrics_2026-05-05_d7t53ofljm6s73bc6bj0.csv`
- SHA256: `84ed542516dd69161ece84cbb936058f78fbf1a443e474d19f25bdacead50c6f`

Lambda trends:

- `data/scpn_fim_hamiltonian/fim_ibm_pilot_lambda_trends_2026-05-05_d7t53ofljm6s73bc6bj0.csv`
- SHA256: `967bcd8d51373e514347351dc52ff35d59bde1eabe7dad8fba9b503460b8171b`

Reproduction command:

```bash
env PYTHONPATH=src /home/anulum/.local/bin/python scripts/analyse_fim_ibm_pilot.py --verify-integrity
```

## Run size

- Circuits: 61
- Main FIM rows: 45
- Readout baseline rows: 16
- Shots: 249,856
- Maximum live transpiled depth: 540
- Maximum live two-qubit gates: 157

## Readout baseline

The readout-only basis-state baseline over all 16 computational basis states gives:

- Mean exact-state retention: `0.9847564697265625`
- Mean magnetisation leakage: `0.015228271484375`
- Mean parity leakage: `0.0150909423828125`

This is sufficient for readout sanity checks and state-specific parity-flip correction. It is not a full `2^n x 2^n` confusion-matrix inversion.

## Descriptive hardware outcome

This pilot does not support a hardware-protection claim for the SCPN/FIM term.

For `lambda = 1` relative to `lambda = 0`, across 15 matched state/depth comparisons:

- Mean state-retention delta: `-0.08836263020833333`
- Mean magnetisation-leakage delta: `0.11277669270833333`
- Mean parity-leakage delta: `0.080615234375`
- Lower magnetisation leakage: `0 / 15`
- Higher magnetisation leakage: `15 / 15`

For `lambda = 4` relative to `lambda = 0`, across 15 matched state/depth comparisons:

- Mean state-retention delta: `-0.09288736979166666`
- Mean magnetisation-leakage delta: `0.11559244791666666`
- Mean parity-leakage delta: `0.08427734375`
- Mean parity-leakage delta after state-specific readout-flip correction: `0.08717736719154022`
- Lower magnetisation leakage: `0 / 15`
- Higher magnetisation leakage: `15 / 15`

## Claim boundary

Supported:

- The submitted circuits executed and returned complete count dictionaries.
- The data are sufficient for sanity checks, readout-baseline comparison, and planning repeated randomized campaigns.

Not supported:

- Formal hardware p-values, because there is one hardware sample per lambda/depth/state condition.
- A claim that the FIM term improves hardware coherence.
- A hardware many-body-localisation claim.
- Full confusion-matrix readout mitigation, because only basis-state readout baselines were collected.

## Scientific interpretation

The pilot is valuable as a falsification and design step. Under this circuit family, transpilation, backend calibration, and initial-state set, increasing `lambda` did not reduce measured leakage. The next hardware experiment should therefore be smaller but replicated and randomized, with layout/state randomization and repeated samples per condition before any manuscript-level hardware claim is made.
