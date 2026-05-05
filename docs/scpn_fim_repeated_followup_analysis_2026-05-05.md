# SCPN/FIM repeated IBM follow-up analysis

Date: 2026-05-05

Backend: `ibm_kingston`

Job ID: `d7t5gtaudops7397ikn0`

## Artefacts

Pending submission:

- `data/scpn_fim_hamiltonian/fim_ibm_repeated_followup_pending_2026-05-05_d7t5gtaudops7397ikn0.json`
- SHA256: `b2c183631e1ead2b41120a69eca311d784a067b3ad95489570061d9eac2858f9`

Raw counts:

- `data/scpn_fim_hamiltonian/fim_ibm_repeated_followup_raw_counts_2026-05-05_d7t5gtaudops7397ikn0.json`
- SHA256: `6e4df78f1c679cd29b9c503bf1fecf39be76e707b1e6c4df99bbcc87b8e50d44`

Analysis:

- `scripts/analyse_fim_ibm_repeated_followup.py`
- `data/scpn_fim_hamiltonian/fim_ibm_repeated_followup_analysis_2026-05-05_d7t5gtaudops7397ikn0.json`
- SHA256: `8bd361b360248a0f00880ad23e9d4898d6e4b2c1ac12a60739e977890b0b18db`

Row metrics:

- `data/scpn_fim_hamiltonian/fim_ibm_repeated_followup_row_metrics_2026-05-05_d7t5gtaudops7397ikn0.csv`
- SHA256: `de26e6d9e9332e0fcc72e80948362a03326ab4b20add437f496fc6a1b28fa8c9`

Comparisons:

- `data/scpn_fim_hamiltonian/fim_ibm_repeated_followup_comparisons_2026-05-05_d7t5gtaudops7397ikn0.csv`
- SHA256: `e9f78765d988224c95a40af9435446c12202079eb80e594b0fa315ae8c5569fe`

Reproduction command:

```bash
env PYTHONPATH=src /home/anulum/.local/bin/python scripts/analyse_fim_ibm_repeated_followup.py --verify-integrity
```

## Run size

- Circuits: `166`
- Main repeated FIM rows: `150`
- Readout baseline rows: `16`
- Shots: `339,968`
- Shots per circuit: `2048`
- Wait wall time: `446.0912413597107` seconds
- Maximum live transpiled depth in returned metadata: `540`
- Maximum live two-qubit gates in returned metadata: `158`

## Readout baseline

The readout-only baseline over all 16 computational basis states gives:

- Mean exact-state retention: `0.98272705078125`
- Mean magnetisation leakage: `0.01727294921875`
- Mean parity leakage: `0.017181396484375`

This supports state-specific sanity checks and parity-flip correction. It is not a full confusion-matrix inversion.

## Primary result

The repeated follow-up falsifies the simple hardware-protection interpretation for this backend/circuit family.

For `lambda = 4` relative to `lambda = 0`, across 15 matched state/depth comparisons with five replicates per condition:

- State retention mean delta: `-0.0796875`
- State retention Fisher p: `1.1055758219655449e-48`
- State retention deltas: `0 / 15` positive, `15 / 15` negative

- Magnetisation leakage mean delta: `+0.08955729166666666`
- Magnetisation leakage Fisher p: `7.488374675730006e-55`
- Magnetisation leakage deltas: `14 / 15` positive, `1 / 15` negative

- Parity leakage mean delta: `+0.06151692708333333`
- Parity leakage Fisher p: `6.267550425892641e-48`
- Parity leakage deltas: `13 / 15` positive, `2 / 15` negative

- Readout-corrected parity leakage mean delta: `+0.06439517298321257`
- Readout-corrected parity leakage Fisher p: `6.267550425892548e-48`
- Readout-corrected parity leakage deltas: `13 / 15` positive, `2 / 15` negative

Interpretation: increasing `lambda` from `0` to `4` decreased exact-state retention and increased leakage for nearly all matched conditions. The sign is opposite to a simple FIM hardware-protection claim.

## Claim boundary

Allowed claims:

- The repeated IBM run completed and produced complete raw count dictionaries.
- On `ibm_kingston`, for this n=4 Trotter circuit family, `lambda = 4` increased measured leakage relative to `lambda = 0` in the repeated design.
- This falsifies the simple claim that the tested FIM term improves hardware coherence under this circuit/backend configuration.

Blocked claims:

- No backend-general FIM protection claim.
- No hardware many-body-localisation claim.
- No full confusion-matrix mitigation claim.
- No claim that the negative result invalidates all possible FIM Hamiltonian designs; it applies to this backend, circuit construction, depth set, and observable set.

## Scientific consequence

The SCPN/FIM paper should be framed as:

- A theoretically and numerically defined self-referential Hamiltonian family.
- An offline spectral/entanglement/VQE methods result.
- A hardware falsification of the simple coherence-protection hypothesis for the tested digital Trotter implementation on `ibm_kingston`.

This is still scientifically useful because it prevents overclaiming and identifies that the FIM term, as implemented here, adds circuit complexity/noise faster than any protection effect can be observed on this hardware configuration.
