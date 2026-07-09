# Coupling-Recovery Evidence

- Schema: `scpn_qc_coupling_recovery_evidence_v1`
- Artifact id: `coupling-recovery-evidence-local`
- Classification: `functional_non_isolated`
- Passed: `True`
- Claim boundary: bounded synthetic Kuramoto phase and XY pair-energy time-series recovery with known ground truth; not hardware Hamiltonian learning, provider execution, isolated timing, or arbitrary partial-observation inference

## Executable Rows

| Case | Family | Max error | RMSE | Valid rows | Rank | Passed |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `kuramoto_clean_three_node` | `kuramoto_phase` | 0.00117263 | 0.00112595 | 1.000 | 3 | `True` |
| `kuramoto_noisy_missing_three_node` | `kuramoto_phase` | 0.00403615 | 0.00332037 | 0.870 | 3 | `True` |
| `xy_pair_energy_noisy_missing_three_node` | `xy_pair_energy` | 0.000307398 | 0.000233742 | 0.948 | 3 | `True` |

## Boundary Rows

| Boundary | Status | Reason |
| --- | --- | --- |
| `partial_observation_inference_boundary` | `hard_gap` | arbitrary partial-observation oscillator inference requires identifiability analysis beyond this bounded synthetic suite |
| `hardware_hamiltonian_learning_boundary` | `hard_gap` | provider-backed XY Hamiltonian learning requires raw counts, calibration, and owner-approved hardware tickets |
