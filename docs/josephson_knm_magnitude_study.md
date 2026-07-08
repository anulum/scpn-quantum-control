# Josephson K_nm Magnitude Study

This QWC-5.2 artifact records the Josephson topology-correlation
candidate and the measured-magnitude gates required before any
physical K_nm coupling claim.

## Boundary

Topology-correlation candidate and magnitude-study preregistration only; this artifact does not validate K_nm physical coupling magnitudes.

## Candidate

| field | value |
| --- | --- |
| N | 14 |
| topology | all_to_all |
| topology source | illustrative_all_to_all |
| topology correlation | 0.989947 |
| rounded topology correlation | 0.990 |
| coupling ratio | 0.034448 |
| parameter source | nominal_transmon_literature |
| frequency source | canonical_OMEGA_N_16_prefix |
| E_J/E_C | 60.0 |
| transmon regime | True |
| claim status | topology_candidate_magnitude_blocked |

## Required Calibration Fields

- `system_id`
- `device_or_array_source`
- `coupling_edges_0_indexed`
- `coupling_unit`
- `normalisation`
- `normalisation_locked`
- `value`
- `uncertainty`
- `calibration_timestamp`
- `source_reference`

## Extension Targets

- N=20
- N=30
- N=40

## Promotion Gates

| gate | current status | evidence required |
| --- | --- | --- |
| calibrated_coupling_units | blocked_nominal_parameters_only | coupling_unit must denote calibrated Hz, rad/s, GHz, or derived Josephson energy; every coupling edge must carry source metadata |
| locked_normalisation_and_uncertainty | blocked_no_measured_uncertainty | normalisation_locked=true; per-edge uncertainty is finite and non-negative |
| direct_magnitude_fit | blocked_no_calibrated_artifact | relative RMSE <= 0.05; all reported edges participate in the fit |
| spectral_response | blocked_no_calibrated_artifact | critical-coupling proxy relative difference <= 0.05; weighted adjacency and Laplacian spectra are recorded |
| null_models | blocked_no_calibrated_artifact | node-label null model gate passes; edge-value null model gate passes |

## Blocked Claims

- No K_nm measured-magnitude validation from topology correlation alone.
- No physical-unit Josephson coupling claim from nominal literature parameters.
- No hardware-device coupling-map claim without backend calibration provenance.
- No promotion over EEG or power-grid controls until the same gates pass.

## Next Actions

- Collect or derive a calibrated Josephson/transmon coupling artifact in the declared schema.
- Run the existing K_nm physical-validation audit on the calibrated artifact at the candidate and extension sizes.
- Require direct magnitude, spectral response, uncertainty, and null-model gates before promotion.

## Regeneration

```bash
scpn-bench knm-josephson-magnitude-study
```
