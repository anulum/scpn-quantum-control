# Synchronisation-Witness Evidence

- Schema: `scpn_qc_sync_witness_evidence_v1`
- Artifact id: `sync-witness-evidence-local`
- Classification: `functional_non_isolated`
- Passed: `True`
- Claim boundary: bounded synthetic phase-cloud synchronisation witnesses (harmonic Kuramoto order parameters and exact Vietoris-Rips persistent homology in dimensions 0 and 1 over geodesic phase distances) with known reference regimes; not hardware phase tomography, provider execution, isolated timing, or high-dimensional manifold inference

## Executable Rows

| Case | Regime | Order r1 | Components | Dominant H1 | Passed |
| --- | --- | ---: | ---: | ---: | --- |
| `synchronised_eight_node` | `synchronised` | 0.999908 | 1 | 0 | `True` |
| `desynchronised_eight_node` | `desynchronised` | 4.30874e-17 | 8 | 1.5708 | `True` |
| `clustered_three_group` | `clustered` | 1.24127e-16 | 3 | 0.06 | `True` |

## Boundary Rows

| Boundary | Status | Reason |
| --- | --- | --- |
| `high_dimensional_manifold_boundary` | `hard_gap` | persistent homology beyond one-dimensional phase clouds requires manifold-embedding and identifiability analysis outside this suite |
| `hardware_phase_tomography_boundary` | `hard_gap` | provider-backed phase tomography requires raw counts, calibration, and owner-approved hardware tickets |
