# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Application benchmark plugins

# Application Benchmark Plugins

Application plugins expose domain datasets through the same QPU data
artifact contract used by the Kuramoto-XY pipeline. The built-in
plugins cover EEG, tokamak MHD mode locking, IEEE power-grid
synchronisation, and Friston-style predictive coding.

```python
from scpn_quantum_control.applications import (
    compile_application_problem,
    run_application_benchmark_suite,
)

results = run_application_benchmark_suite()
problem = compile_application_problem("power_grid_ieee5")
```

## Plugin Extras

The optional extras keep domain dependencies off the default install:

| Extra | Intended domain stack |
|-------|------------------------|
| `app-eeg` | EEG/MEG file readers and MNE pipelines. |
| `app-plasma` | HDF5/tabular tokamak or plasma diagnostics. |
| `app-power-grid` | Power-system case readers and grid toolchains. |
| `app-fep` | Structured predictive-coding workflow configuration. |
| `app-benchmarks` | All four application stacks. |

The packaged benchmark JSON files do not require those extras. The
extras are for users who plug in external raw archives and want the
same registry path to build a QPU-ready artifact.

## Packaged Datasets

The in-repo artifacts live in `data/public_application_benchmarks/`.
Each file validates as `QPUDataArtifact`, carries array hashes, and can
be adapted to the public `KuramotoProblem` facade.

| Dataset | Plugin | Pipeline path |
|---------|--------|---------------|
| `eeg_alpha_plv_8ch` | `eeg_alpha` | PLV matrix → QPU artifact → EEG topology benchmark → Kuramoto facade. |
| `iter_mhd_8mode` | `plasma_iter_mhd` | NTM/RWM mode graph → QPU artifact → mode-locking benchmark → Kuramoto facade. |
| `ieee5bus_power_grid` | `power_grid_ieee5` | IEEE 5-bus constants → QPU artifact → grid synchronisation benchmark → Kuramoto facade. |
| `friston_fep_6node` | `friston_fep` | Precision graph + observations → QPU artifact → variational free energy + predictive-coding step. |

Third-party plugins register factories under the
`scpn_quantum_control.application_plugins` entry-point group. A broken
plugin is logged and skipped so one domain adapter cannot block the
rest of the benchmark suite.

## Curated Researcher Workflows

The promoted researcher workflows are deliberately small and
deterministic. They are meant to demonstrate the application boundary,
provenance trail, and QPU-ready artefact format without presenting
compact benchmark matrices as substitutes for raw domain archives.

| Workflow | Promoted artefacts | Provenance boundary | Deterministic regeneration |
|----------|--------------------|---------------------|----------------------------|
| GraphML/CSV topology import | External user-supplied graph or edge table converted to `QPUDataArtifact`. | Bring-your-own topology path; the repository does not ship private third-party graph archives. | Use the application-plugin registry and validate the converted artefact before adapting it to `KuramotoProblem`. |
| EEG alpha PLV | `data/public_application_benchmarks/eeg_alpha_plv_8ch.json`; measured audit artefacts in `data/knm_physical_validation/`, including `eeg_alpha_plv_knm_comparison.json`. | Public-literature benchmark matrix for examples; raw EDF cohorts stay outside Git under `<private-local-record>`. The K_nm audit keeps PLV non-promotional because it is an association observable, not a calibrated coupling magnitude. | `scripts/build_real_eeg_plv_validation_dataset.py` and `scripts/compare_eeg_plv_cohorts.py` regenerate the cohort artefacts; `scripts/run_knm_physical_validation_audit.py --measured data/knm_physical_validation/measured_couplings.json --n-layers 8` regenerates the K_nm comparison. |
| IEEE power grid | `data/public_application_benchmarks/ieee5bus_power_grid.json`; `data/knm_physical_validation/measured_couplings_power_grid_ieee5bus.json`; `data/knm_physical_validation/measured_couplings_power_grid_ieee14bus.json`; `data/knm_physical_validation/power_grid_ieee14bus_knm_comparison.json`. | Public IEEE 5-bus constants are converted to swing-equation coupling; public IEEE 14-bus branch reactance and voltage constants provide a larger voltage-weighted admittance control. Both remain negative/control candidates until the measured-system promotion gate passes. | `scripts/build_power_grid_measured_couplings.py` regenerates the 5-bus artefact; add `--case ieee14` for the 14-bus admittance candidate; run `scripts/run_knm_physical_validation_audit.py --measured ... --n-layers 14` for the comparison. |
| Plasma/tokamak | `data/public_application_benchmarks/iter_mhd_8mode.json`. | Curated ITER-scale mode-locking topology from public MHD literature, not raw discharge traces. | `run_application_benchmark_suite()` includes the packaged plasma benchmark without optional HDF5 dependencies. |
| Notebook and example workflows | `examples/02_kuramoto_xy_demo.py`, `examples/05_vqe_ansatz_comparison.py`, `examples/09_classical_vs_quantum_benchmark.py`, `examples/13_iter_disruption_demo.py`, `examples/18_end_to_end_pipeline.py`, `examples/19_sync_witness_operator.py`, and `examples/20_quantum_persistent_homology.py`. | Notebooks remain narrative wrappers; reusable logic stays in `src/`, `scripts/`, and versioned example files. | Static example tests ensure promoted examples remain parseable, expose `main()`, and are listed in `examples/README.md`. |

For a no-credential smoke path, run:

```bash
.venv-linux/bin/python - <<'PY'
from scpn_quantum_control.applications import run_application_benchmark_suite

results = run_application_benchmark_suite()
print(sorted(results))
PY
```

This command loads the packaged JSON artefacts only. It does not submit
IBM jobs, download raw EEG data, or touch private datasets.
