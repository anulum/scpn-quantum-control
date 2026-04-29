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
