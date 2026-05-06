# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Public application benchmarks

# Public Application Benchmarks

This directory contains small, curated QPU data artifacts for the
application plugin registry. They are designed for reproducible
software validation and documentation examples, not for replacing raw
domain archives.

| Dataset | Domain | Source basis | Plugin extra |
|---------|--------|--------------|--------------|
| `eeg_alpha_plv_8ch.json` | EEG | Alpha-band PLV topology from published functional-connectivity patterns. | `app-eeg` |
| `iter_mhd_8mode.json` | Plasma | ITER-scale NTM/RWM mode-locking topology from public MHD literature. | `app-plasma` |
| `ieee5bus_power_grid.json` | Power grid | IEEE 5-bus public benchmark constants converted to Kuramoto coupling. | `app-power-grid` |
| `friston_fep_6node.json` | FEP | Predictive-coding precision graph and observations from public FEP workflow equations. | `app-fep` |

Each JSON uses the `scpn-quantum-control.qpu-data-artifact.v1` schema,
includes SHA-256 hashes for the numeric arrays, carries a curation
timestamp, and is validated by `tests/test_application_plugins.py`.

The raw-source distinction is explicit:

- EEG and plasma files are compact curated benchmark matrices derived
  from public literature, not raw recordings or discharge traces.
- The power-grid file is a small public benchmark-constant conversion.
- The FEP file is a workflow benchmark with beliefs, observations, and
  sensory precision in metadata; it contains no human-subject data.

Deterministic smoke command:

```bash
.venv-linux/bin/python - <<'PY'
from scpn_quantum_control.applications import run_application_benchmark_suite

results = run_application_benchmark_suite()
print(sorted(results))
PY
```

This command exercises the packaged artefacts only. It does not download
raw EEG files, submit QPU jobs, or require optional domain dependencies.
