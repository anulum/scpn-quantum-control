# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Methods Benchmark Dashboard

# Methods Benchmark Dashboard

This page tracks the benchmark artefacts supporting the Rust/VQE methods
papers and the SCPN/FIM Hamiltonian paper. The rule is artefact-first:
tables and manuscript claims should be regenerated from committed scripts,
JSON summaries, and CSV summaries.

## Reproducibility commands

The `scpn-bench` entry point is the public one-command interface for local
artefact regeneration:

```bash
scpn-bench reproduce-methods
scpn-bench fim-all
scpn-bench all
```

Useful options:

| Option | Purpose |
| --- | --- |
| `--dry-run` | Print selected harnesses without executing them. |
| `--include-gpu` | Include optional GPU harnesses. |
| `--keep-going` | Continue after a failed harness and report all failures. |
| `--no-diff` | Skip the post-run committed-artefact diff summary. |

By default the CLI runs offline harnesses only. IBM preparation and submission
scripts are deliberately excluded from `scpn-bench`; IBM raw-count analyses are
included only where they consume already committed JSON data.

## Current artefact groups

| Group | Primary artefacts | Generator |
| --- | --- | --- |
| Rust core kernels | `data/rust_vqe_methods/rust_core_benchmark_summary_2026-05-05.json` | `scripts/benchmark_rust_core_methods.py` |
| Ansatz construction | `data/rust_vqe_methods/ansatz_benchmark_summary_2026-05-05.json` | `scripts/benchmark_ansatz_methods.py` |
| VQE comparison | `data/rust_vqe_methods/vqe_benchmark_summary_2026-05-05.json` | `scripts/benchmark_vqe_methods.py` |
| Multi-language K_nm parity | `data/rust_vqe_methods/multilang_knm_benchmark_summary_2026-05-05.json` | `scripts/benchmark_multilang_knm_methods.py` |
| Cross-machine CPU | `data/rust_vqe_methods/remote_knm_benchmark_*_2026-05-05.json` | `scripts/benchmark_remote_knm_machine.py` |
| Vertex T4 GPU | `data/rust_vqe_methods/gpu_benchmark_summary_vertex_t4_2026-05-05.json` | `scripts/benchmark_gpu_methods.py` |
| Combined methods summary | `data/rust_vqe_methods/combined_methods_benchmark_summary_2026-05-05.json` | `scripts/summarise_rust_vqe_method_artifacts.py` |
| Ansatz scaling plus tensor-network diagnostics | `data/rust_vqe_methods/ansatz_scaling_tn_summary_2026-05-05.json` | `scripts/benchmark_ansatz_scaling_tn.py` |
| FIM spectra | `data/scpn_fim_hamiltonian/fim_spectrum_summary_2026-05-05.json` | `scripts/analyse_fim_spectrum.py` |
| FIM level spacing | `data/scpn_fim_hamiltonian/fim_level_spacing_summary_2026-05-05.json` | `scripts/analyse_fim_level_spacing.py` |
| FIM entanglement | `data/scpn_fim_hamiltonian/fim_entanglement_summary_2026-05-05.json` | `scripts/analyse_fim_entanglement.py` |
| FIM sector survival | `data/scpn_fim_hamiltonian/fim_sector_survival_summary_2026-05-05.json` | `scripts/analyse_fim_sector_survival.py` |
| FIM VQE | `data/scpn_fim_hamiltonian/fim_vqe_ground_state_summary_2026-05-05.json` | `scripts/benchmark_fim_vqe_ground_state.py` |
| FIM IBM pilot analysis | `data/scpn_fim_hamiltonian/fim_ibm_pilot_analysis_2026-05-05.json` | `scripts/analyse_fim_ibm_pilot.py` |
| FIM IBM repeated analysis | `data/scpn_fim_hamiltonian/fim_ibm_repeated_followup_analysis_2026-05-05.json` | `scripts/analyse_fim_ibm_repeated_followup.py` |

## Current combined artefact hashes

| Artefact | SHA256 |
| --- | --- |
| `combined_methods_benchmark_summary_2026-05-05.json` | `593330a1dd19f495b899be1031ebe3dd4caa07171053aa376c2f761e557c1428` |
| `combined_methods_benchmark_summary_2026-05-05.csv` | `e69b94df590ff06708b3b21245864f74c3df630b514254526dc6c4af3fe24c2f` |

## Individual harness commands

The one-command CLI is preferred for reproducibility checks. Individual
harnesses remain useful when a single table needs to be regenerated during
development:

```bash
python scripts/benchmark_rust_core_methods.py
python scripts/benchmark_ansatz_methods.py
python scripts/benchmark_vqe_methods.py
python scripts/benchmark_multilang_knm_methods.py
python scripts/benchmark_gpu_methods.py
python scripts/summarise_rust_vqe_method_artifacts.py
python scripts/benchmark_ansatz_scaling_tn.py
python scripts/analyse_fim_spectrum.py
python scripts/analyse_fim_level_spacing.py
python scripts/analyse_fim_entanglement.py
python scripts/analyse_fim_sector_survival.py
python scripts/benchmark_fim_vqe_ground_state.py
python scripts/analyse_fim_ibm_pilot.py
python scripts/analyse_fim_ibm_repeated_followup.py
```

Remote or non-local machine artefacts should record the machine identity,
hardware context, command, timestamp, and checksum before being promoted into
`data/rust_vqe_methods/`.

## Implemented CLI behaviour

- Regenerate local deterministic artefacts from committed scripts.
- Keep optional GPU harnesses behind `--include-gpu`.
- Rebuild combined JSON and CSV summaries where a summariser exists.
- Compare regenerated artefacts with committed files.
- Report changed artefacts explicitly instead of silently accepting drift.
- Avoid spending QPU time or submitting hardware jobs.

## Machine provenance

Current promoted benchmark artefacts include:

- Local workstation CPU runs.
- ML350 CPU runs.
- Vertex `n1-standard-4` CPU runs.
- Vertex T4 GPU runs for batched dense expectation validation.

These timings are opportunistic and not isolated benchmark-lab measurements.
They are useful for reproducibility and cross-machine sanity checks, but the
papers should not interpret them as universal hardware performance constants.

## Planned extensions

### Ansatz scaling with tensor-network baselines

The next benchmark extension is an n=6--12 ansatz-scaling study with exact
diagonalisation where feasible and tensor-network references where exact
methods become impractical. The initial harness records circuit scaling for
n=4--12 and MPS truncation diagnostics from exact ground states up to the
configured exact limit. Larger tensor-network rows are marked as skipped until
a non-exact backend is wired and validated. Current outputs:

- `data/rust_vqe_methods/ansatz_scaling_tn_summary_*.json`
- `data/rust_vqe_methods/ansatz_scaling_summary_*.csv`
- `data/rust_vqe_methods/tn_truncation_summary_*.csv`

### Analog XY bridge

The analog bridge should start as an optional Pulser / Bloqade design spike for
neutral-atom XY mappings. It should remain separate from the default digital
Qiskit workflow until the mapping assumptions, dependencies, and reproducibility
artefacts are documented.
