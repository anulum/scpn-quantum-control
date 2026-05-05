# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Methods Benchmark Dashboard

# Methods Benchmark Dashboard

This page tracks the benchmark artefacts supporting the Rust/VQE methods
papers. The rule is artefact-first: tables and manuscript claims should be
regenerated from committed scripts, JSON summaries, and CSV summaries.

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

## Current combined artefact hashes

| Artefact | SHA256 |
| --- | --- |
| `combined_methods_benchmark_summary_2026-05-05.json` | `593330a1dd19f495b899be1031ebe3dd4caa07171053aa376c2f761e557c1428` |
| `combined_methods_benchmark_summary_2026-05-05.csv` | `e69b94df590ff06708b3b21245864f74c3df630b514254526dc6c4af3fe24c2f` |

## Existing regeneration commands

Until the one-command CLI is implemented, regenerate artefacts through the
individual harnesses:

```bash
python scripts/benchmark_rust_core_methods.py
python scripts/benchmark_ansatz_methods.py
python scripts/benchmark_vqe_methods.py
python scripts/benchmark_multilang_knm_methods.py
python scripts/benchmark_gpu_methods.py
python scripts/summarise_rust_vqe_method_artifacts.py
```

Remote or non-local machine artefacts should record the machine identity,
hardware context, command, timestamp, and checksum before being promoted into
`data/rust_vqe_methods/`.

## Planned one-command interface

The planned CLI should provide:

```bash
scpn-bench reproduce-methods
scpn-bench diff-artifacts
scpn-bench all
```

Expected behaviour:

- Regenerate local deterministic artefacts from committed scripts.
- Detect unavailable optional backends and record structured skip reasons.
- Rebuild combined JSON and CSV summaries.
- Compare regenerated artefacts with committed files.
- Report changed artefacts explicitly instead of silently accepting drift.

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
methods become impractical. Candidate outputs:

- `data/rust_vqe_methods/ansatz_scaling_tn_summary_*.json`
- `data/rust_vqe_methods/ansatz_scaling_tn_summary_*.csv`

### Analog XY bridge

The analog bridge should start as an optional Pulser / Bloqade design spike for
neutral-atom XY mappings. It should remain separate from the default digital
Qiskit workflow until the mapping assumptions, dependencies, and reproducibility
artefacts are documented.
