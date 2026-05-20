# Rust/VQE Methods Paper Submission Checklist

Date: 2026-05-06

This checklist freezes the submission boundary for the Rust/VQE methods paper.
It indexes committed artefacts only and does not introduce new benchmark
results. Manuscript tables must be copied from generated JSON/CSV summaries or
from the `scpn-bench` dashboard output, never hand-edited from memory.

## Submission Scope

Supported:

- Python/Qiskit orchestration with Rust/PyO3 hot-path kernels.
- Topology-informed ansatz construction from the support of `K_ij`.
- `n=4` small VQE comparison under equal optimiser budget and seed policy.
- Multi-language `K_nm` parity checks across Python/NumPy, Rust/PyO3, Julia,
  and Go where the corresponding runtime is available.
- Cross-machine CPU timing artefacts for local, ML350, and Vertex contexts.
- Vertex T4 GPU batched dense-expectation validation.
- Optional n=4--12 ansatz-scaling and tensor-network diagnostic artefacts.
- One-command offline reproduction through `scpn-bench reproduce-methods`.

Not supported:

- A universal end-to-end VQE speedup claim.
- The obsolete `5401x` speedup headline.
- Quantum advantage.
- A proof that the topology-informed ansatz is globally superior.
- Publication of opportunistic shared-machine timings as hardware constants.
- Any benchmark number that is absent from committed JSON/CSV artefacts.

## Required Claim Wording

Use the conservative claim:

> The package provides an artefact-first Kuramoto-XY workflow in which selected
> CPU hot paths are accelerated with Rust/PyO3, topology-informed ansatze are
> benchmarked against generic baselines on small systems, and all promoted
> performance tables are regenerated from committed JSON/CSV artefacts.

Avoid the stronger claim:

> The Rust engine accelerates the full quantum simulation or VQE workflow by a
> fixed factor.

Any speedup statement must name the measured kernel, machine context, benchmark
date, and artefact source.

## Committed Artefact Index

| Item | Path |
|------|------|
| Methods paper plan | `docs/publication/rust_vqe_methods_paper_plan_2026-05-05.md` |
| Public benchmark dashboard | `docs/methods_benchmark_dashboard.md` |
| Rust core summary | `data/rust_vqe_methods/rust_core_benchmark_summary_2026-05-05.json` |
| Ansatz construction summary | `data/rust_vqe_methods/ansatz_benchmark_summary_2026-05-05.json` |
| VQE benchmark summary | `data/rust_vqe_methods/vqe_benchmark_summary_2026-05-05.json` |
| VQE aggregate CSV | `data/rust_vqe_methods/vqe_benchmark_aggregate_2026-05-05.csv` |
| Multi-language summary | `data/rust_vqe_methods/multilang_knm_benchmark_summary_2026-05-05.json` |
| Combined summary | `data/rust_vqe_methods/combined_methods_benchmark_summary_2026-05-05.json` |
| Vertex T4 GPU summary | `data/rust_vqe_methods/gpu_benchmark_summary_vertex_t4_2026-05-05.json` |
| Local GPU summary | `data/rust_vqe_methods/gpu_benchmark_summary_aaarthuus_local_2026-05-05.json` |
| ML350 CPU summary | `data/rust_vqe_methods/remote_knm_benchmark_ml350_2026-05-05.json` |
| Vertex CPU summary | `data/rust_vqe_methods/remote_knm_benchmark_vertex_n1_standard_4_2026-05-05.json` |
| Paper source | `paper/submissions/submission_003_rust_vqe_methods/rust_vqe_methods.tex` |
| Paper PDF | `paper/submissions/submission_003_rust_vqe_methods/rust_vqe_methods.pdf` |

## Generator Scripts

| Artefact group | Generator |
|----------------|-----------|
| Rust core kernels | `scripts/benchmark_rust_core_methods.py` |
| Ansatz construction | `scripts/benchmark_ansatz_methods.py` |
| VQE comparison | `scripts/benchmark_vqe_methods.py` |
| Multi-language parity | `scripts/benchmark_multilang_knm_methods.py` |
| GPU dense expectations | `scripts/benchmark_gpu_methods.py` |
| Cross-machine summaries | `scripts/benchmark_remote_knm_machine.py` |
| Combined summary | `scripts/summarise_rust_vqe_method_artifacts.py` |
| Ansatz scaling plus tensor-network diagnostics | `scripts/benchmark_ansatz_scaling_tn.py` |

## Preferred Reproduction Gate

Run before updating manuscript tables:

```bash
scpn-bench reproduce-methods
```

Optional heavier gates:

```bash
scpn-bench reproduce-methods --include-scaling
scpn-bench reproduce-methods --include-gpu
```

The CLI must report any artefact drift before a paper number is changed. A
local rerun on a shared workstation can change timing rows; such drift must be
reviewed as machine-context evidence, not automatically promoted.

## Final Manual Pre-Upload Gate

Before arXiv, SoftwareX, or related upload:

- Rebuild `paper/submissions/submission_003_rust_vqe_methods/rust_vqe_methods.pdf` from `paper/submissions/submission_003_rust_vqe_methods/rust_vqe_methods.tex`.
- Verify every table number against the committed JSON/CSV artefact cited in
  the caption or text.
- Verify the GitHub URL is `https://github.com/anulum/scpn-quantum-control`.
- Verify all benchmark commands use the public CLI or the listed generator
  scripts.
- Verify the limitations section states Amdahl's-law scope: Rust accelerates
  selected hot paths, not the whole Qiskit/transpilation/VQE workflow.
- Verify opportunistic timing caveats are visible near any CPU/GPU timing table.
- Verify GPU claims are restricted to batched dense expectation workloads.
- Verify no IBM QPU, hardware advantage, or live-provider claim is introduced
  by the methods paper.
- Verify the AI disclosure, if required by the venue, is minimal and venue
  compliant.

## QPU Boundary

No QPU time is required for this methods submission package. The paper may cite
the DLA parity hardware package as an application case study, but benchmark
tables in this methods paper are generated from local/remote classical artefacts
and committed raw-count analyses only.
