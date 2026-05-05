# Rust/VQE software-methods paper plan

Date: 2026-05-05

## Working title

`A Rust-accelerated, topology-informed workflow for Kuramoto--XY quantum simulation and NISQ benchmarking`

## Recommended venue

Primary: SoftwareX or Journal of Open Source Software if framed as a software
paper.

Alternate: Journal of Computational Physics if the benchmark matrix is expanded
and the numerical method contribution is made central.

## Current safe claim boundary

Safe now:

- Python/Qiskit orchestration plus Rust/PyO3 hot-path kernels.
- Topology-informed ansatz generation from the support of `K_ij`.
- Command-provenance performance notes exist in `docs/pipeline_performance.md`.
- Hardware reproducibility workflow is proven by the DLA parity package:
  raw counts, metadata, job IDs, SHA256 hashes, and reproducer scripts.

Now supported by generated artefacts:

- Opportunistic local direct Rust `build_knm` speedups for `n=4..32`, with
  exact parity against the Python reference on tested matrices. These are not
  final publication-grade timing numbers because the workstation was shared and
  CPU load was not isolated.
- A multi-language comparator harness now reports Python/NumPy, Rust/PyO3,
  Julia, and Go timing lanes. Mojo is detected locally but recorded as
  `detected_not_benchmarked` because no stable Mojo `K_ij` kernel is implemented
  in the harness yet.
- `n=4`, two-repetition ansatz/VQE comparison over three seeds and equal
  optimiser budgets. The topology-informed ansatz has lower median and mean
  relative energy error in this small benchmark.

Still not safe:

- Do not claim `5401x` speedup; the fresh harness does not reproduce that
  headline.
- Do not claim ansatz superiority up to 20 qubits until rerun over larger `n`.
- Do not claim a universal coherence wall until hardware depth-scaling is
  re-analysed with calibration metadata and confidence intervals.
- Do not mix simulator-only, hardware-only, and old internal campaign claims
  without a provenance table.

## Minimum benchmark matrix before submission

Completed initial artefacts:

- `scripts/benchmark_rust_core_methods.py`
- `scripts/benchmark_ansatz_methods.py`
- `scripts/benchmark_vqe_methods.py`
- `scripts/benchmark_multilang_knm_methods.py`
- `data/rust_vqe_methods/rust_core_benchmark_summary_2026-05-05.json`
- `data/rust_vqe_methods/rust_core_benchmark_summary_2026-05-05.csv`
- `data/rust_vqe_methods/ansatz_benchmark_summary_2026-05-05.json`
- `data/rust_vqe_methods/ansatz_benchmark_summary_2026-05-05.csv`
- `data/rust_vqe_methods/vqe_benchmark_summary_2026-05-05.json`
- `data/rust_vqe_methods/vqe_benchmark_rows_2026-05-05.csv`
- `data/rust_vqe_methods/vqe_benchmark_aggregate_2026-05-05.csv`
- `data/rust_vqe_methods/multilang_knm_benchmark_summary_2026-05-05.json`
- `data/rust_vqe_methods/multilang_knm_benchmark_summary_2026-05-05.csv`

Remaining benchmark matrix before submission:

1. `build_knm` Python vs Rust:
   - `n in {4, 8, 16, 32, 64}`
   - 1000 iterations or enough to exceed timer noise
   - report mean, median, p95, speedup
   - initial pass complete, but keep rerunnable for final environment capture

2. Hamiltonian construction:
   - dense and sparse paths where applicable
   - `n in {3, 4, 6, 8, 10}` for exact dense comparisons
   - compare numeric equality to Qiskit/SciPy baseline

3. Ansatz construction:
   - `K_ij`-informed vs `TwoLocal` vs `EfficientSU2`
   - parameter count, two-qubit gate count, transpiled depth
   - `n in {3, 4, 6, 8}`
   - initial construction/transpile pass complete

4. VQE quality:
   - equal optimiser, equal iteration budget, multiple seeds
   - exact reference where classically tractable
   - report energy error and convergence distribution
   - initial `n in {3,4}` pass complete; larger `n` remains optional

5. Reproducibility packaging:
   - generated JSON/CSV benchmark artefacts
   - SHA256 hashes
   - one command to regenerate figures/tables

## Manuscript scaffold

Created:

- `paper/rust_vqe_methods.tex`

The scaffold is intentionally conservative. It includes placeholder tables and
explicit TODO-style language where fresh benchmark data is required.

## Suggested next implementation step

Create figures under `figures/rust_vqe_methods/` from the generated CSV files,
then decide whether to compile the manuscript as REVTeX or move it to the target
venue template.

## Activated follow-up work from final methods review (2026-05-05)

These items extend the methods-paper artefact discipline into executable
publication infrastructure and the next validation layer. They are not claims in
the current paper until the corresponding artefacts exist.

### M1. One-command reproducibility CLI

Goal: provide a single public command that regenerates every benchmark artefact
used by the methods papers and reports whether the regenerated outputs match the
committed files.

Candidate commands:

```bash
scpn-bench reproduce-methods
scpn-bench diff-artifacts
scpn-bench all
```

Initial command scope:

- Run the local CPU-safe benchmark harnesses for Rust kernels, ansatz
  construction, VQE aggregates, and multi-language coupling-matrix parity.
- Re-run the combined summariser for `data/rust_vqe_methods/`.
- Detect unavailable optional backends, such as CUDA, remote Vertex, ML350, Go,
  Julia, or Mojo, and record a structured skip reason instead of failing the
  local reproducibility run.
- Emit JSON and CSV summaries plus SHA256 checksums.
- Diff regenerated artefacts against committed artefacts and produce a concise
  changed-file report.

Acceptance criteria:

- `scpn-bench reproduce-methods` regenerates all local deterministic artefacts
  from committed scripts.
- `scpn-bench diff-artifacts` exits cleanly when artefacts match and returns a
  non-zero status with a file-level diff summary when they do not.
- The command never silently overwrites committed benchmark artefacts without a
  visible provenance record.

### M2. Public benchmark dashboard

Goal: expose the latest benchmark artefacts, machine provenance, regeneration
commands, hashes, and caveats through the public documentation site.

Initial deliverable:

- Add `docs/methods_benchmark_dashboard.md`.
- Link it from `mkdocs.yml`.
- Show current artefact groups: Rust core, ansatz, VQE, multi-language parity,
  local/ML350/Vertex CPU, Vertex T4 GPU, and combined summaries.
- Include the existing script-level regeneration commands now, and reserve the
  future CLI commands from M1 as the preferred interface.

Acceptance criteria:

- The dashboard renders through MkDocs.
- Every table points to a committed JSON or CSV artefact.
- Opportunistic shared-machine timing caveats are visible on the page, not only
  in the manuscript.

### M3. Ansatz scaling with tensor-network baselines

Goal: extend the n=4 VQE methods signal into a scaling study with independent
classical references.

Initial design:

- Evaluate n=6, n=8, n=10, and n=12 where computationally feasible.
- Compare topology-informed ansatz families against generic baselines used in
  the methods paper.
- Use exact diagonalisation only while feasible, then switch to tensor-network
  references through quimb or ITensor-style MPS workflows.
- Store generated summaries as
  `data/rust_vqe_methods/ansatz_scaling_tn_summary_*.json` and `.csv`.

Acceptance criteria:

- The benchmark harness records Hamiltonian parameters, ansatz parameters,
  optimiser settings, reference method, seed policy, machine provenance, and
  wall-clock context.
- The paper or dashboard only reports values regenerated from those artefacts.
- Results are framed as classical scaling and ansatz-design evidence, not as
  quantum advantage.

### M4. Analog XY bridge: Pulser and Bloqade

Goal: explore an analog-native backend for Kuramoto-XY dynamics where neutral
atom platforms may represent oscillator-network structure with less digital
Trotter overhead.

Initial design:

- Add an optional research spike for Pulser and Bloqade mappings.
- Keep dependencies optional and feature-gated; the core package must not depend
  on analog-platform packages by default.
- Start with a design document and a minimal mapper prototype before adding
  benchmark claims.

Acceptance criteria:

- The bridge states the exact XY mapping assumptions and platform constraints.
- Any generated example is reproducible from committed parameters.
- The documentation clearly separates digital Qiskit workflows from analog
  neutral-atom exploration.
