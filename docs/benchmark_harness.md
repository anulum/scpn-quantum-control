# Benchmark Harness

## What this page is for

This page provides the reproducible entry point for public benchmark
reconstruction. It keeps low-cost datasets, loaders, and tolerance checks aligned
so published benchmark rows can be regenerated without submitting jobs.

The benchmark harness is the public no-QPU reproduction layer for published SCPN Quantum Control datasets. The first S5 benchmark covers the Phase 1 DLA-parity raw-count campaign.

## Python API

```python
from scpn_quantum_control.benchmark_harness import run_phase1_benchmark

result = run_phase1_benchmark(baselines_backend="numpy")
print(result.dataset.n_circuits_total)
print(result.reproduction.fisher.chi2)
print(result.classical_reference.max_abs_leakage)
```

## Command

```bash
scpn-bench s5-benchmark-suite
```

The command regenerates:

- `data/s5_benchmark_harness/phase1_benchmark_harness_2026-05-06.json`
- `docs/campaigns/benchmark_harness_phase1_2026-05-06.md`

## Scope

The harness loads committed raw-count JSON files, recomputes the published Phase 1 statistics, and checks the noiseless classical parity-conservation reference. It does not submit QPU jobs, does not contact IBM or other providers, and does not claim quantum advantage.

## Extension Rule

New public benchmarks must include raw-data provenance, a typed loader, a reproducer with tolerances, a classical or simulator baseline where scientifically meaningful, and an artefact emitted by `scpn-bench`.

## Tier Benchmark Provenance

The Kuramoto tier benchmark provenance layer records `git rev-parse HEAD` and
`rustc --version` only after resolving each executable to an absolute,
executable file path. Missing or non-executable tools are recorded as
`unknown` for the commit and `absent` for `rustc`, preserving offline
reconstruction without running partially resolved commands from ambient `PATH`
state.

## Registry

```python
from scpn_quantum_control.benchmark_harness import list_benchmark_families

for family in list_benchmark_families():
    print(family.benchmark_id, family.status)
```

```bash
scpn-bench s5-benchmark-registry
```

The registry marks planned entries as planned, not implemented. Planned rows have no command or generated artefact until their raw data, loader, reproducer, tolerance bundle, and baseline are committed.

## Phase-QNode Affinity Metadata

Differentiable Phase-QNode benchmark evidence uses
`run_phase_qnode_affinity_benchmark(...)` and the
`tools/run_phase_qnode_affinity_benchmark.py` CLI. The raw JSON records command,
requested CPU affinity, observed process affinity, isolation method, host load
before and after, CPU/governor or frequency context, dependency versions,
warmups, repetitions, runner metadata, and timing rows.

Rows are labelled `isolated_affinity` only when reserved CPU affinity, low host
load, matching observed process affinity, governor or frequency metadata, fixed
command metadata, and absence of heavy concurrent jobs are all recorded.
GitHub-hosted CI is explicitly non-promotional. The promotion gate is the
remote GitHub Actions lane on a self-hosted runner labelled `self-hosted`,
`linux`, and `isolated-benchmark`; the workflow fails unless its uploaded
evidence reports `isolated_affinity`. Otherwise the result is
`functional_non_isolated` or `hard_gap` and must not be used as a production
throughput or latency claim. The helper
`tools/setup_isolated_benchmark_runner.py` prints or installs the required
runner configuration on the reserved Linux x64 benchmark host.
`validate_phase_qnode_affinity_artifact(...)` is the attachment gate for those
raw JSON files: it hashes the artefact into a deterministic
`phase-qnode-affinity:<sha>` benchmark ID, counts raw timing rows, checks
host-isolation metadata, and reports `promotion_ready=True` only for
`isolated_affinity` artefacts with `production_benchmark=True` and no recorded
isolation failures.
