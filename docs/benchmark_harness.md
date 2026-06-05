# Benchmark Harness

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
CPU affinity, isolation method, host load before and after, CPU/governor
context when available, dependency versions, warmups, repetitions, and timing
rows.

Rows are labelled `isolated_affinity` only when reserved CPU affinity, low host
load, governor or frequency metadata, fixed command metadata, and absence of
heavy concurrent jobs are all recorded. Otherwise the result is
`functional_non_isolated` or `hard_gap` and must not be used as a production
throughput or latency claim. The GitHub Actions lane must run on a self-hosted
runner labelled `self-hosted`, `linux`, and `isolated-benchmark`; the helper
`tools/setup_isolated_benchmark_runner.py` prints or installs that runner
configuration.
