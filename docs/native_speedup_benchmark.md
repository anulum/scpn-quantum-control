# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Native dense-Hamiltonian speedup benchmark

# Native Speedup Benchmark

This page documents the reproducible Rust-vs-Qiskit benchmark for dense
XY-Hamiltonian construction — the operation the retired "5401× faster than
Qiskit" headline referred to. It follows the unified GOTM benchmark standard
(`agentic-shared/BENCHMARK_STANDARD.md`) and mirrors the SCPN-CONTROL regression
apparatus.

## What is measured, and what is claimed

The harness builds the dense XY Hamiltonian two ways from the *same* `K`/`omega`
and parity-checks the results:

- **`rust_pyo3`** — `scpn_quantum_engine.build_xy_hamiltonian_dense`, the real
  `float64` operator kernel.
- **`qiskit_sparsepauliop`** — `knm_to_hamiltonian(K, omega).to_matrix()`.

Each backend is warmed up and then sampled with repeats; P50/P95/P99 and
throughput are recorded with full provenance (CPU model, Rust release profile
read from `Cargo.toml`, commit, CPU affinity, load average, peak RSS).

**The numbers are environment-dependent.** They swing with CPU pinning, BLAS
threading, and host load, so the artefacts are marked
`production_claim_allowed: false` — they are a reproducible *local regression
guard*, not a published performance claim. The earlier "5401×" figure was a
cold-start artefact: an un-warmed Qiskit first-call timed at ~20.9 ms. With
warm-up the Rust kernel advantage is large for small systems and shrinks as the
dense `2^n × 2^n` fill dominates. The production `knm_to_dense_matrix` wrapper
additionally casts `float64 → complex128`; that cast dominates at large `L` and
is a downstream cost, excluded from this construction-kernel comparison.

## Declared-hardware baseline (committed)

Committed at `benchmarks/baselines/native_speedup.json`, measured on the GOTM
workstation (i5-11600K @ 3.90 GHz), pinned to one core for tight samples:

| System | Rust kernel p50 | Qiskit p50 | Speedup (p50) | Parity |
|--------|-----------------|------------|---------------|--------|
| L=4 (16×16)     | 2.79 µs    | 269.5 µs   | 96.5× | ✓ |
| L=8 (256×256)   | 23.1 µs    | 779.0 µs   | 33.7× | ✓ |
| L=10 (1024×1024)| 635.3 µs   | 2131.3 µs  | 3.35× | ✓ |
| L=12 (4096×4096)| 42.2 ms    | 93.0 ms    | 2.20× | ✓ |

## CI evidence (side by side)

The [`Native Speedup Benchmark`](https://github.com/anulum/scpn-quantum-control/actions/workflows/benchmark-native-speedup.yml)
workflow regenerates the report on a fixed `ubuntu-latest` runner (nightly and
on demand via `workflow_dispatch`) and uploads it as the
`native-speedup-benchmark` artefact. Because a hosted runner's CPU differs from
the declared-hardware baseline, the regression gate runs in **evidence-only**
mode there: it collects the verdict (and flags `hardware_mismatch`) but does not
block. Real gating is intended on declared or self-hosted hardware whose baseline
was captured on the same CPU. Download the CI artefact to compare its numbers
against the declared-hardware baseline above.

## Reproduce it yourself — and share your results

```bash
pip install -e ".[accelerated]"        # builds + installs the Rust engine
python scripts/benchmark_native_speedup.py --pin-core <idle-core> \
    --json-out my_report.json
python tools/benchmark_native_speedup_gate.py \
    --report my_report.json \
    --baseline benchmarks/baselines/native_speedup.json
```

The gate validates payload and baseline tamper digests, applies the
[`benchmarks/native_speedup_thresholds.toml`](https://github.com/anulum/scpn-quantum-control/blob/main/benchmarks/native_speedup_thresholds.toml)
policy direction-aware (latency upper-bounded, throughput lower-bounded), and
fails closed on missing/tampered evidence. On a different CPU it reports
`hardware_mismatch` rather than a misleading pass/fail. We welcome reproductions
on other hardware — open an issue with your report JSON and we will add it.

## Regenerating the committed baseline

```bash
python scripts/benchmark_native_speedup.py --pin-core <idle-core> \
    --write-baseline benchmarks/baselines/native_speedup.json
```

The baseline carries `evidence_class: local_regression`,
`production_claim_allowed: false`, full provenance, and a `baseline_sha256`
tamper digest.
