# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — local differentiable benchmark evidence

# Local Differentiable Benchmark Evidence — 2026-06-16T09:55Z

This directory records local differentiable-programming benchmark evidence
captured on host `aaarthuus`. The evidence is `functional_non_isolated` only.
It is not production benchmark evidence and must not be used for throughput,
latency, energy, market-facing, or promotion claims.

## Hardware And Isolation Context

- CPU: 11th Gen Intel(R) Core(TM) i5-11600K @ 3.90GHz.
- Logical CPUs: 12, online set `0-11`.
- OS: Ubuntu 24.04.4 LTS, Linux `6.17.0-35-generic`.
- Hardware: ASRock H510 Pro BTC+.
- Affinity: `taskset -c 2`.
- Benchmark CPU: `2`.
- Governor during final run: `powersave`.
- Final benchmark frequency context: `2333.561 MHz`.
- Final host load before run: `[5.37109375, 4.50732421875, 5.01318359375]`.
- Final host load after run: `[5.5546875, 4.57373046875, 5.02880859375]`.
- Host readiness: `ready=false`; blockers are recorded in
  `host_readiness.json`.
- Accelerator claim: CPU-only. No CUDA, ROCm, GPU, provider, or QPU execution
  is claimed.

The isolated benchmark gate remains closed because CPU 2 used the `powersave`
governor and host load exceeded the isolated threshold.

## 2026-07-04 Catalyst Row Schema Refresh

The `diff-qnode-external-comparison.json` and
`diff-qnode-ci-evidence-schema-v1.{json,csv,md}` artefacts were refreshed on
2026-07-04 to add the dedicated Catalyst compiler-workflow comparison row and
the required `catalyst_comparison` row-schema field. The refresh is still
`functional_non_isolated`: no CPU affinity or isolation method was requested,
the governor was `powersave`, host load was
`[23.6416015625, 20.76220703125, 15.38671875]` before the run and
`[22.671875, 20.640625, 15.404296875]` after the run, and
`heavy_jobs_running=true`. No provider, QPU, GPU, CUDA, or ROCm execution is
claimed.

## Commands

Framework overlay installation:

```bash
PYTHONPATH=src ./.venv/bin/python scripts/install_differentiable_framework_overlay.py \
  --overlay-path /home/anulum/.cache/scpn-qc-framework-site-py312 \
  --manifest-path /home/anulum/.cache/scpn-qc-framework-site-py312/framework_overlay_manifest.json \
  --install
```

Enzyme-JAX dependency installation:

```bash
uv python install 3.9
uv venv /home/anulum/.cache/scpn-qc-enzyme-py39 --python 3.9
uv pip install --python /home/anulum/.cache/scpn-qc-enzyme-py39/bin/python \
  'jax==0.4.30' 'jaxlib==0.4.30' 'enzyme-ad==0.0.6'
```

Final benchmark rerun:

```bash
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu \
  TF_ENABLE_ONEDNN_OPTS=0 SCPN_BENCH_ACCELERATOR_BACKEND=cpu \
  SCPN_ENZYME_RUNNER=/media/anulum/GOTM/aaa_God_of_the_Math_Collection/03_CODE/SCPN-QUANTUM-CONTROL/data/differentiable_phase_qnode/local_benchmark_20260616T0955Z/enzyme_jax_runner.py \
  ENZYME_LLVM_PLUGIN=/home/anulum/.cache/scpn-qc-enzyme-py39/lib/python3.9/site-packages/enzyme_ad/jax/enzyme_call.so \
  PYTHONPATH=src:/home/anulum/.cache/scpn-qc-framework-site-py312 \
  taskset -c 2 ./.venv/bin/python scripts/run_differentiable_benchmark_evidence.py \
  --output-dir /tmp/scpn_qc_bench_20260616/differentiable_evidence_overlay_enzyme_final \
  --cpu-affinity 2 \
  --isolation-method taskset
```

Phase-QNode affinity benchmark:

```bash
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu \
  TF_ENABLE_ONEDNN_OPTS=0 \
  PYTHONPATH=src:/home/anulum/.cache/scpn-qc-framework-site-py312 \
  taskset -c 2 ./.venv/bin/python tools/run_phase_qnode_affinity_benchmark.py \
  --repetitions 50 \
  --warmups 10 \
  --reserved-cpus 2 \
  --output /tmp/scpn_qc_bench_20260616/phase_qnode_affinity_overlay.json
```

## Current External-Comparison Rows

The 2026-07-04 schema-refresh rows record:

- JAX `0.6.2` / `jaxlib 0.6.2`: value error `0.0`, gradient error `0.0`.
- PyTorch `2.11.0`: value error `0.0`, gradient error `0.0`.
- TensorFlow / TensorFlow CPU: `dependency_missing`; install
  `tensorflow-cpu` to rerun this row.
- PennyLane `0.44.1`: value error `1.1102230246251565e-16`,
  gradient error `5.551115123125783e-17`.
- Enzyme: `dependency_missing`; the local `enzyme` executable was visible, but
  `enzyme_ad` was not installed and `SCPN_ENZYME_RUNNER` was not configured.
- Catalyst: `dependency_missing`; the dedicated `catalyst_comparison` payload
  records unestablished qjit/MLIR/QIR workflow parity, first-order compiled
  differentiation scope, control-flow gaps, finite-shot limitations, and
  unsupported provider routes
  (`finite_shot_provider_jobs`, `hardware_qpu_execution`,
  `cloud_provider_submission`).

## SHA-256

```text
ca6c0fd44d9fb5e73f1894c3e2dc3633d791cb2459e0f0129b5ff84cb4a9b165  diff-qnode-ci-evidence-schema-v1.csv
cc33ef5ce749782c33f24746a35eb611b289caa0ef7bbf1d833da18b643d7561  diff-qnode-ci-evidence-schema-v1.json
4e14853a0abd142d304eda1c54479fc3a75ff7f20c9e157681003b9ce051d379  diff-qnode-ci-evidence-schema-v1.md
9b0ba07249cc9fd7315c185b424ed1377aeb1f68b006b73e158ffefe23e46ea8  diff-qnode-external-comparison.json
93202eb606f2712e6a55d597d2c8f1890b802d86cdf216cb5c24281572123d00  enzyme_jax_runner.py
2770738675e8ac3fbf3edd5f8b004a3c0d2621fd3324b77aa3a238437b947d32  enzyme_py39_freeze.txt
11a15a483d2f8f602b8d052dc1cf0824d37a86a47853a66b1cda1ed93caa56c6  framework_overlay_freeze.txt
abec1f7734625add06bf2669ef812502df6b231e141fdb0686f7cddfdd7d9c57  framework_overlay_manifest.json
16bc50f32fc5bb7fbca2bb4282dfcc4c3ac9198d33afa920220938c593b30b68  host_readiness.json
e5e0a86508e365e8d6a88e16ad171abc2cf481782c2a94249e5ad8d66ab1bdf1  phase_qnode_affinity.json
ff3a8277b453fdb146fd1e526f7742f25922246ff78dd8969be1f2331699305b  enzyme_ad/jax/enzyme_call.so
```
