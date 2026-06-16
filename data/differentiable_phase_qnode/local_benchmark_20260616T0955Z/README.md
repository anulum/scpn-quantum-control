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

## Dependency Results

The framework rows succeeded against the bounded SCPN reference:

- JAX `0.10.1` / `jaxlib 0.10.1`: value error `0.0`, gradient error `0.0`.
- PyTorch `2.12.0+cpu`: value error `0.0`, gradient error `0.0`.
- TensorFlow CPU `2.21.0`: value error `0.0`, gradient error `0.0`.
- PennyLane `0.45.0`: value error `1.1102230246251565e-16`,
  gradient error `5.551115123125783e-17`.

The Enzyme dependency is no longer missing: `enzyme-ad==0.0.6` is installed in
the Python 3.9 Enzyme environment, and the native Enzyme-JAX extension is
available at
`/home/anulum/.cache/scpn-qc-enzyme-py39/lib/python3.9/site-packages/enzyme_ad/jax/enzyme_call.so`.
The Enzyme row remains a hard gap with `failure_class=runtime_error` because
Enzyme-JAX fails during MHLO lowering:

```text
invalid properties {limit_indices = dense<1> : tensor<1xi64>, start_indices = dense<0> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} for op mhlo.slice
```

This is an installed-toolchain runtime gap, not a dependency-missing gap.

## SHA-256

```text
34323060bd2d94329bae1e164ca9c8ea31a5f88d5b27dca6b03dfb74101fc1ec  diff-qnode-ci-evidence-schema-v1.csv
15ed583b9f55b3cf0a5e3a023ea99af477d76cd1312ed745a23a52f8cad4be1a  diff-qnode-ci-evidence-schema-v1.json
4e14853a0abd142d304eda1c54479fc3a75ff7f20c9e157681003b9ce051d379  diff-qnode-ci-evidence-schema-v1.md
c45c7a03353dd52a5a8a539de95d71094f4fa3750ea170251038c05a3232123b  diff-qnode-external-comparison.json
93202eb606f2712e6a55d597d2c8f1890b802d86cdf216cb5c24281572123d00  enzyme_jax_runner.py
2770738675e8ac3fbf3edd5f8b004a3c0d2621fd3324b77aa3a238437b947d32  enzyme_py39_freeze.txt
11a15a483d2f8f602b8d052dc1cf0824d37a86a47853a66b1cda1ed93caa56c6  framework_overlay_freeze.txt
abec1f7734625add06bf2669ef812502df6b231e141fdb0686f7cddfdd7d9c57  framework_overlay_manifest.json
16bc50f32fc5bb7fbca2bb4282dfcc4c3ac9198d33afa920220938c593b30b68  host_readiness.json
e5e0a86508e365e8d6a88e16ad171abc2cf481782c2a94249e5ad8d66ab1bdf1  phase_qnode_affinity.json
ff3a8277b453fdb146fd1e526f7742f25922246ff78dd8969be1f2331699305b  enzyme_ad/jax/enzyme_call.so
```
