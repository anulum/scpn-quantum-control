<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts & Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- scpn-quantum-control — Frontier Campaign Manifest -->

# Frontier Campaign 2026 - 8 Tests (Batch 4)

Pre-flight checklist and parameter file manifest.

## Required parameter files (`params/` directory)

Parameter files are local transport caches. Generate them from the
bridge/provenance contract with:

```bash
python scripts/frontier_campaign_2026/generate_params.py --output-dir params
```

Use `--allow-synthetic --seed <n>` only for interface smoke tests. Those
outputs are labelled in `PARAMETER_PROVENANCE.json` and are not
publication-safe QPU inputs.

| Test | Script | Required `.npy` files |
|------|--------|-----------------------|
| T1   | test_quantum_advantage_scaling.py   | `scale_Knm_20x20.npy`, `scale_omega_20.npy`, `scale_Knm_40x40.npy`, `scale_omega_40.npy`, `scale_Knm_80x80.npy`, `scale_omega_80.npy`, `scale_Knm_160x160.npy`, `scale_omega_160.npy` |
| T2   | test_live_scneurocore_loop.py       | *(none — live stream via `scpneurocore.bridge.load_live_stream`)* |
| T3   | test_sync_distillation.py           | `distill_Knm_12x12.npy` |
| T4   | test_multi_backend_distributed.py   | `distributed_Knm_20x20.npy` |
| T5   | test_dla_tensor_network.py          | `tn_Knm_64x64.npy` |
| T6   | test_rl_pulse_optimization.py       | *(none — RL-driven)* |
| T7   | test_pt_symmetric_kuramoto.py       | `pt_Knm_12x12.npy` |
| T8   | test_logical_sync_protection.py     | `logical_Knm_12x12.npy` |

## Result files written (relative to this directory)

| Test | Output file |
|------|-------------|
| T1   | `results/quantum_advantage_scaling.json` |
| T2   | `results/live_scneurocore_loop.json` |
| T3   | `results/sync_distillation.json` |
| T4   | `results/multi_backend_distributed.json` |
| T5   | `results/dla_tensor_network.json` |
| T6   | `results/rl_pulse_optimization.json` |
| T7   | `results/pt_symmetric_kuramoto.json` |
| T8   | `results/logical_sync_protection.json` |

## LaTeX templates

`latex/fig_quantum_advantage_scaling.tex` — figure for T1

## Module dependencies

| Test | Extra module beyond core |
|------|--------------------------|
| T1   | `scpn_quantum_control.accel.rust_kuramoto_classical.run_large_n` |
| T2   | `scpneurocore.bridge.load_live_stream` |
| T5   | `scpn_quantum_control.analysis.dla_truncated_tn` |
| T6   | `scpn_quantum_control.analysis.RLPulseOptimizer` |

## Launch status

Resolved prerequisites:

- `OTOC` is exported from `scpn_quantum_control.analysis`.
- `StructuredAnsatz.from_kuramoto` treats `lambda_fim` as a concrete
  float, not a repeated Qiskit `Parameter`, so multi-step Trotter runs
  do not hit parameter-name collisions.

Implementation-gated tests:

- T5 (`dla_truncated_tn`) is a fail-fast interface until a real
  DLA-truncated tensor-network implementation and validation suite
  exist.
- T6 (`RLPulseOptimizer`) is a fail-fast interface until a real
  optimiser, objective, and replayable training trace exist.

The orchestrator records those implementation gates as failures rather
than substituting synthetic scientific outputs.

## Run order

Run individually or via the orchestrator `run_frontier_campaign.sh`.
Ensure `SCPN_IBM_TOKEN` env var is set before launching.
Do not import local mock injectors during hardware runs.

## IBM backend note

Target: `ibm_heron_r2` (routes to `ibm_fez` or `ibm_kingston`).
Shots budgeted: 6000–15000 per circuit depending on test.
T1 skips 160-qubit QPU submission on 156-qubit Heron hardware and still
records the classical baseline.
