<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts & Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- scpn-quantum-control — Sophisticated Campaign Manifest -->

# Sophisticated Campaign 2026 - 8 Tests (Batch 3)
# Pre-flight checklist and parameter file manifest

## Required parameter files (`params/` directory)

Parameter files are local transport caches. Generate them from
source-backed bridge artifacts where possible; deterministic synthetic
files must be explicitly labelled as smoke-test data and are not
publication-safe QPU inputs.

| Test | Script | Required `.npy` files |
|------|--------|-----------------------|
| T1   | test_fusion_hybrid_stabilizer.py    | `tokamak_Knm_16x16.npy`, `tokamak_omega.npy` |
| T2   | test_brain_scale_bridging.py        | `c_elegans_subnetwork_14x14.npy` |
| T3   | test_sync_resource_theory.py        | `resource_Knm_12x12.npy` |
| T4   | test_autonomous_discovery.py        | *(none — uses runner directly)* |
| T5   | test_quantum_internet_timing.py     | `internet_timing_20x20.npy` |
| T6   | test_collective_thermo_engines.py   | `thermo_Knm_16x16.npy` |
| T7   | test_hypergraph_nonreciprocal.py    | `hyper_pairwise.npy`, `hyper_3body.npy`, `hyper_directed.npy` |
| T8   | test_logical_sync_encoding.py       | `logical_Knm_12x12.npy` |

## Result files written (relative to this directory)

| Test | Output file |
|------|-------------|
| T1   | `results/fusion_hybrid_stabilizer.json` |
| T2   | `results/brain_scale_bridging.json` |
| T3   | `results/sync_resource_theory.json` |
| T4   | `results/autonomous_universality_classes.json` |
| T5   | `results/quantum_internet_timing.json` |
| T6   | `results/collective_thermo_engines.json` |
| T7   | `results/hyper_nonreciprocal.json` |
| T8   | `results/logical_sync_encoding.json` |

## LaTeX templates

`latex/fig_fusion_hybrid_stabilizer.tex` — figure for T1

## Known ready-to-run status

The following library fixes are required before QPU launch:

- `OTOC` class now exported from `scpn_quantum_control.analysis`
- `StructuredAnsatz.from_kuramoto` lambda_fim Qiskit parameter name conflict fixed

## Run order

Run individually or via the orchestrator `run_sophisticated_campaign.sh`.
Ensure `SCPN_IBM_TOKEN` env var is set before launching.
Do not import local mock injectors during hardware runs.

## IBM backend note

Target: `ibm_heron_r2` (routes to `ibm_fez` or `ibm_kingston`).
Shots budgeted: 6000–15000 per circuit depending on test.
