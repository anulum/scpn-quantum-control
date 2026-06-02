# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Onboarding

# Onboarding

This page is the shortest route to understanding what
`scpn-quantum-control` is, what it is useful for, and where the claim
boundaries sit.

## One-Sentence Description

`scpn-quantum-control` is a quantum-control and differentiable-computation
workbench for coupled oscillator networks, centred on the Kuramoto-XY mapping
and backed by simulator workflows, Rust acceleration, hardware-result ledgers,
release gates, and explicit scientific claim boundaries.

## What Problem It Solves

Many physical and engineered systems are networks of coupled oscillators:
electrical grids, plasma control loops, Josephson arrays, brain rhythms,
chemical oscillators, and synchronising control systems. The software provides
a common route for those systems:

1. express the coupling topology as a matrix `K_nm`;
2. express oscillator detuning or local drive as `omega`;
3. compile the corresponding XY Hamiltonian and circuits;
4. run local simulation or provider-backed execution when permitted;
5. measure synchronisation, entanglement, topology, and control observables;
6. preserve raw evidence and claim boundaries for papers, releases, and
   downstream integration.

## Core Capabilities

| Capability | What it means | Production route |
|---|---|---|
| Kuramoto-XY compiler | Converts arbitrary `K_nm` and `omega` into Hamiltonians, dense matrices, circuits, and order-parameter measurements. | `scpn_quantum_control.kuramoto_core` |
| Hardware evidence ledger | Separates theory, simulator output, raw-count hardware evidence, mitigated evidence, and unpromoted artefacts. | `docs/hardware_status_ledger.md` |
| Rust acceleration | Speeds up selected Hamiltonian, expectation, pulse, symmetry, and compiler-AD hot paths. | `scpn-quantum-engine` |
| Differentiable programming | Provides supported scalar, vector, and matrix AD primitives with fail-closed unsupported boundaries. | `scpn_quantum_control.compiler.mlir` and `scpn_quantum_control.differentiable` |
| Paper 0 source register | Preserves source-bounded validation fixtures and claim boundaries for Paper 0. | `scpn_quantum_control.paper0` |
| Release gates | Make public release decisions repeatable instead of narrative. | `tools/audit_release_readiness.py` and `scpn-bench` gates |

## Recommended First Paths

| Goal | Path |
|---|---|
| Run the first local experiment | [Quickstart](quickstart.md) |
| Understand the learning sequence | [Tutorials](tutorials.md) |
| Use the stable API | [Stable Facades API](stable_facades_api.md) |
| Bring a custom oscillator network | [Physics-First Kuramoto-XY](physics_first_kuramoto_xy.md) |
| Inspect notebooks | [Interactive Notebooks](notebooks.md) |
| Build or install the Rust engine | [Rust Engine](rust_engine.md) |
| Evaluate release readiness | [Release Readiness Gate](release_readiness.md) |
| Understand claim limits | [Hardware Status Ledger](hardware_status_ledger.md) and [Falsification](falsification.md) |

## Application Lanes

The framework is designed for application lanes where synchronisation and
coupled dynamics matter:

- **Quantum algorithm research:** XY Hamiltonians, VQE, Trotterisation,
  witness operators, topology, DLA parity, and mitigation workflows.
- **Plasma and control systems:** ITER disruption and control-facing benchmark
  contracts.
- **Power systems:** graph-coupled oscillator and IEEE-bus candidate
  workflows.
- **EEG/MEG and biological rhythms:** source-bounded synchronisation
  candidates with strict non-clinical claim boundaries.
- **Quantum hardware operations:** raw-count evidence packs, provider
  readiness, and no-QPU gates before new hardware spend.
- **Differentiable computation:** gradient-bearing primitive kernels used for
  optimisation and transform-composition experiments.

## What Is Mature

- Local simulator workflows for Kuramoto-XY examples and notebooks.
- Stable facade route for first-path users.
- Public documentation for hardware evidence status and claim classes.
- Rust acceleration for selected hot paths.
- Release-readiness and hardware-result-pack gates.
- Source-bounded Paper 0 register and generated fixtures.
- Supported differentiable scalar, vector, and matrix primitive surfaces, plus
  inspectable native program-AD lowering reports for supported scalar traces
  with strict no-tie native `where`/selection/`clip` support,
  2x2/3x3/4x4/5x5 expression determinant native lowering, helper-backed
  6x6 through 16x16 determinant native lowering, static square/rectangular trace
  native lowering, static diagonal gather/scatter native lowering, static dense
  inverse native lowering through 4x4, static vector solve native lowering
  through 5x5, matrix-RHS solve native lowering through 4x4, 2x2 product
  native lowering, and an introspectable native linalg support contract for
  service gating.

## What Remains Bounded

- Broad quantum advantage is not claimed.
- Clinical, biological, or consciousness claims are not externally validated.
- Hardware evidence must name committed raw-count artefacts and pass the
  relevant ledger gates before promotion.
- General native MLIR/LLVM/JIT AD over arbitrary programs is still an open
  engineering frontier; supported primitives and supported scalar traces
  execute through bounded kernels, and unsupported paths report the blocked
  operation before failing closed.
- Paper 0 ingestion records source structure and generated fixtures; it is not
  an external validation of the propositions.

## Commercial Route

The repository is AGPL-3.0-or-later with a commercial licence available.
Open research, teaching, and AGPL-compatible redistribution use the public
licence. Closed-source products, proprietary services, SaaS integrations, and
embedded deployments require a commercial licence grant before distribution or
network-service use.

Contact: `protoscience@anulum.li`.
