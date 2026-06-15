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

## Why this exists

The package is intended for teams that cannot treat every quantum workflow as
equivalent. It separates:

- **simulation science** (fast iteration),
- **method verification** (reproducible checks and baselines),
- **hardware evidence** (raw-count-backed claims), and
- **commercial readiness** (stable facades, release gates, and deployment boundaries).

The goal is a single path from idea to auditable outcome, where every higher-cost
route has a defined evidence burden. A useful result in this repository is not
only a plot or a notebook output; it is a result with inputs, code path,
dependency context, claim class, and promotion rule.

## One-Sentence Description

`scpn-quantum-control` is a quantum-control and differentiable-computation
workbench for coupled oscillator networks, centred on the Kuramoto-XY mapping
and backed by simulator workflows, Rust acceleration, hardware-result ledgers,
release gates, and explicit scientific claim boundaries.

## Who this helps in practice

| Role | Outcome |
|---|---|
| Researcher | turn coupled-oscillator models into structured comparisons with clear limits |
| Hardware operator | run campaign planning and raw-count review without mixing simulator and hardware claims |
| Product engineer | adopt stable facades and migrate into integrations with low risk |
| Compliance reviewer | inspect explicit claim classes and verify promotion rules before release |
| Commercial evaluator | see which parts are ready for pilots, which are research-only, and where a commercial licence is required |

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
| Differentiable programming | Provides parameter-shift VQE building blocks, supported scalar/vector/matrix AD primitives, native lowering reports, and fail-closed unsupported boundaries. | `scpn_quantum_control.phase.param_shift`, `scpn_quantum_control.compiler.mlir`, and `scpn_quantum_control.differentiable` |
| Paper 0 source register | Preserves source-bounded validation fixtures and claim boundaries for Paper 0. | `scpn_quantum_control.paper0` |
| Release gates | Make public release decisions repeatable instead of process-only statements. | `tools/audit_release_readiness.py` and `scpn-bench` gates |

## Adoption Checklist

Before using the package in a paper, product prototype, or integration, make
these decisions explicit:

| Question | Recommended answer path |
|---|---|
| Is my system naturally represented as coupled oscillators? | Start with `K_nm`, `omega`, and [Physics-First Kuramoto-XY](physics_first_kuramoto_xy.md). |
| Do I need gradients? | Start with [Differentiable Tutorials](differentiable_tutorials.md), then use [Differentiable Programming](differentiable_programming.md) and the gradient support matrix for route decisions. |
| Am I citing hardware? | Use only rows named in [Hardware Status Ledger](hardware_status_ledger.md) and committed raw-count artefacts. |
| Am I building a closed product or service? | Review the AGPL/commercial licence boundary before distribution or network use. |
| Am I using SCPN-specific biological or consciousness language? | Keep it source-bounded unless external validation artefacts exist. |

## Recommended First Paths

| Goal | Path |
|---|---|
| Run the first local experiment | [Quickstart](quickstart.md) |
| Understand the learning sequence | [Tutorials](tutorials.md) |
| Use the stable API | [Stable Facades API](stable_facades_api.md) |
| Bring a custom oscillator network | [Physics-First Kuramoto-XY](physics_first_kuramoto_xy.md) |
| Train or inspect gradients | [Differentiable Tutorials](differentiable_tutorials.md), [Differentiable Programming](differentiable_programming.md), [Quantum Gradients](quantum_gradients.md), and [Differentiable API](differentiable_api.md) |
| Reproduce framework parity | Build the CPU-only framework overlay, run the external comparison suite, and inspect the [claim ledger](https://github.com/anulum/scpn-quantum-control/blob/main/data/differentiable_phase_qnode/claim_ledger.md). |
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
  optimisation, VQE training experiments, compiler-backed AD, CPU-only
  framework parity, and external ML-framework comparison rows.

## Market-fit framing

The strongest adoption case is not a claim of near-term quantum advantage. The
adoption case is governed uncertainty reduction for domains where coupled
dynamics matter and where uncontrolled evidence would be expensive:

| Market problem | Why this package matters |
|---|---|
| Quantum pilots are hard to compare | Common `K_nm`/`omega` contracts, classical references, and claim classes make pilots reviewable. |
| Hardware runs are costly and easy to overstate | No-QPU gates, provider readiness, and raw-count ledgers prevent simulator output from becoming hardware language. |
| Optimisation stacks hide fallback behaviour | Gradient support matrices and fail-closed routes make unsupported AD visible before integration. |
| Research notebooks do not become products cleanly | Stable facades, API docs, examples, and release gates provide a migration path from exploration to integration. |

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
  6x6 through 19x19 determinant native lowering, static square/rectangular trace
  native lowering, static diagonal gather/scatter native lowering, static dense
  inverse native lowering through 6x6, static vector and matrix-RHS solve
  native lowering through 6x6, 2x2 product native lowering, and an
  introspectable native linalg support contract for service gating.
- Parameter-shift gradient helpers for callable expectation objectives,
  gradient-descent VQE examples, provider-gradient readiness ledgers, and a
  unified differentiable-readiness audit for reviewer-facing support evidence.
- The Phase-QNode differentiable lane now has a reproducible CPU framework
  overlay profile, external comparison rows for JAX, PyTorch, TensorFlow,
  PennyLane, and optional Enzyme/compiler AD runner evidence with strict JSON,
  timeout, toolchain, and correctness gates, plus a claim ledger that requires
  artefact IDs before promotion.

## What Remains Bounded

- Broad quantum advantage is not claimed.
- Clinical, biological, or consciousness claims are not externally validated.
- Hardware evidence must name committed raw-count artefacts and pass the
  relevant ledger gates before promotion.
- Phase-QNode performance remains SOTA-candidate until a self-hosted
  `isolated-benchmark` CI runner uploads an `isolated_affinity` artefact; local
  and GitHub-hosted rows are diagnostic only, and CUDA/ROCm claims require
  explicit visible-device metadata rather than CPU fallback.
- General native MLIR/LLVM/JIT AD over arbitrary programs is still an open
  engineering frontier; supported primitives and supported scalar traces
  execute through bounded kernels, and unsupported paths report the blocked
  operation before failing closed.
- Native determinant traces at `20x20` and wider are intentionally
  fail-closed after a strict native verification failure at the current helper
  formulation.
- Native quotient-linalg full-output inverse and matrix-RHS solve traces at
  `7x7` and wider are intentionally unsuitable for the current native path.
  `5x5` through `6x6` reuse one determinant/adjugate helper per static matrix;
  the `7x7` full-output promotion attempt exceeded the focused native gate, so
  wider traces fail closed until a native factorisation helper replaces
  adjugate replay. This limitation is useful research evidence, not a silent
  runtime fallback.
- Full gradient tape semantics, public JAX/PyTorch/TensorFlow adapters,
  PennyLane/Qiskit migration bridges, backend-aware hardware gradient
  planning, QNN/QGNN/QSNN production examples, and analog oscillator mapping
  are planned roadmap surfaces until their tests, docs, and support matrix
  entries are complete.
- Paper 0 ingestion records source structure and generated fixtures; it is not
  an external validation of the propositions.

## Commercial Route

The repository is AGPL-3.0-or-later with a commercial licence available.
Open research, teaching, and AGPL-compatible redistribution use the public
licence. Closed-source products, proprietary services, SaaS integrations, and
embedded deployments require a commercial licence grant before distribution or
network-service use.

Contact: `protoscience@anulum.li`.
