# scpn-quantum-control

![header](figures/header.png)

[![CI](https://github.com/anulum/scpn-quantum-control/actions/workflows/ci.yml/badge.svg)](https://github.com/anulum/scpn-quantum-control/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/anulum/scpn-quantum-control/branch/main/graph/badge.svg)](https://codecov.io/gh/anulum/scpn-quantum-control)
[![License: AGPL-3.0](https://img.shields.io/badge/License-AGPL--3.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://python.org)
[![Qiskit 2.2+](https://img.shields.io/badge/qiskit-2.2%2B-6929C4.svg)](https://qiskit.org)
[![Website](https://img.shields.io/badge/website-anulum.li%2Fscpn--quantum--control-38bdf8.svg)](https://anulum.li/scpn-quantum-control/)
[![Docs](https://img.shields.io/badge/docs-GitHub%20Pages-blue.svg)](https://anulum.github.io/scpn-quantum-control)
[![OpenSSF Best Practices](https://www.bestpractices.dev/projects/12290/badge)](https://www.bestpractices.dev/projects/12290)
[![OpenSSF Scorecard](https://api.securityscorecards.dev/projects/github.com/anulum/scpn-quantum-control/badge)](https://securityscorecards.dev/viewer/?uri=github.com/anulum/scpn-quantum-control)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![mypy](https://img.shields.io/badge/type--checked-mypy-blue.svg)](https://mypy-lang.org/)
[![Tests](https://img.shields.io/badge/tests-CI%20gated-brightgreen.svg)](https://github.com/anulum/scpn-quantum-control/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/scpn-quantum-control)](https://pypi.org/project/scpn-quantum-control/)
[![PyPI Downloads](https://img.shields.io/pypi/dm/scpn-quantum-control)](https://pypi.org/project/scpn-quantum-control/)
[![All-time Downloads](https://static.pepy.tech/badge/scpn-quantum-control)](https://pepy.tech/project/scpn-quantum-control)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18821929.svg)](https://doi.org/10.5281/zenodo.18821929)
[![Hardware: ibm_kingston](https://img.shields.io/badge/hardware-ibm__kingston%20Heron%20r2-blueviolet.svg)]()
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/anulum/scpn-quantum-control/blob/main/notebooks/01_kuramoto_xy_dynamics.ipynb)

> **Active Development** — scpn-quantum-control is under intensive development.
> The public status wording is anchored to the
> [Hardware Status Ledger](docs/hardware_status_ledger.md),
> which separates theory, simulator, unmitigated hardware, mitigated hardware,
> and noise-limited claims. Current promoted hardware evidence is narrowed to
> artefact-backed `ibm_fez` baseline rows, the April/May 2026
> `ibm_kingston` DLA parity raw-count datasets, and the May 2026 SCPN/FIM
> falsification artefacts. Stable core contracts and backend capability
> artefacts are now part of release/repro hardening and are kept separate from
> non-artefact scientific claims. APIs may evolve as this work progresses.

**Version:** 0.9.12
**Status:** Kuramoto-XY compiler + hardware runners + analysis stack + bounded differentiable-programming surface | generated capability inventory below | CI coverage gate 90% | IBM Heron r2 evidence ledgered

## What this repository is for

`scpn-quantum-control` is an evidence-governed workbench for turning
coupled-oscillator models into quantum-control experiments, simulator studies,
gradient-bearing optimisation loops, and auditable hardware-result packages.
It is for teams that need a repeatable path from a physical model to a result
that can be reviewed, reproduced, cited, or used in an application pilot.

If your team needs reproducible research or product-grade experimentation, the
repository provides:

- a stable route from `K_nm`/`omega` problem definitions to simulators;
- deterministic artefact-led evidence for method claims;
- bounded optimisation and differentiable-programming surfaces for training,
  verification, and convergence analysis;
- hardware campaign management that separates committed raw-count rows from
  simulator and scoped-failure classes.

It is positioned for use-cases where decision quality depends on evidence:
engineering teams that need a consistent pathway from model specification to
observable contracts, and research teams that need explicit failure modes rather
than implied coverage.

## What you can do first

| Need | First path | Output |
|---|---|---|
| Understand the software | [Onboarding](docs/onboarding.md) -> [Quickstart](docs/quickstart.md) | A small local Kuramoto-XY run and a clear claim boundary. |
| Bring your own coupled system | [Physics-First Kuramoto-XY](docs/physics_first_kuramoto_xy.md) | A validated `K_nm`/`omega` problem compiled to simulator-ready quantum objects. |
| Train or inspect gradients | [Differentiable Tutorials](docs/differentiable_tutorials.md) -> [Differentiable Programming](docs/differentiable_programming.md) -> [Quantum Gradients](docs/quantum_gradients.md) | Exact, finite-shot, framework-comparison, or fail-closed gradient evidence. |
| Review hardware claims | [Hardware Status Ledger](docs/hardware_status_ledger.md) -> [Hardware Result Packs](docs/hardware_result_packs.md) | Raw-count-backed evidence or an explicit blocked promotion route. |
| Evaluate adoption | [API Overview](docs/api.md) -> [Release Readiness Gate](docs/release_readiness.md) | Stable integration surfaces, release gates, and licensing boundaries. |

## Application and commercial value

The practical value is not "quantum black box" experimentation; it is the
ability to reduce integration risk while preserving future hardware options.
This package is designed for organisations that want to:

- de-risk algorithm ideas on simulators and strong classical baselines first;
- compare evidence classes before claiming hardware-level results;
- standardise proof surfaces (contracts, manifests, and ledgers) across pilots;
- move between R&D notebooks and integration-friendly stable facades without
  collapsing into undocumented internal APIs.

Application lanes include synchronisation diagnostics, control prototyping,
quantum-algorithm research, hardware campaign governance, gradient-informed
optimisation, and evidence packages for due diligence. Commercial value comes
from reducing unclear research risk: every promoted result must have a named
artefact, every unsupported route fails closed, and closed-source or SaaS use
has a defined commercial licensing route.

<!-- capability-snapshot:start -->
<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Generated by tools/capability_manifest.py; do not edit counts by hand. -->

# scpn-quantum-control Capability Inventory

| Surface | Current inventory |
|---|---:|
| Package version | 0.9.12 |
| Public API exports | 717 |
| Python source modules | 437 |
| Public Python classes | 824 |
| Domain package families | 31 |
| API documentation pages | 0 |
| Rust PyO3 function bindings | 147 |
| Rust source modules | 38 |
| Notebook files | 98 |
| Example files | 30 |
| Optional extras | 42 |
| Python test files | 744 |
| Public documentation pages | 247 |
| GitHub Actions workflows | 20 |

Evidence boundary: this snapshot is a static inventory. Performance, coverage, hardware, and scientific-fidelity claims require their own committed evidence artefacts.
<!-- capability-snapshot:end -->

---

## Status Snapshot — 2026-05-18

| Area | Public status |
|---|---|
| Generic compiler surface | `scpn_quantum_control.kuramoto_core` validates arbitrary `K_nm`/`omega` inputs and compiles Hamiltonians, dense matrices, Trotter circuits, and order-parameter measurements. |
| Release and reproducibility scope | Stable core contracts and backend capability artefacts for Kuramoto-XY synchronisation are included in release/readiness checks and promoted only with deterministic evidence manifests. |
| Hardware evidence | `ibm_fez` baseline rows are legacy artefact-backed observations; `ibm_kingston` Phase 1, Phase 2 A+G, Phase 2 B-C, and popcount DLA datasets are promoted with raw-count artefacts. The SCPN/FIM `ibm_kingston` result is promoted as a negative/falsification result for the tested digital circuit family. |
| Simulator and methods evidence | BKT, OTOC, Floquet, MBL, FIM, VQE, GPU, tensor-network, and classical comparison claims stay marked as simulator/classical/methods unless a hardware artefact is named. Generated benchmark artefacts are indexed from the benchmark dashboard and reproducibility CLI. |
| Licence boundary | The possible lightweight core split is documented, but all in-repository code remains AGPL/commercial unless a future release changes metadata and SPDX headers. |

For claim classes, raw-artefact pointers, and promotion rules, see the
[Hardware Status Ledger](docs/hardware_status_ledger.md).

## Plain-Language Summary

`scpn-quantum-control` turns a coupled-oscillator network into quantum
circuits, simulator workflows, hardware-result ledgers, and analysis tools.
SCPN here denotes the **Scale-Coupled Phase Network** — a generalised-Kuramoto
coupling model whose `K_nm` matrix provides a structured, reproducible test
problem. The first supported physics lane is the Kuramoto-XY mapping: provide a
coupling matrix `K_nm` and oscillator frequencies `omega`, then compile the
matching XY Hamiltonian, run local or provider-backed circuits, measure
synchronisation, and compare the result against classical and exact-simulation
baselines.

The repository is not only a circuit generator. It also carries the operational
surfaces needed to use the software responsibly:

- stable public facades for notebooks and integrations;
- Rust acceleration for hot numerical kernels;
- differentiable-programming kernels and conformance rows for supported
  scalar, vector, matrix, broadcast, selection, shape, assembly, cumulative, and reduction
  primitives;
- no-QPU release gates and hardware-result pack verifiers;
- claim-boundary documentation that separates simulator, hardware, and open
  scientific questions.

## Who It Is For

| User | Primary value | Start here |
|---|---|---|
| Quantum algorithm researcher | Compile oscillator networks into XY circuits and inspect synchronisation, entanglement, topology, and control observables. | [Quickstart](docs/quickstart.md), [Tutorials](docs/tutorials.md) |
| Applied physicist or control engineer | Convert domain coupling graphs into reproducible quantum/simulator experiments while retaining classical baselines. | [Physics-First Kuramoto-XY](docs/physics_first_kuramoto_xy.md), [Application Benchmark Plugins](docs/application_benchmarks.md) |
| Hardware experiment operator | Run or replay provider-backed campaigns only when the evidence ledger, raw counts, and release gates permit the claim. | [Hardware Guide](docs/hardware_guide.md), [Hardware Result Packs](docs/hardware_result_packs.md) |
| Software integrator | Use stable facade APIs instead of binding to internal package layout. | [Stable Facades API](docs/stable_facades_api.md), [API Overview](docs/api.md) |
| Commercial evaluator | Assess the route from research prototype to product lane: reproducible simulations, hardware-result governance, Rust kernels, and dual licensing. | [Onboarding](docs/onboarding.md), [Release Readiness Gate](docs/release_readiness.md) |

## Application and Commercial Value

The practical value is a disciplined bridge between coupled-system physics,
quantum hardware experimentation, and control-facing software:

- **Synchronisation diagnostics:** explore where oscillator networks lock,
  decohere, or separate into sectors.
- **Control prototyping:** map power-grid, plasma, EEG/MEG, Josephson-array,
  and other coupled-system candidates into common `K_nm`/`omega` workflows.
- **Hardware evidence management:** keep raw-count evidence, simulator output,
  and open claims separated before public release or paper citation.
- **Differentiable computation:** use supported compiler-AD primitives for
  gradient-bearing scalar, vector, and matrix kernels while unsupported paths
  fail closed.
- **Gradient-informed quantum optimisation:** use the current
  parameter-shift VQE building blocks and compiler/program-AD kernels as the
  base for gradient-trained Kuramoto-XY objectives, bounded phase-QNN
  classifiers, quantum spiking neural networks, and future framework adapters.
- Bounded phase-QNN and QSNN training now expose structured convergence
  diagnostics and parameter-shift evaluation accounting through
  `train_parameter_shift_qnn_classifier(...)` and
  `QSNNTrainer.train_with_diagnostics(...)`.
- The bounded phase-QNN route also exposes its MSE loss, multi-frequency
  parameter-shift gradient, finite-difference verification, and named
  external-gradient agreement records through
  `verify_parameter_shift_qnn_classifier_gradient(...)`.
- `run_parameter_shift_qnn_conformance_suite(...)` bundles QNN training,
  gradient replay, external-gradient agreement hooks, and explicit unsuitable
  scenario records for reviewer-facing evidence packs.
- `run_parameter_shift_qnn_optimizer_benchmark_suite(...)` compares the bounded
  QNN parameter-shift trainer against finite-difference, SGD, Adam, L-BFGS-B,
  diagonal-Fisher natural-gradient, seeded SPSA, and deterministic
  derivative-free baselines as non-isolated functional evidence only.
- `run_parameter_shift_qnn_convergence_suite(...)` records deterministic
  bounded-QNN convergence cases with loss-drop thresholds, accuracy thresholds,
  parameter-shift evaluation accounting, and unsuitable-scenario records.
- `estimate_parameter_shift_qnn_finite_shot_gradient(...)` and
  `run_parameter_shift_qnn_finite_shot_convergence_suite(...)` add seeded
  finite-shot simulator evidence for bounded-QNN gradients and noisy-gradient
  convergence with shot counts, confidence radii, replay seeds, and explicit
  non-hardware claim boundaries.
- `verify_parameter_shift_qnn_framework_agreement(...)` and
  `run_parameter_shift_qnn_framework_agreement_suite(...)` compare bounded
  QNN parameter-shift gradients with caller-supplied or deterministic
  framework-style references while explicitly recording that this is not
  native framework autodiff through simulator kernels.
- `phase_qnode_tape(...)` and `run_phase_qnode_tape_readiness_suite()` add
  QNode-style differentiable execution records for supported phase objectives,
  seeded finite-shot replay, and provider-boundary routes that fail closed
  before hardware submission.
- `PhaseQNodeCircuit`, `execute_phase_qnode_circuit(...)`, and
  `parameter_shift_phase_qnode_gradient(...)` execute the registered local
  statevector subset (`rx/ry/rz/phase`, Pauli Clifford gates, controlled
  rotations, controlled-H/S/T, Toffoli/CCZ/Fredkin, `swap`, `rxx/ryy/rzz`)
  against Pauli products and sparse weighted Pauli Hamiltonians with exact
  operation-list decompositions for registered Toffoli/Fredkin gates, a
  validated sparse Ising-chain Hamiltonian builder, and strict route support
  reports for blocked value, gradient, metric, and Fisher paths.
- `PhaseQNodeDensityCircuit`, `PhaseQNodeNoiseChannel`, and
  `execute_phase_qnode_density_matrix(...)` execute the same registered local
  unitary family through density matrices plus bounded single-qubit Kraus
  channels (`bit_flip`, `phase_flip`, `depolarizing`,
  `amplitude_damping`), returning trace, purity, density entries, support
  reports, and explicit non-gradient/non-metric/non-hardware claim boundaries.
- `run_phase_qnode_framework_parity_suite()` runs bounded local statevector
  parity scenarios through SCPN plus installed JAX, PyTorch, TensorFlow, and
  PennyLane backends. The default single-qubit row is joined by
  `scenario="registered_two_qubit_entangling_statevector"`, which exercises a
  registered two-qubit entangling Phase-QNode tensor path and records value,
  gradient, dtype/device metadata, and dependency-sparse classifications without
  provider execution, finite-shot sampling, hardware gradients, or unrestricted
  simulator-autodiff claims.
- `build_pennylane_qnode_from_phase_qnode(...)` and
  `check_pennylane_phase_qnode_round_trip(...)` generate bounded PennyLane
  QNodes from registered local `PhaseQNodeCircuit` declarations and verify
  value/gradient parity with explicit device, shot, and diff-method metadata.
- `run_phase_qnode_affinity_benchmark(...)` records command, affinity, host
  load, CPU/runtime/dependency metadata, warmups, repetitions, and raw timing
  rows; evidence is labelled `isolated_affinity` only when the isolation policy
  passes, otherwise `functional_non_isolated`.
- `run_differentiable_model_training_evidence_suite()` packages seeded
  registered QNN, QGNN, QSNN, Kuramoto-XY, open-system-control, and
  inverse-coupling-recovery local training cases with loss reduction and
  finite-difference gradient-agreement evidence.
- `run_registered_differentiable_training_suite_audit()` records which
  requested differentiable training-suite lanes are actually evidenced:
  registered local QNN/QGNN/QSNN/Kuramoto-XY/open-system-control/
  inverse-coupling-recovery pass without promoting arbitrary architectures,
  provider hardware, or benchmark-performance claims.
- `execute_phase_qnode_transform(...)` and
  `run_phase_qnode_transform_readiness_suite()` execute supported scalar local
  QNode transforms for `grad`, `value_and_grad`, `hessian`, `jvp`, `vjp`,
  `jacfwd`, and `jacrev`, while preserving fail-closed vectorized, hardware,
  and arbitrary-framework boundaries.
- `execute_phase_qnode_vector_jacobian(...)`,
  `execute_phase_qnode_vmap_grad(...)`, and
  `run_phase_qnode_vector_transform_readiness_suite()` add deterministic native
  vector-output QNode Jacobians and host-side manual `vmap(grad)` over scalar
  parameter-shift objectives. Provider vectorization, framework-native `vmap`,
  finite-shot batched gradients, and hardware execution remain fail-closed.
- `execute_provider_qnode_transform(...)`,
  `execute_provider_qnode_vmap_grad(...)`, and
  `run_provider_qnode_transform_readiness_suite()` connect provider callback
  expectation samples to QNode transform evidence for `grad`, `value_and_grad`,
  `jvp`, `vjp`, scalar `jacfwd`/`jacrev`, and manual `vmap(grad)`. Finite-shot
  provider routes propagate variance and shot metadata; live hardware submission
  remains policy-gated and fail-closed by default.
- **Product route:** AGPL use is available for open research; proprietary
  deployment uses the commercial licence route described below.

Open boundaries remain explicit: the package does not claim broad quantum
advantage, clinical validation, or externally validated SCPN biology. It
provides a reproducible computational workbench and the governance required to
promote claims only when the evidence exists.

## Why Teams Would Adopt It

The package is useful when a team needs a single accountable route from a
coupled-system model to quantum-control evidence:

| Need | What the repository supplies | Boundary to respect |
|---|---|---|
| Turn oscillator data into quantum experiments | `K_nm`/`omega` validation, XY Hamiltonians, circuits, and local execution paths. | The mapping is a quantum XY analogue of synchronisation dynamics, not a proof of nonlinear classical equivalence for every system. |
| Optimise trainable objectives | Parameter-shift VQE, composed phase objectives, gradient certificates, and supported compiler/program-AD kernels. | Unsupported gates, adapters, backends, shapes, and transform nests fail closed. |
| Compare simulators, hardware, and papers | Raw-count ledgers, result packs, no-QPU gates, benchmark manifests, and reproducibility docs. | Claims must cite committed evidence artefacts; simulator output is not hardware validation. |
| Prepare a product or research integration | Stable facades, provider capability records, Rust acceleration hooks, API docs, tutorials, and licence boundary text. | Closed-source network services or embedded products need the commercial licence route. |

Market-facing applications include quantum-control prototyping, power-grid and
plasma synchronisation studies, EEG/MEG rhythm modelling under non-clinical
claim limits, quantum-hardware campaign governance, gradient-informed VQE/QNN
research, and reproducible benchmark publication.

## Differentiable Programming Route

The differentiable-programming lane is now documented as a first-path product
surface because gradient evidence is central to quantum optimisation, machine
learning integration, and control. The bounded Phase-QNode promotion state is
SOTA-candidate until the claim ledger, external comparison rows, and isolated
CI benchmark artefacts all pass:

| Layer | Current status | Where to start |
|---|---|---|
| Quantum parameter-shift | Supported through `scpn_quantum_control.phase.param_shift` for callable expectation objectives, structured `PhaseVQE` gradients, and local gradient-descent VQE examples. | [Quantum Gradients](docs/quantum_gradients.md), [Variational Methods](docs/variational.md) |
| Program and compiler AD | Supported for registered scalar, vector, and matrix primitives with native lowering reports, deterministic alias/effect metadata summaries including bounded local rebinding/list-alias metadata, local object-attribute aliases, expression-rebinding aliases, branch-local control-path alias blockers, loop-carried scalar state metadata, executed array-view aliases, `program_ad_effect_ir.v1` round-trip conformance through `program_ad_ir_roundtrip_contracts`, runtime/source control-region and `ProgramADPhiNode` conformance through `program_ad_control_phi_metadata_contracts`, finite/dtype/shape result checks, trainable-mask derivative zeroing, bounded reverse-adjoint generation provenance with `ProgramADAdjointStep` rows through `program_adjoint_replay_provenance_contracts`, fail-closed derivative-losing `sign`/`heaviside` contracts, elementwise primitive conformance for bounded absolute-value, domain, denominator, and inverse-trig boundary contracts, structured numeric primitive conformance for product/interpolation/signal/stencil contracts, cumulative primitive conformance for bounded `cumsum`/`cumprod`/`diff` traces, assembly primitive conformance for like-constructors and static stack conveniences, reduction primitive conformance for bounded statistical, trapezoid, unique-selector, and scalar-q order-statistic reductions, shape primitive conformance for bounded reshape, axis movement, rank promotion, repeat/tile, roll/rot90, and flip-family contracts, broadcast primitive conformance for bounded `broadcast_to`, `broadcast_arrays`, and binary rank-broadcasting paths, selection primitive conformance for bounded static selection folds, strict sort, `where`, and `clip` paths with dynamic masks/selectors, ties, and integer-output selector differentiation still fail-closed, bounded 2x2 static-linalg native lowering, wider static-linalg MLIR-runtime-only verification, and fail-closed unsupported boundaries. Alias/effect summaries, IR round-trip evidence, control/phi provenance, generated adjoint-step provenance, and static bytecode/source frontend preflight with unsupported-semantics source/region/bytecode diagnostics are metadata-bound local evidence, not non-executed branch adjoints, full compiler phi lowering, a complete static alias lattice, executable Rust/LLVM/JIT lowering, or full reverse-mode compiler AD. | [Differentiable Programming](docs/differentiable_programming.md), [Differentiable API](docs/differentiable_api.md) |
| Gradient support matrix | Executable support planning now covers registered gates, observables, backends, transforms, and ML/provider adapters with explicit blocked reasons and alternatives. | [Quantum Gradients](docs/quantum_gradients.md), [Differentiable API](docs/differentiable_api.md) |
| Unified readiness ledger | `run_differentiable_readiness_audit()` aggregates the support matrix, transform nesting, QNode tape/transform suites, provider gradients, hardware policy, and provider hardware-preparation audit into one reviewer-facing pass/fail ledger. | [Differentiable Programming](docs/differentiable_programming.md), [Differentiable API](docs/differentiable_api.md) |
| Transform nesting governance | Executable planning now separates supported local `grad`, `value_and_grad`, `hessian`, nested-grad, tape, scalar `jvp`, scalar `vjp`, scalar `jacfwd`, scalar `jacrev`, vector-output native Jacobian execution, native manual `vmap(grad)`, whole-program `grad(vmap(f))` over trace-aware leaves, local Hessian over a whole-program AD scalar objective, JVP/VJP over whole-program AD Hessian transforms, and provider-callback QNode transforms from blocked framework-vectorized, adapter-nested, finite-shot curvature, malformed-provider, and hardware nesting routes. | [Quantum Gradients](docs/quantum_gradients.md), [Differentiable API](docs/differentiable_api.md) |
| Provider-gradient readiness | Executable audit evidence distinguishes deterministic callbacks, finite-shot callbacks, multi-frequency rules, hardware-blocked routes, unknown backends, malformed finite-shot samples, and policy-bound hardware-preparation records. | [Quantum Gradients](docs/quantum_gradients.md), [Differentiable API](docs/differentiable_api.md) |
| Hardware-gradient policy readiness | Executable dry-run policy decisions now gate hardware-gradient preparation by provider/backend allowlist, shot budget, required evidence IDs, and live-execution ticket status. `prepare_provider_hardware_parameter_shift_gradient(...)` packages that approval into provider-preparation evidence, and `run_provider_hardware_gradient_preparation_audit()` verifies supported and blocked preparation routes without submitting QPU jobs. | [Quantum Gradients](docs/quantum_gradients.md), [Differentiable API](docs/differentiable_api.md) |
| Differentiable claim ledger | The Phase-QNode evidence ledger maps implementation, tests, artefact IDs, documentation, known gaps, and promotion status; no promoted claim is accepted without an artefact ID, and support-surface alignment checks keep ledger paths consistent with the generated capability manifest. | [Differentiable Programming](docs/differentiable_programming.md), [Claim Ledger](data/differentiable_phase_qnode/claim_ledger.md) |
| Differentiable public claim table | Public-facing differentiable wording is generated from the committed ledger. Every current row is bounded-candidate only, and the table blocks hardware, provider, QPU, GPU, production-performance, and `isolated_affinity` claims until promotion evidence exists. | [Public Claim Table](data/differentiable_phase_qnode/public_claim_table_20260616.md) |
| Differentiable SOTA scorecard | `run_differentiable_sota_scorecard()` scores the lane against named JAX, PyTorch, PennyLane, Qiskit Runtime, Catalyst, Enzyme, Rust Program AD, provider/hardware, benchmark, docs/API, and adoption baselines. Every current category remains behind-baseline governance evidence until promoted ledger rows and isolated benchmark artefacts exist. | [Differentiable Programming](docs/differentiable_programming.md), [SOTA Scorecard](data/differentiable_phase_qnode/differentiable_sota_scorecard_20260620.md) |
| Differentiable Rust/Python inventory | `run_differentiable_rust_python_inventory()` classifies differentiable Python, Rust, compiler, provider, hardware, metadata, and deprecation surfaces before broad rustification. Rows record owner modules, tests, docs, benchmark status, mypy targets, docstring status, Rust parity, polyglot status, and blockers without promoting Rust, LLVM/JIT, provider, hardware, GPU, or isolated benchmark claims. | [Differentiable Programming](docs/differentiable_programming.md), [Rust/Python Inventory](data/differentiable_phase_qnode/differentiable_rust_python_inventory_20260620.md) |
| Differentiable external-validation lock | The external-validation package records exact SHA-256 digests for runtime, development, Python 3.10-3.13 CI, CPU framework-overlay, and Enzyme-runner lockfiles. The artefact is reviewer reproduction evidence only and remains `functional_non_isolated`. | [Differentiable Programming](docs/differentiable_programming.md), [Environment Lock](data/differentiable_phase_qnode/external_validation_environment_lock_20260616.md) |
| Differentiable CI reproducibility | The differentiable framework workflow runs sparse and full CPU profiles across Python 3.10-3.13, enforces the module-specific test audit, uploads scheduled benchmark metadata, and exposes a manual optional GPU contract lane that remains `functional_non_isolated`. | [Differentiable Programming](docs/differentiable_programming.md), [Workflow](.github/workflows/differentiable-frameworks.yml) |
| Differentiable artefact bundle | The external-validation package records a reproducible manifest over the committed claim ledger, public claim table, environment lock, domain dataset closure, gradient comparison, maturity audit, and local benchmark evidence. The bundle is checksum provenance only and remains `functional_non_isolated`. | [Artefact Bundle](data/differentiable_phase_qnode/external_validation_artifact_bundle_20260616.md) |
| Differentiable external-validation report | The technical report summarizes the comparison package, provider-family status, reproducibility artefacts, and remaining promotion blockers without upgrading any row beyond bounded-candidate evidence. | [External Validation Report](docs/differentiable_external_validation_report.md) |
| Hardening-slice gate | `run_differentiable_hardening_slice_gate(...)` records the required Ruff, mypy, module-specific pytest, test-quality audit, claim-ledger validation, and benchmark-classification checks for each differentiable hardening slice. CI, local preflight, and the pre-push hook additionally enforce a module-specific strict-mypy ratchet across the closed differentiable API, claim-ledger, benchmark-evidence, QNN/QGNN/QSNN training and evidence satellites, objective/domain evidence, optimizer-baseline, backend selection, parameter-shift/VQE foundations, structured-ansatz/methodology/benchmark/Kuramoto/UPDE solver foundations, typed trajectory-result containers, layered ADAPT-VQE, Trotter-error bounds, framework-overlay, provider/hardware-gradient safety, Phase-QNode, framework-bridge, transform-nesting, external-comparison, XY compiler, and PennyLane import modules while repository-wide strict mode remains open debt. The same gates now enforce a scoped NumPy-style Ruff docstring ratchet for the differentiable external-validation, module-hardening audit, and hardening-slice gate surfaces while repository-wide docstring enforcement remains open debt. It is checklist/classification evidence only, not benchmark execution. | [Differentiable Programming](docs/differentiable_programming.md), [Differentiable API](docs/differentiable_api.md) |
| Module-hardening audit | `run_differentiable_module_hardening_audit()` discovers every differentiable/gradient/QNode/bridge/compiler module in the promotion scope and verifies a module-specific test plus declared fail-closed diagnostics for each. | [Differentiable Programming](docs/differentiable_programming.md), [Differentiable API](docs/differentiable_api.md) |
| Bounded phase-QNN training | A deterministic data-reuploading binary classifier is available through `train_parameter_shift_qnn_classifier(...)` with multi-frequency parameter-shift descent, prediction evidence, accuracy, convergence certificates, finite-difference gradient verification, seeded finite-shot gradient uncertainty and noisy-convergence evidence, optional named external-gradient agreement records, a conformance suite with unsuitable-scenario evidence, deterministic convergence suites, non-isolated optimizer-baseline comparisons across parameter-shift, finite-difference, SGD, Adam, L-BFGS-B, diagonal-Fisher natural-gradient, seeded SPSA, and derivative-free grid routes, and caller-supplied framework-gradient agreement checks. | [Quantum Gradients](docs/quantum_gradients.md), [Differentiable API](docs/differentiable_api.md) |
| Registered Phase-QNode family | Local statevector execution, density-matrix execution with bounded single-qubit Kraus channels, arbitrary-depth registered circuit builders with deterministic depth/resource profiles, registered GHZ-chain and hardware-efficient multi-qubit templates, controlled-H/S/T plus Toffoli/CCZ/Fredkin gates with exact Toffoli/Fredkin decompositions, sparse Ising-chain Hamiltonian construction, parameter-shift gradients for pure-state routes, framework parity rows, native JAX deterministic statevector value-and-gradient plus `grad`/`value_and_grad`/`jacfwd`/`jacrev`/`hessian`/`jvp`/`vjp`/`vmap`/`jit` transform lowering for registered local circuits, native JAX PyTree transform lowering with flattened Hessian symmetry evidence for structured registered local circuit parameters, native JAX `pmap` sharding transform lowering with one row per local device, native PyTorch deterministic statevector value-and-gradient lowering, native PyTorch `torch.func.grad`/`jacrev`/`vmap` transform lowering, native PyTorch non-fullgraph `torch.compile` value-and-gradient lowering on CPU, verified SCPN MLIR-runtime lowering adapters, and isolated-affinity benchmark metadata are available for the declared gate/observable subset. Unsupported gates, dynamic/provider paths, native LLVM/JIT lowering, interpreter fallback success, noisy-channel gradients/metrics, registered PyTorch fullgraph `torch.compile` lowering, incompatible CUDA/device execution, finite-shot native framework lowering, and unregistered observables fail closed with support reports. | [Differentiable API](docs/differentiable_api.md), [Benchmark Harness](docs/benchmark_harness.md) |
| ML framework and tape roadmap | Gradient tape, QNode-style tape records, backend gradient planning, provider-safe callback execution with shot/variance accounting, convergence certificates, optional JAX host-callback parameter-shift interop, deterministic registered Phase-QNode JAX statevector transform lowering, deterministic registered Phase-QNode JAX PyTree transform lowering, deterministic registered Phase-QNode JAX pmap/sharding transform lowering, deterministic registered Phase-QNode PyTorch statevector, `torch.func`, and non-fullgraph `torch.compile` transform lowering, PyTorch module/transform/compiler/device maturity routing, PennyLane gradient-agreement checks, TensorFlow host-boundary tensor bridges, and bounded framework parity rows are available. Full provider-backed QNode migration bridges, finite-shot native framework lowering, dynamic-circuit lowering, compatible CUDA/device artefacts, registered PyTorch fullgraph `torch.compile` lowering, and arbitrary architectures remain staged surfaces, not yet advertised as production-complete. | [Differentiable Roadmap](docs/differentiable_roadmap.md) |

This matters commercially because optimisation users do not only need circuits.
They need gradients, convergence evidence, framework interop, reproducible
benchmarks, and clear failure modes. The SCPN route aims to combine
Kuramoto-XY physics, quantum-control objectives, hardware-result governance,
and compiler-backed AD under one support matrix.

Rust polyglot parity includes a claim-bounded Program AD IR metadata parser in
`scpn_quantum_engine::program_ad_ir` plus
`program_ad_effect_ir_metadata_summary(...)` and
`program_ad_effect_ir_interpret_forward(...)` plus
`program_ad_effect_ir_interpret_value_and_gradient(...)` for PyO3 consumers.
Metadata summaries validate `program_ad_effect_ir.v1` evidence only; Rust
replay is bounded to opcode-bearing scalar primitive-family forward and
value+gradient rows, including executed runtime branch metadata when matched by
runtime phi provenance. It still fails closed on legacy opcode-free metadata,
aliases, mutation, arrays, source-level/non-executed branch semantics, general
Program AD execution, LLVM/JIT execution, hardware, and performance promotion.
Python callers can use `scpn_quantum_control.program_ad_rust_bridge` for the
typed fail-closed wrappers; `scpn_quantum_control.differentiable` re-exports
the same symbols for backward compatibility.
Static whole-program bytecode/source frontend inspection now lives in
`scpn_quantum_control.whole_program_frontend`; `scpn_quantum_control.differentiable`
and the package root re-export `compile_whole_program_frontend(...)` and its
report objects for compatibility. The frontend remains no-execution preflight
metadata, not executable Rust, LLVM, JIT, provider, hardware, or benchmark
evidence.
Python compiler interchange lowers captured `program_ad_effect_ir.v1` records
into deterministic `scpn_diff.program_ad_*` MLIR-style operations through
`compile_whole_program_ad_trace_to_mlir(...)`, validated by
`program_ad_mlir_interchange_contracts`. This remains metadata interchange, not
executable Rust, LLVM, JIT, provider, hardware, or performance evidence.

The first production-grade differentiable workflows are deliberately bounded:

1. train small VQE objectives with parameter-shift gradients through `PhaseVQE.solve(gradient_method="parameter_shift")`;
2. verify gradients against finite differences and analytic references through `verify_parameter_shift_gradient(...)` and `verify_vqe_parameter_shift_gradient(...)`;
3. train bounded phase-QNN classifiers through `train_parameter_shift_qnn_classifier(...)`, verify their QNN-specific gradients through `verify_parameter_shift_qnn_classifier_gradient(...)`, record seeded finite-shot uncertainty through `estimate_parameter_shift_qnn_finite_shot_gradient(...)`, package evidence with `run_parameter_shift_qnn_conformance_suite(...)`, certify deterministic local convergence with `run_parameter_shift_qnn_convergence_suite(...)`, replay seeded finite-shot convergence with `run_parameter_shift_qnn_finite_shot_convergence_suite(...)`, compare local optimizer baselines with `run_parameter_shift_qnn_optimizer_benchmark_suite(...)`, and record caller-supplied framework-gradient agreement with `verify_parameter_shift_qnn_framework_agreement(...)`;
4. execute registered local Phase-QNode circuits with `execute_phase_qnode_circuit(...)`, compare installed framework parity with `run_phase_qnode_framework_parity_suite()`, lower deterministic registered statevector value-and-gradient routes into native JAX with `jax_phase_qnode_value_and_grad(...)`, audit registered JAX native transforms with `jax_phase_qnode_native_transform_audit(...)`, audit structured-parameter JAX PyTree transforms with `jax_phase_qnode_pytree_transform_audit(...)`, audit local-device JAX pmap/sharding transforms with `jax_phase_qnode_sharding_transform_audit(...)`, lower deterministic registered statevector value-and-gradient routes into native PyTorch with `torch_phase_qnode_value_and_grad(...)`, audit registered PyTorch `torch.func` transforms with `torch_phase_qnode_transform_audit(...)`, record PyTorch module/transform/compiler/device maturity with `run_torch_ecosystem_maturity_audit(...)`, and lower supported subsets to textual MLIR metadata with `lower_phase_qnode_circuit_to_mlir(...)`;
5. use compiler/program-AD kernels for supported classical objectives;
6. evaluate hardware-gradient preparation with `evaluate_hardware_gradient_policy(...)`
   and `run_hardware_gradient_policy_readiness_suite()` before any provider job
   is prepared;
7. summarize the focused readiness suites with `run_differentiable_readiness_audit()`;
8. document unsupported gates, backends, shapes, and dynamic program paths
   before they can mislead users.

Future releases will extend this route toward native framework gradients beyond
the current bounded JAX/PyTorch/TensorFlow and registered JAX/PyTorch statevector bridges, full PennyLane/Qiskit
migration bridges beyond agreement checks, finite-shot and provider-backed
native lowering, quantum neural networks, analog oscillator mappings,
open-system gradients, benchmark leaderboards, and real-time feedback control.

## Richer Presentation

For a richer presentation of the Phase 1 hardware results, methodology
deep-dives, interactive plots, and architecture diagrams, see the
project website:

**[anulum.li/scpn-quantum-control](https://anulum.li/scpn-quantum-control/)**

Direct entry points:

- [Onboarding](docs/onboarding.md)
  — what the project is, who should use it, what is mature, and what remains
  claim-bound
- [Hardware Status Ledger](docs/hardware_status_ledger.md)
  — claim classes, campaign evidence paths, and publication hygiene rules
- [Hardware Result Packs](docs/hardware_result_packs.md)
  — offline manifest and integrity verifier for promoted IBM raw-count datasets
- [Physics-First Kuramoto-XY](docs/physics_first_kuramoto_xy.md)
  — start from arbitrary oscillator networks before SCPN-specific layers
- [Differentiable Programming](docs/differentiable_programming.md)
  — current AD surface, support boundaries, and user routes
- [Quantum Gradients](docs/quantum_gradients.md)
  — parameter-shift, VQE gradients, verification tests, and planned backend
  gradient planner
- [Differentiable API](docs/differentiable_api.md)
  — public `scpn_quantum_control.differentiable` reference and usage map
- [Differentiable Roadmap](docs/differentiable_roadmap.md)
  — staged plan for tape, framework adapters, QNN/QGNN/QSNN, analog mapping,
  benchmarks, verification, and dashboards
- [Stable Facades API](docs/stable_facades_api.md)
  — mkdocstrings reference for first-path public facades
- [Phase 1 Results](https://anulum.li/scpn-quantum-control/phase1-results.html)
  — raw-count reproduction of the DLA parity asymmetry on
  ibm_kingston, April 2026, with full Welch table and interactive
  Plotly plot
- [Reproducibility Manifest](https://anulum.li/scpn-quantum-control/reproducibility.html)
  — per-commit pinning, IBM job IDs, dependency constraints, rerun
  protocol
- [Method: GUESS Mitigation](https://anulum.li/scpn-quantum-control/method-guess.html)
  — symmetry-guided ZNE, shot-budget-free for the XY Hamiltonian
- [Method: DLA Parity Theorem](https://anulum.li/scpn-quantum-control/method-dla-parity.html)
  — $\mathfrak{su}(2^{n-1}) \oplus \mathfrak{su}(2^{n-1})$
  decomposition and hardware reproduction path
- [Method: Pulse Shaping](https://anulum.li/scpn-quantum-control/method-pulse-shaping.html)
  — ICI three-level (1,665× Rust) and (α, β)-hypergeometric
  (44× Rust)
- [The Science](https://anulum.li/scpn-quantum-control/science.html)
  — plain-language primer on SCPN, Kuramoto-XY, and why the DLA
  parity result matters
- [Methods Benchmark Dashboard](docs/methods_benchmark_dashboard.md)
  — generated Rust/VQE, GPU, tensor-network, FIM, and reproducibility
  artefact index
- [Roadmap](ROADMAP.md)
  — canonical active work queue and completed release-safety tasks

---

## Quick Start

```bash
pip install scpn-quantum-control
```

```python
import numpy as np
from scpn_quantum_control.phase.xy_kuramoto import QuantumKuramotoSolver

# 8 oscillators, exponential-decay coupling, heterogeneous frequencies
N = 8
K = 0.5 * np.exp(-0.3 * np.abs(np.subtract.outer(range(N), range(N))))
omega = np.linspace(0.8, 1.2, N)

# Simulate: Trotter evolution → order parameter R(t)
solver = QuantumKuramotoSolver(N, K, omega)
result = solver.run(t_max=1.0, dt=0.1)
print(f"Final R = {result['R'][-1]:.3f}")
# → R rises from ~0.3 (incoherent) toward 1.0 (synchronised)
```

No IBM credentials needed — runs on local statevector simulator.
Pass any coupling matrix; the built-in SCPN benchmark is just one example.

---

## What This Package Does

**A Kuramoto-XY compiler and hardware-evidence workbench for heterogeneous
coupled oscillators.** The repository contains legacy `ibm_fez` baseline
artefacts, promoted `ibm_kingston` DLA parity datasets, and SCPN/FIM
falsification artefacts with raw counts, job IDs, integrity checks, and
count-to-statistic reproduction harnesses.

The package provides:

1. **A Kuramoto-to-quantum compiler** — any coupling matrix K_nm and natural
   frequencies omega compile directly into executable Qiskit circuits for IBM
   hardware. Rust-accelerated dense Hamiltonian construction is faster than the
   Qiskit `SparsePauliOp` path for small systems (a parity-checked local
   regression guard, ~96× at L=4 on the i5-11600K baseline, shrinking with
   system size) — reproducible and gated; see
   [Native Speedup Benchmark](docs/native_speedup_benchmark.md).

2. **Tracked research module families** probing the synchronisation phase
   transition — synchronisation witnesses, OTOC scrambling, Krylov complexity,
   persistent homology, DLA parity theorem, and more. Novel constructions and
   first applications are documented in the
   research-gems and API pages; exact file counts use the package table below.

3. **Hardware evidence with claim classes** — legacy `ibm_fez` baseline rows,
  promoted `ibm_kingston` DLA parity datasets, and the SCPN/FIM negative
  hardware result are separated from simulator-only, frontier, queued-job, and
  aggregate-only outputs.
   Stable core contracts and backend capability artefacts are included in this
   hardening boundary and are replayed via reproducibility tooling.


Think of it as a **quantum microscope for synchronisation**: classical Kuramoto
tells you *when* oscillators lock in step; this package tells you *what the
quantum state looks like* at the transition, *how entangled it is*, *how fast
information scrambles*, and *whether the system thermalises*.

> **Advanced benchmark:** The built-in SCPN 16-layer coupling matrix (Paper 27)
> provides a heterogeneous-frequency benchmark for structured
> oscillator-network experiments. Publication-facing claims should treat this
> as a classical complex-network input to quantum-inspired Hamiltonian,
> tensor-network, topological, and DLA analyses, not as a quantum-biological or
> clinical causation claim. See
> [SCPN Foundations](https://anulum.github.io/scpn-quantum-control/theory/).

## Key Results

### Hardware Evidence

| Result | Value |
|--------|-------|
| `ibm_kingston` DLA parity Phase 1 | 342 circuits, raw counts, job IDs, integrity checks, and reproduction harness in `data/phase1_dla_parity/` |
| `ibm_kingston` DLA parity Phase 2 | Promoted A+G `n=4` replication, B-C mixed `n=6,8` scaling, and popcount controls in `data/phase2_dla_parity/`, `data/phase2_scaling_bc/`, and `data/phase2_popcount_control/`; no DLA-parity-only or monotone-scaling claim |
| `ibm_kingston` SCPN/FIM hardware test | Pilot and repeated follow-up artefacts in `data/scpn_fim_hamiltonian/`; promoted as a negative result for simple digital `lambda=4` hardware protection on the tested circuit family |
| `ibm_fez` baseline rows | Legacy QPU observations retained in `results/ibm_hardware_2026-03-28/` and `results/march_2026/`; quote only artefact-backed values through the ledger |
| Quarantined IBM outputs | V2/frontier/queued-job/aggregate-only artefacts are not promoted until raw counts, retrieval manifests, and analysis code are reviewed |

### Simulation

| Result | Value |
|--------|-------|
| Critical coupling K_c(∞) | **≈ 2.2** (BKT finite-size scaling) |
| DTC with heterogeneous ω | **15/15** amplitudes show subharmonic response |
| OTOC scrambling | **4× faster** at K=4 vs K=1 (n=8) |
| Schmidt gap transition | **K = 3.44** (n=8, 60-point resolution) |
| DLA dimension formula | **2^(2N-1) − 2** (exact, all N) |

### Software

| Metric | Value |
|--------|-------|
| Rust engine bindings | **141** exported `#[pyfunction]` bindings in the tracked Rust crate; low-level helper `fn` definitions are an implementation detail. |
| Source package surface | **903** tracked Python source files under `src/scpn_quantum_control`, excluding package initialisers. |
| Research module families | Analysis, phase, hardware, bridge, mitigation, QEC, applications, forecasting, and benchmark families; exact current counts are listed in the package map below. |
| Publication figures | **17** (simulation + hardware, including the Phase 1 DLA parity panels and exact-simulation crossover) |
| Test suite | CI-gated suite at a **90%** aggregate coverage gate (`--cov-fail-under=90`); the non-refactor source tree is at 100% line coverage. See the generated capability inventory above for the current tracked test-file count. |
| Reproducibility CLI | `scpn-bench reproduce-methods`, `scpn-bench fim-all`, and `scpn-bench all` regenerate committed methods/FIM artefacts without IBM submission |

### Exact-Simulation Wall-Time (Not broad quantum-advantage claim)

This section covers exact Hilbert-space simulation crossover only.
No broad observable-level quantum-advantage claim is closed yet.
Any broader advantage claim requires comparison against state-of-the-art
tensor-network or GPU baselines and explicit accounting for data-loading and
state-preparation cost.

No quantum advantage at n ≤ 16 in this exact-simulation path. Classical ODE is
faster for all accessible sizes. The value of the quantum approach is
characterisation (entanglement, MBL, witnesses), not speed.

| Method | n=4 | n=8 | n=12 | n=16 |
|--------|----:|----:|-----:|-----:|
| Classical Kuramoto ODE (scipy) | 0.4 ms | 1.4 ms | 2.8 ms | ~11 ms |
| Exact diagonalisation (numpy eigh) | 0.1 ms | 164 ms | 26.8 s | OOM (32 GB) |
| Qiskit statevector | ~50 ms | ~2 s | ~minutes | impractical |
| Rust Hamiltonian + numpy eigh | 0.02 ms | 30 ms | ~5 s | ~2 min (est.) |
| IBM hardware (per-job, 4000 shots) | ~5 s | ~10 s | ~20 s | ~40 s |

Measured on Ubuntu 24.04, AMD Ryzen, 32 GB RAM. Rust speedup applies to
Hamiltonian construction only; the eigh bottleneck is LAPACK in all cases.

### Publications

- [Preprint: Quantum Kuramoto-XY on 156-qubit processor](https://anulum.github.io/scpn-quantum-control/preprint/)
- [Paper: Synchronisation Witness Operators](https://anulum.github.io/scpn-quantum-control/paper_sync_witnesses/) (novel NISQ-ready formalism)
- [Paper: DLA Parity Theorem](https://anulum.github.io/scpn-quantum-control/paper_dla_parity/) (exact closed-form)

## Background: Kuramoto → XY Mapping

Any network of N coupled Kuramoto oscillators can be represented by a linear
Kuramoto-XY Hamiltonian analogue or embedding. This is not a claim that a
gate-model circuit directly Trotterises the nonlinear classical Kuramoto ODE;
direct nonlinear simulation requires an explicit Koopman, Carleman, or
equivalent embedding. The built-in SCPN example uses 16 oscillators with a
coupling matrix K_nm:

```
K_nm = K_base * exp(-alpha * |n - m|)
```

with K_base = 0.45, alpha = 0.3, and empirical calibration anchors
(K[1,2] = 0.302, K[2,3] = 0.201, K[3,4] = 0.252, K[4,5] = 0.154).
Cross-hierarchy boosts link distant layers (L1-L16 = 0.05, L5-L7 = 0.15).
See `docs/equations.md` for the full parameter set.

![Knm coupling matrix](figures/knm_heatmap.png)
*The 16×16 K_nm coupling matrix. White annotations: calibration anchors from
Paper 27 Table 2. Cyan annotations: cross-hierarchy boosts (L1↔L16, L5↔L7).
Exponential decay with distance is visible along the diagonal.*

The classical dynamics follow the Kuramoto ODE:

```
d(theta_i)/dt = omega_i + sum_j K_ij sin(theta_j - theta_i)
```

The working quantum analogue uses the XY Hamiltonian

```
H = -sum_{i<j} K_ij (X_i X_j + Y_i Y_j) - sum_i omega_i Z_i
```

where X, Y, Z are Pauli operators. Superconducting transmon devices can compile
XX+YY interactions through native-gate decompositions, making quantum hardware
a useful test bed for the corresponding Hamiltonian dynamics. The order parameter R — a
measure of global synchronization — is extracted from qubit expectations:
R = (1/N)|sum_i (<X_i> + i<Y_i>)|.

![Layer coherence vs coupling strength](figures/layer_coherence_vs_coupling.png)
*Coherence R as a function of coupling strength K_base across 16 SCPN layers.
Strongly-coupled layers (L3, L4, L10) synchronize first; weakly-coupled L12
lags behind, consistent with the exponential decay in K_nm.*

**Reference**: M. Sotek, *Self-Consistent Phenomenological Network: Layer
Dynamics and Coupling Structure*, Working Paper 27 (2025). Manuscript in
preparation.

## Hardware Results (ibm_fez, February 2026)

| Experiment | Qubits | Depth | Hardware | Exact | Error |
|------------|--------|-------|----------|-------|-------|
| VQE ground state | 4 | 12 CZ | -6.2998 | -6.3030 | **0.05%** |
| Kuramoto XY (1 rep) | 4 | 85 | R=0.743 | R=0.802 | 7.3% |
| Qubit scaling | 6 | 147 | R=0.482 | R=0.532 | 9.3% |
| UPDE-16 snapshot | 16 | 770 | R=0.332 | R=0.615 | 46% |
| QAOA-MPC (p=2) | 4 | -- | -0.514 | 0.250 | -- |

Full results with all 12 decoherence data points: [`results/HARDWARE_RESULTS.md`](results/HARDWARE_RESULTS.md)

**Key findings:**

- VQE with K_nm-informed ansatz achieves 0.05% error on 4-qubit subsystem
- Coherence wall at depth 250-400 on Heron r2 — shallow Trotter (1 rep) beats deep Trotter on NISQ devices

![Trotter depth tradeoff](figures/trotter_tradeoff.png)
*More Trotter repetitions improve mathematical accuracy but increase circuit
depth. On NISQ hardware, decoherence from the extra gates outweighs the
Trotter error reduction. Optimal strategy: fewest reps that capture the physics.*

- 16-layer UPDE snapshot on real hardware — per-layer structure partially tracks coupling topology (L12 collapse, L3 resilience at the extremes; Spearman rho = -0.13 across all layers)

![UPDE-16 per-layer expectations](figures/upde16_layer_bars.png)
*Per-layer X-basis expectations from the 16-qubit UPDE snapshot on ibm_fez.
L12 (most weakly coupled) shows near-complete decoherence; strongly-coupled
layers (L3, L4, L10) maintain coherence.*

- 12-point decoherence curve from depth 5 to 770 with exponential decay fit

![Decoherence curve](figures/decoherence_curve.png)
*Hardware-to-exact ratio R_hw/R_exact vs circuit depth. The three regimes:
near-perfect readout (depth < 25), linear decoherence (85-400), and
noise-dominated (> 400).*

## Package Map

Counts below are tracked Python source files under `src/scpn_quantum_control`,
excluding package initialisers.

```mermaid
graph TD
    subgraph Foundation
        bridge["bridge/ (13)\nK_nm → Hamiltonian\ncross-repo adapters"]
    end

    subgraph "Core Physics"
        phase["phase/ (29)\nTrotter, VQE, ADAPT-VQE\nVarQITE, Floquet DTC"]
        analysis["analysis/ (58)\nWitnesses, QFI, PH\nOTOC, Krylov, magic"]
    end

    subgraph "Applications"
        control["control/ (11)\nQAOA-MPC, residual VQLS-GS\nPetri nets, ITER"]
        qsnn["qsnn/ (7)\nQuantum spiking\nneural networks"]
        apps["applications/ (13)\nFMO, power grid\nJosephson, EEG, ITER"]
    end

    subgraph "Hardware & QEC"
        hw["hardware/ (63)\nIBM runner, backends\nGPU offload, cutting"]
        mit["mitigation/ (12)\nZNE, PEC, DD\nZ2 post-selection"]
        qec["qec/ (13)\nToric code, surface code\nrep code, error budget"]
    end

    subgraph "Field Theory"
        gauge["gauge/ (5)\nWilson loops, vortices\nCFT, universality"]
        crypto["crypto/ (6)\nBB84, Bell tests\ntopology-auth QKD"]
    end

    bridge --> phase
    bridge --> analysis
    bridge --> control
    bridge --> qsnn
    phase --> analysis
    phase --> apps
    hw --> phase
    mit --> hw
    qec --> hw
    analysis --> gauge

    style bridge fill:#6929C4,color:#fff
    style analysis fill:#d4a017,color:#000
    style phase fill:#6929C4,color:#fff
```

| Subpackage | Modules | Purpose |
|------------|:-------:|---------|
| `analysis` | 58 | Synchronisation probes: witnesses, QFI, PH, OTOC, Krylov, magic, BKT, DLA |
| `hardware` | 63 | IBM Quantum runner, plugin backends registry, AsyncHardwareRunner, trapped-ion backend, GPU offload, circuit cutting, fast sparse, qubit mapper (DynQ), provenance |
| `phase` | 29 | Time evolution: Trotter, VQE, ADAPT-VQE, VarQITE, AVQDS, QSVT, Floquet DTC, Lindblad |
| `applications` | 13 | FMO photosynthesis, power grid, Josephson array, EEG, ITER, quantum EVS |
| `bridge` | 13 | K_nm → Hamiltonian, cross-repo adapters (sc-neurocore, SSGF, orchestrator) |
| `control` | 11 | QAOA-MPC, residual-certified VQLS Grad-Shafranov, Petri nets, ITER disruption, topological optimiser |
| `mitigation` | 12 | ZNE, PEC, dynamical decoupling, Z2 parity, CPDR, symmetry verification, GUESS, compound |
| `qec` | 13 | Toric code, repetition code UPDE, surface code, biological surface code, error budget, multi-scale, syndrome flow |
| `benchmarks` | 7 | Classical vs quantum scaling, MPS baseline, GPU baseline, AppQSim |
| `crypto` | 6 | BB84, Bell tests, topology-authenticated QKD, key hierarchy |
| `identity` | 6 | VQE attractor, coherence budget, entanglement witness, fingerprint |
| `qsnn` | 7 | Quantum spiking neural networks (LIF, trace STDP, synapses, dynamic coupling, training, neuromorphic bridge) |
| `gauge` | 5 | U(1) Wilson loops, vortex detection, CFT, universality, confinement |
| `psi_field` | 4 | U(1) compact lattice gauge: lattice, infoton, observables, SCPN mapping |
| `ssgf` | 4 | SSGF quantum integration |
| `accel` | 3 | Multi-language dispatcher + Julia tier (Rust → Julia → Python fallback chain) |
| `dla_parity` | 4 | DLA parity helpers and campaign analysis support |
| `fep` | 2 | Friston Free Energy Principle: variational free energy, predictive coding |
| `forecasting` | 1 | Held-out synchronisation forecasting over hardware traces and source-backed topology replays |
| `benchmark_harness` | 4 | Reproducible benchmark harness entry points |
| `tcbo` | 1 | TCBO quantum observer |
| `pgbo` | 1 | PGBO quantum bridge |
| `l16` | 1 | Layer 16 quantum director |

## Quick Start

```bash
pip install scpn-quantum-control
```

**Any coupling network** — bring your own K and omega:

```python
from scpn_quantum_control import QuantumKuramotoSolver, build_kuramoto_ring

K, omega = build_kuramoto_ring(6, coupling=0.5, rng_seed=42)
solver = QuantumKuramotoSolver(6, K, omega)
result = solver.run(t_max=1.0, dt=0.1, trotter_per_step=2)
print(f"R(t): {result['R']}")
```

**Built-in SCPN network** (16 oscillators from Paper 27):

```python
from scpn_quantum_control import QuantumKuramotoSolver, build_knm_paper27, OMEGA_N_16

K = build_knm_paper27(L=4)
solver = QuantumKuramotoSolver(4, K, OMEGA_N_16[:4])
result = solver.run(t_max=0.5, dt=0.1, trotter_per_step=2)
```

**Detect synchronization** with witness operators:

```python
from scpn_quantum_control.analysis.sync_witness import evaluate_all_witnesses

# After running X-basis and Y-basis circuits on IBM hardware:
results = evaluate_all_witnesses(x_counts, y_counts, n_qubits=4)
for name, w in results.items():
    print(f"{name}: {'SYNCHRONIZED' if w.is_synchronized else 'incoherent'}")
```

For development (editable install with test/lint tooling):

```bash
pip install -e ".[dev]"
pre-commit install
pytest tests/ -v
```

### Hardware execution (requires IBM Quantum credentials)

```bash
pip install -e ".[ibm]"
python run_hardware.py --experiment kuramoto --qubits 4 --shots 10000
```

## Data Flow

The pipeline from coupling matrix to measurement follows a fixed sequence:

```mermaid
graph LR
    A["K_nm\ncoupling matrix"] --> B["knm_to_hamiltonian()\nSparsePauliOp"]
    B --> C["Trotter / VQE\nQuantumCircuit"]
    C --> D["Transpile\nnative gates"]
    D --> E["Execute\nAer / IBM"]
    E --> F["Parse counts\n⟨X⟩, ⟨Y⟩, ⟨Z⟩"]
    F --> G["Order parameter\nR(t)"]

    style A fill:#6929C4,color:#fff
    style G fill:#2ecc71,color:#000
```

## Examples

29 standalone scripts in [`examples/`](examples/):

| # | Script | What it demonstrates |
|:-:|--------|---------------------|
| 01 | `qlif_demo` | Quantum LIF neuron: membrane → Ry rotation → spike |
| 02 | `kuramoto_xy_demo` | 4-oscillator Kuramoto dynamics, R(t) trajectory |
| 03 | `qaoa_mpc_demo` | QAOA binary MPC: quadratic cost → Ising Hamiltonian |
| 04 | `qpetri_demo` | Quantum Petri net: tokens evolve in superposition |
| 05 | `vqe_ansatz_comparison` | Three ansatze benchmarked on 4-qubit Hamiltonian |
| 06 | `zne_demo` | Zero-noise extrapolation with unitary folding |
| 07 | `crypto_bell_test` | CHSH inequality violation certification |
| 08 | `dynamical_decoupling` | DD pulse sequence insertion (XY4, X2, CPMG) |
| 09 | `classical_vs_quantum_benchmark` | Scaling crossover analysis |
| 10 | `identity_continuity_demo` | VQE attractor basin stability |
| 11 | `pec_demo` | Probabilistic error cancellation |
| 12 | `trapped_ion_demo` | Ion trap noise model comparison |
| 13 | `iter_disruption_demo` | ITER plasma disruption classification |
| 14 | `quantum_advantage_demo` | Advantage threshold estimation |
| 15 | `qsnn_training_demo` | QSNN training loop with parameter-shift |
| 16 | `fault_tolerant_demo` | Repetition code UPDE |
| 17 | `snn_ssgf_bridges_demo` | Cross-repo bridge roundtrips |
| 18 | `end_to_end_pipeline` | Complete K_nm → IBM → analysis pipeline |
| 19 | `sync_witness_operator` | Synchronisation witness operator demo |
| 20 | `quantum_persistent_homology` | Persistent homology analysis |
| 21 | `biological_qec_scpn16` | Biological surface code on 16-layer SCPN |
| 22 | `quantum_neuromorphic_bridge` | QSNN quantum LIF + trace STDP + dynamic coupling bridge |
| 23 | `differentiable_api_workflow` | Unified differentiable API, diagnostics, compiler report, and bounded QNN training |
| 24 | `differentiable_benchmark_reproduction` | Local differentiable benchmark evidence bundle reproduction with non-isolated classification |

All examples run on statevector simulation (no QPU needed).

## Notebooks

98 tracked Jupyter notebooks in [`notebooks/`](notebooks/) — including the
core tutorials and retained investigation notebooks. Core notebooks:

| # | Notebook | Level | Key Output |
|:-:|----------|:-----:|------------|
| 01 | Kuramoto XY Dynamics | Beginner | R(t) trajectory, quantum-classical overlay |
| 02 | VQE Ground State | Beginner | Energy convergence, ansatz comparison |
| 03 | Error Mitigation | Intermediate | ZNE extrapolation plot |
| 04 | UPDE 16-Layer | Intermediate | Per-layer R bar chart |
| 05 | Crypto & Entanglement | Intermediate | CHSH S-parameter, QKD QBER |
| 06 | PEC Error Cancellation | Advanced | PEC vs ZNE, overhead scaling |
| 07 | Quantum Advantage | Advanced | Scaling crossover prediction |
| 08 | Identity Continuity | Advanced | Attractor basin, fingerprint |
| 09 | ITER Disruption | Domain | Feature distributions, accuracy, CONTROL bridge contract |
| 10 | QSNN Training | Advanced | Loss curve, weight evolution |
| 11 | Surface Code Budget | Advanced | QEC resource estimation |
| 12 | Trapped Ion Comparison | Advanced | Noise model comparison |
| 13 | Cross-Repo Bridges | Integration | Phase roundtrip, adapter demos |

All run on local AerSimulator. No IBM credentials needed.

## Architecture

```
scpn_quantum_control/
├── analysis/       58 modules — synchronisation probes
├── hardware/       63 modules — IBM runner, backends, GPU, cutting, provenance
├── phase/          29 modules — time evolution + variational + Lindblad
├── bridge/         13 modules — K_nm → quantum objects + cross-repo
├── applications/   13 modules — physical system benchmarks
├── control/        11 modules — QAOA-MPC, residual VQLS-GS, Petri, ITER, topological
├── mitigation/     12 modules — ZNE, PEC, DD, Z2, CPDR, symmetry
├── qec/            13 modules — error correction + biological surface code
├── benchmarks/      7 modules — performance baselines
├── identity/        6 modules — identity continuity analysis
├── qsnn/            7 modules — quantum spiking neural networks + neuromorphic bridge
├── crypto/          6 modules — QKD, Bell tests, key hierarchy
├── gauge/           5 modules — U(1) gauge theory probes
├── ssgf/            4 modules — SSGF quantum integration
├── tcbo/            1 module  — TCBO quantum observer
├── pgbo/            1 module  — PGBO quantum bridge
├── l16/             1 module  — Layer 16 quantum director
└── scpn_quantum_engine/  Rust crate (PyO3 0.29, 141 exported PyO3 bindings)
```

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| qiskit | >= 2.2,<3.0 | Circuit construction, transpilation |
| qiskit-aer | >= 0.15,<1.0 | Statevector + noise simulation |
| numpy | >= 1.24 | Array operations |
| scipy | >= 1.10 | Sparse linear algebra, optimisation |
| networkx | >= 3.0 | Graph algorithms (QEC decoder) |

Optional:
- `matplotlib >= 3.5` for visualisation
- `qiskit-ibm-runtime >= 0.40,<1.0` for hardware execution
- `cupy >= 12.0` for GPU-accelerated simulation

## Limitations

- **NISQ benchmarking only.** Current hardware results are proof-of-concept.
  Circuit depths >400 hit the Heron r2 coherence wall; the 16-layer UPDE
  snapshot (46% error) confirms this. Real tokamak control requires <1 ms
  deterministic latency on radiation-hardened hardware — cloud QPUs cannot
  provide that.
- **SCPN is an unpublished model.** The 16-layer coupling structure comes
  from a 2025 working paper (Sotek, Paper 27) with no external citations
  yet. The Kuramoto→XY mapping is standard physics; the specific K_nm
  parameterisation is not independently validated.
- **Small-scale broad advantage not demonstrated.** At N=4-16 qubits,
  classical ODE solvers outperform quantum simulation in both speed and
  accuracy. The exact Hilbert-space crossover is a resource boundary, not
  a demonstrated general advantage.
- **IBM hardware claim hygiene.** Do not cite queued-job, placeholder,
  aggregate-only, or frontier JSON as hardware validation. The promoted
  raw-count campaign is `data/phase1_dla_parity/`; legacy `ibm_fez`
  observations must name their committed artefact row.

## Documentation

Full docs at **[anulum.github.io/scpn-quantum-control](https://anulum.github.io/scpn-quantum-control)**:

- [Onboarding](docs/onboarding.md) — project purpose, user routes, application value, and claim boundaries
- [Installation](docs/installation.md) — pip install + all optional extras
- [Quickstart](docs/quickstart.md) — first experiment in 5 minutes
- [Differentiable Tutorials](docs/differentiable_tutorials.md) — runnable gradient workflow with diagnostics, compiler report, and bounded QNN training
- [Differentiable Programming](docs/differentiable_programming.md) — bounded AD surface, gradients, compiler kernels, and roadmap boundaries
- [Quantum Gradients](docs/quantum_gradients.md) — parameter-shift and gradient-evidence route for VQE and quantum-control objectives
- [Differentiable API](docs/differentiable_api.md) — public differentiable namespace and support matrix
- [Differentiable Roadmap](docs/differentiable_roadmap.md) — staged gradient, adapter, benchmark, verification, and control roadmap
- [Tutorials](docs/tutorials.md) — 4-level learning path, 14 tutorials
- [Stable Facades API](docs/stable_facades_api.md) — first-path public API for notebooks, tutorials, and integrations
- [API Overview](docs/api.md) — stable facade route first, advanced module references second
- [Research Gems](docs/research_gems.md) — **33 analysis modules with theory and API**
- [Equations](docs/equations.md) — every equation in the codebase
- [Architecture](docs/architecture.md) — dependency graph + 20 subpackages
- [Analysis API](docs/analysis_api.md) — advanced reference for 46 analysis modules
- [Phase API](docs/phase_api.md) — advanced reference for 29 evolution algorithms
- [Application Benchmark Plugins](docs/application_benchmarks.md) — EEG, plasma, power-grid, and FEP datasets through QPU artefacts
- [Classical Baselines](docs/classical_baselines.md) — SciPy ODE, QuTiP Lindblad, and MPS TEBD provenance surfaces
- [Hardware Guide](docs/hardware_guide.md) — IBM Quantum setup
- [Notebooks](docs/notebooks.md) — 98 tracked notebooks (47 core + 51 Colab)
- [Bridges](docs/bridges_api.md) — cross-repo integrations
- [Language Policy](docs/language_policy.md) — Rust / Julia / Go / Mojo accel chain
- [Pipeline Performance](docs/pipeline_performance.md) — every module's measured wall-time + multi-language benchmarks
- [Issue Triage](docs/triage.md) — label taxonomy, SLAs, routing
- [Falsification](docs/falsification.md) — 8 named claims + falsifiers

## Related Repositories

| Repository | Description |
|-----------|-------------|
| [sc-neurocore](https://github.com/anulum/sc-neurocore) | Classical SCPN spiking neural network engine (v3.13.3, 2155+ tests) |
| `scpn-fusion-core` | Classical SCPN algorithms: Kuramoto solvers, coupling calibration, transport (v3.9.3, 3300+ tests) |
| [scpn-phase-orchestrator](https://github.com/anulum/scpn-phase-orchestrator) | SCPN phase orchestration: regime detection, UPDE engine, Petri-net supervisor (v0.5.0, 2321 tests) |
| `scpn-control` | SCPN control systems: plasma MPC, disruption mitigation (v0.18.0, 3015 tests) |

## Citation

```bibtex
@software{scpn_quantum_control,
  title  = {scpn-quantum-control: Quantum-Native SCPN Phase Dynamics and Control},
  author = {Sotek, Miroslav},
  year   = {2026},
  url    = {https://github.com/anulum/scpn-quantum-control},
  doi    = {10.5281/zenodo.18821929}
}
```

## License

[AGPL-3.0-or-later](LICENSE) — commercial license available.

Dual licensing is explicit: the public repository is
`AGPL-3.0-or-later`, and proprietary integration requires a separate
commercial licence grant. The generic Kuramoto-XY facade is documented
as a possible future core-package boundary, but it is not a separate
permissive package today. Until an explicit release changes SPDX
headers and package metadata, all in-repository code remains under the
AGPL/commercial terms above.

| Use case | Route |
|----------|-------|
| Academic research, teaching, and individual experiments | Use the AGPL terms in `LICENSE`. |
| AGPL-compatible open-source redistribution | Use the AGPL terms and preserve notices/source obligations. |
| Closed-source product, internal proprietary tool, SaaS, consulting deliverable, or embedded deployment | Obtain a commercial licence before distribution or network service use. |
| Future lightweight core package | Not available yet; no permissive relicensing is implied by the facade docs. |

### Commercial Licensing

AGPL-3.0 requires derivative works and network-service modifications to
provide corresponding source under the AGPL. If you need to integrate
scpn-quantum-control into proprietary software without publishing your
source code, use the commercial route:

1. Email **protoscience@anulum.li** with organisation name, product or
   service description, deployment model, expected users, and whether
   source redistribution is required.
2. Select the licence tier below or request an enterprise quote.
3. Execute the commercial licence grant before shipping the proprietary
   integration or offering it as a network service.

| Tier | Price | Includes |
|------|-------|----------|
| **Indie** | CHF 49/month | Single developer, one product |
| **Pro** | CHF 199/month | Team up to 10, unlimited products |
| **Perpetual** | CHF 999 one-time | Permanent license, one major version |
| **Enterprise** | Custom | SLA, priority support, custom modules |

Reference files: [`LICENSE`](LICENSE), [`NOTICE.md`](NOTICE.md),
[`docs/core_package_boundary.md`](docs/core_package_boundary.md), and
[`docs/licensing_faq.md`](docs/licensing_faq.md).

Contact: **protoscience@anulum.li** | [anulum.li](https://www.anulum.li)

---

<p align="center">
  <a href="https://www.anulum.li">
    <img src="docs/assets/anulum_logo_company.jpg" width="180" alt="ANULUM">
  </a>
  &nbsp;&nbsp;&nbsp;&nbsp;
  <a href="https://www.anulum.li">
    <img src="docs/assets/fortis_studio_logo.jpg" width="180" alt="Fortis Studio">
  </a>
  <br>
  <em>Developed by <a href="https://www.anulum.li">ANULUM</a> / Fortis Studio</em>
</p>
