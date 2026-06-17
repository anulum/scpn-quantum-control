# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Differentiable Programming

# Differentiable Programming

`scpn-quantum-control` treats differentiability as a product surface, not a hidden implementation detail. The goal is to make coupled-oscillator quantum control trainable, testable, and explainable across quantum gradients, classical program AD, compiler-backed kernels, and future ML-framework adapters.

## Business-facing value

The differentiable lane is where optimisation-heavy teams get practical value: a
bounded route from objectives to gradient diagnostics that is explicit about what is
analytic, approximate, stochastic, and unsupported.

In practical terms, this means:

- training and convergence evidence can be produced without waiting on hardware;
- optimisation routes can be compared against finite-difference and multi-framework
  references;
- unsupported or blocked cases stay visible in the API evidence contract.
- finite-difference result artefacts carry an explicit diagnostic-only claim
  boundary so they cannot be promoted as analytic, parameter-shift,
  native-framework, whole-program AD, provider, hardware, or production benchmark
  evidence.

This section is intentionally conservative. It prefers explicit fail-closed boundaries
over hidden fallback because enterprises depend on predictable failure modes.

## Why this matters

A useful quantum-control framework must answer four questions quickly:

1. Can the objective produce a gradient?
2. Is that gradient analytic, approximate, stochastic, or unsupported?
3. Does the gradient help an optimiser converge on a known problem?
4. Can another tool or framework reproduce the same derivative?

This repository now documents those questions directly. Current support is deliberately bounded so users can trust the surfaces that are advertised.

## Current capability map

| Surface | Status | Evidence route |
|---|---|---|
| Unified differentiable API | Available through `scpn_quantum_control.differentiable_api` for value, gradient, Jacobian, Hessian, support, diagnostics, compile, local conformance benchmark, and GUI/audit-dashboard status reports using one JSON-ready evidence envelope. Dashboard status rows label executable, metadata-only, diagnostic, conformance-backed, planned, blocked, and unsupported routes without promoting Program AD metadata, bounded `program_ad_effect_ir.v1` round-trip parsing, higher-order transform evidence, Rust/LLVM compiler paths, provider routes, hardware routes, or local conformance rows beyond their evidence class. | [Differentiable API](differentiable_api.md) |
| Parameter-shift gradients | Available for callable scalar objectives, structured `PhaseVQE` gradients, local gradient-descent VQE examples, metric-aware natural-gradient descent, multi-start optimizer comparison evidence, and compatible composed phase-control objectives through `scpn_quantum_control.phase`. | [Quantum Gradients](quantum_gradients.md), [Variational Methods](variational.md) |
| Objective evidence | Available for composed phase-control objectives through finite-difference agreement, compatibility-gate checks, and local training certificates. | [Differentiable API](differentiable_api.md), [Quantum Gradients](quantum_gradients.md) |
| Objective execution planning | Available for composed objectives through fail-closed routing between pure parameter-shift, hybrid term-gradient, hardware-blocked, and unsupported backend routes. | [Differentiable API](differentiable_api.md), [Quantum Gradients](quantum_gradients.md) |
| Gradient support matrix | Available for executable planning across registered gates, observables, backends, transforms, and adapters with explicit alternatives for blocked combinations. | [Differentiable API](differentiable_api.md), [Quantum Gradients](quantum_gradients.md) |
| Unified readiness ledger | Available through `run_differentiable_readiness_audit()` as one JSON-ready reviewer ledger aggregating support, transform, QNode, provider, hardware-policy, and provider hardware-preparation audits. | [Differentiable API](differentiable_api.md), [Quantum Gradients](quantum_gradients.md) |
| Transform nesting governance | Available for bounded first-order, value-and-gradient, deterministic local Hessian, nested-grad Hessian, tape, scalar local JVP/VJP, scalar local jacfwd/jacrev, deterministic native vector-output Jacobian execution, native manual `vmap(grad)`, program-AD `grad(vmap(f))` over trace-aware leaves, and provider-callback QNode transforms with finite-shot uncertainty propagation, with fail-closed framework vectorization, adapter-nesting, malformed-provider, finite-shot curvature, and hardware boundaries. | [Differentiable API](differentiable_api.md), [Quantum Gradients](quantum_gradients.md) |
| Provider-gradient readiness | Available as an executable support matrix for deterministic callbacks, finite-shot callbacks, multi-frequency rules, hardware-blocked routes, unknown backends, malformed samples, and policy-bound hardware-preparation evidence. | [Differentiable API](differentiable_api.md), [Quantum Gradients](quantum_gradients.md) |
| Hardware-gradient policy readiness | Available as a fail-closed dry-run policy layer for provider/backend allowlists, shot and evaluation budgets, evidence IDs, and live-ticket gating before hardware-gradient job preparation. Provider preparation records and a provider hardware-preparation audit suite can be generated without executing QPU jobs. | [Differentiable API](differentiable_api.md), [Quantum Gradients](quantum_gradients.md) |
| Compiler/program AD | Available for supported scalar, vector, and matrix primitives with registry contracts, lowering reports, native executable kernels on bounded paths, verified whole-program determinant lowering through `19x19`, 2x2 `matrix_power(..., 2)` and 2x2 `multi_dot` native lowering, trace-aware eager `vmap` slicing/stacking inside whole-program objectives, fail-closed finite/dtype/shape checks, trainable-mask zeroing at derivative result boundaries including whole-program forward and adjoint result containers, deterministic `program_ad_effect_ir.v1` metadata parsing, metadata-only phi records for runtime and source-level control joins, deterministic alias/effect metadata summaries over emitted Program AD IR, IR-format and replay-count provenance on supported scalar adjoint replay results, and explicit whole-program Python-semantics diagnostics. Python scalar `abs()` and NumPy absolute-value tracing share the registered fail-closed zero-cusp policy. `np.sort` has a registered strict-total-order selection contract for bounded trace dispatch; equal values and `np.argsort` index selection fail closed. `np.median`, scalar-`q` `np.quantile`, and scalar-`q` `np.percentile` have registered order-statistic reduction contracts with static q/axis/method validation and strict-order selection boundaries. Emitted alias metadata distinguishes mutation-version edges from bounded local scalar rebinding, local list-alias rebinding/mutation metadata, and supported executed array-view aliases for reshape, ravel, basic slicing, take, and transpose. Wider concrete static `matrix_power` and rectangular `multi_dot` lowering remains verified only through MLIR-runtime executable rules; native LLVM/JIT and Rust promotion fails closed until independently verified executable kernels exist. Norm and linalg conditioning diagnostics report zero-norm, rank-threshold, repeated-spectrum, and ill-conditioned boundaries before callers rely on sensitive direct derivative rules. Closures, default arguments, keyword-only parameters, `*args`, `**kwargs`, and generator expressions are reported as accepted semantics; materialized comprehensions, captured object/dataclass attributes, recursion, generator functions, context managers, exception control flow, and decorators fail closed before execution. Phi records and adjoint replay counts are provenance only; alias/effect summaries remain `metadata_only_no_general_alias_lattice`, not non-executed branch adjoints, full reverse-mode compiler AD, or a complete static alias lattice. | [Differentiable API](differentiable_api.md), [Quickstart](quickstart.md) |
| Primitive registry | Available for derivative, batching, lowering, shape, dtype, and nondifferentiability contracts on supported primitive identities. | `scpn_quantum_control.differentiable` |
| Reverse replay and program traces | Available for supported captured operations with source/bytecode feature reports and named Python-semantics accept/reject diagnostics; unsupported arbitrary Python remains fail-closed. | Support reports and module-specific tests |
| Finite-difference diagnostics | Available for scalar gradients, vector Jacobians, JVP/VJP contractions, Hessians, and HVPs as local smooth-objective diagnostics. Result artefacts expose `claim_boundary` metadata and remain non-promotional evidence, not analytic, parameter-shift, native-framework, whole-program AD, provider, hardware, or production benchmark claims. | `scpn_quantum_control.differentiable`, [Quantum Gradients](quantum_gradients.md) |
| JAX, PyTorch, TensorFlow adapters | Optional parameter-shift value-and-gradient bridges plus a fail-closed ML parity audit are available for supported phase objectives; native framework autodiff through arbitrary simulators remains open. | [Differentiable Roadmap](differentiable_roadmap.md), [Quantum Gradients](quantum_gradients.md) |
| Gradient tape | MVP available for supported phase parameter-shift records, plus QNode-style tape records for deterministic, seeded finite-shot, and provider-boundary evidence; arbitrary Python and programme-IR tape semantics remain open. | [Quantum Gradients](quantum_gradients.md), [Differentiable Roadmap](differentiable_roadmap.md) |
| Registered Phase-QNode circuit family | Available for the declared local gate/observable subset with arbitrary-depth registered circuit builders, deterministic depth/resource profiles, registered GHZ-chain and hardware-efficient multi-qubit templates, controlled-H/S/T plus Toffoli/CCZ/Fredkin gates, exact Toffoli/Fredkin operation-list decompositions, statevector execution, density-matrix execution with bounded single-qubit Kraus channels, analytic parameter-shift gradients for pure-state routes, framework parity rows, verified SCPN MLIR-runtime lowering adapters, and affinity-labelled benchmark metadata. Unsupported gates, observables, provider paths, dynamic routes, native LLVM/JIT lowering, noisy-channel gradients/metrics, and interpreter fallback success claims fail closed with support reports. | [Differentiable API](differentiable_api.md), [Benchmark Harness](benchmark_harness.md) |
| QNN/QGNN/QSNN training lane | A bounded phase-QNN binary classifier, QNN-specific finite-difference gradient verification, deterministic multi-seed convergence envelopes, bounded loss-landscape grids, seeded finite-shot gradient uncertainty and noisy-convergence evidence, named external-gradient agreement records, dedicated caller-supplied framework-gradient agreement checks, deterministic convergence-suite evidence, conformance-suite evidence with unsuitable-scenario records, non-isolated optimizer-baseline comparisons across parameter-shift, finite-difference, SGD, Adam, L-BFGS-B, diagonal-Fisher natural-gradient, seeded SPSA, and derivative-free grid routes, QSNN parameter-shift training evidence, and a registered medium QNN/QGNN/QSNN/Kuramoto-XY training evidence suite are available locally; arbitrary QNN/QGNN/QSNN stacks and production convergence notebooks remain planned. | [Differentiable API](differentiable_api.md), [Quantum Gradients](quantum_gradients.md) |

## Evidence Promotion Lane

The differentiable Phase-QNode lane is SOTA-candidate until the committed claim
ledger, isolated CI benchmark artefact, and external comparison rows all pass.
The ledger is committed at
`data/differentiable_phase_qnode/claim_ledger.json` with a reviewer summary in
`data/differentiable_phase_qnode/claim_ledger.md`.
The public-safe wording table is generated from that ledger at
`data/differentiable_phase_qnode/public_claim_table_20260616.md`. Use
`render_public_claim_table(...)` and `validate_public_claim_table(...)` when a
release note, README, package description, or reviewer response needs
claim-boundary language. Every current row is a bounded candidate, so the table
blocks hardware, provider, QPU, GPU, production-performance, and
`isolated_affinity` claims until promoted evidence exists.
The current public technical report is
[Differentiable External-Validation Technical Report](differentiable_external_validation_report.md).
It summarizes the comparison package, provider-family status, reproducibility
artefacts, and promotion blockers without upgrading any row beyond bounded
candidate evidence.
`validate_differentiable_support_surface_alignment()` checks that each ledger
row still points to existing implementation, test, and documentation surfaces
and that source/test/docs paths are present in the generated capability
manifest. This is a consistency gate only; it does not promote hardware,
provider, or performance claims.
`run_differentiable_hardening_slice_gate(...)` records the required closeout
checklist for each differentiable hardening slice: focused Ruff formatting and
linting, mypy over changed source targets, module-specific pytest targets,
the repository test-quality audit, claim-ledger validation, and benchmark
classification smoke cases. It also verifies that GitHub-hosted runners remain
`functional_non_isolated`, incomplete isolated-runner metadata remains a
`hard_gap`, complete self-hosted isolated metadata is the only
`isolated_affinity` path, and silent accelerator fallback remains a hard gap.
The gate is planning and classification evidence only; it does not run shell
commands or promote benchmark artefacts.
`run_differentiable_module_hardening_audit()` discovers the differentiable
module promotion scope from the committed patterns, compares it with the
registered hardening map, and verifies that every module has module-specific
tests plus declared fail-closed diagnostic surfaces. This closes the local
module-inventory portion of the hardening lane; formal proof, provider
execution, hardware execution, and isolated benchmark promotion remain separate
evidence gates.

Optional framework parity uses an explicit CPU-only overlay instead of the
repository `jax` extra, because that extra resolves to `jax[cuda12]`.
`run_phase_qnode_framework_parity_suite()` now exposes explicit scenarios:
the default `single_qubit_ry_rx_pauli_z` compatibility row and
`registered_two_qubit_entangling_statevector`, which executes a registered
two-qubit entangling Phase-QNode statevector tensor path across installed JAX,
PyTorch, TensorFlow, and PennyLane backends. Both routes remain local parity
evidence only; they do not promote provider execution, finite-shot sampling,
hardware gradients, or unrestricted simulator-autodiff claims.

```bash
PYTHONPATH=src:. python scripts/install_differentiable_framework_overlay.py \
  --overlay-path "${XDG_CACHE_HOME:-$HOME/.cache}/scpn-qc-framework-site-py312" \
  --manifest-path /tmp/scpn-qc-framework-overlay.json \
  --install
```

The generated manifest prints the exact `PYTHONPATH` for parity runs, records
package versions when verification succeeds, and lists only CPU wheels:
`jax[cpu]`, `torch`, `tensorflow-cpu`, and `pennylane`.

The external-validation package also has an exact environment lock manifest at
`data/differentiable_phase_qnode/external_validation_environment_lock_20260616.json`
with a reviewer summary at
`data/differentiable_phase_qnode/external_validation_environment_lock_20260616.md`.
`build_external_validation_environment_lock()` records SHA-256 digests, byte
sizes, line counts, and pinned-package counts for the runtime, development,
Python 3.10-3.13 CI, CPU framework overlay, and Python 3.9 Enzyme runner
lockfiles. `validate_external_validation_environment_lock()` rechecks those
digests against the current checkout. This is reproducibility evidence only:
the artefact remains `functional_non_isolated` and does not promote hardware,
provider, GPU, QPU, production-performance, or `isolated_affinity` benchmark
claims.
The reproducible artefact-bundle manifest is committed at
`data/differentiable_phase_qnode/external_validation_artifact_bundle_20260616.json`
with a reviewer summary at
`data/differentiable_phase_qnode/external_validation_artifact_bundle_20260616.md`.
`build_external_validation_artifact_bundle()` records SHA-256 digests for the
claim ledger, public claim table, environment lock, domain dataset closure,
identical-circuit comparison, PyTorch maturity audit, and local benchmark
evidence files. `validate_external_validation_artifact_bundle()` rechecks those
digests against the current checkout. The bundle is checksum provenance only
and remains `functional_non_isolated`.

Differentiable CI reproducibility is split into explicit sparse, full, optional
GPU-contract, scheduled metadata, and isolated-runner lanes. The sparse and full
CPU profiles run across Python 3.10, 3.11, 3.12, and 3.13 using the pinned
per-version Linux requirement locks. Full profiles build the CPU-only framework
overlay for `jax[cpu]`, `torch`, `tensorflow-cpu`, and `pennylane`; sparse
profiles keep the baseline dependency surface. The same workflow runs the
module-specific test-quality audit after the differentiable parity tests, so
new differentiable tests cannot be hidden in a generic coverage bucket. The
manual optional GPU lane runs GPU request/fail-closed contract tests on a
GitHub-hosted runner and uploads a `functional_non_isolated` JSON record; it is
not live GPU, provider, QPU, or production-performance evidence.

Benchmark artefacts written by
`scripts/run_differentiable_benchmark_evidence.py` are CI evidence only.
External comparison artefacts written by
`write_differentiable_external_comparison(...)` record row payloads,
dependency versions, toolchain metadata, failure classes, and local Python/host
metadata, but they are still classified as `functional_non_isolated`.
The benchmark evidence script writes `diff-qnode-external-comparison.json`
beside the benchmark bundle and records that artefact's ID in the bundle, so CI
artifacts retain the complete comparison evidence chain without upgrading local
correctness rows into performance claims.
GitHub-hosted runners are classified as `functional_non_isolated`; production
performance wording requires a self-hosted runner labelled
`isolated-benchmark`, explicit CPU affinity, observed process affinity that
matches the requested CPU set, host-load context, governor or frequency
context, and no concurrent heavy jobs. The remote CI job is the benchmark gate:
the repository may not promote the claim unless that job uploads an artefact
classified as `isolated_affinity`. Unconfigured Enzyme tooling is a recorded
`dependency_missing` hard gap, not a hidden success. When
`SCPN_ENZYME_RUNNER` is configured and LLVM/Enzyme tooling is present, the
external comparison row sends a strict JSON request, enforces a timeout, records
runner toolchain metadata, and accepts success only when value and gradient
match the SCPN analytic reference. Accelerator benchmark claims are also
fail-closed: the benchmark evidence bundle always records explicit accelerator
metadata. CPU-only runs are labelled CPU-only; CUDA or ROCm requested through
`SCPN_BENCH_ACCELERATOR_BACKEND` must expose matching visible-device metadata
(`SCPN_BENCH_ACCELERATOR_DEVICE_IDS`, `CUDA_VISIBLE_DEVICES`,
`ROCR_VISIBLE_DEVICES`, `HIP_VISIBLE_DEVICES`, or JAX CUDA device discovery)
or the artefact is classified as `hard_gap` with
`silent_accelerator_fallback`.

Self-hosted runner preparation is explicit:

```bash
PYTHONPATH=src:. python tools/setup_isolated_benchmark_runner.py \
  --repo anulum/scpn-quantum-control
```

The helper prints the labels, runner directory, runner version, and download
URL without mutating the host. Add `--install` only on the reserved Linux x64
benchmark host. A claim is still not promoted until the CI artefact itself
reports `isolated_affinity`. If the repository has no registered
self-hosted runner with the `isolated-benchmark` label, the benchmark gate is
not executable and the claim remains unpromoted.

## User routes

| Goal | Recommended path |
|---|---|
| Train a small VQE objective | `phase.param_shift` -> [Quantum Gradients](quantum_gradients.md) -> [Variational Methods](variational.md) |
| Train and verify a bounded QNN classifier | `phase.qnn_training` -> `train_parameter_shift_qnn_classifier(...)` -> `verify_parameter_shift_qnn_classifier_gradient(...)` -> `estimate_parameter_shift_qnn_finite_shot_gradient(...)` -> `run_parameter_shift_qnn_conformance_suite(...)` -> `run_parameter_shift_qnn_convergence_suite(...)` -> `run_parameter_shift_qnn_multi_seed_convergence_suite(...)` -> `run_parameter_shift_qnn_loss_landscape_suite(...)` -> `run_parameter_shift_qnn_finite_shot_convergence_suite(...)` -> `run_parameter_shift_qnn_framework_agreement_suite(...)` -> `run_parameter_shift_qnn_optimizer_benchmark_suite(...)` -> [Quantum Gradients](quantum_gradients.md) |
| Execute and compare a registered Phase-QNode | `phase.qnode_circuit` -> `execute_phase_qnode_circuit(...)` -> `parameter_shift_phase_qnode_gradient(...)` -> `run_phase_qnode_framework_parity_suite()` -> `lower_phase_qnode_circuit_to_mlir(...)` -> `compile_phase_qnode_circuit_to_mlir_runtime(...)` |
| Inspect compiler-backed AD | `differentiable_compile_report(...)` -> [Quickstart](quickstart.md) differentiable primitive path -> [Differentiable API](differentiable_api.md) |
| Follow the complete differentiable tutorial | `examples/23_differentiable_api_workflow.py`, `examples/24_differentiable_benchmark_reproduction.py` -> [Differentiable Tutorials](differentiable_tutorials.md) |
| Build a custom primitive | `CustomDerivativeRule` -> `CustomDerivativeRegistry` -> primitive contract tests |
| Decide whether a gradient stack can run | `explain_differentiability(...)`, `differentiable_support_report(...)`, `plan_gradient_support(...)`, `plan_gradient_transform_nesting(...)`, `plan_quantum_gradient_backend(...)`, `run_phase_qnode_tape_readiness_suite()`, `run_provider_gradient_readiness_audit(...)`, `run_hardware_gradient_policy_readiness_suite()`, `run_provider_hardware_gradient_preparation_audit()`, and `run_differentiable_readiness_audit()` |
| Prepare ML-framework integration | Follow [Differentiable Roadmap](differentiable_roadmap.md) until adapter tests land |

## Production-Readiness Rubric

A differentiable workflow should be treated as production-ready only when all
of these checks are true:

| Check | Required evidence |
|---|---|
| Mathematical derivative contract | Parameter-shift, analytic derivative, adjoint replay, or compiler-AD rule is named and registered. |
| Shape and dtype contract | Primitive registry or support matrix admits the target tensor rank, dtype, backend, and static arguments. |
| Verification | Finite-difference, analytic, or independent framework agreement is recorded for a small representative case. |
| Optimiser behaviour | Descent or convergence diagnostics exist for the target objective class. |
| Backend policy | Simulator, finite-shot simulator, hardware, or adapter route declares shot, variance, budget, evidence-ID, ticket, and blocked-state policy. |
| Documentation | Unsupported or unsuitable scenarios are listed with alternatives and no silent fallback. |

This is the current standard for claiming enterprise-grade differentiable
behaviour in this repository. Anything below that bar is documented as staged,
experimental, or unsupported.

`explain_differentiability(...)` is the first inspection call for unsupported
or ambiguous routes. It returns a fail-closed diagnostic report with the exact
blocked reasons, suggested alternatives, bounded framework dependency rows,
device capability rows, backend planning rows, and the underlying support-plan
payload. It is intentionally non-executing planning evidence: no objective,
provider callback, hardware job, or performance benchmark is run by the
diagnostic.

## Design principles

- Fail closed when a derivative mode is unsupported.
- Separate exact, approximate, finite-shot, and roadmap gradient modes.
- Do not silently treat analytic classical penalties as parameter-shift quantum terms.
- Keep shape, dtype, backend, and primitive support inspectable.
- Compare gradients against finite differences, analytic references, and cross-framework references where practical.
- Document failed or unsuitable scenarios because they are research evidence.

## Immediate production targets

The next differentiable-programming implementation rounds should prioritise:

1. larger registered Phase-QNode parity families beyond the current arbitrary-depth registered local subset with controlled-gate decomposition coverage;
2. multi-start convergence studies on known ground states and VQE systems, extending the current phase-optimizer comparison audit with derivative-free baselines;
3. native framework agreement beyond the registered Phase-QNode parity family and bounded QNN records;
4. broader PennyLane adapter round-trip tests beyond caller-supplied framework-gradient agreement checks;
5. broader QNN/QGNN/QSNN convergence notebooks beyond the bounded local phase-QNN conformance, deterministic convergence, deterministic multi-seed, bounded loss-landscape, seeded finite-shot, named optimizer-baseline suites, QSNN tests, and registered medium evidence suite;
6. public tutorials for Kuramoto-XY VQE gradients and coupling learning beyond
   the current unified differentiable API tutorial;
7. executable implementations for still-blocked framework-native nested routes where the physics contract is clear; native vector-output Jacobian, provider-callback QNode transforms, and manual `vmap(grad)` now have bounded local evidence.

## Unsupported boundaries

Unsupported does not mean ignored. Current public boundaries include:

- arbitrary Python/NumPy program AD beyond supported trace operations;
- full native compiler AD for every MLIR/LLVM/JIT path;
- complete gradient tape semantics beyond supported phase parameter-shift and QNode-style phase records;
- public JAX/PyTorch/TensorFlow adapters;
- hardware gradient jobs without hardware-gradient policy approval, required evidence IDs, and live-execution ticketing where applicable;
- provider callbacks that omit finite-shot variance or shifted-sample provenance;
- unsupported gate, observable, transform, adapter, or backend combinations returned by `plan_gradient_support(...)`;
- unsupported transform nesting returned by `plan_gradient_transform_nesting(...)`;
- arbitrary quantum neural architectures beyond the bounded local phase-QNN classifier, its QNN-specific gradient verifier, deterministic multi-seed envelope, bounded loss-landscape scans, finite-shot simulator evidence, conformance, convergence, framework-agreement, and named optimizer-baseline suites, and declared QSNN training routes;
- gates without registered generator spectra;
- dynamic topology changes that invalidate parameter indexing;
- static dense native determinant traces at `20x20` and wider until a stronger determinant-partial helper is verified;
- wide native quotient-linalg traces beyond the documented support profile.

See [Differentiable Roadmap](differentiable_roadmap.md) for the staged closure plan.
