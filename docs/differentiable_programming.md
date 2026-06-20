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
| Transform nesting governance | Available for bounded first-order, value-and-gradient, deterministic local Hessian, nested-grad Hessian, tape, scalar local JVP/VJP, scalar local jacfwd/jacrev, deterministic native vector-output Jacobian execution, native manual `vmap(grad)`, exact custom JVP/VJP rules under eager `vmap`, program-AD `grad(vmap(f))` over trace-aware leaves, JVP/VJP over `vmap` of whole-program AD gradients, local Hessian over a whole-program AD scalar objective, JVP/VJP over whole-program AD Hessian transforms, and provider-callback QNode transforms with finite-shot uncertainty propagation, with fail-closed framework vectorization, adapter-nesting, malformed-provider, finite-shot curvature, and hardware boundaries. | [Differentiable API](differentiable_api.md), [Quantum Gradients](quantum_gradients.md) |
| Provider-gradient readiness | Available as an executable support matrix for deterministic callbacks, finite-shot callbacks, multi-frequency rules, hardware-blocked routes, unknown backends, malformed samples, and policy-bound hardware-preparation evidence. | [Differentiable API](differentiable_api.md), [Quantum Gradients](quantum_gradients.md) |
| Hardware-gradient policy readiness | Available as a fail-closed dry-run policy layer for provider/backend allowlists, shot and evaluation budgets, evidence IDs, and live-ticket gating before hardware-gradient job preparation. Provider preparation records and a provider hardware-preparation audit suite can be generated without executing QPU jobs. | [Differentiable API](differentiable_api.md), [Quantum Gradients](quantum_gradients.md) |
| Compiler/program AD | Available for supported scalar, vector, and matrix primitives with registry contracts, lowering reports, native executable kernels on bounded paths, verified whole-program determinant lowering through `19x19`, 2x2 `matrix_power(..., 2)` and 2x2 `multi_dot` native lowering, trace-aware eager `vmap` slicing/stacking inside whole-program objectives, fail-closed finite/dtype/shape checks, trainable-mask zeroing at derivative result boundaries including whole-program forward and adjoint result containers, deterministic `program_ad_effect_ir.v1` metadata parsing, metadata-only phi records for runtime and source-level control joins backed by `program_ad_control_phi_metadata_contracts`, deterministic alias/effect metadata summaries and static alias-lattice readiness reports over emitted Program AD IR, generated `ProgramADAdjointStep` provenance with finite local pullback scales, cotangent-flow rows, reverse effect-order rows, executed runtime control/phi row bindings, IR-format provenance, and replay-count provenance on supported scalar adjoint generation results, and explicit whole-program Python-semantics diagnostics. Python scalar `abs()` and NumPy absolute-value tracing share the registered fail-closed zero-cusp policy. `np.sign` and `np.heaviside` have registered derivative-losing elementwise contracts that fail closed before trace execution. Product, interpolation, signal, and stencil primitives (`np.inner`, `np.outer`, `np.matmul`, `np.tensordot`, `np.einsum`, `np.interp`, `np.convolve`, `np.correlate`, and `np.gradient`) have registry-backed conformance through `structured_numeric_primitive_contracts`; dynamic interpolation grids, singular stencil spacing, and Rust/LLVM executable promotion remain fail-closed. Cumulative primitives (`np.cumsum`, `np.cumprod`, and `np.diff`) have registry-backed conformance through `cumulative_primitive_contracts`; dynamic axis promotion and Rust/LLVM executable promotion remain fail-closed. Assembly primitives (`np.zeros_like`, `np.ones_like`, `np.full_like`, `np.hstack`, `np.vstack`, `np.column_stack`, and `np.dstack`) have registry-backed conformance through `assembly_primitive_contracts`; dynamic shape assembly and Rust/LLVM executable promotion remain fail-closed. Reduction primitives (`np.sum`, `np.prod`, `np.mean`, `np.var`, `np.std`, `np.trapezoid`, `np.max`, `np.min`, `np.median`, scalar-`q` `np.quantile`, and scalar-`q` `np.percentile`) have registry-backed conformance through `reduction_primitive_contracts`; dynamic axes, dynamic q, tie boundaries, zero-variance standard deviation, and Rust/LLVM executable promotion remain fail-closed. Shape primitives (`np.reshape`, `np.ravel`, `np.transpose`, `np.expand_dims`, `np.squeeze`, `np.swapaxes`, `np.moveaxis`, `np.repeat`, `np.atleast_1d`, `np.atleast_2d`, `np.atleast_3d`, `np.tile`, `np.roll`, `np.rot90`, `np.flip`, `np.flipud`, and `np.fliplr`) have registry-backed conformance through `shape_primitive_contracts`; dynamic shape arguments, invalid axes, and Rust/LLVM executable promotion remain fail-closed. Broadcast primitives (`np.broadcast_to`, `np.broadcast_arrays`, and binary elementwise rank broadcasting) have registry-backed conformance through `broadcast_primitive_contracts`; dynamic output shapes, incompatible shapes, subclass propagation, and Rust/LLVM executable promotion remain fail-closed. `np.sort` has a registered strict-total-order selection contract for bounded trace dispatch; equal values fail closed. `np.select`, `np.piecewise`, `np.choose`, `np.compress`, and `np.extract` have registered static selection-fold contracts with shape, dtype, mask/selector/branch metadata, batching, lowering metadata, and fail-closed dynamic-mask/dynamic-selector boundaries. `np.argmax`, `np.argmin`, trace-array `.argmax()`/`.argmin()`, and `np.argsort` have registered integer-output selection contracts that validate shape, dtype, static-axis/kind, batching, and lowering metadata before failing closed as nondifferentiable selectors. `np.take_along_axis`, `np.delete`, `np.pad`, and `np.insert` have registered static indexing/gather/assembly contracts with direct JVP/VJP factories and conformance row `indexing_static_gather_contracts`; dynamic indices, dynamic insertion values, non-constant padding modes, and Rust/LLVM executable promotion remain fail-closed. `np.flipud` and `np.fliplr` have registered rank-checked fixed-axis shape contracts. `np.zeros_like`, `np.ones_like`, and `np.full_like` have registered reference-shape/scalar-fill assembly contracts for derivative-preserving constant arrays. `np.hstack`, `np.vstack`, `np.column_stack`, and `np.dstack` have registered static-shape assembly contracts with fixed-shape direct JVP/VJP factories. `np.var`/`np.std` and trace-array `.var()`/`.std()` have registered static axis/ddof reduction contracts with positive-denominator and zero-variance standard-deviation boundaries. `np.max`/`np.min` and trace-array `.max()`/`.min()` have registered unique-selector reduction contracts. `np.median`, scalar-`q` `np.quantile`, and scalar-`q` `np.percentile` have registered order-statistic reduction contracts with static q/axis/method validation and strict-order selection boundaries. Emitted alias metadata distinguishes mutation-version edges from bounded local scalar rebinding, bounded expression-rebinding aliases, local object-attribute aliases, branch-local control-path alias blockers, local list-alias rebinding/mutation metadata, bounded loop-carried scalar state metadata, supported executed array-view aliases for reshape, ravel, basic slicing, take, transpose, squeeze, expand_dims, atleast rank promotion, swapaxes, moveaxis, repeat source reuse, tile source reuse, roll, rot90, flip, flipud, and fliplr, plus static rank-1 slice-mutation source indices for supported Program AD views. Wider concrete static `matrix_power` and rectangular `multi_dot` lowering remains verified only through MLIR-runtime executable rules; native LLVM/JIT and Rust promotion fails closed until independently verified executable kernels exist. Norm and linalg conditioning diagnostics report zero-norm, rank-threshold, repeated-spectrum, and ill-conditioned boundaries before callers rely on sensitive direct derivative rules. Closures, default arguments, keyword-only parameters, `*args`, `**kwargs`, generator expressions, and plain list comprehensions without dynamic filters are reported as accepted semantics; filtered comprehensions, set/dict comprehensions, captured object/dataclass attributes, recursion, generator functions, context managers, exception control flow, and decorators fail closed before execution. Phi records and generated adjoint steps are provenance only; static alias-lattice reports are bounded to emitted Program AD IR and record non-executed phi inputs and branch-local control-path aliases as blockers, not non-executed branch adjoints, captured/global object-attribute aliasing, full reverse-mode compiler AD, or executable compiler lowering. | [Differentiable API](differentiable_api.md), [Quickstart](quickstart.md) |
| Primitive registry | Available for derivative, batching, lowering metadata, shape, dtype, static-argument, nondifferentiability, and effect contracts on supported primitive identities. `program_ad_registry_dispatch_coverage_report()` exposes registry-dispatched coverage for 118 declared Program AD primitives across 12 families, and the dashboard row `program_ad_registry_dispatch_coverage` is backed by `program_ad_registry_dispatch_contracts` when local conformance runs. Registry contracts live in `scpn_quantum_control.program_ad_registry`; `scpn_quantum_control.differentiable` re-exports them for compatibility. Boundary-sensitive elementwise primitives are surfaced through `program_ad_elementwise_primitives` backed by `elementwise_boundary_contracts`; unsupported domain boundaries and derivative-losing `sign`/`heaviside` kernels remain blocked. Static selection folds, strict sort, `where`, and `clip` are surfaced through `program_ad_selection_primitives` backed by `selection_piecewise_contracts`; dynamic masks/selectors, ties, integer-output selector differentiation, Rust/LLVM executable lowering, hardware, and performance promotion remain blocked. | `scpn_quantum_control.program_ad_registry`, `scpn_quantum_control.differentiable` |
| Reverse replay and program traces | Available for supported captured operations with source/bytecode feature reports, a first-class static bytecode/source compiler frontend preflight through `compile_whole_program_frontend()`, bytecode basic blocks, source regions, source-bytecode line maps with source-relative and absolute CPython line coordinates, symbol-scope entries, deterministic frontend digests, and named Python-semantics accept/reject diagnostics. The static frontend implementation lives in `scpn_quantum_control.whole_program_frontend` with `scpn_quantum_control.differentiable` kept as the compatibility facade. Unsupported-semantics diagnostics also bind fail-closed constructs to source-relative lines, optional absolute file lines, source-region IDs, and bytecode offsets when available. `program_ad_ir_roundtrip_conformance` is backed by `program_ad_ir_roundtrip_contracts`, checking stable `program_ad_effect_ir.v1` parser reconstruction of emitted SSA/effect/control/phi metadata. `program_ad_control_phi_metadata` is backed by `program_ad_control_phi_metadata_contracts`, checking runtime/source control-region and `ProgramADPhiNode` provenance against analytic and adjoint references. `program_ad_reverse_adjoint_replay` is backed by `program_adjoint_replay_provenance_contracts`, checking `ProgramADAdjointResult` gradient parity, generated `ProgramADAdjointStep` rows, finite local pullback scales, cotangent-flow rows, reverse effect-order rows, and replay node/effect/runtime control/phi row bindings against `program_ad_effect_ir.v1`. Unsupported arbitrary Python, executable compiler lowering, non-executed branch adjoints, full reverse-mode compiler AD, Rust/LLVM executable lowering, hardware, and performance promotion remain fail-closed. | Support reports and module-specific tests |
| Finite-difference diagnostics | Available for scalar gradients, vector Jacobians, JVP/VJP contractions, Hessians, and HVPs as local smooth-objective diagnostics. Result artefacts expose `claim_boundary` metadata and remain non-promotional evidence, not analytic, parameter-shift, native-framework, whole-program AD, provider, hardware, or production benchmark claims. | `scpn_quantum_control.differentiable`, [Quantum Gradients](quantum_gradients.md) |
| JAX, PyTorch, TensorFlow adapters | Optional parameter-shift value-and-gradient bridges, bounded phase-QNN native framework routes, deterministic registered local Phase-QNode statevector value-and-gradient plus flat, PyTree, and pmap/sharding native-transform lowering for JAX, including PyTree Hessian symmetry evidence and one-row-per-local-device sharding checks, deterministic registered local Phase-QNode statevector, `torch.func.grad`/`jacrev`/`vmap`, and non-fullgraph `torch.compile` transform lowering for PyTorch, plus fail-closed ML parity and PyTorch module/transform/compiler/device maturity audits are available for supported phase objectives; native framework autodiff through provider, finite-shot, dynamic-circuit, incompatible CUDA/device, registered PyTorch fullgraph `torch.compile`, hardware, and arbitrary-simulator routes remains open. | [Differentiable Roadmap](differentiable_roadmap.md), [Quantum Gradients](quantum_gradients.md) |
| Gradient tape | MVP available for supported phase parameter-shift records, plus QNode-style tape records for deterministic, seeded finite-shot, and provider-boundary evidence; arbitrary Python and programme-IR tape semantics remain open. | [Quantum Gradients](quantum_gradients.md), [Differentiable Roadmap](differentiable_roadmap.md) |
| Registered Phase-QNode circuit family | Available for the declared local gate/observable subset with arbitrary-depth registered circuit builders, deterministic depth/resource profiles, registered GHZ-chain and hardware-efficient multi-qubit templates, controlled-H/S/T plus Toffoli/CCZ/Fredkin gates, exact Toffoli/Fredkin operation-list decompositions, statevector execution, density-matrix execution with bounded single-qubit Kraus channels, analytic parameter-shift gradients for pure-state routes, framework parity rows, native JAX statevector value-and-gradient plus flat, PyTree, and pmap/sharding native-transform lowering rows with Hessian symmetry evidence, native PyTorch statevector, `torch.func`, and non-fullgraph `torch.compile` transform lowering rows, verified SCPN MLIR-runtime lowering adapters, and affinity-labelled benchmark metadata. Unsupported gates, observables, provider paths, dynamic routes, native LLVM/JIT lowering, registered PyTorch fullgraph `torch.compile` lowering, incompatible CUDA/device routes, noisy-channel gradients/metrics, finite-shot native framework lowering, and interpreter fallback success claims fail closed with support reports. | [Differentiable API](differentiable_api.md), [Benchmark Harness](benchmark_harness.md) |
| QNN/QGNN/QSNN training lane | A bounded phase-QNN binary classifier, QNN-specific finite-difference gradient verification, deterministic multi-seed convergence envelopes, bounded loss-landscape grids, seeded finite-shot gradient uncertainty and noisy-convergence evidence, named external-gradient agreement records, dedicated caller-supplied framework-gradient agreement checks, deterministic convergence-suite evidence, conformance-suite evidence with unsuitable-scenario records, non-isolated optimizer-baseline comparisons across parameter-shift, finite-difference, SGD, Adam, L-BFGS-B, diagonal-Fisher natural-gradient, seeded SPSA, and derivative-free grid routes, QSNN parameter-shift training evidence, and a registered medium QNN/QGNN/QSNN/Kuramoto-XY training evidence suite are available locally; arbitrary QNN/QGNN/QSNN stacks and production convergence notebooks remain planned. | [Differentiable API](differentiable_api.md), [Quantum Gradients](quantum_gradients.md) |

Rust polyglot parity includes `scpn_quantum_engine::program_ad_ir`, a
serde-backed `program_ad_effect_ir.v1` metadata parser with the PyO3
`program_ad_effect_ir_metadata_summary(...)` export, a bounded scalar forward
interpreter exposed as `program_ad_effect_ir_interpret_forward(...)`, and a
bounded scalar value+gradient replay exposed as
`program_ad_effect_ir_interpret_value_and_gradient(...)`. Rust replay only
executes opcode-bearing scalar primitive-family `program_ad_effect_ir.v1` rows
emitted by current Python traces, including executed runtime branch metadata
when matched by runtime phi provenance. Legacy opcode-free metadata, aliases,
mutation, array semantics, source-level and non-executed branch semantics,
general Program AD execution, LLVM/JIT lowering, hardware execution, and
performance promotion remain fail-closed.
Python integration is isolated in
`scpn_quantum_control.program_ad_rust_bridge`, which owns the typed wrapper
dataclasses, native-extension fail-closed handling, JSON payload validation,
and NumPy input normalisation. The legacy differentiable-programming facade
continues to re-export those bridge symbols.
Python compiler interchange lowers captured `program_ad_effect_ir.v1` records
into deterministic `scpn_diff.program_ad_ssa`,
`scpn_diff.program_ad_effect`, `scpn_diff.program_ad_alias_edge`,
`scpn_diff.program_ad_control_region`, and `scpn_diff.program_ad_phi`
operations through `compile_whole_program_ad_trace_to_mlir(...)`. The
`program_ad_mlir_interchange_contracts` row validates that metadata lowering as
local conformance only; executable Rust, LLVM, JIT, provider, hardware, and
performance promotion remain blocked.

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
`run_differentiable_sota_scorecard()` adds the category scorecard that keeps
the promotion discussion tied to named external baselines: JAX transforms,
PyTorch autograd/compile, PennyLane QNode/device/plugin breadth, Qiskit Runtime
provider gradients, Catalyst compiler workflows, Enzyme compiler AD, Rust
Program AD, provider/hardware gradients, benchmark promotion, docs/API
maintainability, and adoption/licensing. The committed artefacts live at
`data/differentiable_phase_qnode/differentiable_sota_scorecard_20260620.json`
and `data/differentiable_phase_qnode/differentiable_sota_scorecard_20260620.md`.
Every current category remains `behind_baseline`; the scorecard is governance
evidence only and does not promote performance, provider, QPU, GPU, hardware,
or `isolated_affinity` claims.
`run_differentiable_rust_python_inventory()` adds the rustification surface
inventory required before broad Rust migration. It classifies each current
differentiable route as `rust_backed`, `python_reference`, `metadata_only`,
`compiler_native_not_rust`, `provider_blocked`, `hardware_blocked`, or
`deprecate_before_promotion`; records owner modules, tests, docs, benchmark
status, mypy targets, docstring status, Rust parity, and polyglot status; and
keeps every row non-promotional until matching claim-ledger, benchmark, and
provider/hardware evidence exists. The committed artefacts live at
`data/differentiable_phase_qnode/differentiable_rust_python_inventory_20260620.json`
and
`data/differentiable_phase_qnode/differentiable_rust_python_inventory_20260620.md`.
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
CI, local preflight, and the pre-push hook enforce a separate strict-mypy
ratchet over the differentiable API, claim-ledger, benchmark-evidence,
hardening-gate, QNN/QGNN/QSNN training and evidence satellites,
objective/domain evidence, optimizer-baseline, backend selection, parameter-shift/VQE foundations,
structured-ansatz/methodology/benchmark/Kuramoto/UPDE solver foundations,
typed trajectory-result containers,
layered ADAPT-VQE,
Trotter-error bounds,
framework-overlay, external-validation, Phase-QNode, gradient, provider/hardware-gradient safety, framework-bridge,
transform-nesting, external-comparison, XY compiler, and PennyLane import
modules that have been closed module-by-module. That ratchet is
module-specific governance; repository-wide `mypy --strict` remains open until
the rest of the codebase is migrated.
`run_differentiable_module_hardening_audit()` discovers the differentiable
module promotion scope from the committed patterns, compares it with the
registered hardening map, and verifies that every module has module-specific
tests plus declared fail-closed diagnostic surfaces. This closes the local
module-inventory portion of the hardening lane. CI, local preflight, and the
pre-push hook also enforce a scoped NumPy-style Ruff docstring ratchet for this
audit module, `run_differentiable_hardening_slice_gate(...)`, and their
module-specific tests. Formal proof, provider execution, hardware execution,
isolated benchmark promotion, and repository-wide docstring enforcement remain
separate evidence gates.

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

CI, local preflight, and the pre-push hook now include the external-validation
module and its module-specific tests in the scoped NumPy-style Ruff docstring
ratchet alongside the module-hardening audit and hardening-slice gate surfaces.
That ratchet covers the manifest builders, checksum validators, renderer
contracts, and fail-closed drift branches while repository-wide docstring
enforcement remains open rollout debt.

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
| Execute and compare a registered Phase-QNode | `phase.qnode_circuit` -> `execute_phase_qnode_circuit(...)` -> `parameter_shift_phase_qnode_gradient(...)` -> `run_phase_qnode_framework_parity_suite()` -> `jax_phase_qnode_value_and_grad(...)` -> `jax_phase_qnode_native_transform_audit(...)` / `jax_phase_qnode_pytree_transform_audit(...)` / `jax_phase_qnode_sharding_transform_audit(...)` / `torch_phase_qnode_value_and_grad(...)` / `torch_phase_qnode_transform_audit(...)` / `run_torch_ecosystem_maturity_audit(...)` -> `lower_phase_qnode_circuit_to_mlir(...)` -> `compile_phase_qnode_circuit_to_mlir_runtime(...)` |
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
- Treat PennyLane finite-shot metadata as non-coercive: analytic mode is
  `shots=None`, and finite-shot mode requires an explicit positive integer
  before plugin/device dispatch.
- Require PennyLane provider-plugin artefact `execution_mode` values to be
  provider-scoped while still rejecting hardware/QPU execution modes.
- Keep PennyLane plugin-matrix route evidence canonical: statuses are
  `passed`, `blocked`, or `failed`, and route metadata is control-clean.
- Pair PennyLane provider-gradient parity evidence to the same provider
  execution artefact, circuit fingerprint, backend, and shot policy before
  marking provider-gradient parity as passed.
- Require ticketed live hardware, allowlist, shot-budget, raw-count,
  calibration, and metadata evidence before PennyLane hardware-plugin execution
  can pass; keep promotion blocked until isolated benchmark evidence exists.
- Split Qiskit Runtime evidence into no-submit primitive metadata and live QPU
  EstimatorV2/SamplerV2 artefacts; live QPU evidence must carry ticket,
  backend-allowlist, shot-budget, ISA/transpiled-circuit, Runtime result, and
  metadata digests before the live-ticket gate can pass. Build those live-QPU
  artefacts from captured Runtime metadata with the no-submit Qiskit helper;
  raw-count replay and calibration/statevector comparison artefacts must be
  attached to the same Runtime QPU provider/backend/circuit/live-ticket chain
  before their gates can pass. Use the provider evidence bundle when attaching
  Runtime QPU, raw-count, calibration, and isolated benchmark artefacts
  together; omitting the isolated benchmark ID keeps benchmark promotion
  blocked. Attach provider-gradient workflow artefacts for the complete
  parameter-shift, finite-difference, LCU, SPSA, QGT, and QFI method set before
  the Qiskit maturity audit can pass the provider-gradient workflow gate.
- Compare gradients against finite differences, analytic references, and cross-framework references where practical.
- Document failed or unsuitable scenarios because they are research evidence.

## Immediate production targets

The next differentiable-programming implementation rounds should prioritise:

1. larger registered Phase-QNode parity families beyond the current arbitrary-depth registered local subset with controlled-gate decomposition coverage;
2. multi-start convergence studies on known ground states and VQE systems, extending the current phase-optimizer comparison audit with derivative-free baselines;
3. native framework agreement beyond the registered Phase-QNode parity family, registered JAX/PyTorch statevector lowering, and bounded QNN records;
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
- public JAX/PyTorch/TensorFlow adapters beyond the documented bounded-model and deterministic registered statevector routes;
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
