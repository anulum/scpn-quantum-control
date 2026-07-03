# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Differentiable API

# Differentiable API

This page maps the public differentiable-programming namespace and the related quantum-gradient entry points. It is an API guide, not a proof that every exported symbol is production-ready for every backend. Always pair an API call with the support matrix and tests for the target primitive, backend, shape, dtype, and transform.

The API contract is deliberately fail-closed: a supported route should return a
structured plan, result, certificate, or diagnostic; an unsupported route should
return a blocked plan or raise a targeted error instead of silently substituting
finite differences or pretending that a hardware/provider gradient exists.

## Public namespaces

| Namespace | Role |
|---|---|
| `scpn_quantum_control.diff` / `scpn.diff` | Canonical first-path namespace for `grad`, `value_and_grad`, `jacfwd`, `jacrev`, `jacobian`, `hessian`, `jvp`, `vjp`, `vmap`, `jit_or_explain`, `gradient_tape`, and `differentiable_circuit`. It wraps existing supported local routes and returns fail-closed diagnostics for unsupported JIT, provider, hardware, and performance routes. |
| `scpn_quantum_control.differentiable_api` | Unified façade for value, gradient, Jacobian, Hessian, support, diagnostics, compile, local conformance benchmark, transform-algebra, and dashboard-status reports with one JSON evidence envelope. |
| `scpn_quantum_control.differentiable` | AD data structures, compatibility re-exports, optimisation helpers, program-AD metadata, and support reports. |
| `scpn_quantum_control.differentiable_transform_algebra` | Executable local metamorphic gate for `grad(vmap(f))`, `vmap(grad(f))`, `jacrev`/`jacfwd`, Hessian symmetry, JVP/VJP duality, linearity, chain rule, phase-periodic parameter-shift wraparound, sparse/masked parameters, dtype/broadcast replay, and fail-closed unsupported transform boundaries. |
| `scpn_quantum_control.program_ad_registry` | Primitive identity, custom-derivative registry, transform-contract, and registry-dispatch coverage contracts for Program AD primitive families. |
| `scpn_quantum_control.phase.param_shift` | Parameter-shift gradient helper and gradient-descent VQE example. |
| `scpn_quantum_control.phase.coupling_learning` | Differentiable coupling inference from observation models with convergence and finite-difference agreement certificates. |
| `scpn_quantum_control.phase.gradient_descent` | Generic parameter-shift gradient descent with line-search traces and convergence certificates. |
| `scpn_quantum_control.phase.qnn_training` | Bounded data-reuploading phase-QNN classifier training with multi-frequency parameter-shift descent, prediction evidence, and accuracy certificates. |
| `scpn_quantum_control.phase.qnn_convergence` | Deterministic bounded-QNN convergence evidence with loss-drop thresholds, accuracy thresholds, parameter-shift evaluation accounting, multi-seed initial-condition envelopes, and unsuitable-scenario records. |
| `scpn_quantum_control.phase.qnn_finite_shot` | Seeded finite-shot simulator evidence for bounded-QNN gradients and noisy-gradient convergence with replay seeds, shot counts, uncertainty radii, and non-hardware claim boundaries. |
| `scpn_quantum_control.phase.qnn_framework_bridge_matrix` | Fail-closed support matrix for bounded phase-QNN framework bridges, separating implemented JAX/PyTorch/TensorFlow routes from arbitrary simulator autodiff and hardware-gradient gaps. |
| `scpn_quantum_control.phase.qnn_framework_agreement` | Caller-supplied QNN framework-gradient agreement checks for JAX/PyTorch/TensorFlow/PennyLane/Qiskit-style references, complemented by a same-circuit conformance table that keeps finite-shot, provider-plan, and hardware-execution rows blocked until their required artefacts exist. |
| `scpn_quantum_control.phase.qnn_loss_landscape` | Deterministic bounded-QNN loss-landscape grids with parameter-shift gradient norms, sampled minima, loss spans, and non-hardware claim boundaries. |
| `scpn_quantum_control.phase.domain_benchmark_datasets` | Synthetic exact-answer differentiable domain datasets for bounded phase-QNN and two-oscillator Kuramoto-XY cases, plus published public-domain QPU artefact references for EEG, plasma, power-grid, and FEP Kuramoto conversion checks. |
| `scpn_quantum_control.phase.natural_gradient` | Metric-aware parameter-shift descent with damped solves, metric validation, line-search traces, and convergence certificates. |
| `scpn_quantum_control.phase.optimizer_audit` | Multi-start optimizer comparison evidence for parameter-shift descent and natural-gradient descent. |
| `scpn_quantum_control.phase.objectives` | Composable differentiable phase-control objectives with term-wise gradients and fail-closed parameter-shift compatibility. |
| `scpn_quantum_control.phase.objective_audit` | Finite-difference agreement, compatibility-gate, and training evidence for composed objectives. |
| `scpn_quantum_control.phase.objective_planner` | Fail-closed execution planning for pure parameter-shift, hybrid term-gradient, hardware, and unsupported composed-objective routes. |
| `scpn_quantum_control.qsnn.training` | QSNN parameter-shift gradients, full-batch descent, and training convergence evidence. |
| `scpn_quantum_control.phase.gradient_backend` | Backend gradient capability declarations, fail-closed planner, shot policy, and hardware-safe defaults. |
| `scpn_quantum_control.phase.differentiable_readiness` | Unified reviewer-facing readiness ledger over the focused differentiable support, transform, provider, QNode, hardware-policy, and provider hardware-preparation audits. |
| `scpn_quantum_control.phase.provider_gradient` | Provider callback parameter-shift execution plus policy-bound hardware-gradient preparation records that never submit QPU jobs. Finite-shot callback samples must carry sample seed, shot-batch, and source-class provenance before variance and shot counts can clear; the executor stamps each plus/minus record with parameter index, shift index, direction, shift, coefficient, and shifted-parameter digest metadata. |
| `scpn_quantum_control.phase.provider_hardware_gradient_audit` | Executable audit suite for approved and blocked provider hardware-gradient preparation routes with zero hardware execution and zero produced hardware gradients. |
| `scpn_quantum_control.phase.hardware_gradient_policy` | Hardware-gradient preparation policy with provider/backend allowlists, shot/evaluation budget accounting, required evidence IDs, dry-run approval, and live-ticket gating. |
| `scpn_quantum_control.phase.hardware_gradient_campaign` | No-submit campaign specs for XY parameter-shift VQE and seeded SPSA hardware-gradient validation, including named Heron r2 backend allowlists, evidence IDs, shot/evaluation budgets, raw-count replay schemas, statevector-reference requirements, and policy-evaluated dry-run plans. |
| `scpn_quantum_control.phase.hardware_gradient_publication` | Publication package scaffold for the planned XY hardware-gradient paper, covering preregistration, methods sections, raw artefact map, draft claim-ledger rows, same-circuit benchmark placeholders, and no-submit claim boundaries. |
| `scpn_quantum_control.phase.gradient_support_matrix` | Executable support planning for gates, observables, backends, transforms, and ML/provider adapters. |
| `scpn_quantum_control.phase.transform_nesting` | Fail-closed transform-nesting planner for local, tape, ML-adapter, vectorized, and hardware gradient routes. |
| `scpn_quantum_control.phase.provider_gradient_audit` | Executable provider-gradient readiness audit for deterministic, finite-shot, multi-frequency, hardware-blocked, unknown-backend, and malformed-sample routes. |
| `scpn_quantum_control.phase.provider_hardware_safety_audit` | Aggregate differentiable provider/hardware safety gate over provider-gradient readiness, provider hardware-gradient preparation, provider QNode transforms, QNode tape records, and hardware-gradient campaign readiness. It verifies zero hardware execution and zero hardware-gradient production, then keeps promotion blocked until a freshness-bounded `DifferentiableProviderHardwareEvidenceChain` binds live-ticket, provider/backend/job/circuit metadata, allowlist, shot budget, raw-count replay, calibration snapshot, statevector comparison, and isolated benchmark artefacts into one validated chain. |
| `scpn_quantum_control.phase.gradient_tape` | Context-managed recording of supported deterministic and finite-shot quantum-gradient evaluations. |
| `scpn_quantum_control.phase.qnode_tape` | QNode-style differentiable tape records for supported phase objectives, seeded finite-shot replay with serialized plus/minus shifted-sample provenance, and provider-boundary routes that fail closed before hardware submission. |
| `scpn_quantum_control.phase.qnode_circuit` | Registered local Phase-QNode statevector and density-matrix circuit family with supported gates, bounded single-qubit Kraus noise channels, controlled-gate decomposition helpers, arbitrary-depth registered circuit builders with deterministic depth/resource profiles, multi-qubit template constructors, dense Hermitian observables, Pauli observables, Pauli covariance observables, sparse Pauli Hamiltonians, sparse Ising-chain Hamiltonian construction, gate-aware parameter-shift evaluation planning, parameter-shift gradients for pure-state routes, exact computational-basis classical Fisher metrics with optional local finite-shot uncertainty and raw-count replay evidence, pure-state QFI/Fubini-Study metrics, natural-gradient metric providers, and strict route support reports for value, density, gradient, metric, and Fisher paths. |
| `scpn_quantum_control.phase.qnode_framework_parity` | Bounded real-framework parity suite for SCPN, JAX, PyTorch, TensorFlow, and PennyLane with dependency-sparse classifications. |
| `scpn_quantum_control.phase.qnode_affinity_benchmark` | Affinity-labelled local benchmark metadata harness for registered Phase-QNode execution, including raw timing rows, host isolation context, and fail-closed raw-artifact attachment validation. |
| `scpn_quantum_control.phase.qnode_transforms` | Executable scalar local QNode transform evidence for `grad`, `value_and_grad`, `hessian`, `hessian_vector_product`, `jvp`, `vjp`, `jacfwd`, and `jacrev`, with real-only complex/W boundaries and fail-closed vectorized/provider/framework-native boundaries. |
| `scpn_quantum_control.phase.qnode_vector_transforms` | Executable deterministic native vector-output QNode `jvp`, `vjp`, vector Hessian tensor, Jacobian evidence for `jacfwd`/`jacrev`, plus host-side manual `vmap(grad)` over scalar local parameter-shift objectives, with real-only complex/W boundaries and fail-closed finite-shot, hardware, provider, and framework-native vectorization boundaries. |
| `scpn_quantum_control.phase.qnode_provider_transforms` | Provider-callback QNode transform evidence for scalar `grad`, `value_and_grad`, `jvp`, `vjp`, scalar `jacfwd`/`jacrev`, and manual `vmap(grad)` with shifted-sample records, finite-shot uncertainty propagation, and fail-closed hardware policy. |
| `scpn_quantum_control.phase.qiskit_bridge` | Qiskit shifted-circuit generation, deterministic local Statevector parameter-shift gradients, finite-shot surrogate uncertainty, no-submit Runtime primitive metadata, ticketed Runtime QPU execution metadata, a freshness-bounded provider evidence chain for matching Runtime QPU/raw-count/calibration artefacts, captured provider-gradient workflow artefacts for parameter-shift, finite-difference, LCU, SPSA, QGT, and QFI methods, and a maturity audit that pairs raw-count replay plus live calibration/statevector comparison evidence to the same Runtime QPU provider/backend/job/circuit/primitive/observable/parameter/live-ticket chain while rejecting stale bundle evidence, duplicate workflow artefact IDs, and isolated benchmark promotion. |
| `scpn_quantum_control.differentiable_framework_overlay` | CPU-only overlay manifest, installer, verifier, and CLI for reproducible JAX, PyTorch, TensorFlow, and PennyLane parity environments. |
| `scpn_quantum_control.benchmarks.differentiable_external_comparison` | External comparison rows and JSON artefact writing for JAX `value_and_grad`/`vmap` support, PyTorch `torch.func`, TensorFlow `GradientTape`, PennyLane QNodes, optional LLVM/Enzyme runner AD, and optional Catalyst qjit/MLIR/QIR runner workflows with strict JSON, timeout, toolchain, correctness gates, dependency-version metadata, and explicit hard-gap rows for unsupported batching, transform, dtype, and hardware-device routes. |
| `scpn_quantum_control.benchmarks.differentiable_evidence` | CI benchmark evidence writer with runner metadata, CPU affinity, host-load, governor/frequency, heavy-job, explicit accelerator metadata, silent CPU-fallback detection, classification, and artefact-ID fields. |
| `scpn_quantum_control.benchmarks.differentiable_isolated_benchmark_plan` | Reserved-host isolated benchmark batch planner over the current non-isolated differentiable benchmark/evidence artefacts. It records rerun commands, self-hosted runner labels, expected outputs, host blockers, and claim boundaries without executing benchmarks or promoting performance claims. |
| `scpn_quantum_control.differentiable_claim_ledger` | Claim-ledger parser, Markdown renderer, public claim-table renderer, public-language guard, and support-surface alignment audit that prevent promoted claims without artefact IDs or drift between implementation, tests, docs, and generated capability inventory. |
| `scpn_quantum_control.differentiable_sota_scorecard` | Deterministic category scorecard and release-blocking public-language audit for JAX, PyTorch, PennyLane, Qiskit Runtime, Catalyst, Enzyme, Rust Program AD, provider/hardware gradients, benchmark promotion, docs/API maintainability, and adoption/licensing readiness. Rows stay `behind_baseline` until promoted claim-ledger rows, external comparison evidence, and isolated benchmark artefacts exist; unbounded public SOTA, exceedance, production-performance, or promotion-ready wording fails unless it cites ready scorecard categories with promoted ledger evidence. |
| `scpn_quantum_control.differentiable_competitive_baselines` | Competitive-baseline freshness gate for the SOTA scorecard. It records official upstream source streams for JAX, PyTorch, TensorFlow, PennyLane, Qiskit Algorithms, Catalyst, Enzyme/MLIR, Julia AD, and emerging AD systems; rejects stale rows after the configured freshness window; and combines fresh-baseline validation with the public-language audit so promotion wording cannot outrun current external-source evidence. |
| `scpn_quantum_control.differentiable_rust_python_inventory` | Deterministic rustification inventory that classifies differentiable Python, Rust, provider, hardware, compiler, metadata, and deprecation surfaces before broad Rust migration. Rows record owner modules, public APIs, tests, docs, benchmark status, mypy targets, docstring status, Rust parity, polyglot status, and blockers without promoting Rust, LLVM/JIT, provider, hardware, GPU, or isolated benchmark claims. |
| `scpn_quantum_control.differentiable_architecture_map` | Deterministic architecture and Rustification routing map that connects the Rust/Python inventory to SOTA scorecard categories across public API, QNode bridge, Program AD, compiler/native execution, provider/hardware, and benchmark/claim-governance layers. It validates inventory references, SOTA categories, evidence paths, and blocker state before broad Rust migration without promoting Rust, LLVM/JIT, provider, hardware, GPU, performance, or isolated benchmark claims. |
| `scpn_quantum_control.differentiable_dependency_environment_map` | Deterministic dependency and environment evidence map over runtime, development, CI Python matrix, CPU framework overlay, and Enzyme runner lock profiles. It reuses the external-validation environment-lock checksums and keeps the Enzyme runner as a hard gap without promoting framework parity, Enzyme parity, provider execution, hardware execution, GPU execution, performance, or isolated benchmark claims. |
| `scpn_quantum_control.differentiable_external_validation` | Exact external-validation environment lock manifest, reproducible artefact-bundle manifest, checksum validators, and reviewer Markdown renderers for runtime, development, CI, framework-overlay, Enzyme-runner, and committed evidence artefacts. |
| `scpn_quantum_control.phase.jax_bridge` | Optional JAX host-callback adapter for supported phase parameter-shift value-and-gradient calls plus bounded native/custom-VJP JAX phase-QNN evidence, audited no-host-callback JIT/VMAP/PMAP/PyTree boundaries for that narrow model, registered deterministic local Phase-QNode flat/PyTree native transform lowering evidence, AOT/export serialization diagnostics for registered value routes, and a JarvisLabs/cloud validation batch plan for locally blocked JAX GPU and multi-device routes. |
| `scpn_quantum_control.phase.pennylane_bridge` | Optional PennyLane gradient-agreement checker for caller-supplied PennyLane/QNode gradient functions. |
| `scpn_quantum_control.phase.torch_bridge` | Optional PyTorch bridge for supported phase parameter-shift value-and-gradient calls, tensor-ready bounded phase-QNN analytic gradient evidence, bounded custom `torch.autograd.Function`, bounded `torch.func.grad`/`vmap`/`jacrev`, bounded `torch.compile`, bounded `nn.Module`/layer wrapper compatibility, deterministic registered local Phase-QNode statevector lowering through native PyTorch autograd, registered local Phase-QNode `torch.func.grad`/`jacrev`/`vmap` transform evidence, broad PyTorch module/transform/compiler/device maturity routing, live CPU-overlay external-comparison artefact validation, and a fail-closed registered Phase-QNode Torch-lowering matrix checked against parameter-shift references and promotion blockers. |
| `scpn_quantum_control.phase.torch_autograd_function` | Bounded PyTorch custom-autograd utilities for the phase-QNN classifier loss. `torch_autograd_function_qnn_loss(...)` returns a scalar tensor whose `Tensor.backward()` gradient is checked against the SCPN parameter-shift reference, while `run_torch_autograd_function_audit(...)` records backward parity, `torch.optim.SGD` integration, and explicit higher-order, CUDA, provider/hardware, arbitrary-simulator, isolated-benchmark, and performance blockers. |
| `scpn_quantum_control.phase.torch_module_state` | Bounded PyTorch module-state utilities for the phase-QNN `nn.Module` route. `validate_torch_bounded_qnn_state_dict(...)` checks candidate `state_dict` keys, shapes, and dtypes without loading them, while `run_torch_module_state_audit(...)` verifies strict module `load_state_dict(strict=True)` replay plus Adam optimizer `state_dict` replay on local CPU-compatible tensors. CUDA device transfer, cross-runtime checkpoint portability, provider, hardware, isolated benchmark, and performance promotion remain blocked. |
| `scpn_quantum_control.phase.torch_device_state` | Bounded PyTorch device-state utilities for the same phase-QNN `nn.Module` route. `run_torch_module_device_state_audit(...)` verifies `module.to("cpu")` state replay through strict `state_dict` loading and classifies `module.to("cuda")` replay only after a real CUDA smoke succeeds. Incompatible CUDA, cross-runtime checkpoint portability, provider, hardware, isolated benchmark, and performance promotion remain blocked. |
| `scpn_quantum_control.phase.torch_checkpoint` | Bounded PyTorch checkpoint utilities for the same phase-QNN `nn.Module` route. `run_torch_module_checkpoint_audit(...)` writes a real `torch.save` checkpoint, reloads it with `torch.load(..., map_location="cpu", weights_only=True)`, and replays strict module plus Adam optimizer state. Cross-runtime checkpoint portability, CUDA checkpoint replay, provider, hardware, isolated benchmark, and performance promotion remain blocked. |
| `scpn_quantum_control.phase.torch_checkpoint_matrix` | Bounded PyTorch long-lived checkpoint matrix utilities for the same phase-QNN `nn.Module` route. `run_torch_long_lived_checkpoint_matrix(...)` wraps the bounded checkpoint audit, records the versioned checkpoint schema, tensor metadata manifest, runtime fingerprint, and repeated local CPU weights-only loads. Cross-runtime checkpoint replay, CUDA checkpoint replay, external long-lived checkpoint-corpus promotion, provider, hardware, isolated benchmark, and performance promotion remain blocked. |
| `scpn_quantum_control.phase.torch_export` | Bounded PyTorch `torch.export` persistence utilities for the same phase-QNN `nn.Module` route. `run_torch_module_export_audit(...)` exports the module with `torch.export.export(...)`, persists it with `torch.export.save(...)`, reloads it with `torch.export.load(...)`, and replays the local CPU value route through `ExportedProgram.module()`. Gradient export for this `torch.export` route, dynamic-shape export promotion, cross-runtime deployment, CUDA/provider/hardware execution, isolated benchmark, and performance promotion remain blocked. |
| `scpn_quantum_control.phase.torch_export_shape_matrix` | Bounded PyTorch static-shape export matrix utilities for the same phase-QNN `nn.Module` route. `run_torch_export_shape_matrix(...)` exports separate deterministic one- and two-parameter static feature shapes, records per-shape `ExportedProgram` artifact metadata, and keeps dynamic-shape constraints, dynamic-shape replay, cross-runtime deployment, CUDA/provider/hardware execution, isolated benchmark, and performance promotion blocked. |
| `scpn_quantum_control.phase.torch_dynamic_shape_export` | Bounded PyTorch dynamic-batch export utilities for the same phase-QNN `nn.Module` route. `run_torch_dynamic_shape_export_audit(...)` exports one input-driven `ExportedProgram` with symbolic batch constraints, persists it with `torch.export.save(...)`, reloads it with `torch.export.load(...)`, and replays multiple concrete batch sizes locally. Dynamic feature width, dynamic-shape AOTAutograd export, cross-runtime deployment, CUDA/provider/hardware execution, isolated benchmark, and performance promotion remain blocked. |
| `scpn_quantum_control.phase.torch_aot_autograd_export` | Bounded PyTorch AOTAutograd FX persistence utilities for the same phase-QNN loss route. `run_torch_aot_autograd_export_audit(...)` captures forward and backward FX `GraphModule` objects with `torch._functorch.aot_autograd`, saves and reloads the self-produced local artifacts, and replays the loaded backward graph against the SCPN parameter-shift gradient reference. The artifact is a local PyTorch FX pickle, not a stable cross-runtime export format; cross-runtime execution, CUDA replay, dynamic-shape AOTAutograd export, isolated benchmark, and performance promotion remain blocked. |
| `scpn_quantum_control.phase.torch_training_loop_matrix` | Bounded PyTorch training-loop matrix utilities for the same phase-QNN `nn.Module` route. `run_torch_training_loop_matrix(...)` expands the single training-loop audit into deterministic one- and two-parameter scenarios, records loss descent, parameter-update norm, compile-mode coverage, and gradient parity, and keeps CUDA, provider/hardware, arbitrary-architecture, isolated benchmark, and performance promotion blocked. |
| `scpn_quantum_control.phase.tensorflow_bridge` / `scpn_quantum_control.phase.tensorflow_maintenance` | Optional TensorFlow tensor bridge for supported phase parameter-shift value-and-gradient calls plus tensor-ready bounded phase-QNN analytic gradient evidence checked against parameter-shift references. TensorFlow is explicitly maintained as compatibility-only evidence for bounded routes; broad Graph/XLA parity, arbitrary Phase-QNode lowering, provider callbacks, hardware gradients, and performance promotion stay blocked. |
| `scpn_quantum_control.compiler.mlir` | Compiler/program AD lowering, native executable kernel helpers, Phase-QNode MLIR-runtime execution adapters, support-profile reports, and the Enzyme/MLIR maturity audit that records executable SCPN MLIR-runtime correctness, native LLVM/JIT support metadata, local toolchain versions, typed native Enzyme execution evidence, typed MLIR/LLVM correctness evidence, and hard gaps until successful native Enzyme plus isolated benchmark artefacts exist. |

## Common objects

| Object family | Examples | Use |
|---|---|---|
| Unified API evidence | `UnifiedDifferentiableAPIResult`, `DifferentiabilityDiagnosticReport`, `DifferentiableDashboardStatus`, `DifferentiableDashboardCapabilityRow`, `DifferentiableBenchmarkReport`, `TransformAlgebraAudit`, `TransformAlgebraCase`, `differentiable_api`, `differentiable_value`, `differentiable_gradient`, `differentiable_jacobian`, `differentiable_hessian`, `differentiable_support_report`, `explain_differentiability`, `differentiable_compile_report`, `differentiable_frontend_report`, `build_differentiable_benchmark_report`, `differentiable_benchmark_report`, `differentiable_transform_algebra_report`, `differentiable_qfi_fss_report`, `differentiable_dashboard_status`, `differentiable_sota_scorecard_report`, `differentiable_competitive_baseline_refresh_report`, `differentiable_rust_python_inventory_report`, `differentiable_architecture_map_report`, `differentiable_dependency_environment_map_report`, `differentiable_isolated_benchmark_plan_report` | Use one JSON-ready envelope across scalar values, gradients, Jacobians, Hessians, fail-closed support decisions, differentiability diagnostics, compiler planning, static bytecode/source frontend preflight, local conformance evidence, transform-algebra metamorphic evidence, bounded QFI/FSS finite-size-scaling evidence, SOTA scorecard governance, competitive-baseline freshness evidence, Rust/Python rustification inventory governance, architecture/Rustification routing governance, dependency/environment lock governance, isolated benchmark batch planning, and GUI/audit-dashboard status. The local benchmark report builder is isolated in `scpn_quantum_control.differentiable_benchmark_report` and the facade wraps it without changing the public `benchmark_report` payload or upgrading it beyond deterministic non-isolated conformance evidence. The `transform_algebra_report` operation executes local deterministic metamorphic checks for transform composition, including phase-periodic parameter-shift wraparound, and keeps unsupported custom-rule, structured-container, complex-valued objective, and nondifferentiable boundaries blocked instead of promoting finite-difference diagnostics. The `qfi_fss_report` operation returns local dense exact finite-size gap diagnostics with BKT and inverse-size residual evidence while blocking hardware, isolated-performance, and thermodynamic-limit promotion. Dashboard rows preserve `planned`, `metadata_only`, `diagnostic`, `conformance_backed`, `executable`, `blocked`, and `unsupported` labels without upgrading bounded Program AD IR round-trip parsing or static bytecode/source preflight into executable compiler lowering or promoting blocked compiler, Rust, LLVM, provider, or hardware paths. The `sota_scorecard` operation returns `promotion_ready=False` while any category lacks promoted claim-ledger rows and isolated benchmark evidence; the `competitive_baseline_refresh` operation returns a non-promotional upstream-source freshness snapshot and never promotes a scorecard row by itself; the `rust_python_inventory` operation returns `rustification_ready=False` while any surface lacks matching Rust parity, polyglot, benchmark, and claim-ledger evidence; the `architecture_rustification_map` operation returns `rustification_ready=False` while any architecture layer, inventory row, or scorecard category remains blocked; the `dependency_environment_map` operation returns `environment_ready=False` while any dependency profile remains a hard gap or the environment lock is not promotable; the `isolated_benchmark_plan` operation returns `promotion_ready=False` until every row is backed by validated `isolated_affinity` artefacts and has no host or source-classification blockers. The `program_ad_bytecode_source_frontend`, `program_ad_alias_effects`, `program_ad_ir_roundtrip_conformance`, `program_ad_control_phi_metadata`, `program_ad_registry_dispatch_coverage`, `program_ad_reverse_adjoint_replay`, `program_ad_elementwise_primitives`, `program_ad_array_indexing`, `program_ad_linalg_primitives`, `program_ad_structured_primitives`, `program_ad_cumulative_primitives`, `program_ad_assembly_primitives`, `program_ad_reduction_primitives`, `program_ad_shape_primitives`, `program_ad_broadcast_primitives`, and `program_ad_selection_primitives` rows expose local static or conformance evidence through `WholeProgramCompilerFrontendReport`, `WholeProgramBytecodeBasicBlock`, `WholeProgramSourceRegion`, `WholeProgramSourceBytecodeLineMap`, `WholeProgramSymbolScopeEntry`, `WholeProgramUnsupportedSemanticDiagnostic`, `loop_carried_state_alias_metadata_contracts`, `program_ad_ir_roundtrip_contracts`, `program_ad_control_phi_metadata_contracts`, `program_ad_registry_dispatch_contracts`, `program_adjoint_replay_provenance_contracts`, `program_ad_static_alias_lattice_contracts`, `elementwise_boundary_contracts`, `indexing_static_gather_contracts`, `linalg_primitive_contracts`, `structured_numeric_primitive_contracts`, `cumulative_primitive_contracts`, `assembly_primitive_contracts`, `reduction_primitive_contracts`, `shape_primitive_contracts`, `broadcast_primitive_contracts`, and `selection_piecewise_contracts` without promoting hardware, performance, Rust, LLVM, or JIT claims. |
| Primitive identity and rules | `PrimitiveIdentity`, `PrimitiveContract`, `CustomDerivativeRule`, `CustomDerivativeRegistry`, `ProgramADRegistryDispatchCoverageReport`, `ProgramADRegistryDispatchCoverageRow`, `program_ad_registry_dispatch_coverage_report`, `RustProgramADRegistryMetadataMirrorResult`, `mirror_program_ad_registry_metadata_with_rust`, `ProgramADLinalgConditioningDiagnostic`, `diagnose_program_ad_linalg_conditioning` | Bind derivative, batching, lowering metadata, shape, dtype, nondifferentiability, and conditioning-diagnostic rules to supported primitives. Registry contracts now live in `scpn_quantum_control.program_ad_registry`; `scpn_quantum_control.differentiable` re-exports the same objects for compatibility. `program_ad_registry_dispatch_coverage_report()` returns a JSON-ready registry-dispatched coverage report over 118 declared Program AD primitives across 12 families and is backed by the `program_ad_registry_dispatch_contracts` conformance row; runtime dispatch now enforces the same explicit `nondifferentiable_boundary` plus `nondifferentiable_boundary_policy="fail_closed"` metadata before executing traced Program AD primitive paths. `mirror_program_ad_registry_metadata_with_rust()` sends that report through the optional Rust metadata mirror, validates family/facet counts, and reports conservative primitive-name overlap with existing bounded Rust scalar/static-linalg plus elementwise/static-structural replay only. This does not claim executable Rust registry coverage, broad array adjoints, LLVM, JIT, provider, hardware, or performance evidence. The Program AD elementwise registry includes fail-closed derivative-losing `sign` and `heaviside` contracts, plus smooth and boundary-sensitive arithmetic contracts; local boundary-sensitive conformance for `abs`, `absolute`, `log`, `sqrt`, `reciprocal`, `log1p`, `arcsin`, and `arccos` is exposed through `elementwise_boundary_contracts`. Product, interpolation, signal, and stencil registries cover `inner`, `outer`, `matmul`, `tensordot`, `einsum`, `interp`, `convolve`, `correlate`, and `gradient` with static-shape/grid/mode/spacing metadata and fail-closed unsupported boundaries. The Program AD cumulative registry covers bounded `cumsum`, `cumprod`, and `diff` contracts with static-shape/axis metadata and blocked Rust/LLVM executable lowering. The Program AD shape registry covers reshape, ravel, transpose, expand/squeeze, axis movement, repeat/tile, rank promotion, roll, rot90, flip, flipud, and fliplr contracts; local conformance is exposed through `shape_primitive_contracts`. The Program AD selection registry includes `where`, `clip`, strict-total-order `sort`, static selection-fold `select`, `piecewise`, `choose`, `compress`, and `extract`, plus fail-closed integer-output `argmax`, `argmin`, and `argsort` contracts; local selection conformance is exposed through `selection_piecewise_contracts`, with dynamic masks, dynamic selectors, ties, and integer-output selector differentiation still blocked. The Program AD reduction registry includes `sum`, `prod`, `mean`, `var`, `std`, `trapezoid`, unique-selector `max`/`min`, `median`, scalar-`q` `quantile`, and scalar-`q` `percentile`, with positive-denominator/zero-variance and strict-order fail-closed boundaries for sensitive reductions; local conformance is exposed through `reduction_primitive_contracts`. The Program AD assembly registry includes `zeros_like`, `ones_like`, `full_like`, `hstack`, `vstack`, `column_stack`, `dstack`, `broadcast_to`, and `broadcast_arrays` contracts for derivative-preserving constant arrays, static-shape stack convenience assembly, and bounded broadcast expansion; local broadcast conformance is exposed through `broadcast_primitive_contracts`, with dynamic output shapes and subclass propagation still blocked. The linalg diagnostic covers `norm`, `det`, `inv`, `solve`, `matrix_power`, `eig`, `eigh`, `eigvals`, `eigvalsh`, `svd`, and `pinv`; it reports zero-norm, rank-threshold, repeated-spectrum, and ill-conditioned boundaries without changing AD execution or promoting benchmark evidence. |
| Program AD alias/effect metadata | `ProgramADPhiNode`, `ProgramADAliasSet`, `ProgramADAliasEffectAnalysis`, `ProgramADStaticAliasLatticeReport`, `ProgramADUnknownAliasEdge`, `ProgramADControlPathAliasProvenance`, `ProgramADViewAliasProvenance`, `ProgramADListAliasProvenance`, `ProgramADLoopCarriedStateProvenance`, `ProgramADRebindingAliasProvenance`, `parse_program_ad_effect_ir`, `analyze_program_ad_alias_effects`, `program_ad_static_alias_lattice_report` | Parse the bounded `program_ad_effect_ir.v1` JSON evidence emitted by whole-program AD, including metadata-only phi records for runtime and source-level control joins, then summarize deterministic alias components, mutation-effect ordering, mutation-version edges, source aliases, bounded local scalar rebinding, bounded expression-rebinding aliases, local object-attribute aliases, branch-local control-path alias blockers, local list-alias rebinding/mutation metadata, bounded loop-carried scalar state metadata, supported executed array-view aliases, and static rank-1 slice-mutation source indices from emitted `ProgramADEffectIR`. View aliases include reshape, ravel, basic slicing, take, transpose, squeeze, expand_dims, atleast rank promotion, swapaxes, moveaxis, repeat source reuse, tile source reuse, roll, rot90, flip, flipud, and fliplr; parseable source-to-view edges are retained as `ProgramADViewAliasProvenance` rows with operation, view id, output index, source, target, and version metadata. Parseable local list construction, local-name rebinding, and indexed mutation-source edges are retained as `ProgramADListAliasProvenance` rows with list name, target kind, source, target, and version metadata. Parseable loop-carried scalar state edges are retained as `ProgramADLoopCarriedStateProvenance` rows with state name, entry/backedge labels, source, target, and version metadata. Parseable local-name and expression rebinding edges are retained as `ProgramADRebindingAliasProvenance` rows with binding kind, source local name or expression line/label, target name, source, target, and version metadata. Parseable branch-local control-path edges are retained as `ProgramADControlPathAliasProvenance` rows with branch line, branch arm, target label, source, target, and version metadata. The `program_ad_ir_roundtrip_conformance` dashboard row is backed by `program_ad_ir_roundtrip_contracts` and checks stable parser reconstruction of emitted SSA/effect/control/phi metadata. The `program_ad_control_phi_metadata` dashboard row is backed by `program_ad_control_phi_metadata_contracts` and checks runtime/source control-region plus `ProgramADPhiNode` provenance against analytic and adjoint references. The parser and analysis helper fail closed on malformed or unknown metadata. `program_ad_static_alias_lattice_report(...)` builds a JSON-ready static alias-lattice readiness report over emitted IR components and records mutation effects, unknown alias-edge provenance, malformed control-path alias edges, malformed view-alias edges, malformed list-alias edges, malformed loop-carried state edges, malformed rebinding-alias edges, branch-local control-path aliases, frontend unsupported Python semantics with `WholeProgramUnsupportedSemanticDiagnostic` source-region and bytecode-offset provenance, captured/global object-attribute roots/details as static object-model blockers, and non-executed phi inputs as explicit blockers; phi records remain control-join provenance only, not mutation adjoints, malformed control/view/list/loop/rebinding-alias promotion, unknown dynamic alias promotion, arbitrary dynamic-Python frontend lowering, non-executed branch adjoints, captured/global object-attribute alias sets, or a compiler frontend. |
| Program AD Python semantics | `WholeProgramSemanticsReport`, `WholeProgramCompilerFrontendReport`, `WholeProgramBytecodeBasicBlock`, `WholeProgramSourceRegion`, `WholeProgramSourceBytecodeLineMap`, `WholeProgramSymbolScopeEntry`, `WholeProgramUnsupportedSemanticDiagnostic`, `compile_whole_program_frontend`, `whole_program_value_and_grad`, `differentiable_frontend_report`, `differentiable_dashboard_status` | Report accepted closure/default/keyword calling semantics, generator expressions, and plain list comprehensions without dynamic filters. Static bytecode/source frontend ownership lives in `scpn_quantum_control.whole_program_frontend`; runtime whole-program result records live in `scpn_quantum_control.whole_program_ad_result`; `scpn_quantum_control.differentiable` and the package root re-export the same public objects for compatibility. `compile_whole_program_frontend()` inspects bytecode instructions, bytecode basic blocks, source AST features, source regions, source-bytecode line maps, symbol-scope entries, source start/end line bounds, deterministic digests, unsupported-semantics diagnostics, and hard gaps without executing the objective; line-map and unsupported-semantics diagnostic rows use source-relative `line_number` for AST/region joins and preserve CPython/file provenance as `absolute_line_number`. Unsupported semantics become explicit hard gaps with source region IDs and bytecode offsets when available. Filtered comprehensions, set/dict comprehensions, generator functions, context managers, exception control flow, recursion, decorated objectives, and captured object/dataclass attributes fail closed before execution. |
| Forward and reverse AD results | `GradientResult`, `JacobianResult`, `HessianResult`, `JVPResult`, `HVPResult`, `ProgramADAdjointResult`, `ProgramADAdjointStep` | Return structured derivative outputs and diagnostics. Program AD adjoint generation results include generated reverse-adjoint steps, finite local pullback scales, cotangent-flow rows, reverse effect-order rows, replayed node, effect, runtime control/phi row binding, blocked non-executed phi inputs, and IR-format provenance so reviewers can bind reverse-mode evidence to the captured `ProgramADEffectIR`; the dashboard exposes this through `program_ad_reverse_adjoint_replay` backed by `program_adjoint_replay_provenance_contracts`. This remains supported executed-scalar IR generation metadata, not full arbitrary Python reverse-mode compiler AD, non-executed branch adjoints, Rust/LLVM executable lowering, hardware, or performance evidence. |
| Optimisation helpers | `DifferentiableOptimizer`, `NaturalGradientOptimizer`, `LevenbergMarquardtOptimizer` | Drive supported differentiable objectives. |
| Compiler-backed kernels | `compile_*_ad_to_native_llvm_jit`, `compile_whole_program_ad_trace_to_native_llvm_jit`, `compile_phase_qnode_circuit_to_mlir_runtime`, `native_whole_program_ad_linalg_support` | Execute bounded native AD kernels where support reports allow it, including verified static dense determinant lowering through `19x19`, bounded 2x2 `matrix_power(..., 2)` and 2x2 `multi_dot` native lowering, fail-closed `20x20+` reports, and a verified SCPN MLIR-runtime adapter for registered local Phase-QNode value/gradient execution with shape/type checks and blocked interpreter-fallback success claims. Wider concrete static linalg rules such as 3x3 `matrix_power(..., 3)` and rectangular `multi_dot` remain MLIR-runtime-only; native LLVM/JIT and Rust promotion is blocked until independent executable kernel verification exists. |
| Backend and shot planning | `QuantumGradientPlan`, `QuantumGradientBackendCapability`, `ShotAllocationResult`, `GradientFailurePolicy`, `StochasticGradientConfidenceInterval`, `FiniteShotSampleProvenance`, `ParameterShiftSampleRecord`, `gradient_confidence_interval`, `SPSAObjectiveSample`, `SPSAProbeRecord`, `SPSAGradientResult`, `spsa_gradient_estimate`, `ScoreFunctionSampleRecord`, `ScoreFunctionGradientResult`, `score_function_gradient_estimate`, support-profile records | Select supported local gradient methods, propagate finite-shot uncertainty with confidence intervals, source-classed finite-shot sample provenance, shifted-sample records that reconstruct their gradient and covariance contributions, standard-error/covariance consistency checks, fail-closed policy metadata, and explicit no-hardware claim boundaries, run seeded local SPSA probes over caller-supplied objectives, estimate materialised likelihood-ratio score-function gradients, and fail closed for unsafe hardware routes. |
| Hardware-gradient campaign readiness | `HardwareGradientCampaignSpec`, `HardwareGradientReplaySchema`, `HardwareGradientCampaignPlan`, `HardwareGradientCampaignSuite`, `default_hardware_gradient_campaign_specs`, `plan_hardware_gradient_campaign`, `run_hardware_gradient_campaign_readiness_suite` | Prepare no-submit XY hardware-gradient validation campaigns for parameter-shift VQE and seeded SPSA routes with backend allowlists, live-ticket gates, evidence IDs, shot budgets, calibration snapshot requirements, raw-count replay schemas, statevector references, and policy decisions that preserve `hardware_execution == False` until live artefacts exist. |
| Provider/hardware safety audit | `DifferentiableProviderHardwareEvidenceChain`, `DifferentiableProviderHardwareSafetySurface`, `DifferentiableProviderHardwareSafetyAuditResult`, `run_differentiable_provider_hardware_safety_audit` | Aggregate differentiable provider-gradient, provider-preparation, QNode transform, QNode tape, and hardware-gradient campaign surfaces into one promotion gate. Legacy detached artifact IDs are still serialized for compatibility, but promotion readiness now requires a validated UTC-fresh evidence chain with matching live-ticket, provider/backend/job/circuit, allowlist, shot-budget, raw-count replay digest, calibration digest, statevector comparison digest, and isolated benchmark artifact metadata. |
| Hardware-gradient publication package | `HardwareGradientPublicationPackage`, `HardwareGradientPreregistration`, `HardwareGradientMethodSection`, `HardwareGradientArtifactMapEntry`, `HardwareGradientClaimLedgerRow`, `HardwareGradientBenchmarkPlaceholder`, `build_hardware_gradient_publication_package` | Produce a JSON-ready and Markdown-ready publication scaffold for the planned XY hardware-gradient paper while keeping claim rows unpromoted and rejecting injected live-result claims in the no-submit package. |
| Gradient support matrix | `GradientSupportCapability`, `GradientSupportPlan`, `GradientSupportMatrixAuditResult`, `gradient_support_capability`, `list_gradient_support_capabilities`, `plan_gradient_support`, `assert_gradient_support`, `run_gradient_support_matrix_audit` | Decide whether a gate, observable, backend, transform, and adapter combination is supported before execution; blocked combinations carry reasons and alternatives. |
| Transform nesting | `GradientTransformNestingPlan`, `GradientTransformNestingAuditResult`, `plan_gradient_transform_nesting`, `assert_gradient_transform_nesting_supported`, `run_gradient_transform_nesting_audit` | Decide whether transform stacks such as `grad`, `value_and_grad`, `hessian`, `grad` of `grad`, tape, native manual `vmap(grad)`, exact custom JVP/VJP rules under eager `vmap`, JVP/VJP over `vmap` of whole-program AD gradients, local Hessian over a whole-program AD scalar objective, JVP/VJP over whole-program AD Hessian transforms, provider-callback routes, adapter bridges, or hardware routes are safe before execution. |
| Gradient audit evidence | `DifferentiableQuantumAuditReport`, `DifferentiableWorkflowAuditSuiteResult`, `FiniteShotGradientAuditResult`, `MLFrameworkGradientAuditSuiteResult`, `ParameterShiftAnalyticAgreement`, `PhaseGradientBenchmarkSuiteResult`, `ProviderGradientReadinessAuditResult`, `run_differentiable_workflow_audit_suite`, `run_finite_shot_gradient_uncertainty_audit`, `run_ml_framework_gradient_audit`, `run_known_phase_gradient_audit`, `run_parameter_shift_audit_suite`, `run_phase_gradient_benchmark_suite`, `run_provider_gradient_readiness_audit` | Bundle finite-difference agreement, finite-shot uncertainty containment, optional ML-framework parity, analytic-gradient agreement, convergence evidence, coupling-learning checks, provider-readiness checks, and multi-case phase-gradient conformance into reviewer-facing reports. |
| Gradient-training evidence | `ParameterShiftTrainingResult`, `ParameterShiftTrainingCertificate`, `ParameterShiftNaturalGradientResult`, `ParameterShiftNaturalGradientCertificate`, `ParameterShiftQNNTrainingResult`, `ParameterShiftQNNPredictionResult`, `ParameterShiftQNNMultiSeedConvergenceSuiteResult`, `ParameterShiftQNNLossLandscapeSuiteResult`, `QNNOptimizerBaselineResult`, `GenericParameterShiftEvaluationPlan`, `plan_generic_parameter_shift_evaluations`, `DifferentiableDomainBenchmarkDatasetSuite`, `DifferentiableDomainBenchmarkValidationSuite`, `OptimizerComparisonSuiteResult`, `OptimizerConvergenceRecord`, `ParamShiftVQEResult`, `ParamShiftConvergenceDiagnostics` | Certify accepted value descent, metric-aware descent, bounded phase-QNN classification, deterministic multi-seed convergence envelopes, bounded loss-landscape scans, named QNN optimizer baseline evidence, exact-answer domain dataset validation, optimizer comparison evidence, opaque-callable 2N fallback planning, line-search behaviour, exact-gap metadata, and parameter-shift evaluation counts. |
| Objective composition | `ComposedPhaseObjective`, `ObjectiveTerm`, `ObjectiveGradientEvaluation`, `ComposedObjectiveTrainingResult`, `ComposedObjectiveGradientAgreement`, `ComposedObjectiveAuditSuiteResult`, `ComposedObjectiveExecutionPlan`, `ComposedObjectivePlannerAuditResult`, `build_phase_control_objective`, `train_composed_phase_objective`, `verify_composed_objective_gradient`, `run_composed_objective_audit_suite`, `plan_composed_objective_execution`, `run_composed_objective_planner_audit` | Combine energy, fidelity, periodic regularization, symmetry, and smooth safety penalties without misclassifying analytic classical penalties as parameter-shift quantum terms. |
| Coupling-learning evidence | `CouplingLearningResult`, `CouplingGradientVerificationResult`, `learn_couplings_from_observations`, `verify_coupling_parameter_shift_gradient` | Learn symmetric oscillator couplings from parameter-shift-compatible observation models and independently check small smooth gradients against central finite differences. |
| QSNN training evidence | `QSNNTrainingRun`, `QSNNParameterShiftDescentRun` | Attach parameter-shift traces and certificates to quantum neural network training loops. |
| Registered model-training evidence | `DifferentiableModelTrainingEvidenceSuite`, `DifferentiableModelTrainingRecord`, `RegisteredDifferentiableTrainingSuiteAuditResult`, `RegisteredDifferentiableTrainingSuiteRecord`, `run_differentiable_model_training_evidence_suite`, `run_registered_differentiable_training_suite_audit` | Package seeded QNN, QGNN, QSNN, Kuramoto-XY, open-system-control, and inverse-coupling-recovery local training cases with loss reduction and gradient-agreement evidence, then audit the requested training-suite lanes without promoting arbitrary architectures, provider hardware, or benchmark-performance claims. |
| Registered Phase-QNode circuit evidence | `PhaseQNodeCircuit`, `PhaseQNodeDensityCircuit`, `PhaseQNodeNoiseChannel`, `PhaseQNodeDepthProfile`, `PhaseQNodeRegisteredCircuitSpec`, `PhaseQNodeTemplateSpec`, `PhaseQNodeGradientEvaluationPlan`, `PhaseQNodeGradientEvaluationGroup`, `build_registered_phase_qnode_circuit`, `phase_qnode_depth_profile`, `build_phase_qnode_template`, `build_sparse_ising_chain_hamiltonian`, `registered_phase_qnode_templates`, `registered_phase_qnode_decompositions`, `registered_phase_qnode_noise_channels`, `decompose_phase_qnode_controlled_gate`, `DenseHermitianObservable`, `PauliTerm`, `PauliCovarianceObservable`, `SparsePauliHamiltonian`, `PhaseQNodeClassicalFisherResult`, `PhaseQNodeDensityExecutionResult`, `PhaseQNodeMetricTensorResult`, `plan_phase_qnode_parameter_shift_evaluations`, `execute_phase_qnode_circuit`, `execute_phase_qnode_density_matrix`, `parameter_shift_phase_qnode_gradient`, `phase_qnode_gradient_support_report`, `phase_qnode_metric_support_report`, `phase_qnode_computational_basis_fisher_information`, `phase_qnode_computational_basis_fisher_support_report`, `phase_qnode_density_support_report`, `phase_qnode_quantum_fisher_information`, `phase_qnode_natural_gradient_metric`, `phase_qnode_support_report`, `ParityScenario`, `run_phase_qnode_framework_parity_suite`, `PhaseQNodeAffinityArtifactValidation`, `validate_phase_qnode_affinity_artifact`, `run_phase_qnode_affinity_benchmark` | Execute the declared local gate/observable family, including arbitrary-depth registered circuits with depth/resource budget gates, registered GHZ-chain and hardware-efficient multi-qubit templates, controlled-H/S/T plus Toffoli/CCZ/Fredkin gates, exact Toffoli/Fredkin operation-list decompositions, sparse Ising-chain Hamiltonian construction with scalar or site/edge coefficients, density-matrix execution through `bit_flip`, `phase_flip`, `depolarizing`, and `amplitude_damping` Kraus channels, dense Hermitian expectations, exact Pauli covariance values, gate-aware logical-parameter shift planning with multi-frequency repeated-parameter fallback, product-rule covariance gradients for pure states, exact computational-basis classical Fisher metrics with optional multinomial finite-shot standard errors, confidence radii, and strict raw-count replay, and pure-state Fubini-Study/QFI metrics; inspect strict support reports before blocked gradient, metric, Fisher, density, or singular-probability paths; compare installed framework parity with named scenarios, record benchmark-isolation metadata, hash and validate raw Phase-QNode affinity artefacts before claim-ledger attachment, and fail closed for unsupported routes. |
| Rust differentiable parity kernels | `phase_qnode_fubini_study_metric_rust`, `phase_qnode_computational_basis_fisher_rust`, `phase_qnode_vector_jvp_rust`, `phase_qnode_vector_vjp_rust`, `phase_qnode_hessian_vector_product_rust`, `phase_qnode_vector_hessian_tensor_rust`, `phase_qnode_complex_derivative_contract_rust`, `parameter_shift_gradient_uncertainty_rust`, `spsa_gradient_rust`, `score_function_gradient_rust`, `gradient_confidence_interval_rust` | Optional PyO3 parity surface for the promoted deterministic local metric, directional-transform, vector-Hessian, real-only complex-boundary, materialised finite-shot uncertainty, materialised SPSA-record, materialised score-function, and confidence-policy primitives. The kernels operate on materialised state derivatives, Jacobians, Hessians, vector Hessian tensors, shifted means, variances, shot counts, coefficients, SPSA perturbations, rewards, score vectors, gradients, standard errors, or trainable masks and are checked against the Python APIs. They do not execute provider callbacks or hardware jobs. |
| Differentiable promotion evidence | `FrameworkOverlayManifest`, `FrameworkOverlayVerification`, `install_framework_overlay`, `verify_framework_overlay_path`, `BenchmarkIsolationMetadata`, `DifferentiableBenchmarkClassificationCase`, `DifferentiableHardeningGateCheck`, `DifferentiableHardeningSliceGateResult`, `DifferentiableModuleHardeningAuditResult`, `DifferentiableModuleHardeningRecord`, `DifferentiableIsolatedBenchmarkPlan`, `DifferentiableIsolatedBenchmarkPlanRow`, `DifferentiableIsolatedBenchmarkPlanValidation`, `DifferentiableArchitectureMap`, `DifferentiableArchitectureMapLayer`, `DifferentiableArchitectureMapValidation`, `ExternalComparisonArtifact`, `ExternalComparisonRow`, `IdenticalCircuitGradientComparisonArtifact`, `IdenticalCircuitGradientComparisonRow`, `REQUIRED_EXTERNAL_COMPARISON_ROW_FIELDS`, `ClaimLedger`, `ClaimLedgerRow`, `ClaimLedgerValidation`, `DifferentiableSupportSurfaceAlignment`, `DifferentiableSOTAPromotionLanguageAudit`, `ExternalValidationArtifactBundle`, `ExternalValidationArtifactEntry`, `ExternalValidationEnvironmentLock`, `ExternalValidationEnvironmentLockValidation`, `EnvironmentLockfileSummary`, `EnzymeMLIRBenchmarkAttachment`, `EnzymeMLIRCompilerADBreadthArtifact`, `EnzymeMLIRCompilerADBreadthCaseEvidence`, `EnzymeMLIRCompilerADBreadthEvidence`, `run_differentiable_hardening_slice_gate`, `run_differentiable_module_hardening_audit`, `differentiable_module_hardening_registry`, `run_differentiable_isolated_benchmark_plan`, `validate_differentiable_isolated_benchmark_plan`, `render_differentiable_isolated_benchmark_plan_markdown`, `run_differentiable_external_comparison_suite`, `run_identical_circuit_gradient_comparison_suite`, `write_differentiable_external_comparison`, `write_identical_circuit_gradient_comparison`, `build_enzyme_mlir_benchmark_attachment`, `build_enzyme_mlir_compiler_ad_breadth_artifact`, `build_enzyme_mlir_compiler_ad_breadth_evidence`, `build_enzyme_mlir_compiler_ad_breadth_gap_artifact`, `build_external_validation_artifact_bundle`, `build_external_validation_environment_lock`, `load_differentiable_claim_ledger`, `load_differentiable_support_surface_alignment`, `load_external_validation_artifact_bundle`, `load_external_validation_environment_lock`, `render_differentiable_architecture_map_markdown`, `render_differentiable_support_surface_alignment_markdown`, `render_external_validation_artifact_bundle_markdown`, `render_external_validation_environment_lock_markdown`, `render_public_claim_table`, `audit_differentiable_sota_promotion_language`, `run_differentiable_architecture_map`, `validate_differentiable_architecture_map`, `validate_claim_ledger`, `validate_differentiable_support_surface_alignment`, `validate_external_validation_artifact_bundle`, `validate_external_validation_environment_lock`, `validate_public_claim_table` | Reproduce the CPU framework overlay, produce CI-only benchmark bundles, compare external AD frameworks, write non-promotional external-comparison artefacts with an enforced row schema for value error, gradient error, runtime, memory, batching, failure, dependency, toolchain, and claim-boundary fields, write stricter identical-circuit Qiskit/PennyLane gradient artefacts with the same circuit, parameters, observable, and exact-state policy, validate that Phase-QNode claims have implementation, tests, docs, known gaps, artefact IDs, and benchmark IDs, attach raw Enzyme/MLIR compiler-AD breadth artifacts and derived evidence for scalar, vector, matrix, loop, alias, MLIR, LLVM, native Enzyme, and isolated benchmark routes, assemble partial breadth captures into explicit per-case hard-gap artifacts, require a promotion-ready `EnzymeMLIRBenchmarkAttachment` instead of a string-only benchmark ID before Enzyme/MLIR provider-exceedance, plan reserved-host isolated reruns without executing or promoting benchmark rows, check committed support surfaces against the generated capability manifest, render a public-safe claim table from the ledger, reject public SOTA or exceedance wording unless matching scorecard categories and ledger rows are promoted, map architecture layers to Rust/Python inventory rows and scorecard categories before broad Rust migration, record exact external-validation environment lockfile checksums for runtime, development, CI, CPU framework overlay, and Enzyme runner reproduction, record a reproducible checksum manifest over committed differentiable validation artefacts, record the focused per-slice hardening checklist plus benchmark-classification invariants without executing benchmark jobs, and audit every differentiable/gradient/QNode/bridge/compiler module in the promotion scope against module-specific tests and declared fail-closed diagnostics. |
| Bounded QNN framework bridge matrix | `BoundedQNNFrameworkBridgeCapability`, `BoundedQNNFrameworkBridgeMatrixResult`, `run_bounded_qnn_framework_bridge_matrix`, `assert_bounded_qnn_framework_bridge_supported` | Declare implemented bounded JAX/PyTorch/TensorFlow bridge routes, including the bounded JAX custom-VJP route, bounded PyTorch custom-autograd route, bounded PyTorch custom `torch.autograd.Function` backward/optimizer audit route, bounded PyTorch `torch.func` compatibility route, bounded PyTorch `torch.compile` route, bounded PyTorch module/layer wrapper route, bounded PyTorch module/optimizer state replay route, bounded PyTorch device-state replay route, bounded PyTorch checkpoint replay route, bounded PyTorch long-lived checkpoint matrix route, bounded PyTorch training-loop matrix route, bounded PyTorch `torch.export` value replay route, bounded PyTorch export-shape matrix route, bounded PyTorch dynamic-batch export replay route, bounded PyTorch AOTAutograd FX gradient replay route, bounded TensorFlow `GradientTape` route, bounded TensorFlow `tf.function` route, bounded TensorFlow XLA route, and bounded TensorFlow Keras layer route, and fail closed for arbitrary simulator autodiff or live provider hardware-gradient routes. |
| Optional JAX bridge | `PhaseJAXParameterShiftResult`, `PhaseJAXNativeQNNGradientResult`, `PhaseJAXCustomVJPQNNGradientResult`, `PhaseJAXPhaseQNodeStatevectorResult`, `PhaseJAXPhaseQNodeNativeTransformResult`, `PhaseJAXPhaseQNodePyTreeTransformResult`, `PhaseJAXPhaseQNodeShardingTransformResult`, `PhaseJAXJITCompatibilityResult`, `PhaseJAXVMAPCompatibilityResult`, `PhaseJAXShardingCompatibilityResult`, `PhaseJAXPyTreeCompatibilityResult`, `PhaseJAXNestedTransformRoute`, `PhaseJAXNestedTransformAlgebraResult`, `PhaseJAXPhaseQNodeLoweringRoute`, `PhaseJAXPhaseQNodeLoweringMatrixResult`, `PhaseJAXCloudValidationRunSpec`, `PhaseJAXMaturityAuditResult`, `jax_parameter_shift_value_and_grad`, `jax_native_qnn_value_and_grad`, `jax_custom_vjp_qnn_value_and_grad`, `jax_phase_qnode_value_and_grad`, `jax_phase_qnode_native_transform_audit`, `jax_phase_qnode_pytree_transform_audit`, `jax_phase_qnode_sharding_transform_audit`, `plan_jax_cloud_validation_batch`, `run_jax_jit_compatibility_audit`, `run_jax_vmap_compatibility_audit`, `run_jax_sharding_compatibility_audit`, `run_jax_pytree_compatibility_audit`, `run_jax_nested_transform_algebra_audit`, `run_jax_phase_qnode_lowering_matrix`, `run_jax_maturity_audit`, `is_phase_jax_available` | Expose phase parameter-shift value-and-gradient calls to JAX workflows through an explicit host-callback boundary, expose native JAX autodiff evidence for the bounded phase-QNN classifier, expose a bounded JAX `custom_vjp` route whose backward rule is checked against the SCPN parameter-shift gradient, lower registered deterministic local Phase-QNode statevector value-and-gradient and flat/PyTree/native-sharded `grad`/`value_and_grad`/`jacfwd`/`jacrev`/`hessian`/`jvp`/`vjp`/`vmap`/`jit`/`pmap` execution into native JAX without host callbacks, report JIT/VMAP/PMAP/PyTree and bounded nested-transform algebra compatibility, provide a fail-closed registered Phase-QNode JAX-lowering matrix, emit a JarvisLabs/cloud validation run spec for locally blocked JAX GPU and multi-device routes, and aggregate a maturity audit that keeps finite-shot/provider/hardware/dynamic lowering, provider callbacks, hardware gradients, broad arbitrary-provider parity, cloud GPU promotion, and promotion-grade benchmarks blocked until artefacts exist. |
| Optional PennyLane bridge | `PennyLaneGradientAgreementResult`, `PennyLaneQNodeConversionResult`, `PennyLaneRoundTripResult`, `PennyLanePluginMatrixRoute`, `PennyLanePluginMatrixResult`, `PennyLaneProviderEvidenceBundle`, `PennyLaneProviderPluginExecutionArtifact`, `PennyLaneProviderGradientParityArtifact`, `PennyLaneHardwarePluginExecutionArtifact`, `PennyLaneMaturityAuditResult`, `check_pennylane_parameter_shift_agreement`, `build_pennylane_qnode_from_phase_qnode`, `check_pennylane_phase_qnode_round_trip`, `check_pennylane_qnode_round_trip`, `run_pennylane_plugin_matrix`, `run_pennylane_maturity_audit`, `is_phase_pennylane_available` | Compare SCPN parameter-shift gradients against caller-supplied PennyLane callables, generate bounded PennyLane QNodes from registered local `PhaseQNodeCircuit` declarations with explicit device, interface, shot, and diff-method metadata, record a fail-closed plugin/provider matrix owned by `scpn_quantum_control.phase.pennylane_provider_plugin` that passes local `default.qubit` parity routes and optional validated provider-plugin execution, matching-interface/diff-method/shot-policy provider-gradient parity, ticketed hardware-plugin execution artefacts, and freshness-bounded provider bundles, and aggregate agreement/export/import evidence plus grouped parameter-shift evaluation counts while keeping isolated-benchmark promotion blocked until artefacts exist. |
| Optional Qiskit bridge | `QiskitParameterShiftRecord`, `QiskitParameterShiftGradientResult`, `QiskitRuntimePrimitiveExecutionArtifact`, `QiskitRuntimeQPUExecutionArtifact`, `QiskitRawCountReplayArtifact`, `QiskitCalibrationStatevectorComparisonArtifact`, `QiskitProviderGradientWorkflowArtifact`, `QiskitRuntimeQPUProviderEvidenceBundle`, `QiskitMaturityAuditResult`, `generate_qiskit_parameter_shift_circuits`, `execute_qiskit_statevector_parameter_shift`, `execute_qiskit_finite_shot_parameter_shift`, `build_qiskit_provider_gradient_workflow_artifact`, `build_qiskit_runtime_qpu_execution_artifact`, `build_qiskit_runtime_qpu_provider_evidence_bundle`, `run_qiskit_maturity_audit` | Generate fully bound Qiskit parameter-shift circuits, evaluate local Statevector gradients, produce finite-shot provider-contract surrogate uncertainty, validate optional no-submit Runtime primitive metadata plus ticketed Runtime QPU EstimatorV2/SamplerV2 execution from captured metadata, require raw-count replay and calibration/statevector comparison artefacts to match the same Runtime QPU evidence chain, attach captured provider-gradient workflow evidence for the complete parameter-shift, finite-difference, LCU, SPSA, QGT, and QFI method set, require unique workflow artefact IDs plus matching Runtime primitive, observable fingerprint, parameter digest, provider/backend/job/circuit/live-ticket metadata, require method-specific workflow provenance for shift-rule, stencil, LCU-generator, SPSA-perturbation, and QGT/QFI matrix evidence, attach the matching chain as one freshness-bounded provider evidence bundle when needed, reject expired bundles during maturity audit, and aggregate maturity evidence while keeping isolated-benchmark promotion blocked until artefacts exist. |
| Optional PyTorch bridge | `PhaseTorchParameterShiftResult`, `PhaseTorchQNNGradientResult`, `PhaseTorchAutogradQNNGradientResult`, `PhaseTorchAutogradFunctionResult`, `PhaseTorchAutogradFunctionRoute`, `PhaseTorchFuncCompatibilityResult`, `PhaseTorchCompileCompatibilityResult`, `PhaseTorchModuleWrapperAuditResult`, `PhaseTorchModuleStateAuditResult`, `PhaseTorchModuleStateValidationResult`, `PhaseTorchModuleStateRoute`, `PhaseTorchDeviceStateAuditResult`, `PhaseTorchDeviceStateRoute`, `PhaseTorchCheckpointAuditResult`, `PhaseTorchCheckpointRoute`, `PhaseTorchCheckpointMatrixResult`, `PhaseTorchCheckpointMatrixRoute`, `PhaseTorchCheckpointMatrixTensorMetadata`, `PhaseTorchCheckpointRuntimeFingerprint`, `PhaseTorchTrainingLoopMatrixResult`, `PhaseTorchTrainingLoopMatrixRoute`, `PhaseTorchTrainingLoopMatrixRecord`, `PhaseTorchTrainingLoopScenario`, `PhaseTorchExportAuditResult`, `PhaseTorchExportRoute`, `PhaseTorchExportShapeMatrixResult`, `PhaseTorchExportShapeMatrixRoute`, `PhaseTorchExportShapeMatrixRecord`, `PhaseTorchExportShapeScenario`, `PhaseTorchDynamicShapeExportResult`, `PhaseTorchDynamicShapeExportRoute`, `PhaseTorchDynamicShapeExportRecord`, `PhaseTorchDynamicShapeExportReplayCase`, `PhaseTorchAOTAutogradExportResult`, `PhaseTorchAOTAutogradExportRoute`, `PhaseTorchAOTAutogradGraphRecord`, `PhaseTorchEcosystemMaturityRoute`, `PhaseTorchEcosystemMaturityAuditResult`, `PhaseTorchLiveOverlayEvidence`, `PhaseTorchPhaseQNodeStatevectorResult`, `PhaseTorchPhaseQNodeTransformResult`, `PhaseTorchPhaseQNodeLoweringRoute`, `PhaseTorchPhaseQNodeLoweringMatrixResult`, `PhaseTorchMaturityAuditResult`, `torch_parameter_shift_value_and_grad`, `torch_bounded_qnn_value_and_grad`, `torch_autograd_qnn_value_and_grad`, `torch_autograd_function_qnn_loss`, `run_torch_autograd_function_audit`, `torch_phase_qnode_value_and_grad`, `torch_phase_qnode_transform_audit`, `run_torch_func_compatibility_audit`, `run_torch_compile_compatibility_audit`, `run_torch_ecosystem_maturity_audit`, `torch_bounded_qnn_module`, `torch_bounded_qnn_layer`, `run_torch_module_wrapper_audit`, `validate_torch_bounded_qnn_state_dict`, `run_torch_module_state_audit`, `run_torch_module_device_state_audit`, `run_torch_module_checkpoint_audit`, `run_torch_long_lived_checkpoint_matrix`, `run_torch_training_loop_matrix`, `run_torch_module_export_audit`, `run_torch_export_shape_matrix`, `run_torch_dynamic_shape_export_audit`, `run_torch_aot_autograd_export_audit`, `run_torch_phase_qnode_lowering_matrix`, `run_torch_maturity_audit`, `default_torch_training_loop_scenarios`, `default_torch_export_shape_scenarios`, `default_torch_dynamic_shape_export_replay_cases`, `is_phase_torch_available` | Convert supported phase parameter-shift value-and-gradient outputs into PyTorch tensors, provide bounded phase-QNN tensor-gradient evidence, expose a bounded custom `torch.autograd.Function` path, audit `Tensor.backward()` and `torch.optim.SGD` integration for that custom backward against SCPN parameter-shift references, audit bounded `torch.func.grad`/`vmap`/`jacrev`, `torch.compile`, module/layer wrapper compatibility, strict bounded module `state_dict` validation, Adam optimizer-state replay, bounded CPU/CUDA-smoke-gated device-state replay, bounded `torch.save`/`torch.load(weights_only=True, map_location="cpu")` checkpoint replay, bounded long-lived checkpoint matrix diagnostics for schema, tensor metadata, runtime fingerprint, repeated local CPU loads, bounded multi-scenario training-loop matrix diagnostics for loss descent, parameter updates, compile-mode coverage, and gradient parity, bounded `torch.export.export` plus `torch.export.save/load` local value replay, bounded static export-shape matrix diagnostics over separate one- and two-parameter `ExportedProgram` artifacts, bounded dynamic-batch export replay through one input-driven `ExportedProgram`, and bounded local AOTAutograd forward/backward FX graph persistence with loaded backward-gradient replay against SCPN parameter-shift references, lower deterministic registered local Phase-QNode statevector value-and-gradient execution into native PyTorch autograd without host callbacks, audit registered local Phase-QNode `torch.func.grad`/`jacrev`/`vmap` transforms against SCPN parameter-shift references, record broad PyTorch module/transform/compiler/CUDA-device maturity, validate optional live CPU-overlay external-comparison artefacts for the PyTorch route, and aggregate those routes plus a fail-closed registered Phase-QNode lowering matrix into a maturity audit that keeps higher-order custom-autograd transforms, registered `torch.compile`, incompatible CUDA/device routes, cross-runtime AOTAutograd execution, dynamic-shape AOTAutograd export, dynamic feature-width export, cross-runtime checkpoint/export portability, external long-lived checkpoint-corpus promotion, arbitrary-architecture training loops, finite-shot lowering, provider callbacks, hardware lowering, dynamic circuits, full compiler/autograd integration, and isolated benchmark promotion blocked until artefacts exist. |
| Optional TensorFlow bridge | `PhaseTensorFlowParameterShiftResult`, `PhaseTensorFlowQNNGradientResult`, `PhaseTensorFlowGradientTapeCompatibilityResult`, `PhaseTensorFlowFunctionCompatibilityResult`, `PhaseTensorFlowXLACompatibilityResult`, `PhaseTensorFlowKerasLayerWrapperAuditResult`, `PhaseTensorFlowPhaseQNodeLoweringRoute`, `PhaseTensorFlowPhaseQNodeLoweringMatrixResult`, `PhaseTensorFlowMaturityAuditResult`, `PhaseTensorFlowMaintenanceReport`, `PhaseTensorFlowMaintenanceRoute`, `tensorflow_parameter_shift_value_and_grad`, `tensorflow_bounded_qnn_value_and_grad`, `run_tensorflow_gradient_tape_compatibility_audit`, `run_tensorflow_function_compatibility_audit`, `run_tensorflow_xla_compatibility_audit`, `tensorflow_bounded_qnn_keras_layer`, `run_tensorflow_keras_layer_wrapper_audit`, `run_tensorflow_phase_qnode_lowering_matrix`, `run_tensorflow_maturity_audit`, `run_tensorflow_maintenance_decision`, `is_phase_tensorflow_available` | Convert supported phase parameter-shift value-and-gradient outputs into TensorFlow tensors, provide bounded phase-QNN tensor-gradient evidence, audit bounded `GradientTape`/`tf.function`/XLA/Keras layer gradients against parameter-shift references, expose a fail-closed registered Phase-QNode TensorFlow-lowering matrix, and aggregate those routes into a maturity record. `run_tensorflow_maintenance_decision()` records the explicit compatibility-only strategy: bounded TensorFlow routes remain maintained, but broad Graph/XLA parity, arbitrary Phase-QNode TensorFlow lowering, full graph autodiff-through-simulator, provider callbacks, hardware gradients, and isolated benchmark promotion stay blocked until artefacts exist. |

## Unified façade

```python
import numpy as np

from scpn_quantum_control import (
    differentiable_api,
    differentiable_compile_report,
    differentiable_gradient,
    differentiable_qfi_fss_report,
    differentiable_support_report,
    explain_differentiability,
)


def objective(values: np.ndarray) -> float:
    return float(values[0] ** 2 + 3.0 * values[1])


gradient = differentiable_gradient(
    objective,
    np.array([2.0, -1.0], dtype=float),
    method="finite_difference",
)
assert gradient.operation == "gradient"
print(gradient.to_dict()["gradient"])

support = differentiable_support_report(
    gate="ry",
    observable="pauli_expectation",
    n_params=2,
)
assert support.supported

diagnostic = explain_differentiability(
    gate="arbitrary_unitary",
    observable="pauli_expectation",
    backend="hardware",
    shots=1024,
)
assert diagnostic.fail_closed
print(diagnostic.to_dict()["blocked_reasons"])
print(diagnostic.to_dict()["suggested_alternatives"])

compile_report = differentiable_compile_report(
    primitive_identities=("scpn.program_ad.array:getitem@1",)
)
assert compile_report.payload["primitive_count"] == 1

same_gradient = differentiable_api(
    "gradient",
    objective=objective,
    values=np.array([2.0, -1.0], dtype=float),
    method="finite_difference",
)
assert same_gradient.to_dict()["operation"] == "gradient"

dashboard = differentiable_api("dashboard_status")
assert dashboard.payload["status_api_ready"] is True

scorecard = differentiable_api("sota_scorecard")
assert scorecard.payload["promotion_ready"] is False

inventory = differentiable_api("rust_python_inventory")
assert inventory.payload["rustification_ready"] is False

qfi_fss = differentiable_qfi_fss_report(
    system_sizes=[2, 3],
    k_range=np.linspace(0.5, 3.0, 6),
)
assert qfi_fss.operation == "qfi_fss_report"
assert qfi_fss.payload["bkt_fit"]["model"] == "bkt_log_correction"
assert "no hardware" in qfi_fss.claim_boundary
```

`UnifiedDifferentiableAPIResult` is the stable evidence envelope for the façade.
It always carries `operation`, `supported`, `fail_closed`, `method`, derivative
arrays when applicable, a route-specific `payload`, and a claim boundary.
The `qfi_fss_report` payload serializes `FSSResult` with raw finite-size
gap-minimum scans, BKT and inverse-size fit diagnostics, residuals, conditioning
metadata, and a non-promotional claim boundary.
`DifferentiabilityDiagnosticReport` is the reviewer-facing explanation surface:
it carries the request, blocked reasons, suggested alternatives, dependency rows
for bounded framework bridges, device capability rows, backend planning rows,
and the underlying support-plan payload. The diagnostic route is planning
evidence only; it does not execute objectives, provider callbacks, hardware
jobs, or performance benchmarks.
`DifferentiableDashboardStatus` is the backing contract for future GUI or
audit-dashboard layers. Consumers must display each row's `state` and
`claim_boundary` directly: metadata-only Program AD IR, bounded
`program_ad_effect_ir.v1` round-trip parsing, static bytecode/source compiler
frontend preflight with bytecode basic blocks, source regions,
source-bytecode maps, symbol scopes, and
alias/effect rows are not executable compiler lowering;
higher-order
transform algebra is `diagnostic` in the default cheap status call and becomes
`conformance_backed` only when `include_conformance=True` runs the local
benchmark report, conformance rows are local non-performance evidence, and
Rust/LLVM/provider/hardware rows stay blocked until executable artefacts exist.

The façade delegates to existing implemented surfaces rather than weakening
their contracts: finite-difference gradients, Jacobians, and Hessians remain
local diagnostic routes; support reports fail closed for unsupported gate,
observable, backend, transform, or adapter combinations; compile reports are
compiler-planning and MLIR interchange evidence unless the selected primitive
plan has an executable backend; benchmark reports are local conformance rows,
not isolated performance, provider, or hardware execution evidence. The SOTA
scorecard report is governance evidence only; it records category blockers
against named external baselines and cannot promote performance, provider, QPU,
GPU, hardware, or `isolated_affinity` claims.

The Rust engine mirrors the bounded Program AD IR metadata schema through
`scpn_quantum_engine::program_ad_ir` and the PyO3
`program_ad_effect_ir_metadata_summary(...)` and
`program_ad_effect_ir_interpret_forward(...)` plus
`program_ad_effect_ir_interpret_value_and_gradient(...)` exports. The
`program_ad_registry_metadata_mirror(...)` export validates the Python
registry-dispatch coverage snapshot and returns deterministic family/facet
counts plus conservative primitive-name overlap with existing bounded Rust
scalar/static-linalg plus elementwise/static-structural replay. Metadata
summaries remain parser parity for `program_ad_effect_ir.v1`; Rust
value+gradient replay is bounded to opcode-bearing scalar, elementwise
shaped-array, and static structural rows with scalar-to-array broadcasting,
static `reshape`/`ravel`, `broadcast_to`, reversed-axis `transpose`, static-axis
`concatenate`/`stack`, static-axis `sum`/`mean`/`prod`/`var`/`std`/`max`/`min`/`median`
reductions with static `ddof`/`correction` metadata for `var`/`std`,
compact static-grid `trapezoid` reductions with `dx`/`x`/`xfull` metadata,
scalar-`q` `quantile`/`percentile` reductions, and scalar all-axis
`sum`/`mean`/`prod`/`var`/`std`/`max`/`min`/`median` plus compact static-grid
`trapezoid` and scalar-`q` `quantile`/`percentile` objective closure, plus static source-map `index_map:<sN|cVALUE,...>` indexing,
including executed runtime branch metadata when matched by runtime phi
provenance. Legacy opcode-free metadata, aliases, mutation, non-lowered dynamic
indexing semantics, dynamic axes, dynamic trapezoid-grid metadata, dynamic q/method metadata, dynamic ddof/correction
metadata, zero-variance `std` gradients, broad array/linalg adjoints, source-level and non-executed branch
semantics, general Program AD execution, LLVM/JIT differentiated execution,
hardware, provider, and performance routes remain fail-closed.
Python callers may use `scpn_quantum_control.program_ad_rust_bridge` directly
for the typed fail-closed wrappers, while the historical
`scpn_quantum_control.differentiable` facade re-exports the same result
dataclasses, registry metadata mirror result, and helper functions for
compatibility.
`compile_whole_program_ad_trace_to_mlir(...)` lowers captured
`program_ad_effect_ir.v1` records into deterministic
`scpn_diff.program_ad_ssa`, `scpn_diff.program_ad_effect`,
`scpn_diff.program_ad_alias_edge`, `scpn_diff.program_ad_control_region`, and
`scpn_diff.program_ad_phi` interchange operations. The
`program_ad_mlir_interchange` dashboard row and
`program_ad_mlir_interchange_contracts` benchmark row validate metadata
lowering only; they do not promote executable Rust, LLVM, JIT, provider,
hardware, or performance claims.

Registered Phase-QNode callers can preflight each narrow route with
`phase_qnode_support_report(...)`, `phase_qnode_density_support_report(...)`,
`phase_qnode_gradient_support_report(...)`,
`phase_qnode_metric_support_report(...)`, and
`phase_qnode_computational_basis_fisher_support_report(...)`. Execution APIs
raise `PhaseQNodeSupportError` with the same report when a density/noise circuit
is sent to pure-state gradients or metrics, when unsupported gates or
observables are present, or when computational-basis Fisher information is
singular at a zero-probability outcome.

## Minimal parameter-shift call

```python
import numpy as np

from scpn_quantum_control.phase.param_shift import parameter_shift_gradient


def cost(params: np.ndarray) -> float:
    return float(np.cos(params[0]))


grad = parameter_shift_gradient(cost, np.array([0.4], dtype=float))
```

## Minimal Kuramoto-XY VQE gradient call

```python
from scpn_quantum_control.bridge.knm_hamiltonian import OMEGA_N_16, build_knm_paper27
from scpn_quantum_control.phase import PhaseVQE

K = build_knm_paper27(L=2)
omega = OMEGA_N_16[:2]

vqe = PhaseVQE(K, omega, ansatz_reps=1)
result = vqe.solve(maxiter=40, seed=0, gradient_method="parameter_shift")
print(result["gradient_method"], result["n_grad_evals"])
```

The solver switches derivative-free defaults to a gradient-aware local
optimiser for this mode and returns gradient evaluation counts plus the final
gradient norm.

## Minimal convergence certificate

```python
import numpy as np
import jax

from scpn_quantum_control.phase import (
    parameter_shift_gradient_descent,
    parameter_shift_natural_gradient_descent,
    run_parameter_shift_optimizer_comparison,
    validate_param_shift_convergence,
    validate_natural_gradient_training,
    validate_parameter_shift_training,
    vqe_with_param_shift,
)


def cost(params: np.ndarray) -> float:
    return float(np.cos(params[0]) + np.sin(params[1]))


run = vqe_with_param_shift(
    cost,
    n_params=2,
    initial_params=np.array([2.7, -0.4]),
    steps=28,
    learning_rate=0.35,
)
certificate = validate_param_shift_convergence(run, gradient_tolerance=0.08)
assert certificate.monotone_energy

generic_run = parameter_shift_gradient_descent(
    cost,
    np.array([2.7, -0.4]),
    max_steps=28,
    learning_rate=0.35,
)
generic_certificate = validate_parameter_shift_training(
    generic_run,
    gradient_tolerance=0.08,
)
assert generic_certificate.monotone_accepted_values

natural_run = parameter_shift_natural_gradient_descent(
    cost,
    np.array([2.7, -0.4]),
    metric_tensor=np.eye(2),
    max_steps=28,
    learning_rate=0.35,
)
natural_certificate = validate_natural_gradient_training(
    natural_run,
    gradient_tolerance=0.08,
)
assert natural_certificate.monotone_accepted_values

comparison = run_parameter_shift_optimizer_comparison(max_steps=8)
assert comparison.passed
assert comparison.natural_gradient_not_worse_count == comparison.start_count
```

Natural-gradient training accepts an explicit metric tensor or callable metric
and validates shape, symmetry, finite values, conditioning, and descent
direction before applying a damped solve. The identity metric path is recorded
as a preconditioner baseline; it is not promoted as a quantum Fisher extraction
or arbitrary-circuit natural-gradient method.

The optimizer comparison audit is a local functional evidence tool. It runs
multiple starts through ordinary parameter-shift descent and natural-gradient
descent, records certificates for every route, and checks whether the metric
route is no worse than the baseline under the declared tolerance. It is not a
hardware benchmark, throughput result, or proof of global optimality.

## Minimal bounded phase-QNN classifier

```python
import numpy as np

from scpn_quantum_control.phase import (
    run_parameter_shift_qnn_conformance_suite,
    run_parameter_shift_qnn_optimizer_benchmark_suite,
    train_parameter_shift_qnn_classifier,
    verify_parameter_shift_qnn_classifier_gradient,
)


features = np.array([[0.0], [np.pi]], dtype=float)
labels = np.array([0.0, 1.0], dtype=float)

run = train_parameter_shift_qnn_classifier(
    features,
    labels,
    initial_params=np.array([0.8], dtype=float),
    learning_rate=0.7,
    max_steps=80,
    target_loss=0.0,
    target_loss_tolerance=1e-4,
)

assert run.prediction.accuracy == 1.0
assert run.certificate.monotone_accepted_values

verification = verify_parameter_shift_qnn_classifier_gradient(
    features,
    labels,
    run.best_params,
)
assert verification.passed

suite = run_parameter_shift_qnn_conformance_suite()
assert suite.passed
print(suite.case_count, suite.unsuitable_scenario_count)

optimizer_suite = run_parameter_shift_qnn_optimizer_benchmark_suite()
assert optimizer_suite.passed
assert optimizer_suite.evidence_class == "functional_non_isolated"
assert "lbfgs" in optimizer_suite.optimizer_names
```

This is a deliberately bounded local classifier. Each feature column is encoded
as a phase offset, each trainable parameter is a phase response, and the
full-batch MSE loss is trained with an explicit `[1, 2]` multi-frequency
parameter-shift rule because MSE introduces second harmonics. The verification
helper replays the same bounded loss against central finite differences and can
record caller-supplied external gradients under names such as `jax` or
`pennylane`. External agreement payloads include `source_class`,
`native_framework_autodiff`, and claim-boundary fields, and the conformance
suite propagates them per case so deterministic manual references,
caller-supplied adapter gradients, and separately validated native framework
routes stay distinct. The conformance suite bundles three deterministic replay
cases, one convergence case, optional external-gradient hooks, and explicit
unsuitable-scenario records with required-evidence lists and claim-boundary
text for hardware, finite-shot uncertainty, unsupported architecture,
feature/parameter contract, unregistered primitive, and external-provenance
routes. The optimizer benchmark suite compares the parameter-shift trainer with
finite-difference gradient descent, full-batch SGD, Adam, SciPy L-BFGS-B with a
parameter-shift Jacobian, diagonal-Fisher natural gradient, seeded SPSA, and a
deterministic derivative-free grid. Each baseline records best loss, accuracy,
evaluation count, step count, convergence flag, method label, and wall-clock
runtime, but the suite records `functional_non_isolated` evidence only; it is
not a throughput benchmark or hardware performance claim. This route is not an
unrestricted QNN framework, live provider execution path, or proof that
arbitrary feature maps are differentiable.

## Exact-answer differentiable domain datasets

Closure artefact:
`data/differentiable_phase_qnode/domain_benchmark_dataset_closure_20260616.json`.
It combines exact-answer synthetic validation with the published public-domain
artefact validation suite and remains `functional_non_isolated`.

```python
from scpn_quantum_control.phase import (
    load_differentiable_domain_benchmark_datasets,
    run_differentiable_domain_benchmark_dataset_validation,
)


datasets = load_differentiable_domain_benchmark_datasets()
print(datasets.dataset_ids)

validation = run_differentiable_domain_benchmark_dataset_validation()
assert validation.passed
assert validation.evidence_class == "synthetic_exact_answer"
```

The dataset suite is deterministic and code-defined: bounded phase-QNN cases
carry exact probabilities, MSE losses, and parameter-shift gradients; the
two-oscillator Kuramoto-XY case carries the exact order parameter, mean phase,
XY energy, and phase-energy gradient for a $\pi/3$ phase gap. The suite is a
correctness/conformance fixture for differentiable training and benchmark
harnesses. It is not a measured physics dataset, hardware result, or isolated
performance benchmark artefact.

Published public-domain cases are indexed separately and backed by committed
`QPUDataArtifact` files:

```python
from scpn_quantum_control.phase import (
    load_differentiable_published_domain_benchmark_cases,
    run_differentiable_published_domain_benchmark_validation,
)


published = load_differentiable_published_domain_benchmark_cases()
assert "ieee5bus_power_grid" in published.dataset_ids

published_validation = run_differentiable_published_domain_benchmark_validation()
assert published_validation.passed
```

Those rows validate publication-safe artefact metadata and the Kuramoto facade
conversion path for the existing EEG, ITER-style MHD, IEEE 5-bus, and
Friston-style FEP benchmark artefacts. They also preserve source-equation IDs
and formula strings for the PLV, MHD mode-coupling, swing-equation power-flow,
and variational-free-energy transforms so reviewers can see which mathematical
terms were carried into the Kuramoto conversion. They are source-backed
conformance records, not live hardware execution or timing evidence.

Differentiable benchmark evidence also carries explicit accelerator metadata.
By default the evidence is CPU-only and does not imply GPU execution. When
`SCPN_BENCH_ACCELERATOR_BACKEND=cuda` or `rocm` is set, the benchmark bundle
requires matching visible-device metadata before an accelerator claim can be
attached; otherwise classification becomes `hard_gap` with
`silent_accelerator_fallback`.

## Native and tensor framework QNN gradient bridges

The bounded phase-QNN classifier has a narrow native JAX autodiff route,
registered deterministic Phase-QNode circuits have a native JAX statevector
value-and-gradient plus native transform route, and PyTorch has
tensor-gradient evidence routes:

```python
from pathlib import Path

import jax
import numpy as np

from scpn_quantum_control.phase import (
    PauliTerm,
    PhaseQNodeCircuit,
    SparsePauliHamiltonian,
    jax_custom_vjp_qnn_value_and_grad,
    jax_native_qnn_value_and_grad,
    jax_phase_qnode_aot_export_audit,
    jax_phase_qnode_native_transform_audit,
    jax_phase_qnode_pytree_transform_audit,
    jax_phase_qnode_sharding_transform_audit,
    jax_phase_qnode_value_and_grad,
    plan_jax_cloud_validation_batch,
    run_jax_jit_compatibility_audit,
    run_jax_maturity_audit,
    run_jax_pytree_compatibility_audit,
    run_jax_sharding_compatibility_audit,
    run_jax_vmap_compatibility_audit,
    run_torch_compile_compatibility_audit,
    run_torch_ecosystem_maturity_audit,
    run_torch_func_compatibility_audit,
    run_torch_long_lived_checkpoint_matrix,
    run_torch_module_checkpoint_audit,
    run_torch_module_device_state_audit,
    run_torch_module_export_audit,
    run_torch_module_wrapper_audit,
    run_torch_module_state_audit,
    run_torch_autograd_function_audit,
    run_torch_export_shape_matrix,
    run_torch_dynamic_shape_export_audit,
    run_torch_aot_autograd_export_audit,
    run_torch_training_loop_matrix,
    run_tensorflow_function_compatibility_audit,
    run_tensorflow_gradient_tape_compatibility_audit,
    run_tensorflow_keras_layer_wrapper_audit,
    run_tensorflow_xla_compatibility_audit,
    tensorflow_bounded_qnn_value_and_grad,
    tensorflow_bounded_qnn_keras_layer,
    torch_autograd_qnn_value_and_grad,
    torch_autograd_function_qnn_loss,
    torch_bounded_qnn_value_and_grad,
    torch_bounded_qnn_layer,
    torch_bounded_qnn_module,
    torch_phase_qnode_compile_boundary_audit,
    torch_phase_qnode_transform_audit,
    torch_phase_qnode_value_and_grad,
)

features = np.array([[0.0], [np.pi]], dtype=float)
labels = np.array([0.0, 1.0], dtype=float)
params = np.array([0.45], dtype=float)

jax_result = jax_native_qnn_value_and_grad(features, labels, params)
jax_custom_vjp_result = jax_custom_vjp_qnn_value_and_grad(features, labels, params)
jax_jit_audit = run_jax_jit_compatibility_audit(
    features=features,
    labels=labels,
    params=params,
)
jax_vmap_audit = run_jax_vmap_compatibility_audit(
    features=features,
    labels=labels,
    params_batch=np.array([[0.25], [0.45], [0.65]], dtype=float),
)
jax_sharding_audit = run_jax_sharding_compatibility_audit(
    features=features,
    labels=labels,
    params_batch=np.linspace(
        0.25,
        0.65,
        int(jax.local_device_count()),
        dtype=float,
    ).reshape(int(jax.local_device_count()), 1),
)
jax_pytree_audit = run_jax_pytree_compatibility_audit(
    features=np.array([[0.0, 0.2, 0.4], [np.pi, np.pi + 0.2, np.pi + 0.4]]),
    labels=labels,
    params_pytree={
        "encoder": np.array([0.25, 0.45], dtype=float),
        "readout": {"phase": np.array([0.65], dtype=float)},
    },
)
jax_circuit = PhaseQNodeCircuit(
    n_qubits=2,
    operations=(("ry", (0,), 0), ("cnot", (0, 1)), ("rz", (1,), 1)),
    observable=SparsePauliHamiltonian((PauliTerm(1.0, ((0, "z"), (1, "z"))),)),
)
jax_qnode = jax_phase_qnode_value_and_grad(
    jax_circuit,
    np.array([0.17, -0.23], dtype=float),
    jit=True,
)
jax_qnode_transforms = jax_phase_qnode_native_transform_audit(
    jax_circuit,
    np.array([0.17, -0.23], dtype=float),
)
jax_qnode_pytree_transforms = jax_phase_qnode_pytree_transform_audit(
    jax_circuit,
    {
        "parameter_0": np.array([0.17], dtype=float),
        "parameter_1": (np.array([-0.23], dtype=float),),
    },
)
jax_qnode_sharding_transforms = jax_phase_qnode_sharding_transform_audit(
    jax_circuit,
    np.tile(
        np.array([[0.17, -0.23]], dtype=float),
        (int(jax.local_device_count()), 1),
    ),
)
jax_qnode_aot_export = jax_phase_qnode_aot_export_audit(
    jax_circuit,
    np.array([0.17, -0.23], dtype=float),
)
jax_cloud_batch = plan_jax_cloud_validation_batch(runner="jarvislabs")
jax_maturity = run_jax_maturity_audit(
    features=features,
    labels=labels,
    params=params,
    params_batch=np.array([[0.25], [0.45]], dtype=float),
    params_pytree={"phase": params},
)
tf_result = tensorflow_bounded_qnn_value_and_grad(features, labels, params)
tf_tape_audit = run_tensorflow_gradient_tape_compatibility_audit(
    features=features,
    labels=labels,
    params=params,
)
tf_function_audit = run_tensorflow_function_compatibility_audit(
    features=features,
    labels=labels,
    params=params,
)
tf_xla_audit = run_tensorflow_xla_compatibility_audit(
    features=features,
    labels=labels,
    params=params,
)
tf_keras_layer = tensorflow_bounded_qnn_keras_layer(
    features=features,
    labels=labels,
    initial_params=params,
)
tf_keras_audit = run_tensorflow_keras_layer_wrapper_audit(
    features=features,
    labels=labels,
    initial_params=params,
)
torch_result = torch_bounded_qnn_value_and_grad(features, labels, params)
torch_autograd_result = torch_autograd_qnn_value_and_grad(features, labels, params)
torch_autograd_function_audit = run_torch_autograd_function_audit(
    features=features,
    labels=labels,
    initial_params=params,
)
torch_func_audit = run_torch_func_compatibility_audit(
    features=features,
    labels=labels,
    params=params,
    params_batch=np.array([[0.25], [0.45], [0.65]], dtype=float),
)
torch_compile_audit = run_torch_compile_compatibility_audit(
    features=features,
    labels=labels,
    params=params,
)
torch_module = torch_bounded_qnn_module(
    features=features,
    labels=labels,
    initial_params=params,
)
torch_layer = torch_bounded_qnn_layer(
    features=features,
    labels=labels,
    initial_params=params,
)
torch_module_audit = run_torch_module_wrapper_audit(
    features=features,
    labels=labels,
    initial_params=params,
)
torch_module_state_audit = run_torch_module_state_audit(
    features=features,
    labels=labels,
    initial_params=params,
)
torch_module_device_state_audit = run_torch_module_device_state_audit(
    features=features,
    labels=labels,
    initial_params=params,
)
torch_module_checkpoint_audit = run_torch_module_checkpoint_audit(
    features=features,
    labels=labels,
    initial_params=params,
)
torch_checkpoint_matrix = run_torch_long_lived_checkpoint_matrix(
    features=features,
    labels=labels,
    initial_params=params,
)
torch_training_loop_matrix = run_torch_training_loop_matrix()
torch_module_export_audit = run_torch_module_export_audit(
    features=features,
    labels=labels,
    initial_params=params,
    export_path="bounded_phase_qnn_export.pt2",
)
torch_export_shape_dir = Path("bounded_phase_qnn_export_shapes")
torch_export_shape_matrix = run_torch_export_shape_matrix(export_dir=torch_export_shape_dir)
torch_dynamic_shape_export = run_torch_dynamic_shape_export_audit(
    export_path=Path("bounded_phase_qnn_dynamic_shape.pt2"),
)
torch_aot_autograd_export = run_torch_aot_autograd_export_audit(
    features=features,
    labels=labels,
    initial_params=params,
    artifact_dir=Path("bounded_phase_qnn_aot_autograd"),
)
torch_qnode = torch_phase_qnode_value_and_grad(
    jax_circuit,
    np.array([0.17, -0.23], dtype=float),
)
torch_qnode_transforms = torch_phase_qnode_transform_audit(
    jax_circuit,
    np.array([0.17, -0.23], dtype=float),
    params_batch=np.array([[0.17, -0.23], [0.21, -0.19]], dtype=float),
)
torch_qnode_compile_boundary = torch_phase_qnode_compile_boundary_audit(
    jax_circuit,
    np.array([0.17, -0.23], dtype=float),
)
torch_ecosystem = run_torch_ecosystem_maturity_audit()

assert jax_result.passed
assert jax_custom_vjp_result.passed
assert jax_jit_audit.passed
assert not jax_jit_audit.native_qnn_host_callback
assert jax_jit_audit.parameter_shift_host_callback
assert jax_vmap_audit.passed
assert not jax_vmap_audit.native_qnn_host_callback
assert "parameter_shift_host_loop_reference" in jax_vmap_audit.unsupported_native_routes
assert jax_sharding_audit.passed
assert not jax_sharding_audit.native_qnn_host_callback
assert jax_pytree_audit.passed
assert not jax_pytree_audit.native_qnn_host_callback
assert jax_qnode.passed
assert jax_qnode.jitted
assert not jax_qnode.host_callback
assert jax_qnode_transforms.passed
assert not jax_qnode_transforms.host_callback
assert "hessian" in jax_qnode_transforms.transform_names
assert jax_qnode_pytree_transforms.passed
assert not jax_qnode_pytree_transforms.host_callback
assert "hessian" in jax_qnode_pytree_transforms.transform_names
assert jax_qnode_sharding_transforms.passed
assert not jax_qnode_sharding_transforms.host_callback
assert jax_qnode_sharding_transforms.pmapped
assert jax_qnode_aot_export.passed
assert jax_qnode_aot_export.serialized
assert not jax_qnode_aot_export.persistent_export_claim
assert jax_cloud_batch.local_execution_status in {
    "local_accelerator_ready",
    "skipped_incompatible_local_hardware",
}
assert "isolated_benchmark_artifact" in jax_cloud_batch.required_artifacts
assert jax_maturity.bounded_model_ready
assert not jax_maturity.ready_for_provider_exceedance
assert tf_result.passed
assert tf_keras_layer.claim_boundary == "bounded_tensorflow_keras_layer_wrapper"
assert tf_keras_audit.passed
assert tf_keras_audit.keras_layer_supported
assert torch_result.passed
assert torch_autograd_result.passed
assert torch_autograd_result.custom_autograd_function
assert torch_func_audit.passed
assert torch_func_audit.func_vmap_supported
assert torch_compile_audit.passed
assert torch_compile_audit.compiled_gradient_supported
assert torch_module().shape == ()
assert torch_layer.claim_boundary == "bounded_torch_module_layer_wrapper"
assert torch_module_audit.passed
assert torch_module_audit.module_wrapper_supported
assert torch_module_state_audit.passed
assert torch_module_state_audit.route_status("module_state_dict_round_trip") == "passed"
assert torch_module_state_audit.route_status("device_state_transfer") == "blocked"
assert torch_module_device_state_audit.passed
assert torch_module_device_state_audit.route_status("cpu_module_state_transfer") == "passed"
assert torch_module_checkpoint_audit.passed
assert torch_module_checkpoint_audit.route_status("checkpoint_weights_only_cpu_load") == "passed"
assert torch_checkpoint_matrix.passed
assert torch_checkpoint_matrix.route_status("repeated_local_cpu_replay") == "passed"
assert torch_module_export_audit.passed
assert torch_module_export_audit.route_status("exported_program_loaded_cpu_replay") == "passed"
assert torch_qnode.passed
assert torch_qnode_transforms.passed
assert torch_qnode_transforms.func_vmap_supported
assert torch_qnode_compile_boundary.passed
assert torch_qnode_compile_boundary.route_status("non_fullgraph_compile") == "passed"
assert torch_qnode_compile_boundary.route_status("fullgraph_compile") == "blocked"
assert not torch_qnode_compile_boundary.persistent_export_claim
assert torch_ecosystem.route_status("torch_compile_callable") in {"passed", "blocked"}
```

`jax_native_qnn_value_and_grad` expresses the bounded model directly in JAX
operations and compares the JAX `value_and_grad` result against the canonical
SCPN parameter-shift gradient. `jax_custom_vjp_qnn_value_and_grad` registers an
explicit JAX `custom_vjp` for the same bounded loss, keeps `host_callback=False`,
and checks the backward rule against the same parameter-shift reference.
`run_jax_jit_compatibility_audit` runs the bounded native and custom-VJP routes
under `jax.jit`, records that both remain no-host-callback routes, and lists the
parameter-shift bridge as host-callback interop rather than native JIT.
`run_jax_vmap_compatibility_audit` vectorises the same bounded native and
custom-VJP routes over a two-dimensional parameter batch, verifies every row
against SCPN parameter-shift references, and labels those references as a
host-side loop rather than native VMAP.
`run_jax_sharding_compatibility_audit` maps one parameter row per local JAX
device with `jax.pmap`, records whether the run is a single-device smoke or a
multi-device pmap audit, and applies the same no-host-callback and host-loop
reference boundaries.
`run_jax_pytree_compatibility_audit` accepts nested numeric parameter PyTrees,
flattens them into the bounded phase-QNN parameter vector, restores gradients to
the original tree structure, and labels arbitrary simulator PyTree lowering as
unsupported.
`jax_phase_qnode_value_and_grad` lowers registered deterministic local
`PhaseQNodeCircuit` statevector execution into native JAX operations, enables
JAX x64 for complex statevector fidelity, optionally JITs the route, and checks
value and gradient against SCPN statevector plus gate-aware parameter-shift
references without `pure_callback`.
`jax_phase_qnode_native_transform_audit` runs the same registered local
statevector value function through native JAX `grad`, `value_and_grad`,
`jacfwd`, `jacrev`, `hessian`, `jvp`, `vjp`, `vmap`, and `jit` routes, checks
first-order and batched gradients against SCPN parameter-shift references, and
checks Hessian symmetry and JVP/VJP contractions without host callbacks.
`jax_phase_qnode_pytree_transform_audit` accepts nested numeric PyTree
parameters for the same registered local circuit family, flattens them in JAX
tree order, restores native gradients to the caller's PyTree structure, and
checks `grad`, `value_and_grad`, `jacfwd`, `jacrev`, `hessian`, `jvp`, `vjp`,
`vmap`, and `jit` against SCPN parameter-shift references without host
callbacks while reporting flattened Hessian symmetry evidence. Finite-shot,
provider, hardware, dynamic-circuit, and performance-promotion claims remain
blocked.
`jax_phase_qnode_sharding_transform_audit` maps one registered local
statevector value-and-gradient row per local JAX device through `jax.pmap`,
checks each row against SCPN parameter-shift references, and reports
`host_callback=False`. Single-device CPU runs are smoke evidence for the pmap
route, not multi-device performance evidence.
`jax_phase_qnode_aot_export_audit` stages the same registered local value route
through `jax.jit(...).lower(...)`, records StableHLO/compiler metadata, exports
and serializes it through `jax.export.export(...)`, deserializes the blob, and
checks replayed values against SCPN parameter-shift references. It is diagnostic
metadata only: exported VJPs, persistent cross-platform execution, provider,
hardware, isolated benchmark, and performance promotion remain blocked.
`plan_jax_cloud_validation_batch` records the local JAX device count and device
descriptions, classifies local GTX 1060 or single-device routes as
`skipped_incompatible_local_hardware`, and returns the JarvisLabs/cloud
commands, required CUDA/XLA/pmap artefacts, isolated-benchmark artefact
requirement, and claim boundary needed before any JAX GPU, multi-device, or
accelerator-performance claim is promoted. The plan does not submit hardware or
network jobs.
`run_jax_maturity_audit` aggregates the bounded custom-VJP, JIT, VMAP,
PMAP/sharding, PyTree, registered-lowering, and cloud-validation scheduling
audits into one reviewer-facing record. It reports `bounded_model_ready=True`
when those bounded routes pass, but keeps `ready_for_provider_exceedance=False`
until full arbitrary `jacfwd`/`jacrev`/Hessian transform algebra,
finite-shot/provider/hardware routes, hardware/provider callback transform
safety, compatible cloud accelerator artefacts, and isolated benchmark artefacts
exist.
`torch_bounded_qnn_value_and_grad` returns framework tensors from the analytic
bounded-model gradient, while `torch_autograd_qnn_value_and_grad` wraps the same
bounded model in a custom `torch.autograd.Function` and checks its backward rule
against the parameter-shift reference. `torch_autograd_function_qnn_loss(...)`
returns a scalar custom-autograd loss for direct `Tensor.backward()` workflows,
and `run_torch_autograd_function_audit(...)` records backward-gradient parity
plus `torch.optim.SGD` integration while keeping higher-order autograd, CUDA,
provider/hardware, arbitrary-simulator, isolated-benchmark, and performance
routes blocked. `run_torch_func_compatibility_audit`
checks bounded `torch.func.grad`, `torch.func.vmap`, and `torch.func.jacrev`
outputs against single-row and batched parameter-shift references.
`run_torch_compile_compatibility_audit` compiles the bounded PyTorch loss route
and checks the resulting gradient against the same parameter-shift reference.
`torch_bounded_qnn_module` and `torch_bounded_qnn_layer` expose the same bounded
loss through a PyTorch `nn.Module`/layer wrapper, and
`run_torch_module_wrapper_audit` checks the wrapper gradient against the same
reference. `run_torch_module_state_audit` verifies strict bounded-module
`state_dict` replay and Adam optimizer-state replay for the same route, while
`validate_torch_bounded_qnn_state_dict` checks keys, shapes, and dtypes without
loading. `run_torch_module_device_state_audit` verifies CPU `module.to(...)`
state replay and attempts CUDA `module.to(...)` replay only after the installed
PyTorch runtime passes a real CUDA smoke. `run_torch_module_checkpoint_audit`
writes a real PyTorch checkpoint, reloads it on CPU with `weights_only=True`,
and replays strict module plus Adam optimizer state.
`run_torch_long_lived_checkpoint_matrix(...)` records a versioned checkpoint
schema, tensor metadata manifest, runtime fingerprint, and repeated local CPU
weights-only loads for the same bounded route while keeping cross-runtime,
CUDA, and external checkpoint-corpus promotion blocked.
`run_torch_training_loop_matrix(...)` records deterministic bounded
one-parameter and two-parameter training-loop scenarios, loss descent,
parameter-update norms, compile-mode coverage, and gradient parity while
keeping CUDA, provider/hardware, arbitrary-architecture, isolated benchmark, and
performance promotion blocked.
`run_torch_module_export_audit` exports the same bounded module through
`torch.export.export(...)`, persists it with `torch.export.save(...)`, reloads it
with `torch.export.load(...)`, and replays the local CPU value route through
`ExportedProgram.module()`. Incompatible CUDA, cross-runtime checkpoint/export
portability, gradient export for this `torch.export` route, dynamic-shape export
promotion, provider, hardware, isolated benchmark, and performance promotion
remain blocked.
`run_torch_export_shape_matrix(...)` wraps that export route over deterministic
one- and two-parameter static feature shapes, records per-shape
`ExportedProgram` artifact metadata, and keeps broader dynamic-shape promotion
outside the local static-shape matrix boundary.
`run_torch_dynamic_shape_export_audit(...)` exports one input-driven bounded
phase-QNN module with symbolic batch constraints, persists and reloads the
`ExportedProgram`, and replays multiple concrete batch sizes against the SCPN
parameter-shift reference. Dynamic feature width, CUDA, dynamic-shape
AOTAutograd export, cross-runtime checkpoint/export portability,
provider, hardware, isolated benchmark, and performance promotion remain
blocked until dedicated artefacts exist.
`run_torch_aot_autograd_export_audit(...)` compiles the bounded phase-QNN loss
route with PyTorch AOTAutograd, captures the forward and backward FX
`GraphModule` objects, saves and reloads the self-produced local artifacts, and
replays the loaded backward graph against the SCPN parameter-shift gradient.
Those artifacts are local PyTorch FX pickles, not stable cross-runtime export
contracts; CUDA replay, dynamic-shape AOTAutograd export, isolated benchmark,
and performance promotion remain blocked until dedicated artefacts exist.
`torch_phase_qnode_value_and_grad` lowers deterministic registered local
Phase-QNode statevector value-and-gradient execution into native PyTorch
autograd without host callbacks and checks the result against the SCPN
parameter-shift reference. `torch_phase_qnode_transform_audit` runs the same
registered local statevector family through `torch.func.grad`,
`torch.func.jacrev`, and `torch.func.vmap`, checks single-row and batched
gradients against SCPN parameter-shift references, and keeps `host_boundary`
false. `torch_phase_qnode_compile_boundary_audit` executes the registered
Phase-QNode `torch.compile` route in non-fullgraph, dynamic, and fullgraph
modes, records non-fullgraph correctness, and keeps dynamic-shape,
fullgraph compiled-frame, AOTAutograd/export, CUDA, provider, hardware,
isolated-benchmark, and performance promotion blocked until artefacts exist.
`run_torch_ecosystem_maturity_audit` records installed PyTorch
`nn.Module`/`Parameter`, `torch.func`, `torch.compile`, and CUDA-device
capabilities; visible but incompatible CUDA devices remain blocked until a
successful device smoke and device-specific Phase-QNode gradient artefact exist.
`run_torch_maturity_audit` aggregates the bounded analytic tensor,
custom-autograd, `torch.func`, `torch.compile`, module/layer wrapper,
registered statevector lowering, and ecosystem-route evidence into one
reviewer-facing record. It reports `bounded_model_ready=True` only when the
bounded model routes pass, but keeps `ready_for_provider_exceedance=False` until
live overlay execution, registered Phase-QNode `torch.compile` lowering,
compatible CUDA/device evidence, finite-shot/provider/hardware Phase-QNode Torch
lowering, full compiler/autograd integration, and promotion-grade isolated benchmark artefacts
exist.
`tensorflow_bounded_qnn_value_and_grad` returns TensorFlow tensors from the
analytic bounded-model gradient and checks the same reference.
`run_tensorflow_gradient_tape_compatibility_audit` differentiates the same
bounded TensorFlow loss through `GradientTape` and checks the returned gradient
against the parameter-shift reference.
`run_tensorflow_function_compatibility_audit` traces that bounded loss through
`tf.function`, differentiates it with `GradientTape`, and checks the returned
gradient against the same reference.
`run_tensorflow_xla_compatibility_audit` requests `tf.function(jit_compile=True)`
for that bounded loss and checks the gradient against the same reference.
`tensorflow_bounded_qnn_keras_layer` exposes the bounded loss through a Keras
`Layer`, and `run_tensorflow_keras_layer_wrapper_audit` checks its
`GradientTape` gradient against the same reference. These routes are not
arbitrary simulator autodiff claims, not provider-backed hardware gradients, not
broad XLA lowering claims, and not replacements for the broader
framework-agreement surface for caller-supplied models.

Use the bridge matrix before selecting a framework route:

```python
from scpn_quantum_control.phase import run_bounded_qnn_framework_bridge_matrix

matrix = run_bounded_qnn_framework_bridge_matrix()
assert matrix.capability_by_framework("jax").native_framework_autodiff
assert matrix.capability_by_framework("pytorch").tensor_output
assert not matrix.capability_by_framework("generic_simulator_autodiff").supported
```

## Minimal gradient support matrix

```python
from scpn_quantum_control.phase import (
    assert_gradient_support,
    plan_gradient_support,
    run_gradient_support_matrix_audit,
)


plan = plan_gradient_support(
    gate="ry",
    observable="pauli_expectation",
    backend="statevector",
    transform="grad",
    adapter="native",
    n_params=2,
)
assert_gradient_support(plan)

audit = run_gradient_support_matrix_audit()
assert audit.passed
```

The support matrix answers a broader question than the backend planner: it
combines gate, observable, backend, transform, and adapter policy into one
fail-closed decision. Built-in supported plans include local parameter-shift,
finite-shot parameter-shift with variance requirements, JAX host-callback
value-and-gradient, and Qiskit shifted-circuit gradients. Built-in blocked
plans include unregistered gates, unregistered observables, hardware without
policy approval, unsupported transform algebra, and finite-shot Hessian
requests.

## Minimal transform-nesting plan

```python
from scpn_quantum_control.phase import (
    assert_gradient_transform_nesting_supported,
    plan_gradient_transform_nesting,
    run_gradient_transform_nesting_audit,
)


plan = plan_gradient_transform_nesting(("grad", "grad"), n_params=2)
assert_gradient_transform_nesting_supported(plan)

audit = run_gradient_transform_nesting_audit()
assert audit.passed
```

Supported nesting is intentionally bounded. Local `grad`, `value_and_grad`,
deterministic local `hessian`, `grad` of `grad`, single-adapter
`value_and_grad`, phase `gradient_tape`, phase `qnode_tape`, scalar local
`jvp`, scalar local `vjp`, scalar local `jacfwd`, and scalar local `jacrev`
routes are planned as supported. `vmap`, nested ML/provider adapters,
finite-shot curvature, nested tape transforms, vector-output Jacobians, and
hardware nesting fail closed with reasons and alternatives.

## Minimal composed objective

```python
import numpy as np

from scpn_quantum_control.phase import (
    build_phase_control_objective,
    plan_composed_objective_execution,
    run_composed_objective_audit_suite,
    run_composed_objective_planner_audit,
    train_composed_phase_objective,
    validate_composed_objective_training,
)


objective = build_phase_control_objective(
    2,
    energy_weight=1.0,
    fidelity_target=np.zeros(2),
    fidelity_weight=0.2,
    safety_bounds=(-1.0, 1.0),
    safety_weight=0.1,
)

evaluation = objective.evaluate(np.array([0.8, -0.7]))
run = train_composed_phase_objective(objective, np.array([0.8, -0.7]))
certificate = validate_composed_objective_training(run, min_decrease=0.1)

print(evaluation.value, certificate.monotone_accepted_values)

audit = run_composed_objective_audit_suite()
assert audit.passed
assert audit.hybrid_parameter_shift_gate_failed

plan = plan_composed_objective_execution(objective)
planner_audit = run_composed_objective_planner_audit()
assert plan.recommended_entrypoint == "train_composed_phase_objective"
assert planner_audit.passed
```

The objective reports which terms are parameter-shift compatible. Periodic
energy, fidelity, regularization, and symmetry terms are compatible; the smooth
box-safety penalty is analytic-only and makes
`require_parameter_shift_compatible()` fail closed.
`run_composed_objective_audit_suite()` makes that boundary executable: it
checks finite-difference gradient agreement for pure and hybrid objectives,
confirms the pure objective passes the parameter-shift gate, confirms the
hybrid safety objective fails that gate, and records local training
certificates for both.
`plan_composed_objective_execution()` is the fail-closed routing layer: pure
periodic objectives route to local parameter-shift training, hybrid objectives
route to exact term-gradient training, and hardware, unknown backends, or forced
pure parameter-shift on analytic safety terms return unsupported plans.

## Minimal provider-gradient readiness audit

```python
from scpn_quantum_control.phase import run_provider_gradient_readiness_audit


audit = run_provider_gradient_readiness_audit()
assert audit.passed
print(len(audit.supported_records), len(audit.blocked_records))
```

The audit executes three supported local callback routes and three deliberate
failure routes. Supported records cover deterministic statevector
parameter-shift, finite-shot parameter-shift with variance metadata, and
multi-frequency finite-shot parameter-shift. Blocked records cover hardware
without explicit policy approval, unknown provider families, and finite-shot
callbacks that omit sample variance. This is provider-readiness evidence, not a
live hardware-gradient claim.

## Hardware-gradient policy readiness

Use `evaluate_hardware_gradient_policy(...)` before preparing a provider-backed
hardware-gradient job. The policy checks explicit hardware opt-in, provider and
backend allowlists, parameter-shift evaluation count, per-evaluation and total
shot budgets, required evidence identifiers, and live-execution ticket status:

```python
from scpn_quantum_control.phase import (
    HardwareGradientRequest,
    evaluate_hardware_gradient_policy,
)

decision = evaluate_hardware_gradient_policy(
    HardwareGradientRequest(
        provider="ibm_quantum",
        backend="ibm_quantum",
        n_params=2,
        shots=512,
        allow_hardware=True,
        evidence_ids={
            "backend_calibration_id": "calibration-snapshot-id",
            "no_qpu_gate_id": "no-qpu-gate-id",
            "claim_boundary_id": "claim-boundary-id",
            "cost_budget_id": "budget-approval-id",
        },
    )
)
```

An approved dry-run decision means the request is ready for controlled provider
job preparation. `prepare_provider_hardware_parameter_shift_gradient(...)`
packages the same policy decision with provider/backend, shifted-evaluation,
shot-budget, and claim-boundary metadata. It still does not submit hardware
work and does not promote a hardware-gradient claim. Live mode remains blocked
unless a `live_execution_ticket` is present.

For a reviewer-facing support matrix, use
`run_provider_hardware_gradient_preparation_audit()`. It records bounded dry-run,
ticketed live-preparation, missing-evidence, shot-budget, unknown-provider, and
missing-ticket scenarios. Passing the audit means the preparation layer preserves
its declared boundaries; it is not a live hardware-gradient result.

For a whole-lane status ledger, use `run_differentiable_readiness_audit()`. It
aggregates the support matrix, transform nesting, QNode tape/transform suites,
provider gradients, hardware policy, and provider hardware-preparation audit
into one JSON-ready pass/fail record with supported counts, blocked counts, and
hardware-execution counters.

## Minimal QSNN descent certificate

```python
import numpy as np

from scpn_quantum_control.qsnn import QuantumDenseLayer, QSNNTrainer

layer = QuantumDenseLayer(1, 1, seed=42)
trainer = QSNNTrainer(layer, lr=0.4)
run = trainer.train_with_parameter_shift_descent(
    np.array([[1.0]]),
    np.array([[0.0]]),
    max_steps=40,
    min_loss_decrease=1e-4,
)

assert run.certificate.monotone_accepted_values
```

This route is full-batch local-simulator training. Hardware backends remain
disabled by default through the same fail-closed backend planner used by phase
gradients.

## Reviewer-facing gradient audit report

```python
import numpy as np

from scpn_quantum_control.phase import run_known_phase_gradient_audit


report = run_known_phase_gradient_audit(np.array([0.8, -0.5, 0.3]))

print(report.passed)
print(report.finite_difference.max_abs_error)
print(report.analytic.max_abs_error)
print(report.training_certificate.to_dict())
```

The audit report is intended for visible correctness evidence. It combines
parameter-shift versus finite-difference agreement, parameter-shift versus an
analytic gradient, and a deterministic gradient-descent convergence
certificate. The built-in benchmark is a smooth phase-rotation objective,
`mean(1 - cos(theta_i))`, with exact gradient `sin(theta_i) / n`.
Discontinuous objectives, stochastic hardware shots, arbitrary regressors, and
undeclared generator spectra are explicitly outside this report boundary.

For a wider built-in conformance pass, use the benchmark suite:

```python
from scpn_quantum_control.phase import run_phase_gradient_benchmark_suite


suite = run_phase_gradient_benchmark_suite()

print(suite.passed)
print(suite.benchmark_names)
print(suite.worst_gradient_error)
print(suite.unsupported_scenarios)
```

The suite currently covers single-frequency phase rotations, multi-frequency
phase rotations using a declared shift rule, and a coupled pair phase loss.
This is the recommended CI and paper-table entry point when users need visible
evidence that the differentiable-programming surface handles more than one
single-case gradient.

For the full supported workflow audit:

```python
from scpn_quantum_control.phase import run_differentiable_workflow_audit_suite


workflow = run_differentiable_workflow_audit_suite()

print(workflow.passed)
print(workflow.workflow_names)
print(workflow.worst_gradient_error)
print(workflow.best_training_values)
print(workflow.unsupported_scenarios)
```

This single report aggregates phase-gradient conformance, finite-shot
uncertainty containment, coupling-gradient verification, and coupling-learning
training evidence. It is the best current release-note and reviewer-facing
entry point for the supported differentiable-programming surface. It does not
claim arbitrary Python reverse-mode AD, live provider calibration, dynamic
circuit topology, classical regressors without generator spectra, or
mutation-heavy program IR semantics.

For optional ML-framework parity status:

```python
from scpn_quantum_control.phase import (
    run_ml_framework_gradient_audit,
    run_phase_qnode_framework_parity_suite,
)


ml = run_ml_framework_gradient_audit()
registered = run_phase_qnode_framework_parity_suite(
    scenario="registered_two_qubit_entangling_statevector"
)

print(ml.audit_passed)
print(ml.executed_frameworks)
print(ml.unavailable_frameworks)
print(ml.blocked_frameworks)
print(ml.failed_frameworks)
print(registered.scenario, registered.frameworks, registered.passed)
```

This report checks JAX, PyTorch, TensorFlow, and PennyLane routes without
requiring those optional dependencies in the base installation. Importable
adapters are compared against the native parameter-shift reference; missing
dependencies are recorded as fail-closed unavailable. PennyLane remains blocked
unless the caller supplies a QNode gradient callable, because a meaningful
round-trip requires caller-owned circuit semantics.

For finite-shot uncertainty evidence:

```python
import numpy as np

from scpn_quantum_control.phase import run_finite_shot_gradient_uncertainty_audit


def objective(theta: np.ndarray) -> float:
    return float(np.mean(1.0 - np.cos(theta)))


finite_shot = run_finite_shot_gradient_uncertainty_audit(
    objective,
    np.array([0.7, -0.4, 0.2]),
    target_standard_error=0.02,
)

print(finite_shot.passed)
print(finite_shot.max_standard_error)
print(finite_shot.within_confidence)
```

This path verifies uncertainty propagation, shot allocation, and confidence
containment for declared shifted-expectation variances. It is not a live
hardware-sampling, detector-drift, or queue-calibration certificate.
Materialised parameter-shift sample records are also checked against their
plus/minus shifted values, finite-shot variances, shot counts, and trainable
mask before the result contract accepts them; attached records must reconstruct
the published gradient and diagonal covariance. They also require
`sample_seed`, `shot_batch_id`, and an allowlisted `source_class`, either from a
`FiniteShotSampleProvenance` record or a matching mapping, before
caller-supplied finite-shot tensors can become accepted replay evidence.
Stochastic parameter-shift,
SPSA, and score-function result contracts also reject accepted evidence when
standard errors disagree with covariance diagonals or attached confidence
interval bounds disagree with published confidence radii.

## Minimal differentiable coupling learning

```python
import numpy as np

from scpn_quantum_control.phase import (
    learn_couplings_from_observations,
    multi_frequency_parameter_shift_rule,
    verify_coupling_parameter_shift_gradient,
)


def observations(K: np.ndarray) -> np.ndarray:
    return np.array([np.sin(K[0, 1])])


run = learn_couplings_from_observations(
    observations,
    target_observations=np.array([0.0]),
    initial_couplings=np.array([[0.0, 0.8], [0.8, 0.0]]),
    rule=multi_frequency_parameter_shift_rule([2.0]),
    max_steps=80,
)

certificate = verify_coupling_parameter_shift_gradient(
    observations,
    target_observations=np.array([0.0]),
    couplings=np.array([[0.0, 0.8], [0.8, 0.0]]),
    rule=multi_frequency_parameter_shift_rule([2.0]),
)

print(run.learned_coupling_matrix, run.certificate.monotone_accepted_values)
print(certificate.passed, certificate.max_abs_error)
```

This is a bounded differentiable-programming route for sinusoidal or quantum
expectation observation models. The verifier is a small-model diagnostic that
records parameter-shift and finite-difference gradients, absolute errors,
evaluation counts, and edge provenance. It is not an arbitrary
classical-regression, discontinuous-model, shot-noisy hardware, or
production-scale finite-difference claim; hardware backends remain disabled
unless an explicit policy enables them.

## Minimal backend gradient plan

```python
from scpn_quantum_control.phase import plan_quantum_gradient_backend

plan = plan_quantum_gradient_backend("statevector", n_params=4)
assert plan.method == "parameter_shift"
```

For finite-shot simulator diagnostics:

```python
plan = plan_quantum_gradient_backend("qasm_simulator", n_params=4, shots=4096)
assert plan.method == "stochastic_parameter_shift"
```

Hardware backends intentionally return an unsupported plan by default. That is a
safety boundary, not a missing exception.

For seeded local SPSA diagnostics:

```python
import numpy as np

from scpn_quantum_control.differentiable import (
    SPSAObjectiveSample,
    spsa_gradient_estimate,
)


def finite_shot_cost(values: np.ndarray, shots: int | None) -> SPSAObjectiveSample:
    return SPSAObjectiveSample(
        value=float(0.5 * values[0] - 0.25 * values[1]),
        variance=0.04,
        shots=shots,
    )


result = spsa_gradient_estimate(
    finite_shot_cost,
    np.array([0.4, -0.2]),
    perturbation_radius=0.25,
    repetitions=4,
    seed=11,
    shots=400,
)

print(
    result.gradient,
    result.standard_error,
    result.confidence_interval.status,
    result.claim_boundary,
)
```

The SPSA route records each plus/minus perturbation pair and propagates
finite-shot uncertainty only when objective samples provide both variance and
shot counts. It is a seeded local estimator over caller-provided objective
values; it does not submit provider jobs or claim arbitrary hardware-gradient
execution.

For materialised score-function diagnostics:

```python
import numpy as np

from scpn_quantum_control.differentiable import score_function_gradient_estimate

rewards = np.array([2.0, 0.0, 4.0])
score_vectors = np.array([[1.0, 2.0], [-1.0, 0.0], [0.0, 1.0]])

result = score_function_gradient_estimate(
    rewards,
    score_vectors,
    baseline=1.0,
    confidence_z=2.0,
)

print(
    result.gradient,
    result.covariance,
    result.confidence_interval.status,
    result.claim_boundary,
)
```

This is the likelihood-ratio identity
`E[(reward - baseline) * grad log p(sample; theta)]` over materialised finite
samples. It requires finite rewards and score vectors, at least two samples for
empirical uncertainty, and an explicit finite baseline. It does not infer score
vectors from a sampler and does not claim REINFORCE-style gradients for
arbitrary discrete programs.

For explicit confidence-policy checks:

```python
from scpn_quantum_control.differentiable import (
    GradientFailurePolicy,
    gradient_confidence_interval,
)

interval = gradient_confidence_interval(
    gradient=[0.5, -0.25],
    standard_error=[0.02, 0.08],
    confidence_z=2.0,
    trainable=[True, True],
    failure_policy=GradientFailurePolicy(max_standard_error=0.05),
)

print(interval.lower, interval.upper, interval.status, interval.failure_reasons)
```

`GradientFailurePolicy` evaluates active trainable parameters only. A missing
or all-false trainable mask is rejected when trainability is required, while
excess standard error or confidence radius returns a `failed` interval with
machine-readable reasons.
The same confidence status is stored on stochastic parameter-shift result
contracts after shifted-sample reconstruction succeeds, so uncertainty policy
metadata cannot mask inconsistent finite-shot provenance.

## Canonical first-path namespace

```python
import numpy as np

from scpn_quantum_control import diff


def cost(params: np.ndarray) -> float:
    return float(np.sin(params[0]) + params[1] ** 2)


circuit = diff.differentiable_circuit(
    cost,
    name="two_parameter_phase_objective",
    parameter_names=("theta", "bias"),
)

params = np.array([0.3, 0.5])
print(circuit(params))
print(circuit.grad(params, method="finite_difference"))
print(circuit.diagnostics.to_dict()["supported"])
print(diff.jit_or_explain(circuit).to_dict()["fail_closed"])
```

`DifferentiableCircuit` serializes metadata, backend capability, shot policy,
and estimator provenance, but it does not serialize executable Python code.
Unsupported routes fail closed through `DifferentiableCircuitDiagnostics` and
`JITExplanation` instead of falling back silently.

## Minimal gradient tape

```python
import numpy as np

from scpn_quantum_control.phase import gradient_tape


def cost(params: np.ndarray) -> float:
    return float(np.cos(params[0]))


with gradient_tape(backend="statevector") as tape:
    record = tape.record_parameter_shift("one_angle", cost, np.array([0.4]))

print(record.gradient, record.plan.method)
```

The tape records only supported phase-gradient evaluations. Unsupported
hardware routes fail closed through the same backend planner.

QNode-style tape records keep the same boundary but attach reviewer-facing
finite-shot provenance. A finite-shot `PhaseQNodeTapeRecord` serializes
`sample_record_count` plus `sample_records` with term index, parameter index,
trainable mask, plus/minus estimates, variances, shot counts, sample seed,
shot-batch ID, source class, and gradient and variance contributions. These
records prove local stochastic replay; they do not promote provider submission,
live hardware execution, or isolated benchmark performance.

## Minimal JAX host-callback bridge

```python
import numpy as np

from scpn_quantum_control.phase import jax_parameter_shift_value_and_grad


def cost(params: np.ndarray) -> float:
    return float(np.cos(params[0]))


result = jax_parameter_shift_value_and_grad(
    cost,
    np.array([0.4]),
    jit=True,
)
print(result.gradient, result.host_callback)
```

This is an optional interop adapter. It imports JAX only when called and reports
`host_callback=True` for JIT-wrapped parameter-shift execution. Use
`run_jax_jit_compatibility_audit(...)` for the bounded phase-QNN route when a
reviewer needs explicit evidence that native JAX and custom-VJP loss paths JIT
without host callbacks while the generic parameter-shift bridge remains
host-callback interop.

## Minimal PennyLane agreement check

```python
import numpy as np

from scpn_quantum_control.phase import check_pennylane_parameter_shift_agreement


def scpn_cost(params: np.ndarray) -> float:
    return float(np.cos(params[0]))


def pennylane_grad(params: np.ndarray) -> np.ndarray:
    # Usually qml.grad(qnode)(params); written explicitly for compact docs.
    return np.array([-np.sin(params[0])], dtype=float)


agreement = check_pennylane_parameter_shift_agreement(
    scpn_cost,
    pennylane_grad,
    np.array([0.4]),
)
assert agreement.passed
```

The bridge validates caller-supplied agreement. For registered local
`PhaseQNodeCircuit` declarations, it can also build a bounded PennyLane QNode:

```python
from scpn_quantum_control.phase import (
    PauliTerm,
    PhaseQNodeCircuit,
    build_pennylane_qnode_from_phase_qnode,
    check_pennylane_phase_qnode_round_trip,
)

circuit = PhaseQNodeCircuit(
    1,
    (("ry", (0,), 0), ("rx", (0,), 1)),
    PauliTerm(1.0, ((0, "z"),)),
)
conversion = build_pennylane_qnode_from_phase_qnode(circuit, shots=None)
round_trip = check_pennylane_phase_qnode_round_trip(circuit, np.array([0.4, -0.2]))
assert round_trip.passed
print(conversion.device_name, conversion.shots, conversion.diff_method)
```

The generated-QNode route covers the registered static gate family and direct
expectation observables with PennyLane equivalents. It does not claim provider
submission, hardware execution, dynamic circuits, noise models, or covariance
observable conversion.
Generated-QNode `device_name` metadata is trimmed and rejected when empty or
when it contains control characters before PennyLane device creation.
Generated-QNode `interface` and `diff_method` metadata is also constrained to
canonical PennyLane interfaces (`auto`, `autograd`, `jax`, `tf`, `torch`) and
documented QNode diff methods (`adjoint`, `backprop`, `best`, `device`,
`finite-diff`, `hadamard`, `parameter-shift`, `spsa`), so plugin selection
remains explicit and auditable.
`scpn_quantum_control.phase.pennylane_provider_plugin` owns the provider-plugin
artefact types and fail-closed plugin matrix; `pennylane_bridge` re-exports the
same objects for compatibility with older imports.
`PennyLaneProviderPluginExecutionArtifact` validates provider-plugin execution
metadata with non-empty plugin/provider/device/backend identities, a
circuit-fingerprint string, replay metadata when present, explicit PennyLane
`interface`, `diff_method`, and `shot_policy` metadata, canonical interface
values (`auto`, `autograd`, `jax`, `tf`, `torch`) instead of undocumented
aliases, documented QNode diff methods (`adjoint`, `backprop`, `best`,
`device`, `finite-diff`, `hadamard`, `parameter-shift`, `spsa`) instead of
undocumented aliases, `shot_policy="analytic"` with `shots=None` or
`shot_policy="finite_shot"` with a positive shot count, SHA-256 result and
metadata digests, non-hardware execution mode, and
`hardware_execution=False`.
`run_pennylane_plugin_matrix` records local `default.qubit` exact-state,
shot-policy metadata, generated Phase-QNode export, and supported tape-import
routes as passed. Passing a validated provider execution artefact marks
`provider_plugin_execution` as passed, and passing a matching
`PennyLaneProviderGradientParityArtifact` marks
`provider_plugin_gradient_parity` as passed only when provider identity,
circuit fingerprint, PennyLane interface, diff method, and shot policy match.
Hardware-plugin execution remains blocked until its own ticketed artefact is
attached, and isolated-benchmark promotion remains blocked with required
artefacts listed per route.
Passing a `PennyLaneProviderEvidenceBundle` keeps provider execution,
provider-gradient parity, and optional ticketed hardware execution in one
exclusive attachment. The bundle requires explicit `captured_at_utc` and
`valid_until_utc` metadata, rejects inverted freshness windows, rejects
hardware evidence whose provider, circuit fingerprint, or shot count no longer
matches the provider execution chain, rejects provider-gradient parity whose
interface, diff method, or shot policy drifts from the provider execution
chain, and fails closed when the bundle expires before the review cutoff.
Passing a validated `PennyLaneHardwarePluginExecutionArtifact` marks only
`hardware_plugin_execution` as passed; the artefact must include ticket,
allowlist, shot-budget, hardware evidence, raw-count, calibration digest,
calibration capture/expiry timestamps, and metadata provenance before the route
opens. The plugin matrix rejects stale calibration metadata at the review cutoff
before it can open the hardware-plugin route.
`run_pennylane_maturity_audit` aggregates caller-supplied gradient agreement,
caller-supplied QNode value/gradient parity, generated Phase-QNode export
round-trip parity, optional PennyLane tape import round-trip parity, device
metadata, shot policy, diff method, grouped registered Phase-QNode
parameter-shift evaluation counts, optional provider execution,
provider-gradient parity, and hardware execution artefacts, and the plugin
matrix. It reports
`identical_circuit_ready=True` only when the import tape is supplied and all
bounded agreement/export/import routes pass; `ready_for_provider_exceedance`
remains false until isolated-benchmark artefacts exist.
The inverse PennyLane tape import rejects non-finite gate parameters and invalid
round-trip tolerances before executing the imported Phase-QNode comparison, so
import parity remains a bounded local-circuit contract rather than an implicit
provider or hardware claim.

## Minimal PyTorch and TensorFlow tensor bridges

```python
import numpy as np

from scpn_quantum_control.phase import (
    tensorflow_parameter_shift_value_and_grad,
    torch_parameter_shift_value_and_grad,
)


def cost(params: np.ndarray) -> float:
    return float(np.cos(params[0]))


torch_result = torch_parameter_shift_value_and_grad(cost, np.array([0.4]))
tf_result = tensorflow_parameter_shift_value_and_grad(cost, np.array([0.4]))

print(torch_result.torch_gradient, torch_result.host_boundary)
print(tf_result.tensorflow_gradient, tf_result.host_boundary)
```

These bridges are optional tensor-conversion boundaries. They are useful for
framework pipelines that need gradient payloads, but they do not claim native
PyTorch or TensorFlow autodiff through a quantum simulator.

## Minimal custom primitive route

```python
from scpn_quantum_control import CustomDerivativeRule

rule = CustomDerivativeRule(
    name="square",
    value=lambda values: values[0] ** 2,
    derivative=lambda values, tangent: 2.0 * values[0] * tangent[0],
)
```

Production use should add primitive identity, shape, dtype, lowering, batching, nondifferentiability, and fail-closed tests before the primitive is advertised as supported.

## API contract checklist

Every new differentiable API must document:

- input shapes and dtype rules;
- scalar, vector, matrix, batch, and backend support;
- exact versus approximate derivative semantics;
- unsupported gates, transforms, backends, and control flow;
- finite-shot variance or numerical tolerance where relevant;
- reproducibility metadata;
- benchmark or convergence evidence before promotion.
