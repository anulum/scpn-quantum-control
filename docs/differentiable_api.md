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
| `scpn_quantum_control.differentiable_api` | Unified façade for value, gradient, Jacobian, Hessian, support, diagnostics, compile, and local conformance benchmark reports with one JSON evidence envelope. |
| `scpn_quantum_control.differentiable` | AD data structures, primitive registry contracts, optimisation helpers, program-AD metadata, and support reports. |
| `scpn_quantum_control.phase.param_shift` | Parameter-shift gradient helper and gradient-descent VQE example. |
| `scpn_quantum_control.phase.coupling_learning` | Differentiable coupling inference from observation models with convergence and finite-difference agreement certificates. |
| `scpn_quantum_control.phase.gradient_descent` | Generic parameter-shift gradient descent with line-search traces and convergence certificates. |
| `scpn_quantum_control.phase.qnn_training` | Bounded data-reuploading phase-QNN classifier training with multi-frequency parameter-shift descent, prediction evidence, and accuracy certificates. |
| `scpn_quantum_control.phase.qnn_convergence` | Deterministic bounded-QNN convergence evidence with loss-drop thresholds, accuracy thresholds, parameter-shift evaluation accounting, multi-seed initial-condition envelopes, and unsuitable-scenario records. |
| `scpn_quantum_control.phase.qnn_finite_shot` | Seeded finite-shot simulator evidence for bounded-QNN gradients and noisy-gradient convergence with replay seeds, shot counts, uncertainty radii, and non-hardware claim boundaries. |
| `scpn_quantum_control.phase.qnn_framework_bridge_matrix` | Fail-closed support matrix for bounded phase-QNN framework bridges, separating implemented JAX/PyTorch/TensorFlow routes from arbitrary simulator autodiff and hardware-gradient gaps. |
| `scpn_quantum_control.phase.qnn_framework_agreement` | Caller-supplied QNN framework-gradient agreement checks for JAX/PennyLane/PyTorch/TensorFlow-style references, complemented by bounded native/tensor QNN bridge evidence where explicitly documented. |
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
| `scpn_quantum_control.phase.provider_gradient` | Provider callback parameter-shift execution plus policy-bound hardware-gradient preparation records that never submit QPU jobs. |
| `scpn_quantum_control.phase.provider_hardware_gradient_audit` | Executable audit suite for approved and blocked provider hardware-gradient preparation routes with zero hardware execution and zero produced hardware gradients. |
| `scpn_quantum_control.phase.hardware_gradient_policy` | Hardware-gradient preparation policy with provider/backend allowlists, shot/evaluation budget accounting, required evidence IDs, dry-run approval, and live-ticket gating. |
| `scpn_quantum_control.phase.hardware_gradient_campaign` | No-submit campaign specs for XY parameter-shift VQE and seeded SPSA hardware-gradient validation, including named Heron r2 backend allowlists, evidence IDs, shot/evaluation budgets, raw-count replay schemas, statevector-reference requirements, and policy-evaluated dry-run plans. |
| `scpn_quantum_control.phase.hardware_gradient_publication` | Publication package scaffold for the planned XY hardware-gradient paper, covering preregistration, methods sections, raw artefact map, draft claim-ledger rows, same-circuit benchmark placeholders, and no-submit claim boundaries. |
| `scpn_quantum_control.phase.gradient_support_matrix` | Executable support planning for gates, observables, backends, transforms, and ML/provider adapters. |
| `scpn_quantum_control.phase.transform_nesting` | Fail-closed transform-nesting planner for local, tape, ML-adapter, vectorized, and hardware gradient routes. |
| `scpn_quantum_control.phase.provider_gradient_audit` | Executable provider-gradient readiness audit for deterministic, finite-shot, multi-frequency, hardware-blocked, unknown-backend, and malformed-sample routes. |
| `scpn_quantum_control.phase.provider_hardware_safety_audit` | Aggregate differentiable provider/hardware safety gate over provider-gradient readiness, provider hardware-gradient preparation, provider QNode transforms, QNode tape records, and hardware-gradient campaign readiness. It verifies zero hardware execution and zero hardware-gradient production, then keeps promotion blocked until live-ticket, raw-count replay, calibration snapshot, statevector comparison, and isolated benchmark artefacts are attached. |
| `scpn_quantum_control.phase.gradient_tape` | Context-managed recording of supported deterministic and finite-shot quantum-gradient evaluations. |
| `scpn_quantum_control.phase.qnode_tape` | QNode-style differentiable tape records for supported phase objectives, seeded finite-shot replay, and provider-boundary routes that fail closed before hardware submission. |
| `scpn_quantum_control.phase.qnode_circuit` | Registered local Phase-QNode statevector and density-matrix circuit family with supported gates, bounded single-qubit Kraus noise channels, controlled-gate decomposition helpers, arbitrary-depth registered circuit builders with deterministic depth/resource profiles, multi-qubit template constructors, dense Hermitian observables, Pauli observables, Pauli covariance observables, sparse Pauli Hamiltonians, sparse Ising-chain Hamiltonian construction, gate-aware parameter-shift evaluation planning, parameter-shift gradients for pure-state routes, exact computational-basis classical Fisher metrics, pure-state QFI/Fubini-Study metrics, natural-gradient metric providers, and strict route support reports for value, density, gradient, metric, and Fisher paths. |
| `scpn_quantum_control.phase.qnode_framework_parity` | Bounded real-framework parity suite for SCPN, JAX, PyTorch, TensorFlow, and PennyLane with dependency-sparse classifications. |
| `scpn_quantum_control.phase.qnode_affinity_benchmark` | Affinity-labelled local benchmark metadata harness for registered Phase-QNode execution, including raw timing rows and host isolation context. |
| `scpn_quantum_control.phase.qnode_transforms` | Executable scalar local QNode transform evidence for `grad`, `value_and_grad`, `hessian`, `hessian_vector_product`, `jvp`, `vjp`, `jacfwd`, and `jacrev`, with real-only complex/W boundaries and fail-closed vectorized/provider/framework-native boundaries. |
| `scpn_quantum_control.phase.qnode_vector_transforms` | Executable deterministic native vector-output QNode `jvp`, `vjp`, vector Hessian tensor, Jacobian evidence for `jacfwd`/`jacrev`, plus host-side manual `vmap(grad)` over scalar local parameter-shift objectives, with real-only complex/W boundaries and fail-closed finite-shot, hardware, provider, and framework-native vectorization boundaries. |
| `scpn_quantum_control.phase.qnode_provider_transforms` | Provider-callback QNode transform evidence for scalar `grad`, `value_and_grad`, `jvp`, `vjp`, scalar `jacfwd`/`jacrev`, and manual `vmap(grad)` with shifted-sample records, finite-shot uncertainty propagation, and fail-closed hardware policy. |
| `scpn_quantum_control.phase.qiskit_bridge` | Qiskit shifted-circuit generation, deterministic local Statevector parameter-shift gradients, finite-shot surrogate uncertainty, and a maturity audit that aggregates no-submit provider hardware-gradient preparation while keeping live QPU execution, raw-count replay, live calibration/statevector comparison, and isolated benchmark promotion blocked. |
| `scpn_quantum_control.differentiable_framework_overlay` | CPU-only overlay manifest, installer, verifier, and CLI for reproducible JAX, PyTorch, TensorFlow, and PennyLane parity environments. |
| `scpn_quantum_control.benchmarks.differentiable_external_comparison` | External comparison rows and JSON artefact writing for JAX `value_and_grad`/`vmap` support, PyTorch `torch.func`, TensorFlow `GradientTape`, PennyLane QNodes, and optional LLVM/Enzyme runner AD with strict JSON, timeout, toolchain, correctness gates, dependency-version metadata, and explicit hard-gap rows for unsupported batching, transform, dtype, and hardware-device routes. |
| `scpn_quantum_control.benchmarks.differentiable_evidence` | CI benchmark evidence writer with runner metadata, CPU affinity, host-load, governor/frequency, heavy-job, explicit accelerator metadata, silent CPU-fallback detection, classification, and artefact-ID fields. |
| `scpn_quantum_control.differentiable_claim_ledger` | Claim-ledger parser, Markdown renderer, and validation helpers that prevent promoted claims without artefact and benchmark IDs. |
| `scpn_quantum_control.phase.jax_bridge` | Optional JAX host-callback adapter for supported phase parameter-shift value-and-gradient calls plus bounded native/custom-VJP JAX phase-QNN evidence and audited no-host-callback JIT/VMAP/PMAP/PyTree boundaries for that narrow model. |
| `scpn_quantum_control.phase.pennylane_bridge` | Optional PennyLane gradient-agreement checker for caller-supplied PennyLane/QNode gradient functions. |
| `scpn_quantum_control.phase.torch_bridge` | Optional PyTorch bridge for supported phase parameter-shift value-and-gradient calls, tensor-ready bounded phase-QNN analytic gradient evidence, bounded custom `torch.autograd.Function`, bounded `torch.func.grad`/`vmap`/`jacrev`, bounded `torch.compile`, bounded `nn.Module`/layer wrapper compatibility, and a fail-closed registered Phase-QNode Torch-lowering matrix checked against parameter-shift references and promotion blockers. |
| `scpn_quantum_control.phase.tensorflow_bridge` | Optional TensorFlow tensor bridge for supported phase parameter-shift value-and-gradient calls plus tensor-ready bounded phase-QNN analytic gradient evidence checked against parameter-shift references. |
| `scpn_quantum_control.compiler.mlir` | Compiler/program AD lowering, native executable kernel helpers, Phase-QNode MLIR-runtime execution adapters, support-profile reports, and the Enzyme/MLIR maturity audit that records executable SCPN MLIR-runtime correctness, native LLVM/JIT support metadata, local toolchain versions, and hard gaps until native Enzyme plus isolated benchmark artefacts exist. |

## Common objects

| Object family | Examples | Use |
|---|---|---|
| Unified API evidence | `UnifiedDifferentiableAPIResult`, `DifferentiabilityDiagnosticReport`, `differentiable_api`, `differentiable_value`, `differentiable_gradient`, `differentiable_jacobian`, `differentiable_hessian`, `differentiable_support_report`, `explain_differentiability`, `differentiable_compile_report`, `differentiable_benchmark_report` | Use one JSON-ready envelope across scalar values, gradients, Jacobians, Hessians, fail-closed support decisions, differentiability diagnostics, compiler planning, and local conformance evidence. |
| Primitive identity and rules | `PrimitiveIdentity`, `PrimitiveContract`, `CustomDerivativeRule`, `CustomDerivativeRegistry` | Bind derivative, batching, lowering, shape, dtype, and nondifferentiability rules to supported primitives. |
| Forward and reverse AD results | `GradientResult`, `JacobianResult`, `HessianResult`, `JVPResult`, `HVPResult`, `ProgramADAdjointResult` | Return structured derivative outputs and diagnostics. |
| Optimisation helpers | `DifferentiableOptimizer`, `NaturalGradientOptimizer`, `LevenbergMarquardtOptimizer` | Drive supported differentiable objectives. |
| Compiler-backed kernels | `compile_*_ad_to_native_llvm_jit`, `compile_whole_program_ad_trace_to_native_llvm_jit`, `compile_phase_qnode_circuit_to_mlir_runtime`, `native_whole_program_ad_linalg_support` | Execute bounded native AD kernels where support reports allow it, including verified static dense determinant lowering through `19x19`, fail-closed `20x20+` reports, and a verified SCPN MLIR-runtime adapter for registered local Phase-QNode value/gradient execution with shape/type checks and blocked interpreter-fallback success claims. |
| Backend and shot planning | `QuantumGradientPlan`, `QuantumGradientBackendCapability`, `ShotAllocationResult`, `GradientFailurePolicy`, `StochasticGradientConfidenceInterval`, `ParameterShiftSampleRecord`, `gradient_confidence_interval`, `SPSAObjectiveSample`, `SPSAProbeRecord`, `SPSAGradientResult`, `spsa_gradient_estimate`, `ScoreFunctionSampleRecord`, `ScoreFunctionGradientResult`, `score_function_gradient_estimate`, support-profile records | Select supported local gradient methods, propagate finite-shot uncertainty with confidence intervals, shifted-sample records, fail-closed policy metadata, and explicit no-hardware claim boundaries, run seeded local SPSA probes over caller-supplied objectives, estimate materialised likelihood-ratio score-function gradients, and fail closed for unsafe hardware routes. |
| Hardware-gradient campaign readiness | `HardwareGradientCampaignSpec`, `HardwareGradientReplaySchema`, `HardwareGradientCampaignPlan`, `HardwareGradientCampaignSuite`, `default_hardware_gradient_campaign_specs`, `plan_hardware_gradient_campaign`, `run_hardware_gradient_campaign_readiness_suite` | Prepare no-submit XY hardware-gradient validation campaigns for parameter-shift VQE and seeded SPSA routes with backend allowlists, live-ticket gates, evidence IDs, shot budgets, calibration snapshot requirements, raw-count replay schemas, statevector references, and policy decisions that preserve `hardware_execution == False` until live artefacts exist. |
| Hardware-gradient publication package | `HardwareGradientPublicationPackage`, `HardwareGradientPreregistration`, `HardwareGradientMethodSection`, `HardwareGradientArtifactMapEntry`, `HardwareGradientClaimLedgerRow`, `HardwareGradientBenchmarkPlaceholder`, `build_hardware_gradient_publication_package` | Produce a JSON-ready and Markdown-ready publication scaffold for the planned XY hardware-gradient paper while keeping claim rows unpromoted and rejecting injected live-result claims in the no-submit package. |
| Gradient support matrix | `GradientSupportCapability`, `GradientSupportPlan`, `GradientSupportMatrixAuditResult`, `gradient_support_capability`, `list_gradient_support_capabilities`, `plan_gradient_support`, `assert_gradient_support`, `run_gradient_support_matrix_audit` | Decide whether a gate, observable, backend, transform, and adapter combination is supported before execution; blocked combinations carry reasons and alternatives. |
| Transform nesting | `GradientTransformNestingPlan`, `GradientTransformNestingAuditResult`, `plan_gradient_transform_nesting`, `assert_gradient_transform_nesting_supported`, `run_gradient_transform_nesting_audit` | Decide whether transform stacks such as `grad`, `value_and_grad`, `hessian`, `grad` of `grad`, tape, native manual `vmap(grad)`, JVP/VJP, provider-callback routes, adapter bridges, or hardware routes are safe before execution. |
| Gradient audit evidence | `DifferentiableQuantumAuditReport`, `DifferentiableWorkflowAuditSuiteResult`, `FiniteShotGradientAuditResult`, `MLFrameworkGradientAuditSuiteResult`, `ParameterShiftAnalyticAgreement`, `PhaseGradientBenchmarkSuiteResult`, `ProviderGradientReadinessAuditResult`, `run_differentiable_workflow_audit_suite`, `run_finite_shot_gradient_uncertainty_audit`, `run_ml_framework_gradient_audit`, `run_known_phase_gradient_audit`, `run_parameter_shift_audit_suite`, `run_phase_gradient_benchmark_suite`, `run_provider_gradient_readiness_audit` | Bundle finite-difference agreement, finite-shot uncertainty containment, optional ML-framework parity, analytic-gradient agreement, convergence evidence, coupling-learning checks, provider-readiness checks, and multi-case phase-gradient conformance into reviewer-facing reports. |
| Gradient-training evidence | `ParameterShiftTrainingResult`, `ParameterShiftTrainingCertificate`, `ParameterShiftNaturalGradientResult`, `ParameterShiftNaturalGradientCertificate`, `ParameterShiftQNNTrainingResult`, `ParameterShiftQNNPredictionResult`, `ParameterShiftQNNMultiSeedConvergenceSuiteResult`, `ParameterShiftQNNLossLandscapeSuiteResult`, `QNNOptimizerBaselineResult`, `GenericParameterShiftEvaluationPlan`, `plan_generic_parameter_shift_evaluations`, `DifferentiableDomainBenchmarkDatasetSuite`, `DifferentiableDomainBenchmarkValidationSuite`, `OptimizerComparisonSuiteResult`, `OptimizerConvergenceRecord`, `ParamShiftVQEResult`, `ParamShiftConvergenceDiagnostics` | Certify accepted value descent, metric-aware descent, bounded phase-QNN classification, deterministic multi-seed convergence envelopes, bounded loss-landscape scans, named QNN optimizer baseline evidence, exact-answer domain dataset validation, optimizer comparison evidence, opaque-callable 2N fallback planning, line-search behaviour, exact-gap metadata, and parameter-shift evaluation counts. |
| Objective composition | `ComposedPhaseObjective`, `ObjectiveTerm`, `ObjectiveGradientEvaluation`, `ComposedObjectiveTrainingResult`, `ComposedObjectiveGradientAgreement`, `ComposedObjectiveAuditSuiteResult`, `ComposedObjectiveExecutionPlan`, `ComposedObjectivePlannerAuditResult`, `build_phase_control_objective`, `train_composed_phase_objective`, `verify_composed_objective_gradient`, `run_composed_objective_audit_suite`, `plan_composed_objective_execution`, `run_composed_objective_planner_audit` | Combine energy, fidelity, periodic regularization, symmetry, and smooth safety penalties without misclassifying analytic classical penalties as parameter-shift quantum terms. |
| Coupling-learning evidence | `CouplingLearningResult`, `CouplingGradientVerificationResult`, `learn_couplings_from_observations`, `verify_coupling_parameter_shift_gradient` | Learn symmetric oscillator couplings from parameter-shift-compatible observation models and independently check small smooth gradients against central finite differences. |
| QSNN training evidence | `QSNNTrainingRun`, `QSNNParameterShiftDescentRun` | Attach parameter-shift traces and certificates to quantum neural network training loops. |
| Registered model-training evidence | `DifferentiableModelTrainingEvidenceSuite`, `DifferentiableModelTrainingRecord`, `RegisteredDifferentiableTrainingSuiteAuditResult`, `RegisteredDifferentiableTrainingSuiteRecord`, `run_differentiable_model_training_evidence_suite`, `run_registered_differentiable_training_suite_audit` | Package seeded QNN, QGNN, QSNN, and Kuramoto-XY local training cases with loss reduction and gradient-agreement evidence, then audit the requested training-suite lanes so open-system control and inverse-coupling recovery remain explicitly blocked until dedicated evidence exists. |
| Registered Phase-QNode circuit evidence | `PhaseQNodeCircuit`, `PhaseQNodeDensityCircuit`, `PhaseQNodeNoiseChannel`, `PhaseQNodeDepthProfile`, `PhaseQNodeRegisteredCircuitSpec`, `PhaseQNodeTemplateSpec`, `PhaseQNodeGradientEvaluationPlan`, `PhaseQNodeGradientEvaluationGroup`, `build_registered_phase_qnode_circuit`, `phase_qnode_depth_profile`, `build_phase_qnode_template`, `build_sparse_ising_chain_hamiltonian`, `registered_phase_qnode_templates`, `registered_phase_qnode_decompositions`, `registered_phase_qnode_noise_channels`, `decompose_phase_qnode_controlled_gate`, `DenseHermitianObservable`, `PauliTerm`, `PauliCovarianceObservable`, `SparsePauliHamiltonian`, `PhaseQNodeClassicalFisherResult`, `PhaseQNodeDensityExecutionResult`, `PhaseQNodeMetricTensorResult`, `plan_phase_qnode_parameter_shift_evaluations`, `execute_phase_qnode_circuit`, `execute_phase_qnode_density_matrix`, `parameter_shift_phase_qnode_gradient`, `phase_qnode_gradient_support_report`, `phase_qnode_metric_support_report`, `phase_qnode_computational_basis_fisher_information`, `phase_qnode_computational_basis_fisher_support_report`, `phase_qnode_density_support_report`, `phase_qnode_quantum_fisher_information`, `phase_qnode_natural_gradient_metric`, `phase_qnode_support_report`, `run_phase_qnode_framework_parity_suite`, `run_phase_qnode_affinity_benchmark` | Execute the declared local gate/observable family, including arbitrary-depth registered circuits with depth/resource budget gates, registered GHZ-chain and hardware-efficient multi-qubit templates, controlled-H/S/T plus Toffoli/CCZ/Fredkin gates, exact Toffoli/Fredkin operation-list decompositions, sparse Ising-chain Hamiltonian construction with scalar or site/edge coefficients, density-matrix execution through `bit_flip`, `phase_flip`, `depolarizing`, and `amplitude_damping` Kraus channels, dense Hermitian expectations, exact Pauli covariance values, gate-aware logical-parameter shift planning with multi-frequency repeated-parameter fallback, product-rule covariance gradients for pure states, exact computational-basis classical Fisher metrics, and pure-state Fubini-Study/QFI metrics; inspect strict support reports before blocked gradient, metric, Fisher, density, or singular-probability paths; compare installed framework parity, record benchmark-isolation metadata, and fail closed for unsupported routes. |
| Rust differentiable parity kernels | `phase_qnode_fubini_study_metric_rust`, `phase_qnode_computational_basis_fisher_rust`, `phase_qnode_vector_jvp_rust`, `phase_qnode_vector_vjp_rust`, `phase_qnode_hessian_vector_product_rust`, `phase_qnode_vector_hessian_tensor_rust`, `phase_qnode_complex_derivative_contract_rust`, `parameter_shift_gradient_uncertainty_rust`, `spsa_gradient_rust`, `score_function_gradient_rust`, `gradient_confidence_interval_rust` | Optional PyO3 parity surface for the promoted deterministic local metric, directional-transform, vector-Hessian, real-only complex-boundary, materialised finite-shot uncertainty, materialised SPSA-record, materialised score-function, and confidence-policy primitives. The kernels operate on materialised state derivatives, Jacobians, Hessians, vector Hessian tensors, shifted means, variances, shot counts, coefficients, SPSA perturbations, rewards, score vectors, gradients, standard errors, or trainable masks and are checked against the Python APIs. They do not execute provider callbacks or hardware jobs. |
| Differentiable promotion evidence | `FrameworkOverlayManifest`, `FrameworkOverlayVerification`, `install_framework_overlay`, `verify_framework_overlay_path`, `BenchmarkIsolationMetadata`, `ExternalComparisonArtifact`, `ExternalComparisonRow`, `run_differentiable_external_comparison_suite`, `write_differentiable_external_comparison`, `load_claim_ledger`, `validate_claim_ledger` | Reproduce the CPU framework overlay, produce CI-only benchmark bundles, compare external AD frameworks, write non-promotional external-comparison artefacts, and validate that Phase-QNode claims have implementation, tests, docs, known gaps, artefact IDs, and benchmark IDs. |
| Bounded QNN framework bridge matrix | `BoundedQNNFrameworkBridgeCapability`, `BoundedQNNFrameworkBridgeMatrixResult`, `run_bounded_qnn_framework_bridge_matrix`, `assert_bounded_qnn_framework_bridge_supported` | Declare implemented bounded JAX/PyTorch/TensorFlow bridge routes, including the bounded JAX custom-VJP route, bounded PyTorch custom-autograd route, bounded PyTorch `torch.func` compatibility route, bounded PyTorch `torch.compile` route, bounded PyTorch module/layer wrapper route, bounded TensorFlow `GradientTape` route, bounded TensorFlow `tf.function` route, bounded TensorFlow XLA route, and bounded TensorFlow Keras layer route, and fail closed for arbitrary simulator autodiff or live provider hardware-gradient routes. |
| Optional JAX bridge | `PhaseJAXParameterShiftResult`, `PhaseJAXNativeQNNGradientResult`, `PhaseJAXCustomVJPQNNGradientResult`, `PhaseJAXJITCompatibilityResult`, `PhaseJAXVMAPCompatibilityResult`, `PhaseJAXShardingCompatibilityResult`, `PhaseJAXPyTreeCompatibilityResult`, `PhaseJAXNestedTransformRoute`, `PhaseJAXNestedTransformAlgebraResult`, `PhaseJAXPhaseQNodeLoweringRoute`, `PhaseJAXPhaseQNodeLoweringMatrixResult`, `PhaseJAXMaturityAuditResult`, `jax_parameter_shift_value_and_grad`, `jax_native_qnn_value_and_grad`, `jax_custom_vjp_qnn_value_and_grad`, `run_jax_jit_compatibility_audit`, `run_jax_vmap_compatibility_audit`, `run_jax_sharding_compatibility_audit`, `run_jax_pytree_compatibility_audit`, `run_jax_nested_transform_algebra_audit`, `run_jax_phase_qnode_lowering_matrix`, `run_jax_maturity_audit`, `is_phase_jax_available` | Expose phase parameter-shift value-and-gradient calls to JAX workflows through an explicit host-callback boundary, expose native JAX autodiff evidence for the bounded phase-QNN classifier, expose a bounded JAX `custom_vjp` route whose backward rule is checked against the SCPN parameter-shift gradient, report JIT/VMAP/PMAP/PyTree and bounded nested-transform algebra compatibility, provide a fail-closed registered Phase-QNode JAX-lowering matrix, and aggregate a maturity audit that keeps arbitrary simulator lowering, full arbitrary Phase-QNode `jacfwd`/`jacrev`/Hessian algebra, provider callbacks, hardware gradients, and promotion-grade benchmarks blocked until artefacts exist. |
| Optional PennyLane bridge | `PennyLaneGradientAgreementResult`, `PennyLaneQNodeConversionResult`, `PennyLaneRoundTripResult`, `PennyLaneMaturityAuditResult`, `check_pennylane_parameter_shift_agreement`, `build_pennylane_qnode_from_phase_qnode`, `check_pennylane_phase_qnode_round_trip`, `check_pennylane_qnode_round_trip`, `run_pennylane_maturity_audit`, `is_phase_pennylane_available` | Compare SCPN parameter-shift gradients against caller-supplied PennyLane callables, generate bounded PennyLane QNodes from registered local `PhaseQNodeCircuit` declarations with explicit device, shot, and diff-method metadata, and aggregate agreement/export/import evidence plus grouped parameter-shift evaluation counts while keeping plugin, provider, hardware, and isolated-benchmark promotion blocked until artefacts exist. |
| Optional PyTorch bridge | `PhaseTorchParameterShiftResult`, `PhaseTorchQNNGradientResult`, `PhaseTorchAutogradQNNGradientResult`, `PhaseTorchFuncCompatibilityResult`, `PhaseTorchCompileCompatibilityResult`, `PhaseTorchModuleWrapperAuditResult`, `PhaseTorchPhaseQNodeLoweringRoute`, `PhaseTorchPhaseQNodeLoweringMatrixResult`, `PhaseTorchMaturityAuditResult`, `torch_parameter_shift_value_and_grad`, `torch_bounded_qnn_value_and_grad`, `torch_autograd_qnn_value_and_grad`, `run_torch_func_compatibility_audit`, `run_torch_compile_compatibility_audit`, `torch_bounded_qnn_module`, `torch_bounded_qnn_layer`, `run_torch_module_wrapper_audit`, `run_torch_phase_qnode_lowering_matrix`, `run_torch_maturity_audit`, `is_phase_torch_available` | Convert supported phase parameter-shift value-and-gradient outputs into PyTorch tensors, provide bounded phase-QNN tensor-gradient evidence, expose a bounded custom `torch.autograd.Function` path, audit bounded `torch.func.grad`/`vmap`/`jacrev`, `torch.compile`, and module/layer wrapper compatibility, and aggregate those routes plus a fail-closed registered Phase-QNode lowering matrix into a maturity audit that keeps arbitrary statevector lowering, finite-shot lowering, provider callbacks, hardware lowering, dynamic circuits, full compiler/autograd integration, live overlay execution, and isolated benchmark promotion blocked until artefacts exist. |
| Optional TensorFlow bridge | `PhaseTensorFlowParameterShiftResult`, `PhaseTensorFlowQNNGradientResult`, `PhaseTensorFlowGradientTapeCompatibilityResult`, `PhaseTensorFlowFunctionCompatibilityResult`, `PhaseTensorFlowXLACompatibilityResult`, `PhaseTensorFlowKerasLayerWrapperAuditResult`, `PhaseTensorFlowMaturityAuditResult`, `tensorflow_parameter_shift_value_and_grad`, `tensorflow_bounded_qnn_value_and_grad`, `run_tensorflow_gradient_tape_compatibility_audit`, `run_tensorflow_function_compatibility_audit`, `run_tensorflow_xla_compatibility_audit`, `tensorflow_bounded_qnn_keras_layer`, `run_tensorflow_keras_layer_wrapper_audit`, `run_tensorflow_maturity_audit`, `is_phase_tensorflow_available` | Convert supported phase parameter-shift value-and-gradient outputs into TensorFlow tensors, provide bounded phase-QNN tensor-gradient evidence, audit bounded `GradientTape`/`tf.function`/XLA/Keras layer gradients against parameter-shift references, and aggregate those routes into a maturity record that keeps arbitrary Phase-QNode TensorFlow lowering, full graph autodiff-through-simulator, provider callbacks, hardware gradients, and isolated benchmark promotion blocked until artefacts exist. |

## Unified façade

```python
import numpy as np

from scpn_quantum_control import (
    differentiable_api,
    differentiable_compile_report,
    differentiable_gradient,
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
```

`UnifiedDifferentiableAPIResult` is the stable evidence envelope for the façade.
It always carries `operation`, `supported`, `fail_closed`, `method`, derivative
arrays when applicable, a route-specific `payload`, and a claim boundary.
`DifferentiabilityDiagnosticReport` is the reviewer-facing explanation surface:
it carries the request, blocked reasons, suggested alternatives, dependency rows
for bounded framework bridges, device capability rows, backend planning rows,
and the underlying support-plan payload. The diagnostic route is planning
evidence only; it does not execute objectives, provider callbacks, hardware
jobs, or performance benchmarks.

The façade delegates to existing implemented surfaces rather than weakening
their contracts: finite-difference gradients, Jacobians, and Hessians remain
local diagnostic routes; support reports fail closed for unsupported gate,
observable, backend, transform, or adapter combinations; compile reports are
compiler-planning and MLIR interchange evidence unless the selected primitive
plan has an executable backend; benchmark reports are local conformance rows,
not isolated performance, provider, or hardware execution evidence.

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

The bounded phase-QNN classifier has a narrow native JAX autodiff route and a
PyTorch tensor-gradient evidence route:

```python
import numpy as np

from scpn_quantum_control.phase import (
    jax_custom_vjp_qnn_value_and_grad,
    jax_native_qnn_value_and_grad,
    run_jax_jit_compatibility_audit,
    run_jax_maturity_audit,
    run_jax_pytree_compatibility_audit,
    run_jax_sharding_compatibility_audit,
    run_jax_vmap_compatibility_audit,
    run_torch_compile_compatibility_audit,
    run_torch_func_compatibility_audit,
    run_torch_module_wrapper_audit,
    run_tensorflow_function_compatibility_audit,
    run_tensorflow_gradient_tape_compatibility_audit,
    run_tensorflow_keras_layer_wrapper_audit,
    run_tensorflow_xla_compatibility_audit,
    tensorflow_bounded_qnn_value_and_grad,
    tensorflow_bounded_qnn_keras_layer,
    torch_autograd_qnn_value_and_grad,
    torch_bounded_qnn_value_and_grad,
    torch_bounded_qnn_layer,
    torch_bounded_qnn_module,
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
`run_jax_maturity_audit` aggregates the bounded custom-VJP, JIT, VMAP,
PMAP/sharding, and PyTree audits into one reviewer-facing record. It reports
`bounded_model_ready=True` when those bounded routes pass, but keeps
`ready_for_provider_exceedance=False` until arbitrary quantum-kernel JAX
lowering, full transform-nesting algebra, hardware/provider callback transform
safety, and isolated benchmark artefacts exist.
`torch_bounded_qnn_value_and_grad` returns framework tensors from the analytic
bounded-model gradient, while `torch_autograd_qnn_value_and_grad` wraps the same
bounded model in a custom `torch.autograd.Function` and checks its backward rule
against the parameter-shift reference. `run_torch_func_compatibility_audit`
checks bounded `torch.func.grad`, `torch.func.vmap`, and `torch.func.jacrev`
outputs against single-row and batched parameter-shift references.
`run_torch_compile_compatibility_audit` compiles the bounded PyTorch loss route
and checks the resulting gradient against the same parameter-shift reference.
`torch_bounded_qnn_module` and `torch_bounded_qnn_layer` expose the same bounded
loss through a PyTorch `nn.Module`/layer wrapper, and
`run_torch_module_wrapper_audit` checks the wrapper gradient against the same
reference.
`run_torch_maturity_audit` aggregates the bounded analytic tensor,
custom-autograd, `torch.func`, `torch.compile`, and module/layer wrapper routes
into one reviewer-facing record. It reports `bounded_model_ready=True` only when
those bounded routes pass, but keeps `ready_for_provider_exceedance=False` until
live overlay execution, arbitrary registered Phase-QNode Torch lowering, full
compiler/autograd integration, and promotion-grade isolated benchmark artefacts
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
from scpn_quantum_control.phase import run_ml_framework_gradient_audit


ml = run_ml_framework_gradient_audit()

print(ml.audit_passed)
print(ml.executed_frameworks)
print(ml.unavailable_frameworks)
print(ml.blocked_frameworks)
print(ml.failed_frameworks)
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
`run_pennylane_maturity_audit` aggregates caller-supplied gradient agreement,
caller-supplied QNode value/gradient parity, generated Phase-QNode export
round-trip parity, optional PennyLane tape import round-trip parity, device
metadata, shot policy, diff method, and grouped registered Phase-QNode
parameter-shift evaluation counts. It reports `identical_circuit_ready=True`
only when the import tape is supplied and all bounded agreement/export/import
routes pass; `ready_for_provider_exceedance` remains false until plugin-matrix,
provider-plugin, hardware-execution, and isolated-benchmark artefacts exist.

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
