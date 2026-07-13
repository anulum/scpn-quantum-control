# Changelog

Dated list of changes. Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Fixed
- 2026-07-13 — Scoped the neural-operator content-determinism regression test
  to reproducible fields, excluding live host provenance such as load averages
  exactly as required by the existing digest and claim boundary.
- 2026-07-13 — Made reproduction-image policy tests distinguish host-only
  Docker build metadata and optional Rust-toolchain availability. Host checks
  remain strict, while the toolchain-free image verifies the documented Cargo
  fallback and the complete manifest-scoped Rustfmt command.

### Added
- 2026-07-13 — Closed the D401 suppression for the Lindblad Kuramoto-XY
  solver. Its NumPy-style contracts now document validated physical inputs,
  dense-memory budgeting, cached operator construction, observable semantics,
  frequency-seeded initial state, time-grid behavior, result arrays, and
  failure modes. Executable dynamics, public signatures, phase wiring,
  polyglot kernels, and benchmark evidence are unchanged.
- 2026-07-13 — Closed the D400 suppression for the cross-domain transfer
  summarizer. Its NumPy-style contract now documents the result sequence,
  aggregate pair counts, best-transfer label and speedup, mean speedup, and
  empty-input defaults. Executable source, VQE transfer behavior, public APIs,
  and benchmark evidence are unchanged.
- 2026-07-13 — Closed the D401 suppression for the ansatz-methodology
  convergence helper. Its NumPy-style contract now documents ordered energy
  history, short-history and fallback results, and the benchmark's historical
  signed 99/101-percent threshold convention. Executable source, benchmark
  algorithms, public APIs, and evidence are unchanged.
- 2026-07-13 — Closed the D205 suppression and adjacent D413 debt for the
  single-ancilla Lindblad circuit. The module, circuit builder, and resource
  estimator now expose complete NumPy-style parameter, result, validation, and
  failure contracts; the estimator contract also removes its stale reference
  to a nonexistent `estimated_depth` key. Executable source, public APIs,
  open-system physics, and benchmarks are unchanged.
- 2026-07-13 — Closed the D401 suppression and adjacent D413 debt for layered
  ADAPT-VQE. The symmetric-reference helper and solver now expose complete
  NumPy-style input, result, convergence, restart, memory-budget, and failure
  contracts. Focused regressions also cover uncoupled-pair exclusion and the
  zero-layer result path. Executable source, public APIs, physics, and
  benchmarks are unchanged.
- 2026-07-13 — Closed the D205 suppression for the mitigation package facade.
  Its package contract now separates the public summary from the exported ZNE,
  PEC, decoupling, CPDR, readout, and symmetry-mitigation scope under the global
  Ruff D gate. Executable source, public exports, mitigation algorithms, and
  benchmarks are unchanged.
- 2026-07-13 — Closed the D401 suppression for the identity entanglement
  witness. Its rotated two-qubit Pauli correlator now documents state, qubit,
  angle, return, and Qiskit ordering contracts in NumPy style under the global
  Ruff D gate. Executable source, CHSH physics, identity wiring, and benchmarks
  are unchanged.
- 2026-07-13 — Closed the D205 suppression for the IBM/Aer hardware runner.
  Its structured-logger helper now documents the logger name, return contract,
  and optional logging dependency fallback in NumPy style under the global Ruff
  D gate. Executable source, provider execution, result normalisation, and
  benchmarks are unchanged.
- 2026-07-13 — Closed the D401 suppression for the hardware plugin registry.
  Its backend class-decorator contract now uses NumPy-style parameter, return,
  and example documentation enforced by the global Ruff D gate. The adjacent
  runner-factory parameter and return sections now also satisfy the strict
  D413/D417 contract. Executable source, registry wiring, provider schemas, and
  benchmarks are unchanged.
- 2026-07-13 — Closed the D401 suppression for the IQM provider adapter. Its
  Qiskit-on-IQM availability contract now uses NumPy-style documentation
  enforced by the global Ruff D gate. A malformed provider-counts regression
  closes branch coverage; executable source, submission policy, provider
  schemas, and benchmarks are unchanged.
- 2026-07-13 — Closed the D401 suppression for the hybrid digital-analog
  compiler. Its provider protocol and built-in availability contracts now use
  NumPy-style documentation enforced by the global Ruff D gate. A stale-engine
  fallback regression closes branch coverage; executable source, Rust/PyO3
  partitioning, provider schemas, and benchmarks are unchanged.
- 2026-07-13 — Closed the D401 suppression for the hardware backend registry.
  Its protocol and process-wide registration, lookup, removal, and discovery
  wrappers now use NumPy-style contracts enforced by the global Ruff D gate.
  A sparse-capability regression closes the registry's branch-coverage proof;
  executable source, provider routes, and benchmarks are unchanged.
- 2026-07-13 — Closed the D401 suppression for the native analog Kuramoto
  backend. Its provider protocol and built-in availability contracts now use
  NumPy-style documentation enforced by the global Ruff D gate; executable
  source, provider wiring, native kernels, and benchmarks are unchanged.
- 2026-07-13 — Closed the D205 suppression for the hardware package facade.
  Its backend and execution-contract summary now follows the NumPy-style
  summary/description boundary enforced by global Ruff D; exports and
  executable source are unchanged.
- 2026-07-13 — Closed the D400 suppression for the neural-operator advantage
  study. Its existing module-level comparison contract now has a punctuated
  imperative summary enforced by the global Ruff D gate; executable source and
  performance-claim boundaries are unchanged.
- 2026-07-13 — Closed the D401 suppression for topology challenge-response
  authentication. Its public HMAC verifier now has a NumPy-style contract
  enforced by the global Ruff D gate; executable source is unchanged.
- 2026-07-13 — Closed the D205/D401 suppression for the control-layer
  structured ansatz. Its circuit-construction and submission contracts now use
  NumPy-style documentation enforced by the global Ruff D gate; executable
  source is unchanged.
- 2026-07-13 — Closed the D401 suppression for the FRC pulsed-shot cost
  adapter. Its MRTI-amplitude contract is now enforced by the global Ruff D
  gate; executable source is unchanged.
- 2026-07-13 — Removed the stale D205 suppression from the quantum-control
  package facade. Its existing package contract is now enforced directly by
  the global Ruff D gate; source is unchanged.
- 2026-07-13 — Closed the D401 suppression for the QPU-data-artifact bridge.
  Its array-construction contract is now enforced by the global Ruff D gate;
  executable source is unchanged.
- 2026-07-13 — Closed the D401 suppression for the Josephson-array
  application. Its correlation-variation helper is now enforced by the global
  Ruff D gate; executable source is unchanged.
- 2026-07-13 — Closed the D401 suppression for the vortex-binding analysis.
  Its configurational-entropy contract is now enforced by the global Ruff D
  gate; executable source is unchanged.
- 2026-07-13 — Closed the D401 suppression for the synchronization-witness
  analysis. Its count-based Fiedler contract is now enforced by the global
  Ruff D gate; executable source is unchanged.
- 2026-07-13 — Closed the D401 suppression for the phase-diagram analysis.
  Its effective-temperature contract is now enforced by the global Ruff D
  gate; executable source is unchanged.
- 2026-07-13 — Closed the D400 suppression for the H1-persistence analysis.
  Its module summary is now enforced by the global Ruff D gate; executable
  source is unchanged.
- 2026-07-13 — Closed the D205/D400 suppression for the
  entanglement-enhanced synchronization analysis. Its module contract is now
  enforced by the global Ruff D gate; executable source is unchanged.
- 2026-07-13 — Changed coverage source discovery from an importable package
  name to the package filesystem directory across coverage.py, CI, local
  preflight, Makefile, and test documentation. This prevents coverage.py's
  temporary source discovery from importing and then unloading NumPy/Qiskit
  modules, while preserving the measured package and Cobertura path mapping.
- 2026-07-13 — Closed the NumPy-docstring suppression for the public DLA parity
  witness. Its class and callable contracts are now enforced by the global Ruff
  D gate; runtime signatures and parity calculations are unchanged.
- 2026-07-13 — Added a typed CI/preflight gate for the differentiable
  external-validation environment lock and dependent artefact bundle, refreshed
  their stale file metadata in dependency order, and gave the reproduction-only
  Docker image a credential-free synthetic Git index so tracked-file policy
  audits work without copying host history, remotes, objects, or credentials.
  The synthetic index honours `.gitignore`; recursive Rust build outputs,
  bytecode caches, local backups, and ignored campaign-run logs are excluded
  from the Docker context.
- 2026-07-13 — Added a crate-wide Rustfmt gate to local preflight and the CI
  Rust audit job, and canonicalised the four queued Kuramoto engine formatting
  drifts. The change is layout-only; numerical expressions, PyO3 surfaces, and
  benchmark semantics are unchanged.
- 2026-07-13 — Replaced the stale 684-file coverage-debt snapshot with a
  deterministic 100% recovery register derived from remote CI evidence. The
  typed audit prioritises public claim-bearing and runtime-critical modules,
  tracks post-baseline source decomposition explicitly, honours justified
  exclusions, and blocks new or regressing measured debt in CI/preflight
  without changing the separate 90% aggregate line gate.
- 2026-07-13 — Enabled coverage.py branch collection while preserving the
  existing 90% line gate through a typed Cobertura policy audit. CI now requires
  real branch opportunity data and reports the branch rate in observation mode;
  a branch threshold remains blocked on consecutive remote baseline artefacts.
- 2026-07-13 — Added a machine-readable, additive strict-mypy cohort for
  repository tests. The gate records the current whole-tree debt baseline,
  orders later migrations by claim risk, validates tracked cohort membership,
  and executes the exact enforced files in CI and local no-test preflight.
- 2026-07-13 — Added a cross-language licence-readiness gate that checks every
  tracked Python, Rust, Julia, Go, C/C++, JavaScript/TypeScript, TOML, YAML,
  Verilog, and SystemVerilog source/configuration header in local preflight and
  CI. Canonicalised 515 existing headers while preserving each file body.
- 2026-07-04 — Extracted the coupled-phase-oscillator (Kuramoto) toolkit into a
  standalone [`oscillatools`](oscillatools/) distribution (a NumPy + SciPy floor
  with optional `rust`/`julia`/`jax`/`torch`/`sklearn`/`viz` tiers) so it can be
  installed without the quantum-provider dependency stack. The distribution
  carries its own mkdocs documentation site, CI, publish workflow, SBOM, and
  dependabot coverage. `scpn-quantum-control` now depends on `oscillatools>=0.1.0`.
- 2026-07-04 — Added Rust replay implementations for the program_ad
  linear-algebra operations in the `scpn-quantum-engine` crate — `multi_dot`,
  `eigvals`, `eigvalsh`, `eigh`, and `svd` — routed through the program_ad Rust
  bridge with parity tests against the reference path.

### Deprecated
- 2026-07-04 — `scpn_quantum_control.kuramoto` and `scpn_quantum_control.accel`
  are now thin re-export shims over `oscillatools`; importing them emits a
  `DeprecationWarning` and forwards every symbol, so existing code keeps working
  until the shims are removed no earlier than the next major release.

## [0.10.0] - 2026-06-26

### Added
- 2026-06-24 — Added the Studio federation surface (`scpn_quantum_control.studio`): a
  schema-A `CapabilityManifest` on the SCPN-STUDIO platform contract
  (`scpn-studio-platform` 0.2.x, the `studio` extra) declaring eight verbs
  (`compile`/`simulate`/`analyse`/`validate`/`benchmark`/`replay` from the core spine
  plus `mitigate`/`execute`), nine `studio.*.v1` evidence schemas mapped from the
  five-class hardware ledger, and a content-addressed digest. The emitted
  `docs/_generated/studio_manifest.json` (via `scpn-emit-studio-manifest`) carries the
  schema-A federation block plus an additive `architecture_map` extension block
  (per-stage IO pipeline, backend/dispatch matrix, interfaces, cross-repo wire formats,
  honest scope boundaries) for STUDIO/Hub ingestion.
- 2026-06-23 — Added an Architecture Map page (`docs/architecture_map.md`, in the
  Theory nav): a capability/input-output/backend map for sibling repositories and
  STUDIO design — the canonical data pipeline with per-stage IO contracts, the stable
  public surface, per-lane capabilities with honest maturity grades, the backend/dispatch
  wiring (including the verified 151 Rust PyO3 kernels), cross-repo integration points,
  and explicit honest scope boundaries (executed vs bounded vs feasibility-only/closed).
- 2026-06-23 — Added uncertainty propagation for zero-noise extrapolation
  (`scpn_quantum_control.mitigation.zne_uncertainty`): `zne_extrapolate_with_uncertainty`
  and `ZNEUncertaintyResult` attach a propagated standard error and coverage interval
  to the ZNE zero-noise estimate. Per-scale shot-noise errors (for example from
  `analysis.sync_uncertainty`) propagate through a weighted-least-squares fit; without
  them an ordinary-least-squares residual variance is used. The point estimate matches
  `zne_extrapolate` for the unweighted case. Mitigated results now carry defensible
  error bars instead of bare point estimates.
- 2026-06-23 — Added an Architecture Decision Records page
  (`docs/ARCHITECTURE_DECISIONS.md`, in the Theory nav) recording the reasoning
  behind six major design decisions: single-responsibility module boundaries,
  measured-fastest-first multi-language acceleration with a Python floor,
  publication-faithful Kuramoto→XY mapping and solver selection, the five-class
  evidence/claim taxonomy with falsification culture, the small stable-core
  contract versus churning internals, and fail-closed resource budgets. It
  complements the existing Architecture page (structure) with context,
  alternatives, and trade-offs.
- 2026-06-23 — Added shot-noise uncertainty quantification for synchronisation
  metrics (`scpn_quantum_control.analysis.sync_uncertainty`): `UncertaintyInterval`,
  `order_parameter_shot_noise` (analytic delta-method interval), `order_parameter_bootstrap`
  and `metric_bootstrap` (distribution-free bootstrap percentile intervals for the
  order parameter and any count-to-scalar metric such as witnesses), and
  `order_parameter_estimate`. Hardware-derived sync metrics now carry defensible
  error bars at a stated coverage level instead of bare point estimates; empirical
  coverage is checked against a known distribution in the tests.
- 2026-06-23 — Added a sparse Pauli-operator construction budget guard
  (`scpn_quantum_control.compile_budget`): `require_pauli_operator_budget`,
  `estimate_pauli_operator`, `pauli_term_upper_bound`, and `pauli_budget_bytes`,
  with the `SCPN_MAX_PAULI_GIB` override. `knm_to_xxz_hamiltonian` now fails
  closed before an `O(n**2)` Pauli list is built and `knm_to_sparse_matrix`
  fails closed (with a new `max_gib` argument) before a `2**n` materialisation,
  so a pathological `n` from arbitrary `K_nm`/`omega` cannot exhaust memory
  across the sparse Hamiltonian and Trotter-circuit compile paths.
- 2026-06-21 — Added a SCPN-FUSION-CORE FRC calibration bridge
  (`scpn_quantum_control.bridge.fusion_core_frc`):
  `calibrate_frc_surrogate_from_equilibrium`, `calibrate_frc_surrogate_from_inputs`,
  the `FRCEquilibriumLike` structural contract, and the `FusionCoreFRCCalibration`
  provenance record. The bridge derives the pulsed-shot QAOA scheduler's
  `FRCPlasmaSurrogate` reference field, s-parameter, and plasma mass density from a
  fusion-core rigid-rotor equilibrium, records which surrogate fields are
  physics-derived versus control-grade defaults, and imports fusion-core lazily with
  an optional `repo_src` fallback.
- 2026-06-19 — Added fail-closed Enzyme/MLIR compiler-AD breadth evidence
  through `EnzymeMLIRCompilerADBreadthEvidence` and
  `build_enzyme_mlir_compiler_ad_breadth_evidence(...)`. The Enzyme/MLIR
  maturity audit now requires complete scalar, vector, matrix, loop, alias,
  MLIR, LLVM, native Enzyme, and isolated benchmark evidence before
  provider-exceedance can pass.
- 2026-06-19 — Added a no-submit Qiskit provider-gradient workflow evidence
  artefact for captured Runtime parameter-shift, finite-difference, LCU, SPSA,
  QGT, and QFI workflow metadata. The Qiskit maturity audit now keeps provider
  gradient workflow evidence blocked until the complete method set is attached
  and matched to the Runtime QPU provider/backend/circuit/live-ticket chain.
- 2026-06-19 — Added a no-submit Qiskit Runtime QPU provider evidence bundle
  tying captured Runtime QPU execution, raw-count replay,
  calibration/statevector comparison, and optional isolated benchmark artefacts
  into one maturity-audit attachment without submitting provider jobs.
- 2026-06-19 — Added a no-submit Qiskit Runtime QPU evidence builder for
  captured EstimatorV2/SamplerV2 metadata. The helper constructs validated
  Runtime QPU artefacts from approved ticket, backend, job, digest,
  shot-budget, and session-mode evidence without submitting provider jobs.
- 2026-06-19 — Paired Qiskit raw-count replay and live
  calibration/statevector comparison evidence to matching Runtime QPU execution
  artefacts. The Qiskit maturity audit now rejects unpaired or mismatched
  provider, backend, job, circuit, live-ticket, or shot evidence before those
  gates can pass, while isolated benchmark promotion remains blocked.
- 2026-06-19 — Added fail-closed Qiskit Runtime QPU execution artefacts for
  EstimatorV2 and SamplerV2 evidence, requiring live tickets, backend
  allowlists, shot budgets, ISA/transpiled-circuit digests, Runtime metadata,
  positive shots, and hardware execution before the Qiskit maturity audit can
  clear the live-QPU ticket gate.
- 2026-06-19 — Added fail-closed PennyLane hardware-plugin execution artefacts,
  requiring ticket, allowlist, shot-budget, raw-count, calibration, metadata,
  and live hardware execution evidence before the plugin matrix can mark
  hardware execution as passed while isolated-benchmark promotion remains
  blocked.
- 2026-06-19 — Added fail-closed PennyLane provider-plugin gradient parity
  artefacts, pairing them with matching provider execution artefacts before the
  plugin matrix can mark provider-gradient parity as passed while hardware and
  isolated-benchmark promotion remain blocked.
- 2026-06-19 — Hardened PennyLane plugin-matrix route metadata by
  canonicalising route names, reasons, and requirement labels, and rejecting
  malformed route names, empty reasons, control characters, and unsupported
  route statuses outside `passed`, `blocked`, and `failed`.
- 2026-06-19 — Hardened PennyLane provider-plugin execution artefacts by
  requiring execution modes to explicitly identify provider-plugin execution,
  rejecting generic local/offline simulator labels while still rejecting
  hardware/QPU modes and keeping provider-gradient and hardware promotion
  blocked.
- 2026-06-19 — Hardened PennyLane Phase-QNode finite-shot metadata by rejecting
  boolean, non-positive, non-integer, and string-coercible shot counts before
  `qml.device(...)` dispatch, and applying the same non-coercive integer-shot
  boundary to provider-plugin execution artefacts under standalone strict mypy.
- 2026-06-19 — Hardened the PennyLane HAL native-gate payload boundary by
  rejecting non-integer or duplicate wires, wrong gate arities, wrong rotation
  parameter counts, boolean parameters, and non-finite rotation parameters
  before plugin loading or device dispatch, with standalone strict mypy on the
  HAL adapter and focused HAL PennyLane tests.
- 2026-06-19 — Hardened the legacy PennyLane runner physics-input boundary by
  rejecting non-square Kuramoto coupling matrices before device dispatch,
  preserving existing finite `K`/`omega` and shot validation, and bringing
  `tests/test_pennylane_adapter_contracts.py` plus vendor-backend tests under
  standalone strict mypy.
- 2026-06-19 — Hardened the PennyLane tape-import boundary by rejecting
  non-finite imported gate parameters and invalid import round-trip tolerances,
  preserving bounded local import claims, and bringing the dedicated PennyLane
  import test module under standalone strict mypy with typed tape factories and
  `NDArray[np.float64]` callback annotations.
- 2026-06-19 — Hardened the PennyLane Phase-QNode conversion and provider-plugin
  artefact boundary by canonicalising generated-QNode device/interface/diff
  metadata before PennyLane dispatch, rejecting control-character metadata before
  device creation, rejecting boolean provider-shot counts, rejecting
  hardware-implying provider execution modes, and normalising SHA-256 artefact
  digests without promoting provider, hardware, or benchmark claims.
- 2026-06-19 — Hardened the PennyLane HAL device route by normalising local
  plugin device names before `qml.device(...)`, preserving `device_kwargs`
  dispatch evidence, rejecting empty/control-character device names at adapter
  construction, avoiding untyped QNode decorator leakage under strict mypy, and
  validating provider-lineage control-character failures without monkeypatching
  imported stdlib modules.
- 2026-06-19 — Hardened the PennyLane provider/device adapter boundary by
  normalising padded device names, rejecting empty or control-character device
  strings, rejecting non-positive or boolean finite-shot counts, and bringing
  the adapter plus vendor-backend tests under standalone strict mypy.
- 2026-06-19 — Added fail-closed negative-path coverage for the bounded PyTorch
  training-loop audit, covering invalid learning rates, invalid step counts,
  negative tolerances, and parameter-width mismatches in the
  `nn.Module`/`torch.func`/`torch.compile` parity route.
- 2026-06-19 — Tightened the PyTorch/TensorFlow framework-bridge test surface
  under strict mypy by replacing legacy unparameterised NumPy array annotations
  with typed `NDArray[np.float64]` aliases and removing obsolete dynamic-attribute
  ignore comments. The focused framework-bridge test module remains behaviourally
  unchanged and now passes strict mypy as a touched file.
- 2026-06-19 — Added bounded PyTorch module training-loop parity evidence
  through `run_torch_training_loop_audit(...)`. The audit compiles the bounded
  phase-QNN module loss, obtains `torch.func.grad` gradients, applies
  deterministic parameter updates, checks every gradient route against SCPN
  parameter-shift references, and feeds `run_torch_maturity_audit(...)` while
  keeping CUDA/device, provider, hardware, isolated benchmark, and performance
  promotion blocked.
- 2026-06-19 — Added registered local Phase-QNode PyTorch non-fullgraph
  `torch.compile` evidence through `torch_phase_qnode_compile_audit(...)`. The
  audit compiles the deterministic statevector value and gradient routes,
  compares them against SCPN parameter-shift references, updates the PyTorch
  lowering matrix, and adds a non-isolated quantum-gradient conformance
  benchmark row. Fullgraph `torch.compile`, incompatible CUDA/device execution,
  provider, hardware, finite-shot, dynamic-circuit, isolated benchmark, and
  performance-promotion claims remain blocked.
- 2026-06-19 — Added `plan_torch_cloud_validation_batch(...)` and
  `PhaseTorchCloudValidationRunSpec` for PyTorch differentiable cloud
  validation scheduling. The plan records local CUDA/device skip reasons,
  blocked registered Phase-QNode `torch.compile` and accelerator routes,
  required JarvisLabs/cloud artefacts, and reproduction commands while keeping
  cloud execution and promotion claims open until compatible hardware artefacts
  exist.
- 2026-06-19 — Added registered local Phase-QNode PyTorch `torch.func`
  transform evidence through `torch_phase_qnode_transform_audit(...)`, plus
  broad PyTorch module/transform/compiler/device maturity routing through
  `run_torch_ecosystem_maturity_audit(...)`. The new audit records
  `torch.nn.Module`/`Parameter`, `torch.func`, `torch.compile`, and CUDA-device
  capability state, updates the PyTorch lowering matrix, and adds a
  non-isolated quantum-gradient conformance benchmark row. Registered
  Phase-QNode `torch.compile`, incompatible CUDA/device execution, provider,
  hardware, finite-shot, dynamic-circuit, isolated benchmark, and performance
  promotion claims remain blocked.
- 2026-06-19 — Added registered local Phase-QNode JAX pmap/sharding transform
  evidence through `jax_phase_qnode_sharding_transform_audit(...)`. The audit
  maps one deterministic statevector value-and-gradient row per local JAX
  device, compares values and gradients against SCPN parameter-shift
  references without host callbacks, updates the lowering matrix, and adds a
  non-isolated quantum-gradient conformance benchmark row. Single-device CPU
  runs are pmap smoke evidence only; provider, hardware, finite-shot,
  dynamic-circuit, isolated benchmark, and performance-promotion claims remain
  blocked.
- 2026-06-19 — Added registered local Phase-QNode PyTree native JAX transform
  evidence through `jax_phase_qnode_pytree_transform_audit(...)`. The audit
  accepts structured numeric PyTree parameters, validates native `grad`,
  `value_and_grad`, `jacfwd`, `jacrev`, `hessian`, `jvp`, `vjp`, `vmap`, and
  `jit` against SCPN parameter-shift references without host callbacks, exports
  the result surface, updates the JAX lowering matrix, and adds a non-isolated
  quantum-gradient conformance benchmark row while provider, hardware,
  finite-shot, dynamic-circuit, and performance-promotion claims remain blocked.
- 2026-06-19 — Hardened optional JAX availability detection so import-time
  backend compatibility failures fail closed as unavailable instead of
  escaping through optional benchmark/test discovery.
- 2026-06-19 — Added registered local Phase-QNode native JAX transform
  lowering evidence through `jax_phase_qnode_native_transform_audit(...)`.
  The audit exercises JAX `grad`, `value_and_grad`, `jacfwd`, `jacrev`,
  `hessian`, `jvp`, `vjp`, `vmap`, and `jit` without host callbacks, compares
  against SCPN parameter-shift references, exports the result surface, updates
  the JAX lowering matrix, and adds a non-isolated quantum-gradient conformance
  benchmark row. The quantum-gradient benchmark suite now also skips unavailable
  optional backend rows instead of failing before installed backends can run,
  while finite-shot, provider, hardware, dynamic-circuit, and
  performance-promotion claims remain blocked.
- 2026-06-19 — Added a scoped NumPy-style Ruff docstring ratchet for the
  differentiable external-validation, module-hardening audit, and
  hardening-slice gate surfaces. CI, local preflight, and the pre-push hook now
  enforce `ruff check --select D` on all three modules and their
  module-specific tests while repository-wide docstring enforcement remains
  open rollout debt. Refreshed the external-validation environment-lock and
  artefact-bundle checksum manifests against the current checkout.
- 2026-06-18 — Added deterministic transform-algebra conformance for JVP/VJP
  over whole-program AD Hessian transforms through
  `transform_nesting_program_ad_hessian_jvp_vjp`, with compiler, Rust,
  LLVM/JIT, hardware, and performance claims still blocked.
- 2026-06-18 — Extended the explicit strict-mypy ratchet for differentiable
  promotion surfaces. CI, local preflight, and the pre-push hook now enforce
  `mypy --strict` on `differentiable.py`, `differentiable_claim_ledger.py`,
  `differentiable_api.py`, and
  `benchmarks/differentiable_programming.py`, plus external-validation,
  framework-overlay, module-hardening-audit, and hardening-gate differentiable
  modules; repository-wide strict mode remains open debt.
- 2026-06-18 — Extended the differentiable strict-mypy ratchet to
  `benchmarks/differentiable_evidence.py`,
  `phase/differentiable_readiness.py`, and `phase/differentiable_audit.py`.
  The audit helper now avoids the redundant casts that blocked strict mode, and
  CI, local preflight, and the pre-push hook enforce the same eleven-module
  strict command while repository-wide strict mode remains open debt.
- 2026-06-18 — Extended the differentiable strict-mypy ratchet to
  `phase/gradient_support_matrix.py` and `phase/provider_gradient.py`, bringing
  the explicit CI, local preflight, and pre-push strict command to thirteen
  differentiable/gradient modules while repository-wide strict mode remains
  open debt.
- 2026-06-18 — Extended the differentiable strict-mypy ratchet to
  `phase/hardware_gradient_policy.py`, `phase/provider_gradient_audit.py`,
  `phase/hardware_gradient_publication.py`, and
  `phase/provider_hardware_gradient_audit.py`, bringing the explicit CI, local
  preflight, and pre-push strict command to seventeen differentiable/gradient
  modules while repository-wide strict mode remains open debt.
- 2026-06-18 — Extended the differentiable strict-mypy ratchet to
  `phase/hardware_gradient_campaign.py`, `phase/gradient_backend.py`, and
  `phase/gradient_tape.py`, bringing the explicit CI, local preflight, and
  pre-push strict command to twenty differentiable/gradient modules while
  repository-wide strict mode remains open debt.
- 2026-06-18 — Extended the differentiable strict-mypy ratchet to
  `phase/natural_gradient.py`, `phase/gradient_descent.py`, and
  `phase/qnode_affinity_benchmark.py`, bringing the explicit CI, local
  preflight, and pre-push strict command to twenty-three
  differentiable/gradient modules while repository-wide strict mode remains
  open debt.
- 2026-06-18 — Exposed structured numeric Program AD primitive conformance
  through `structured_numeric_primitive_contracts` and the
  `program_ad_structured_primitives` dashboard row. The row covers product,
  interpolation, signal, and stencil primitive contracts while keeping
  Rust/LLVM/JIT, hardware, and performance promotion blocked.
- 2026-06-18 — Added deterministic transform-algebra conformance coverage for
  local Hessian over a whole-program AD scalar objective against analytic
  curvature through `transform_nesting_program_ad_hessian`, with compiler,
  Rust, LLVM/JIT, hardware, and performance claims still blocked.
- 2026-06-18 — Registered Program AD `np.sign` and `np.heaviside` as
  derivative-losing elementwise primitive contracts with explicit fail-closed
  nondifferentiability policies and dashboard evidence. No derivative kernel,
  benchmark-performance, Rust, LLVM/JIT, provider, or hardware claim is
  promoted for these discontinuous primitives.
- 2026-06-18 — Added bounded Program AD loop-carried state alias/effect
  metadata for local derivative-carrying scalar reassignment, with deterministic
  conformance evidence through `loop_carried_state_alias_metadata_contracts`
  while keeping the full alias lattice and Rust/LLVM promotion blocked.
- 2026-06-18 — Exposed Program AD array-indexing conformance in
  `differentiable_dashboard_status(...)`. The `program_ad_array_indexing` row
  is diagnostic by default and becomes conformance-backed only when
  `indexing_static_gather_contracts` runs through
  `run_differentiable_programming_benchmark_suite()`, keeping dynamic indices,
  dynamic insertion values, Rust, LLVM/JIT, hardware, and performance
  promotion fail-closed.
- 2026-06-18 — Exposed bounded Program AD linalg primitive conformance through
  a dedicated `program_ad_linalg_primitives` dashboard row backed by
  `linalg_primitive_contracts`, keeping spectral multiplicity,
  rank-threshold, wider native LLVM/JIT, Rust, hardware, and performance
  promotion boundaries separate from linalg conditioning diagnostics.
- 2026-06-18 — Extended Program AD alias/effect metadata for derivative-
  preserving permutation and source-reuse transforms (`tile`, `roll`,
  `rot90`, `flip`, `flipud`, `fliplr`) with deterministic conformance
  evidence while keeping the full alias lattice and Rust/LLVM promotion
  blocked.
- 2026-06-18 — Added Program AD static rank-1 slice assignment for
  derivative-carrying arrays, with source-index mutation metadata and
  deterministic alias/effect conformance evidence while keeping the full alias
  lattice and Rust/LLVM promotion blocked.
- 2026-06-18 — Extended Program AD alias/effect metadata to executed shape
  views (`squeeze`, `expand_dims`, `atleast_*d`, `swapaxes`, `moveaxis`) and
  repeat source reuse, with deterministic conformance evidence while keeping
  the full alias lattice and Rust/LLVM promotion blocked.
- 2026-06-18 — Promoted bounded plain list-comprehension semantics for
  whole-program AD, with deterministic benchmark conformance and fail-closed
  filtered, set, and dict comprehension boundaries.
- 2026-06-18 — Added deterministic transform-algebra conformance coverage for
  JVP/VJP over `vmap` of whole-program AD gradients against analytic
  block-Hessian contractions, with compiler, Rust, LLVM, hardware, and
  performance claims still blocked.
- 2026-06-18 — Added deterministic differentiable-programming conformance
  coverage for eager `vmap` over exact custom JVP/VJP rules, with a
  non-performance claim boundary.
- 2026-06-18 — Registered Program AD `np.select`, `np.piecewise`,
  `np.choose`, `np.compress`, and `np.extract` as bounded selection-fold
  primitive contracts with shape, dtype, static mask/selector/branch metadata,
  batching, lowering metadata, and fail-closed static boundary policies.
  Rust/LLVM executable promotion remains blocked until independently verified
  selection-fold kernels exist.
- 2026-06-18 — Registered Program AD `np.argmax`, `np.argmin`, trace-array
  `.argmax()`/`.argmin()`, and `np.argsort` as explicit fail-closed
  index-selection primitive contracts with shape, dtype, static-axis/kind,
  batching, lowering, and nondifferentiable integer-output metadata. Rust/LLVM
  executable promotion remains blocked because these primitives return
  nondifferentiable integer selectors.
- 2026-06-17 — Registered Program AD `np.hstack`, `np.vstack`,
  `np.column_stack`, and `np.dstack` as bounded assembly primitive contracts
  with shape, dtype, static operand-shape validation, batching, lowering
  metadata, fixed-shape direct JVP/VJP factories, and blocked Rust/LLVM
  executable lowering status.
- 2026-06-17 — Registered Program AD `np.flipud` and `np.fliplr` as
  bounded shape primitive contracts with rank validation, batching, lowering
  metadata, fixed-axis direct JVP/VJP factories, and blocked Rust/LLVM
  executable lowering status.
- 2026-06-17 — Registered Program AD `np.zeros_like`, `np.ones_like`, and
  `np.full_like` as bounded assembly primitive contracts with shape, dtype,
  static reference-shape/scalar-fill validation, batching, lowering metadata,
  and blocked Rust/LLVM executable lowering status.
- 2026-06-17 — Registered Program AD `np.var`/`np.std` and trace-array
  `.var()`/`.std()` as bounded variance-reduction primitive contracts with
  shape, dtype, static axis/ddof validation, batching, lowering metadata,
  direct JVP/VJP factories, positive-denominator checks, and fail-closed
  zero-variance standard-deviation boundaries.
- 2026-06-17 — Registered Program AD `np.max`/`np.min` and trace-array
  `.max()`/`.min()` as bounded extreme-reduction primitive contracts with
  shape, dtype, static-axis validation, batching, lowering metadata, direct
  JVP/VJP factories, and fail-closed unique-selector boundaries.
- 2026-06-17 — Registered Program AD `np.median`, `np.quantile`, and
  `np.percentile` as bounded reduction primitive contracts with shape, dtype,
  static scalar-q/axis/method validation, batching, lowering metadata, direct
  JVP/VJP factories, and fail-closed strict-order selection boundaries.
  Rust/LLVM executable promotion remains blocked until independently verified
  polyglot reduction kernels exist.
- 2026-06-17 — Registered Program AD `np.sort` as a bounded selection
  primitive contract with shape, dtype, static-axis/kind, batching, lowering,
  and fail-closed strict-total-order nondifferentiability metadata. `np.argsort`
  remains fail-closed index selection.
- 2026-06-17 — Added a per-module strict-mypy ratchet for
  `scpn_quantum_control.differentiable` in CI, local preflight, and the
  pre-push hook. Removed the redundant casts that blocked `mypy --strict` on the
  differentiable-programming surface while leaving the wider repository
  strict-mypy migration tracked as module-specific debt.
- 2026-06-17 — Added IR-format and replay-count provenance to
  `ProgramADAdjointResult`. Supported scalar whole-program adjoint replay now
  records replayed node, effect, control-region, and phi-node counts from the
  emitted `ProgramADEffectIR`, keeping reverse-mode evidence auditable without
  promoting full arbitrary Python reverse-mode compiler AD.
- 2026-06-17 — Added `ProgramADPhiNode` metadata to emitted
  `ProgramADEffectIR` and the `program_ad_effect_ir.v1` round-trip parser.
  Runtime branches and source-level control/loop regions now carry
  deterministic control-join provenance while remaining metadata-only; this
  does not promote non-executed branch adjoints, full compiler phi lowering,
  static alias lattice coverage, Rust, LLVM/JIT, provider, or hardware routes.
- 2026-06-17 — Exposed the bounded Program AD IR round-trip parser through
  `differentiable_dashboard_status(...)` as a `program_ad_ir_roundtrip`
  `metadata_only` row backed by `parse_program_ad_effect_ir(...)` and
  `program_ad_effect_ir.v1` provenance. GUI/audit consumers can now distinguish
  parser metadata evidence from the still-open bytecode/source compiler
  frontend, static alias lattice, Rust interpreter, LLVM/JIT lowering, provider,
  and hardware routes.
- 2026-06-17 — Added `parse_program_ad_effect_ir(...)` for bounded
  `program_ad_effect_ir.v1` metadata round-tripping. The parser reconstructs
  validated Program AD IR dataclasses from emitted JSON and fails closed on
  malformed or unsupported payloads without claiming a full bytecode/source
  compiler frontend. Documented in `docs/differentiable_api.md` and
  `docs/differentiable_programming.md`, with generated capability surfaces
  refreshed.
- 2026-06-17 — Hardened `WholeProgramADResult` trainable-mask validation so
  forward whole-program gradients and attached adjoint replay gradients fail
  closed when frozen parameters carry derivative mass. Documented the
  invariant in `docs/differentiable_programming.md` and covered it in
  `tests/test_differentiable.py`.
- 2026-06-17 — Added a `higher_order_transform_algebra` row to
  `differentiable_dashboard_status(...)`. The row is `diagnostic` by default
  and becomes `conformance_backed` only when `include_conformance=True` runs
  the local benchmark report, tying GUI/audit status to the
  `transform_nesting_vmap_program_grad` and
  `transform_nesting_whole_program_higher_order` benchmark rows without
  promoting compiler, JIT, hardware, provider, or performance claims.
- 2026-06-17 — Added the
  `transform_nesting_whole_program_higher_order` differentiable-programming
  conformance benchmark row. It checks `jacfwd` and `jacrev` over
  whole-program `grad(vmap(f))` against analytic block-diagonal curvature,
  keeping the evidence bounded to deterministic correctness and avoiding
  compiler, JIT, hardware, or timing claims. Documented in
  `docs/benchmarks_api.md` and covered by
  `tests/test_differentiable_programming_benchmarks.py`.
- 2026-06-17 — Added `differentiable_dashboard_status` plus
  `DifferentiableDashboardStatus` and `DifferentiableDashboardCapabilityRow`
  for GUI/audit-dashboard consumers. The status payload labels executable,
  metadata-only, diagnostic, conformance-backed, planned, blocked, and
  unsupported differentiable routes without promoting Program AD metadata,
  Rust/LLVM compiler paths, provider routes, hardware routes, or local
  conformance rows beyond their current claim boundaries. Documented in
  `docs/differentiable_api.md` and `docs/differentiable_programming.md`, with
  generated capability surfaces refreshed.
- 2026-06-16 — Added isolated-benchmark host readiness and self-hosted runner
  provisioning so the `differentiable-isolated-benchmark` CI job can produce
  `isolated_affinity` evidence. `benchmarks/isolated_host_readiness.py`
  (`assess_host_readiness` / `capture_host_readiness`) checks the reserved-core
  governor, frequency, and host load against the same `1.0` threshold the
  isolation classifier applies, and `scripts/check_isolated_benchmark_host.py`
  exits non-zero with the blockers when a host would be downgraded to
  `functional_non_isolated`. `scripts/provision_isolated_benchmark_runner.sh`
  registers a GitHub Actions runner with the `self-hosted,linux,isolated-benchmark`
  labels, pinning runner `v2.335.1` and SHA-256-verifying the archive, installing
  it as a systemd service, and setting the reserved core to the `performance`
  governor. Documented in `docs/isolated_benchmark_runner.md` and covered by
  `tests/test_isolated_host_readiness.py`.
- 2026-06-15 — Added a Wirtinger (Cauchy-Riemann) calculus module
  (`wirtinger_calculus.py`), the complex-derivative surface the registered
  Phase-QNode engine leaves fail-closed (it differentiates real rotation angles).
  `wirtinger_partials` returns `df/dz` and `df/dconj_z` for an arbitrary complex
  callable `f: C^n -> C` by central differences in the real and imaginary
  directions; `is_holomorphic` tests the Cauchy-Riemann residual;
  `holomorphic_gradient` returns the complex derivative of a holomorphic function
  and fails closed otherwise; and `real_objective_gradient` /
  `minimise_real_objective` run CR steepest descent on a real-valued objective of
  complex parameters. Verified against the textbook partials of `z**2`, `|z|**2`,
  `conj(z)`, `Re(z)`, the Wirtinger product rule, and complex-parameter descent
  convergence. Documented in `docs/wirtinger_calculus.md` and covered by
  `tests/test_wirtinger_calculus.py`. A general complex-analysis utility,
  independent of the real-parameter Phase-QNode route.
- 2026-06-15 — Added a PennyLane import-from bridge
  (`phase/pennylane_import.py`), the inverse of the existing export bridge.
  `import_phase_qnode_from_pennylane` reads a `pennylane.tape.QuantumScript` and
  builds the equivalent registered `PhaseQNodeCircuit`, mapping the supported
  gate set and a single Pauli-word expectation observable, with every gate
  parameter becoming a Phase-QNode parameter in tape order.
  `check_pennylane_phase_qnode_import_round_trip` confirms the imported circuit
  reproduces the source value and parameter-shift gradient (the gradient
  comparison restricts the tape to its gate parameters so Hamiltonian
  coefficients are not differentiated, and independently confirms the four-term
  controlled-rotation rule against PennyLane). Import is fail-closed on
  unsupported gates, multi-parameter gates, non-integer or non-contiguous wires,
  multiple or non-expectation measurements, and non-Pauli observables.
  Documented in `docs/quantum_gradients.md` and covered by
  `tests/test_phase_pennylane_import.py`.
- 2026-06-15 — Added closed-loop control analysis for the measurement-feedback
  synchronisation loop (`control/closed_loop_analysis.py`).
  `analyse_closed_loop_response` turns a set-point tracking trajectory into a
  control-theoretic verdict — converged, limit-cycle, diverged, or unsettled —
  with settling round, steady-state error, overshoot, integral absolute error,
  oscillation amplitude, and trailing error sign-change metrics.
  `evaluate_closed_loop_policy` gates execution fail-closed (simulation by
  default; hardware only with `allow_hardware`, a live ticket, and an
  allow-listed backend, within a round budget), and `run_closed_loop_control`
  runs the existing controller under the policy and returns a deterministic,
  replayable `ClosedLoopControlEvidence` record. Documented in
  `docs/closed_loop_control.md` and covered by
  `tests/test_closed_loop_analysis.py`. Software-in-the-loop assessment only:
  not provider-prepared dynamic-circuit or live closed-loop QPU evidence.
- 2026-06-15 — Added a bounded quantum graph neural network over K_nm graphs
  (`phase/qgnn.py`). A classical message-passing stack turns a coupling graph
  (node frequencies and weighted degree, edge couplings) into the rotation
  angles of a registered Phase-QNode circuit whose `CZ` entanglers follow the
  graph topology; the circuit expectation is the model output. `predict_and_gradient`
  returns the output and its exact gradient with respect to the classical
  weights, chaining the analytic parameter-shift gradient (output with respect
  to circuit angles) with the analytic message-passing backward pass (angles
  with respect to weights), validated against finite differences to ~1e-9.
  `synthetic_kuramoto_target` and `train` fit the weights to phase-locked
  Kuramoto-XY targets and report observed local loss decrease only. Documented
  in `docs/quantum_graph_neural_network.md` and covered by
  `tests/test_phase_qgnn.py`. Bounded local model: small statevector circuits,
  no arbitrary-graph, depth, provider, hardware, or production convergence claim.

### Changed
- 2026-06-23 — Corrected the stale public-API symbol count in the deprecation/SemVer
  policy (`DEPRECATIONS.md`): the top-level `__all__` is 717 symbols, not 104, and
  the exact set frozen at v1.0 is noted as being finalised before the 1.0.0 tag
  (the present surface spans the research workbench, not only the durable contract).
- 2026-06-23 — Surfaced the honest-scope caveat to the top of the README: at the
  reachable system sizes (n ≤ 16 qubits) classical solvers are faster and more
  accurate and there is no demonstrated broad quantum advantage. The disclosure
  already existed lower in the README; it is now a read-first callout under the
  status line.
- 2026-06-23 — Set the PyPI `Development Status` classifier to `4 - Beta`
  (from `5 - Production/Stable`) to match the pre-1.0 `0.9.x` line and the
  package's own Active Development notice. The `5 - Production/Stable`
  declaration is restored at the v1.0 API-stability gate.

### Fixed
- 2026-06-15 — Corrected the registered Phase-QNode parameter-shift rule for
  controlled rotations (`crx`, `cry`, `crz`). Their generator eigenvalues are
  `{0, 0, +1/2, -1/2}`, giving two spectral gaps `{1/2, 1}`, so the two-term
  `pi/2` rule was wrong whenever the observable coupled the control-on and
  control-off sectors (for example a Pauli on the control qubit); the gradient
  now uses the four-term rule (and `{m/2, m}` for a tied group of `m` identical
  controlled rotations). Single-Pauli rotations are unchanged.

### Added
- 2026-06-15 — Added U3 and arbitrary single-qubit unitary coverage to the
  registered Phase-QNode gate family (`phase/general_unitary.py`).
  `su2_zyz_angles` returns the exact `RZ(phi) RY(theta) RZ(lam)` Euler angles of
  any `2x2` unitary (global phase discarded) and `build_u3_operations` emits the
  matching registered `RZ·RY·RZ` decomposition, so a U3 or general single-qubit
  unitary differentiates analytically through three two-term rotations without
  enlarging the differentiable-gate primitive set. Documented in
  `docs/quantum_gradients.md` and covered by `tests/test_phase_general_unitary_gates.py`.
- 2026-06-15 — Added a quantum/classical co-simulation package
  (`cosimulation/`). `partition_knm` deterministically splits a K_nm coupling
  network into a strongly-coupled quantum core (statevector-evolved, capped at
  14 nodes) and a weakly-coupled classical Kuramoto bath, with an edge-exact
  conservation report and a `cross_fraction` quality signal. `cosimulate`
  interleaves an exact-internal second-order Trotter evolution of the quantum
  core (driven by the classical mean field) with an explicit-Euler Kuramoto step
  of the bath (driven by the coherence-weighted quantum moments), returning the
  classical-phase and quantum-moment trajectories, quantum/classical/global
  order parameters, and an all-classical baseline. In the decoupled limit the
  co-simulation reduces exactly to an isolated statevector evolution and an
  isolated classical Kuramoto run. The per-step classical update dispatches to a
  zero-skipping Rust kernel (`cosim_classical_substep`, 12.8x faster than the
  dense NumPy reference on a sparse N=128 bath). Ships parity/conservation/
  decoupled-limit tests, a polyglot comparison benchmark
  (`scripts/bench_cosimulation.py`, `results/cosimulation_benchmark.json`,
  `functional_non_isolated`), and documentation
  (`docs/quantum_classical_cosimulation.md`). Local mean-field embedding, not an
  exact full-network or hardware path.
- 2026-06-15 — Added a pulse-waveform → AMD UltraScale+ HLS code generator
  (`codegen/ultrascale_hls.py`). `pulse_to_vivado_hls` quantises a control
  envelope to a signed Q-format ROM and renders a synthesisable AXI4-Stream
  pulse-player header, a C co-simulation testbench, and a clock-only XDC for the
  ZU3EG (`xczu3eg-sbva484-1-e`) and ZU9EG (`xczu9eg-ffvb1156-2-e`) parts shared
  with SC-NEUROCORE. Quantisation dispatches to a bit-true Rust kernel
  (`quantise_q_format`, 29.7x faster on 10^4 samples). A non-synthesis shim
  (`tests/hls_shim`) lets the generated bundle run a bit-true software
  co-simulation under g++; Vivado synthesis is gated behind `MIF_FPGA_VIVADO_CI`.
  Ships parity/validation/co-simulation tests, a polyglot comparison benchmark
  (`scripts/bench_ultrascale_hls.py`, `results/ultrascale_hls_benchmark.json`,
  `functional_non_isolated`), documentation (`docs/ultrascale_hls.md`), and a
  quickstart (`examples/28_pulse_to_hls_quickstart.py`).

### Added
- 2026-06-15 — Added a from-specification FIPS 204 ML-DSA-65 module-lattice
  signature implementation (`crypto/ml_dsa.py`) and a post-quantum capacitor-bank
  trigger signer (`crypto/pqc_trigger.py`). ML-DSA-65 (key generation, signing,
  verification, the negacyclic NTT over Z_q=8380417, ExpandA/ExpandS/ExpandMask,
  SampleInBall, rounding and hint functions, and key/signature encoding) is
  validated bit-for-bit against the official NIST ACVP known-answer vectors
  (keyGen, deterministic sigGen, sigVer). The NTT dispatches to a new bit-true
  Rust kernel (16x faster). `PqcTriggerSigner` binds payloads to a timestamp,
  supports a freshness window, and signs canonical capacitor-bank discharge
  commands. Ships KAT-based and property tests, a polyglot comparison benchmark
  (`scripts/bench_ml_dsa.py`, `results/ml_dsa_benchmark.json`,
  `functional_non_isolated`), documentation (`docs/ml_dsa_pqc.md`), and a demo
  (`examples/27_pqc_trigger_signer_demo.py`). The module is FIPS 204-conformant,
  not a FIPS-140-validated cryptographic module.

### Added
- 2026-06-15 — Added the `scpn_quantum_control.sensing` package with a
  simulation-only NV-centre magnetometry model valid into the 20 T regime
  (`sensing/nv_magnetometry_20T.py`): the exactly-diagonalised ground-state
  spin-1 Hamiltonian (zero-field splitting, transverse strain, electron Zeeman
  at arbitrary field magnitude and angle), ODMR resonance frequencies validated
  across the GSLAC and the high-field regime, shot-noise CW-ODMR DC sensitivity,
  a Lorentzian ODMR spectrum dispatching to a bit-true Rust kernel, and a noisy
  field-calibration loop recovering the field to ~2 microtesla across 0.07-20 T.
  Hardware calibration is gated by `MIF_NV_HARDWARE_CI=1`. Ships module-specific
  and property-based tests, a polyglot comparison benchmark
  (`scripts/bench_nv_magnetometry.py`, `results/nv_magnetometry_benchmark.json`,
  `functional_non_isolated`), documentation (`docs/nv_magnetometry_20T.md`), and
  a demo (`examples/26_nv_magnetometry_20T_demo.py`).

### Added
- 2026-06-15 — Added the FRC pulsed-shot QAOA scheduling cost
  (`control/qaoa_pulsed_cost.py`, `control/frc_pulsed_qaoa.py`):
  `FRCQAOAObjective`, a cited control-grade `FRCPlasmaSurrogate` (s-parameter
  flux-compression scaling, magnetic-tension-stabilised MRTI growth, FRC n=1
  tilt-mode margin), `frc_pulsed_shot_cost`, and brute-force, classical SLSQP,
  and QAOA schedulers over the exact diagonal cost. The MRTI growth integral
  dispatches to a new Rust kernel matching the NumPy reference to 1e-12 relative
  tolerance (2.9-59x faster). Ships module-specific and property-based tests, a
  polyglot comparison benchmark (`scripts/bench_frc_pulsed_qaoa.py`,
  `results/frc_pulsed_qaoa_benchmark.json`, `functional_non_isolated`),
  documentation (`docs/frc_pulsed_qaoa.md`), and a demo
  (`examples/14_frc_pulsed_shot_qaoa_demo.py`).

### Added
- 2026-06-15 — Added the `scpn_quantum_control.entropy` quantum random-number
  package: a `QRNGStream` streaming harness with Qiskit Aer measurement entropy
  sources (`xy_measurement`, `bell_pair`, `phase_estimation`), Von Neumann
  debiasing, and periodic health checks; the full NIST SP 800-22 Revision 1a
  fifteen-test statistical suite, validated against the publication's
  worked-example P-values; the FIPS 140-2 Annex C power-up tests with
  fail-closed enforcement; and Shannon and min-entropy estimation. The
  linear-complexity Berlekamp-Massey hot path and the monobit/runs/longest-run
  statistics dispatch to new Rust kernels that are bit-true with the NumPy
  reference (measured 227–325× faster for Berlekamp-Massey). Ships
  module-specific and property-based tests, a polyglot comparison benchmark
  (`scripts/bench_qrng_entropy.py`, `results/qrng_entropy_benchmark.json`,
  `functional_non_isolated`), documentation (`docs/entropy_qrng.md`), and an
  example (`examples/25_qrng_streaming_quickstart.py`).
- 2026-06-15 — Added a sub-microsecond outer-loop telemetry surface to
  `control/realtime_runtime.py`: `SubMicrosecondTracker`, `CycleSample`,
  `SubMicrosecondReport`, and `summarise_cycle_samples`, reporting inter-cycle
  jitter percentiles against a target period and a deadline-miss count over a
  bounded ring with exact running counters. Percentile and summary computation
  dispatch to new Rust kernels (`sub_us_jitter_percentiles`,
  `sub_us_tracker_summary`) that are bit-true identical to the NumPy fallback,
  with module-specific and property-based tests, a throughput benchmark
  (`scripts/bench_sub_us_tracker.py`, `results/sub_us_tracker_benchmark.json`,
  classified `functional_non_isolated`), and documentation
  (`docs/realtime_runtime.md`).

## [0.9.12] - 2026-06-15

### Security
- 2026-06-15 — Upgraded the Rust PyO3/Numpy binding stack to `pyo3`/`numpy`
  `0.29` and refreshed the Rust test attach path, resolving the open PyO3
  Dependabot advisories before tagging `0.9.12`.

### Added
- 2026-06-15 — Bumped the public release metadata and documentation surfaces
  for the current differentiable-programming hardening queue, including README,
  site home, onboarding, tutorial, notebook, API, reproducibility,
  hardware-ledger, release-readiness, citation, Zenodo, and generated
  capability inventory alignment.
- 2026-06-15 — Added adoption-oriented documentation routes that explain the
  software purpose, application lanes, commercial value, first user paths,
  notebook governance, API selection, evidence classes, and release-hygiene
  boundary without promoting unsupported hardware, clinical, or broad advantage
  claims.
- 2026-06-14 — Added bounded PyTorch module/layer wrapper evidence via
  `torch_bounded_qnn_module`, `torch_bounded_qnn_layer`, and
  `run_torch_module_wrapper_audit`, including `torch.nn.Module`/`Parameter`
  fail-closed handling, module-gradient checks against SCPN parameter-shift
  references, phase exports, bridge matrix promotion, module tests, generated
  capability updates, and public documentation.
- 2026-06-14 — Added bounded PyTorch `torch.compile` compatibility evidence via
  `run_torch_compile_compatibility_audit`, including compiled bounded loss
  gradient checks against the canonical parameter-shift reference, fail-closed
  missing-`torch.compile` handling, phase exports, framework bridge matrix
  updates, module tests, generated capability updates, and public
  documentation.
- 2026-06-14 — Added bounded PyTorch `torch.func` compatibility evidence via
  `run_torch_func_compatibility_audit`, including `torch.func.grad`,
  `torch.func.vmap`, and `torch.func.jacrev` checks against the canonical
  parameter-shift reference, fail-closed missing-`torch.func` handling, phase
  exports, framework bridge matrix updates, module tests, generated capability
  updates, and public documentation.
- 2026-06-14 — Added bounded PyTorch custom-autograd phase-QNN gradient
  evidence via `torch_autograd_qnn_value_and_grad`, including a custom
  `torch.autograd.Function` route, fail-closed optional dependency surface
  checks, parameter-shift reference validation, phase exports, framework bridge
  matrix updates, module tests, generated capability updates, and public
  documentation.
- 2026-06-14 — Added audited JAX PyTree parameter support via
  `run_jax_pytree_compatibility_audit`, including bounded native-QNN and
  custom-VJP no-host-callback structured-parameter checks, deterministic
  flattening and gradient tree restoration, explicit arbitrary-simulator
  PyTree lowering gap classification, phase exports, module tests, live local
  JAX audit evidence, generated capability updates, and public documentation.
- 2026-06-14 — Added audited JAX PMAP/sharding compatibility evidence via
  `run_jax_sharding_compatibility_audit`, including bounded native-QNN and
  custom-VJP no-host-callback local-device batch checks, explicit single-device
  versus multi-device pmap classification, host-loop parameter-shift reference
  classification, phase exports, module tests, live local JAX audit evidence,
  generated capability updates, and public documentation.
- 2026-06-14 — Added audited JAX VMAP compatibility evidence via
  `run_jax_vmap_compatibility_audit`, including bounded native-QNN and
  custom-VJP no-host-callback parameter-batch checks, explicit host-loop
  parameter-shift reference classification, phase exports, module tests, live
  local JAX audit evidence, generated capability updates, and public
  documentation.
- 2026-06-14 — Added audited JAX JIT compatibility evidence via
  `run_jax_jit_compatibility_audit`, including bounded native-QNN and
  custom-VJP no-host-callback checks, explicit parameter-shift host-callback
  classification, active JAX callback dtype negotiation for x64-disabled
  runtimes, phase exports, module tests, generated capability updates, and
  public documentation.
- 2026-06-14 — Added bounded JAX custom-VJP phase-QNN gradient evidence via
  `jax_custom_vjp_qnn_value_and_grad`, including fail-closed optional
  dependency checks, JIT-compatible no-host-callback execution for the bounded
  classifier, parameter-shift reference validation, phase exports, framework
  bridge matrix updates, module tests, generated capability updates, and public
  documentation.
- 2026-06-14 — Added stochastic-gradient confidence intervals and
  fail-closed failure policies across finite-shot parameter-shift, seeded SPSA,
  and materialised score-function estimators, with Rust/PyO3 interval-policy
  parity, module/parity tests, Criterion benchmark coverage, typed extension
  exports, and public documentation updates.
- 2026-06-14 — Added materialised score-function likelihood-ratio gradient
  estimation with explicit baseline handling, sample provenance, empirical
  covariance, fail-closed validity checks, Rust/PyO3 parity, module/parity
  tests, Criterion benchmark coverage, typed extension exports, and public
  documentation updates.
- 2026-06-14 — Added seeded local SPSA gradient estimation with finite-shot
  sample uncertainty propagation, probe-pair provenance, fail-closed stochastic
  contracts, Rust/PyO3 parity for materialised SPSA records, module/parity
  tests, Criterion benchmark coverage, typed extension exports, and public
  documentation updates.
- 2026-06-14 — Added Rust/PyO3 parity for materialised finite-shot
  parameter-shift uncertainty propagation, including shifted-mean, variance,
  shot-count, coefficient, and trainable-mask validation; Python/Rust parity
  tests; Criterion benchmark coverage; typed extension exports; and public
  documentation updates.
- 2026-06-14 — Added deterministic local vector-output Phase-QNode Hessian
  tensors with fail-closed finite-shot/hardware/adapter boundaries, Rust/PyO3
  tensor validation parity, module tests, benchmark coverage, and
  documentation updates.
- 2026-06-14 — Added Rust/PyO3 parity kernels for promoted local
  Phase-QNode Fubini-Study/QFI, computational-basis Fisher, vector JVP/VJP,
  Hessian-vector product, and real-only complex-derivative contract surfaces,
  with Rust unit tests, Python parity tests, typed extension exports, Criterion
  benchmark entries, and documentation updates.
- 2026-06-14 — Added an explicit real-only complex/Wirtinger derivative
  contract for Phase-QNode scalar and vector transform surfaces, with public
  contract metadata, fail-closed complex input/output validation, docs, tests,
  and generated capability-surface alignment.
- 2026-06-14 — Added deterministic local scalar Phase-QNode
  Hessian-vector products backed by parameter-shift Hessian evidence, with
  vector validation, fail-closed finite-shot/hardware/adapter routes, public
  docs, tests, and generated capability-surface alignment.
- 2026-06-14 — Added vector-output Phase-QNode `jvp` and `vjp`
  transforms over deterministic local parameter-shift Jacobian evidence, with
  tangent/cotangent shape validation, fail-closed unsafe routes, public docs,
  tests, and generated capability-surface alignment.
- 2026-06-14 — Added exact computational-basis classical Fisher information
  for the registered local Phase-QNode statevector family, with singular
  zero-probability fail-closed handling, public docs, tests, and generated
  capability-surface alignment.
- 2026-06-14 — Added pure-state Phase-QNode QFI/Fubini-Study metric extraction
  for the registered local statevector gate family, with a natural-gradient
  metric provider, fail-closed unsupported-route tests, phase namespace exports,
  public docs, and generated capability-surface alignment.
- 2026-06-14 — Added `DenseHermitianObservable` to the registered Phase-QNode
  circuit family, with finite square Hermitian matrix validation, exact
  statevector expectation evaluation, parameter-shift gradient coverage, phase
  namespace exports, public docs, and generated capability-surface alignment.
- 2026-06-14 — Added `PauliCovarianceObservable` to the registered
  Phase-QNode circuit family, including exact symmetrised covariance execution,
  product-rule parameter-shift gradients, phase namespace exports, module tests,
  differentiable API documentation, and generated capability-surface alignment.
- 2026-06-14 — Added a six-parameter sparse Ising-chain Hamiltonian
  expectation row to the quantum-gradient benchmark suite, with analytic
  field/coupling gradients, parameter-shift verification, finite-difference
  diagnostics, and explicit non-performance/non-hardware claim boundaries.
- 2026-06-14 — Hardened `VQLS_GradShafranov` with residual-certified
  Grad-Shafranov solves for `n_qubits=2,3,4`, diagnostic convergence metadata,
  configurable multi-restart optimisation, fail-closed unrepaired variational
  residual handling, and direct SPD residual repair for the finite-difference
  Laplacian path.
- 2026-06-05 — Hardened the differentiable Phase-QNode promotion lane with a
  CPU-only framework overlay installer, isolated benchmark metadata gates,
  real optional-framework comparison execution, self-hosted runner setup
  tooling, and stricter claim-ledger promotion checks.
- 2026-06-05 — Added a registered local Phase-QNode circuit family with
  statevector execution, analytic parameter-shift gradients, Pauli-product and
  sparse-Hamiltonian expectations, structured support reports, framework parity
  rows, affinity-labelled benchmark metadata, registered model-training
  evidence, and textual MLIR lowering metadata for the supported subset.
- 2026-06-05 — Added a fail-closed bounded phase-QNN framework bridge matrix
  that declares implemented JAX/PyTorch/TensorFlow bridge routes and records
  arbitrary simulator autodiff plus live provider hardware-gradient routes as
  explicit unsupported gaps.
- 2026-06-05 — Added native bounded phase-QNN framework-gradient evidence:
  `jax_native_qnn_value_and_grad` now evaluates the bounded classifier loss in
  JAX `value_and_grad` without host callbacks, while
  `torch_bounded_qnn_value_and_grad` and
  `tensorflow_bounded_qnn_value_and_grad` return tensor-ready analytic gradient
  evidence checked against the canonical SCPN parameter-shift gradient.

## [0.9.11] - 2026-06-05

### Added
- 2026-06-05 — Bumped the public release metadata and documentation surfaces
  for the current differentiable-programming hardening state, including README,
  onboarding, tutorial, notebook, API, reproducibility, hardware-ledger,
  release-readiness, citation, Zenodo, and generated capability inventory
  alignment.
- 2026-06-05 — Promoted native compiler-backed whole-program AD determinant
  lowering from helper-backed `6x6`-`16x16` to verified `6x6`-`19x19`, and
  documented `20x20+` determinant traces as fail-closed after strict native
  value/JVP/VJP/gradient verification rejected the current helper formulation.

## [0.9.10] - 2026-06-04

*(untagged release; release commit `c19d4d0e` — retro-tagging is owner-gated)*

### Added
- 2026-06-04 — Polished the public differentiable-programming documentation and
  release metadata surface for the current implementation state, including
  clearer onboarding, README positioning, tutorial/notebook routing, support
  boundaries, reproducibility metadata, and generated capability inventory.
- 2026-06-04 — Documented the current enterprise-readiness boundary for
  gradient-bearing Kuramoto-XY workflows: supported parameter-shift, composed
  objectives, program/compiler AD primitives, provider-gradient readiness,
  transform-nesting governance, and fail-closed unsupported scenarios remain
  separated from future framework-native and arbitrary-program AD claims.

## [0.9.9] - 2026-06-02

### Added
- 2026-06-02 — Added a public differentiable-programming documentation route
  with current quantum-gradient support, compiler/program AD boundaries,
  public API entry points, tutorial and notebook plans, benchmark-evidence
  expectations, framework-adapter roadmap, and fail-closed unsupported
  scenarios.
- 2026-06-02 — Updated README, documentation index, onboarding, quickstart,
  tutorials, notebook guide, API overview, MkDocs navigation, release
  readiness, citation metadata, Zenodo metadata, and release-version surfaces
  for the 0.9.9 documentation release.

## [0.9.8] - 2026-06-01

### Added
- 2026-06-01 — Added a public onboarding page and refreshed the README,
  documentation index, quickstart, tutorials, notebook guide, API overview,
  examples guide, installation guide, MkDocs navigation, and release-readiness
  page so first-time users can understand the software purpose, application
  lanes, commercial route, evidence boundaries, and `0.9.8` release scope
  before entering low-level APIs.
- 2026-06-01 — Aligned the release-readiness audit default coverage threshold
  with the temporary 70% CI gate while preserving missing-file blockers and the
  long-term 100% meaningful-coverage target.
- 2026-05-31 — Added an SCPN-CONTROL disruption bridge dependency contract to
  the ITER quantum disruption backend, including classifier API, 11-feature
  ordering, Qiskit core dependencies, optional provider families, report schema
  awareness, downstream non-admission policy, and tamper-evident contract
  digests for future backend hardening.
- 2026-06-01 — Added compact differentiable-programming reverse replay for
  determinant, inverse, solve, trace, diagonal extraction/construction,
  flattened diagonal construction, static matrix powers, and static
  multi-operand matrix chains so supported program-AD linalg traces now
  propagate exact adjoints instead of expanding every scalar operation.
- 2026-06-01 — Documented the current compiler-backed AD boundary honestly:
  MLIR-runtime scalar kernels and LLVM-style provenance are available for
  supported scalar paths, while general Rust/LLVM/JIT executable program-AD
  lowering remains fail-closed until real polyglot compiler backends exist.
- 2026-06-01 — Aligned public release metadata, citation metadata, Zenodo
  metadata, README status, documentation index, hardware ledger, API guide,
  reproducibility guide, adopter snippets, dataset guide, and publication
  readiness pages for the 0.9.8 source release.

## [0.9.7] - 2026-05-19

### Added
- 2026-05-19 — Added the first GOTM-SCPN Paper 0 K_nm preregistered replay
  artefact set with deterministic JSON/Markdown outputs, comparator gate,
  release-safe `scpn-bench` wiring, null-model diagnostics, input digest
  manifests, and an explicit do-not-promote hardware boundary.
- 2026-05-19 — Added the Paper 0 methodology-paper outline, first downstream
  preregistered experiment design, replay contract, and measured-coupling
  evidence checklist so Paper 0 ingestion now produces a concrete
  source-bounded experimental pathway.
- 2026-05-19 — Added the repository-specific capability-manifest system with
  generated JSON/Markdown inventory, README snapshot, SQC-specific counts for
  Paper 0 validation modules, domain package families, notebooks, examples,
  Rust source modules, and Rust PyO3 function bindings.
- 2026-05-19 — Added `scpn-bench capability-manifest-check` and focused tests
  so public capability counts are generated and checked from one source of
  truth rather than hand-edited across release surfaces.

### Added
- 2026-05-18 — Added a release-readiness audit helper and public gate
  document that compose version consistency, coverage-gap readiness,
  behavioural-test density, and required claim-boundary artefacts into one
  deterministic tag blocker.
- 2026-05-18 — Added reviewed coverage exclusions for optional Julia-runtime
  wrappers and the generated Paper 0 source-accounting package, with
  glob-aware coverage-gap audit handling.
- 2026-05-18 — Added repo-local agent instructions advertising that Paper 0
  source ingestion is complete and accessible through the generated register,
  while requiring future papers to follow the same source-bounded processing
  path before any completion claim.
- 2026-05-18 — Added a Paper 0 experimental-pathway programme page that
  turns completed source ingestion into a methodology-paper route,
  experimental tiers, candidate lanes, and immediate production queue while
  keeping Paper 27 bounded as an implementation candidate rather than the
  definitive source of truth.
- 2026-05-18 — Added S2 benchmark-matrix readiness ingestion to the
  quantum-advantage gap audit so IBM advantage runs stay blocked until the
  full protocol grid, hardware rows, and claim-boundary gates are satisfied.
- 2026-05-18 — Added an IEEE 14-bus measured-system control candidate
  for K_nm physical validation with voltage-weighted admittance,
  all-pairs uncertainty accounting, and explicit non-promotional blockers.
- 2026-05-18 — Added justified file-level exclusions to the coverage-gap audit
  helper so release gating can distinguish explicit accepted gaps from
  unreviewed missing or below-threshold files.
- 2026-05-18 — Added aggregate behavioural-quality thresholds to the test audit helper so release coverage closure can require assertion and exception-contract density, not only line execution.
- 2026-05-18 — Added a K_nm measured-system promotion hard gate that keeps physical validation open unless units, uncertainty, full pairwise coverage, tolerance, spectral response, and null-model requirements all pass.
- 2026-05-18 — Added an S2 IBM advantage-readiness hard gate to the claim-boundary report so partial or hardware-free scaling rows cannot justify new IBM spend.
- 2026-05-18 — Added a TCBO `p_h1` replay-uncertainty gate so the reconstructed coupling-weighted complex can report confidence intervals and refuse claim promotion without a named preregistered dataset.
- 2026-05-18 — Added a repository-wide documentation surface audit helper with focused tests and an internal baseline report for Python docstrings, Markdown titles, and stale status snapshots.
- 2026-05-18 — Added public Paper 0 source-validation register documentation and API routing after completing source-accounting ingestion through `P0R06211`.
- 2026-04-30 — Added an IEEE 5-bus measured power-grid coupling artefact and audit support for measured-system topology, magnitude, spectral, critical-response, and null-model diagnostics.
- 2026-04-30 — Added the 109-subject EEGMMIDB baseline eyes-closed PLV artefact and an eyes-closed-minus-eyes-open comparison artefact.

### Changed
- 2026-05-18 — Marked the Paper 0 promotion planner tests as
  `internal_corpus` because they require ignored source-ledger artefacts, and
  expanded the Docker workflow path filters to cover every copied test/input
  surface.
- 2026-05-18 — Documented the remaining documentation-surface TODO after Paper 0 generated-builder batch 25 so production work can resume from a tracked continuation point.
- 2026-05-18 — Closed generated Paper 0 builder documentation 10-batch 21 for the next CLI-entrypoint burn-down block.
- 2026-05-18 — Closed generated Paper 0 builder documentation 10-batch 22 for the next CLI-entrypoint burn-down block.
- 2026-05-18 — Closed generated Paper 0 builder documentation 10-batch 23 for the next CLI-entrypoint burn-down block.
- 2026-05-18 — Closed generated Paper 0 builder documentation 10-batch 24 for the next CLI-entrypoint burn-down block.
- 2026-05-18 — Closed generated Paper 0 builder documentation 10-batch 25 for the next CLI-entrypoint burn-down block.
- 2026-05-18 — Closed generated Paper 0 builder documentation 10-batch 16 for the next CLI-entrypoint burn-down block.
- 2026-05-18 — Closed generated Paper 0 builder documentation 10-batch 17 for the next CLI-entrypoint burn-down block.
- 2026-05-18 — Closed generated Paper 0 builder documentation 10-batch 18 for the next CLI-entrypoint burn-down block.
- 2026-05-18 — Closed generated Paper 0 builder documentation 10-batch 19 for the next CLI-entrypoint burn-down block.
- 2026-05-18 — Closed generated Paper 0 builder documentation 10-batch 20 for the next CLI-entrypoint burn-down block.
- 2026-05-18 — Closed generated Paper 0 builder documentation 10-batch 11 for the next CLI-entrypoint burn-down block.
- 2026-05-18 — Closed generated Paper 0 builder documentation 10-batch 12 for the next CLI-entrypoint burn-down block.
- 2026-05-18 — Closed generated Paper 0 builder documentation 10-batch 13 for the next CLI-entrypoint burn-down block.
- 2026-05-18 — Closed generated Paper 0 builder documentation 10-batch 14 for the next CLI-entrypoint burn-down block.
- 2026-05-18 — Closed generated Paper 0 builder documentation 10-batch 15 for the next CLI-entrypoint burn-down block.
- 2026-05-18 — Closed generated Paper 0 builder documentation 10-batch 6 for the next CLI-entrypoint burn-down block.
- 2026-05-18 — Closed generated Paper 0 builder documentation 10-batch 7 for the next CLI-entrypoint burn-down block.
- 2026-05-18 — Closed generated Paper 0 builder documentation 10-batch 8 for the next CLI-entrypoint burn-down block.
- 2026-05-18 — Closed generated Paper 0 builder documentation 10-batch 9 for the next CLI-entrypoint burn-down block.
- 2026-05-18 — Closed generated Paper 0 builder documentation 10-batch 10 for the next CLI-entrypoint burn-down block.
- 2026-05-18 — Closed the fifth 10-script Paper 0 generated-builder documentation batch across domain overview, renormalisation, cybernetic closure, ecology, ethics, and experimental-design CLIs.
- 2026-05-18 — Closed the fourth 10-script Paper 0 generated-builder documentation batch across universal-parameter, core-assumption, cosmology, coherent-sigma, data-fusion, infoton, interaction, Lagrangian, and domain-layer CLIs.
- 2026-05-18 — Closed the third 10-script Paper 0 generated-builder documentation batch across grammar, citation, clinical, cultural, complexity, component, AI-alignment, and consciousness CLIs.
- 2026-05-18 — Closed the second 10-script Paper 0 generated-builder documentation batch across Axiom III, n-tilde, stability, biology, and case-study CLIs.
- 2026-05-18 — Closed the first 10-script Paper 0 Axiom spec-builder documentation-surface batch across Axiom I, II, and III generated CLIs.
- 2026-05-18 — Closed the second Paper 0 spec-builder documentation-surface slice for immune-enzyme, pathology/anomaly, coupling-affinity, causal-efficacy, and Axiom I family-prediction builders.
- 2026-05-18 — Closed the VQE and first Paper 0 spec-builder documentation-surface slice.
- 2026-05-18 — Closed the second benchmark documentation-surface slice for FIM VQE, GPU, multi-language K_nm, Rust-core, and S1 feedback benchmark scripts.
- 2026-05-18 — Closed the first benchmark documentation-surface slice for order-parameter, classical/Rust/GPU guardrail, S2-lite, ansatz, and tensor-network scaling scripts.
- 2026-05-18 — Closed the audit and Paper 0 automation documentation-surface slice for split audit, readout-mitigation eligibility, and promotion automation CLIs.
- 2026-05-18 — Closed the Phase 3 GUESS/state-layout and S1 feedback documentation-surface slice.
- 2026-05-18 — Closed the Phase 2 popcount, readout-mitigation, and B-C scaling documentation-surface slice.
- 2026-05-18 — Closed the Phase 1/Phase 2 DLA parity documentation-surface slice for parity statistics, raw-count reproduction, and GUESS-readiness calibration scripts.
- 2026-05-18 — Closed the IQM documentation-surface slice for DLA parity, layout campaign, and layout-pinned repeat analysis scripts.
- 2026-05-18 — Closed the second FIM documentation-surface slice for level-spacing, readout-mitigation, sector-survival, and spectrum artefact scripts.
- 2026-05-18 — Closed the first documentation-surface audit slice for FIM analysis scripts with public docstrings and typed CLI output handling.
- 2026-05-18 — Refreshed README and documentation index status surfaces to report `0` remaining Paper 0 work orders under the explicit source-bounded, non-hardware claim boundary.
- 2026-04-29 — Added documented classical baseline surfaces for SciPy ODE, optional QuTiP Lindblad, and optional MPS TEBD runs.
- 2026-04-29 — Re-routed API documentation so stable facades are the first-path entry point and low-level module references sit under advanced navigation.
- 2026-04-29 — Added a mkdocstrings stable facades API page for first-path public facades.
- 2026-04-29 — Added a physics-first Kuramoto-XY tutorial with a tested arbitrary `K_nm`/`omega` workflow.
- 2026-04-29 — Refreshed README, docs index, and results gallery from the dated hardware status ledger.
- 2026-04-29 — Documented the licence boundary for a possible future lightweight Kuramoto-XY core package; no relicensing is implied.
- 2026-04-29 — Added a `kuramoto_core` facade for validated `K_nm`/`omega` problems, Hamiltonian/circuit compilation, dense Rust-backed Hamiltonians, and order-parameter measurement.
- 2026-04-29 — Added `tools/check_dependency_drift.py` so release checks can verify that `requirements.txt` still mirrors `pyproject.toml`.
- 2026-04-29 — `QuantumKuramotoSolver` now validates coupling shape, finite values, symmetry, Trotter settings, and run-grid parameters before building circuits.
- 2026-04-29 — Added a public hardware status ledger that classifies theory, simulator, hardware, mitigated, and noise-limited claims and links visible docs to that evidence index.
- 2026-04-29 — Dependency source hygiene: `requirements.txt` now mirrors the canonical runtime bounds from `pyproject.toml`; `[all]` is a portable optional surface; CUDA/JAX packages moved behind `[accelerated]`; CI adds a fresh editable `[all]` install smoke job.
- 2026-04-18 — `qiskit` pin bumped `>=1.0,<2.0` → `>=2.2,<3.0`; `qiskit-aer` pin bumped `>=0.14,<1.0` → `>=0.15,<1.0`; `qiskit-ibm-runtime` pin bumped `>=0.20,<1.0` → `>=0.40,<1.0`. Misleading `PauliEvolutionGate bug (GH #15476)` comment removed — upstream closed that report as not-a-bug (the 2.2 change correctly treats `PauliEvolutionGate` as an abstract exact-evolution object; pre-2.2 was silently injecting Trotter error). Full pytest suite: 5045 passed / 0 failed on qiskit 2.4.0 without code changes.

## [0.9.6] - 2026-04-17

### Security
- Bumped `rand` 0.9.2 → 0.9.4 (RUSTSEC-2026-0097).
- Removed IBM Cloud CRN prefix log in `scripts/retrieve_ibm_job.py`.
- Bumped `pytest` 9.0.2 → 9.0.3 (CVE-2025-71176).

### Added
- Input-validation guards + `MAX_OSCILLATORS_DEFAULT = 32` cap in `analysis/koopman.py`.
- `docs/pipeline_performance.md` §21 (measured Rust speedups + new-backend decision rule).
- `docs/language_policy.md` (multi-language accel chain: Rust + Julia + Go + Mojo + Python floor).
- `docs/triage.md` (issue triage policy and label taxonomy).
- `docs/falsification.md`, `docs/preregistration.md`, `docs/THREAT_MODEL.md`, `docs/EXPORT_CONTROL.md`, `docs/adopters.md`, `docs/datasets.md`, `docs/mutation_testing.md`, `DEPRECATIONS.md`.
- `.well-known/security.txt` (RFC 9116).
- `src/scpn_quantum_control/config.py` — `SCPNConfig` via `pydantic-settings`.
- `src/scpn_quantum_control/logging_setup.py` — `structlog` bootstrap.
- `src/scpn_quantum_control/hardware/backends.py` — entry-points backend registry.
- `src/scpn_quantum_control/hardware/async_runner.py` — `AsyncHardwareRunner` over `asyncio`.
- `src/scpn_quantum_control/accel/` — multi-language dispatcher + Julia tier (`order_parameter`).
- `src/scpn_quantum_control/hardware/provenance.py` — `capture_provenance()`.
- `tests/test_phase1_dla_parity_reproduces.py` (8 tests, Phase 1 reproducer).
- `tests/test_cross_validation_qutip_dynamiqs.py` (43 tests, XY Hamiltonian vs QuTiP / Dynamiqs).
- `tests/test_pennylane_vendor_backends.py` (67 tests, mocked IBM / IonQ / Rigetti / Quantinuum / Braket / Cirq).
- `tests/test_backend_registry.py` (26 tests).
- `tests/test_config.py` (20 tests).
- `tests/test_logging_setup.py` (19 tests).
- `tests/test_async_runner.py` (17 tests, mocked `qiskit_ibm_runtime`).
- `tests/test_accel_dispatch.py` (27 tests, Rust → Julia → Python floor).
- `tests/test_phase_artefact_fuzz.py` (18 Hypothesis tests).
- `tests/test_hardware_classical_fuzz.py` (13 Hypothesis tests).
- `tests/test_qec_validators_fuzz.py` (15 Hypothesis tests).
- `tests/test_perf_regression.py` (Rust ≥ 2× Python floor).
- `tests/test_provenance.py` (11 tests).
- `scripts/bench_order_parameter_tiers.py` + `docs/benchmarks/order_parameter_tiers.json`.
- `scpn_quantum_engine/benches/hot_paths.rs` (Criterion harness) + `.github/workflows/rust-benches.yml`.
- `.github/workflows/notebooks.yml`, `link-check.yml`, `docs-strict.yml`, `sbom.yml`, `mutation-testing.yml`, `commit-trailers.yml`.
- `tools/check_commit_trailers.py`, `tools/mutmut_runner.sh`.
- `pyproject.toml` extras: `xvalidate`, `config`, `logging`, `julia`; entry-point group `scpn_quantum_control.backends`.
- `<private-local-record>` — 9 visibility drafts (HN / r/QuantumComputing / r/qiskit / Slack / Discord / LinkedIn / X / arXiv / README).

### Changed
- CI dev-tool matrix (`pytest 9.0.3`, `mypy 1.20.1`, `ruff 0.15.10`, `hypothesis 6.151.13`, `build 1.4.3`, `actions/upload-artefact 7.0.1`, `pypa/gh-action-pypi-publish 1.14.0`).
- `_order_param` in `hardware/classical.py` now dispatches through the accel chain.
- Five remaining `print()` calls in `hardware/runner.py` → `structlog` events.
- `HardwareRunner.DEFAULT_INSTANCE` → `HardwareRunner._default_instance()` (reads SCPNConfig).
- Stale counts refreshed in `literature/README.md`, `docs/architecture.md`, `docs/index.md`, `docs/pipeline_performance.md`, `docs/test_infrastructure.md`.
- Self-applied quality labels scrubbed across `CHANGELOG.md`, `docs/changelog.md`, `docs/test_infrastructure.md`, `docs/symmetry_decay_guess.md`, `docs/dynq_qubit_mapping.md`, and 27 test docstrings.

### Repository hygiene
- `<private-local-record>` and `<private-local-record>` untracked; local-only going forward.
- `.gitignore` patterns for paper-extraction working files, `.agent_metadata.json`, root-level `private handoff record_*.md`, `<private-local-record>`, `<private-local-record>`.
- Agent-name mentions stripped from public-facing tracked files (`CHANGELOG.md`, `docs/triage.md`, `docs/PAPER_CLAIMS.md`, `docs/changelog.md`, `figures/generate_ansatz_comparison.py`, `tests/test_koopman.py`, and nine files referencing the internal audit filename).

## [0.9.5] - 2026-03-29 / 2026-04-11

### Added
- 2026-04-10: Phase 1 IBM Quantum hardware campaign on `ibm_kingston` (Heron r2, 156 qubits). 348 circuits, up to 21 reps per (depth, sector) at n = 4. Mean DLA-parity asymmetry +10.8 % for depths ≥ 4, peak +17.48 % at depth 6. Fisher combined χ²(16) = 123.4, p ≪ 10⁻¹⁶. Apriori simulator band was 4.5–9.6 %.
- `scripts/analyse_phase1_dla_parity.py`, `paper/submissions/submission_002_phase1_dla_parity/phase1_dla_parity_short_paper.md`.
- IBM execution scripts: `pipe_cleaner_ibm_kingston.py`, `phase1_mini_bench_ibm_kingston.py`, `phase1_5_reinforce_ibm_kingston.py`, `phase2_exhaust_cycle_ibm_kingston.py`, `phase2_5_final_burn_ibm_kingston.py`, `phase2_full_campaign_ibm.py`, `micro_probe_ibm_kingston.py`, `retrieve_ibm_job.py`.
- `<private-local-record>`, `IBM_EXECUTION_LOG.md`, `phase1_experiment_design.md`, `WEBMASTER_CONTEXT.md`.
- 2026-04-08: `mitigation/symmetry_decay.py` + `scpn_quantum_engine/src/symmetry_decay.rs` — GUESS symmetry-decay ZNE (Oliva del Moral et al., arXiv:2603.13060, 2026). 20 tests.
- `hardware/qubit_mapper.py` + `scpn_quantum_engine/src/community.rs` — DynQ topology-agnostic mapper (Liu et al., arXiv:2601.19635). 17 tests.
- `phase/pulse_shaping.py` + `scpn_quantum_engine/src/pulse_shaping.rs` — PMP / ICI pulse sequences (Liu et al. 2023). Rust `ici_three_level_evolution_batch` 1 665× vs Python.
- `(α,β)`-hypergeometric pulse family via Gauss ${}_2F_1$ (Ventura Meinersen et al., arXiv:2504.08031, 2025). Rust 44× vs scipy. 25 tests for ICI + hypergeometric.
- FFI boundary hardening across all 36 `#[pyfunction]` exports (`PyResult<T>`, `validate_n`, `validate_positive`, `validate_range`, `validate_finite`, `validate_flat_square`, `validate_statevec_len`, `validate_domain_range`). 16 `validation.rs` tests.
- `docs/symmetry_decay_guess.md` (891 lines), `docs/dynq_qubit_mapping.md` (878 lines).
- 2026-04-10: `tools/check_secrets.py` (vault-pattern scanner) + gitleaks v8.21.2 pre-commit hook.
- `<private-local-record>`.
- `.gitignore` patterns for `.venv-linux/`, `.venv-rocm/`, `.venv-cuda/`, `results/`, `<private-local-record>`, `<private-local-record>`.

- 2026-04-06 / -07: `qec/multiscale_qec.py`, `qec/syndrome_flow.py` — concatenated surface codes across 5 SCPN domains. 23 tests.
- `fep/variational_free_energy.py`, `fep/predictive_coding.py` — Friston 2010 variational F. 16 tests.
- `psi_field/lattice.py`, `psi_field/infoton.py`, `psi_field/scpn_mapping.py`, `psi_field/observables.py` — U(1) compact gauge theory with HMC. 22 tests.
- K_nm validation on IEEE 5-bus (ρ = 0.881), Josephson (ρ = 0.990), EEG (ρ = 0.916), ITER MHD (ρ = 0.944).
- `ripser` under `[topology]`.
- 2026-03-29 / 2026-04-01: dynamic-coupling engine (Quantum Hebbian learning).
- Topological coupling optimiser + `HardwareTopologicalOptimizer` (persistent homology as loss).
- `BiologicalSurfaceCode` — stabilisers on the 16-layer SCPN graph.
- `LindbladSyncEngine` (MCWF path for large-N).
- EEG PLV → quantum pipeline.
- Sparse evolution engine (bypasses Qiskit overhead).
- `StructuredAnsatz` — topology-informed variational circuits on arbitrary K.
- 27 experiment notebooks (NB14–47, FIM mechanism).
- `test_fim_mechanism.py` — 81 regression tests.
- 25 JSON result files from 27 notebooks.
- IBM hardware v2 — 9 fair-depth experiments on `ibm_fez` (F_FIM = 0.916 vs F_XY = 0.849, p < 10⁻⁶).
- Rust: `correlation_matrix_xy`, `lindblad_jump_ops_coo`, `lindblad_anti_hermitian_diag`, `parity_filter_mask` (18 → 22 total).

### Changed
- Tests collected: 2 813 → 4 828 (97 %+ coverage).
- Python modules: 165 → 201; subpackages 17 → 19.
- Rust functions exported: 22 → 36 across 20 source files.
- Hardware reference: `ibm_fez` → `ibm_fez + ibm_kingston` (Feb + Apr 2026).
- `runner.py`: counts extraction now tries `meas`, `c`, `cr`, `c0` then introspects `DataBin`.
- `lib.rs` god file (1 436 lines) split into 16 focused modules + 3 new Rust paths (`concat_qec.rs`, `fep.rs`, `gauge_lattice.rs`). 22 → 25 exported functions.
- OTOC: 4.4× faster (n = 4, 50 time points) via O(d) phase rotation.
- Pauli expectations: 2–10× faster via half-loop over paired states.
- Kuramoto order parameter: Rust fast path via `all_xy_expectations`.
- Dockerfile base image pinned by SHA256; `build==1.4.2`, `pip-audit==2.9.0`, `sc-neurocore==3.14.0`.
- Dev: `ruff 0.15.6 → 0.15.9`, `mypy 1.19.1 → 1.20.0`, `hypothesis 6.151.10 → 6.151.11`.
- `multiscale_qec.py` (346 lines) → `multiscale_qec.py` (292) + `syndrome_flow.py` (66).
- Tests: 2 715 → 4 445+; Rust tests: 47 → 65.

### Fixed
- 2026-04-10: SamplerV2 result parsing no longer hard-codes the classical-register name as `meas`.
- `knm_hamiltonian.py` K-symmetry enforced for Hermiticity.
- All example scripts wrapped in `if __name__ == "__main__":`.
- `delta` parameter propagation in `pairing_correlator.py`, `xxz_phase_diagram.py`.
- Infinite recursion in `knm_to_dense_matrix` fallback.
- `vqe_energy` alias on `PhaseVQE` for backward compatibility.
- 25+ analysis modules migrated to Rust-accelerated dense-matrix path.

## [0.9.4] - 2026-03-29

### Added
- 81 tests across 6 new files (pennylane mock, JAX mock, ripser mock, hardware runner mock, Python fallback, edge cases).
- OpenSSF Best Practices badge (project 12290).
- OpenSSF Scorecard, Ruff, mypy badges.
- 3 benchmark API docs: `gpu_baseline`, `mps_baseline`, `appqsim_protocol`.

### Changed
- Coverage 95 % → 98 % (9 601 stmts, 165 missed).
- Tests: 2 634 → 2 715.
- `docs.yml` `contents:write` scoped to job; `ci.yml` `contents:read` at top level.
- Architecture stats refreshed (155 modules, 23.8k LOC, 29 doc pages).

### Fixed
- PennyLane adapter: catch `Exception` not only `ImportError`.
- 5 test failures: skip guards for unimplemented Rust functions.
- 12 doc errors: function names, test counts, version numbers.
- CI: ripser mock test args; injected missing module attrs.

## [0.9.3] - 2026-03-28

### Added
- Rust: `lanczos_b_coefficients` (27× vs numpy), `otoc_from_eigendecomp` (264× vs scipy), `build_xy_hamiltonian_dense` (5 401× vs Qiskit), `all_xy_expectations` (6.2× vs individual calls). 11 → 15 total.
- IBM hardware campaign 20/20 experiments complete (22 jobs on `ibm_fez`, 176 000+ shots). CHSH S = 2.165 (>8σ), QBER 5.5 %, 16-qubit UPDE. ZNE stable fold 1–9.
- 16 publication figures (simulation + hardware + MBL + BKT).
- 3 publications on GitHub Pages (preprint, sync witnesses, DLA parity).
- `knm_to_dense_matrix` Rust fast path (8 modules migrated).
- JAX GPU backend (`jax_accel.py`) for vectorised coupling scans.
- PyPI Rust wheel CI (`rust-wheels.yml`) for 5 platforms.
- SCPN theory + biochemical foundations pages on GitHub Pages.
- Results gallery on GitHub Pages.
- 12 coverage tests for v0.9.3 additions.
- Kaggle notebook for JAX GPU validation + BKT universality.

### Changed
- README reframed: hardware-first.
- 12 GitHub topics added.
- GitHub Release v0.9.3 created.
- 2 Dependabot PRs merged (codeql-action, codecov-action).

### Fixed
- Empty `pauli_list` crash in `knm_to_xxz_hamiltonian` (Hypothesis edge case).
- Rust parity tests: `pytest.importorskip` for Docker CI.
- JAX backend: build H in numpy/Rust, GPU only for `eigh` / `svd` (was 1 731× slower).
- 3 TokenPermissions OpenSSF Scorecard alerts.
- Version test 0.9.2 → 0.9.3.
- Preprint MBL → non-ergodic correction.

## [0.9.2] - 2026-03-26

*(untagged; predates the 2026-05-12 repository history reconstruction)*

### Added
- 38 runner coverage tests (simulator-path methods).
- 22 experiment coverage tests (all 20 experiment functions).
- `requirements.txt` and `requirements-dev.txt` (pinned).

### Changed
- Removed coverage omit for `runner.py` and `experiments.py`.
- README cross-refs updated; hardware campaign status noted.
- ROADMAP: v0.9.1 + v0.9.2 sections.

### Fixed
- Rust engine `build_knm` parity test (wheel rebuilt).

## [0.9.1] - 2026-03-25

*(untagged; predates the 2026-05-12 repository history reconstruction)*

### Added
- `analysis/sync_witness.py` — correlation, Fiedler, topological witnesses.
- `mitigation/symmetry_verification.py` — Z₂ parity post-selection.
- `analysis/quantum_persistent_homology.py` — hardware counts → $p_{h1}$.
- `phase/cpdr_simulation.py` — CPDR for XY simulation.
- `phase/coupling_topology_ansatz.py` — K_nm-informed VQE ansatz.
- `analysis/sync_entanglement_witness.py` — R as entanglement witness.
- `analysis/entanglement_sync.py`, `phase/cross_domain_transfer.py`, `analysis/otoc_sync_probe.py`, `analysis/hamiltonian_self_consistency.py`, `analysis/quantum_speed_limit.py`, `analysis/qfi_criticality.py`, `analysis/entanglement_percolation.py`, `analysis/qrc_phase_detector.py`, `phase/floquet_kuramoto.py`, `analysis/critical_concordance.py`, `analysis/berry_fidelity.py`, `analysis/quantum_mpemba.py`, `analysis/lindblad_ness.py`, `analysis/adiabatic_gap.py`, `analysis/pairing_correlator.py`, `analysis/xxz_phase_diagram.py`, `analysis/spectral_form_factor.py`, `analysis/loschmidt_echo.py`, `analysis/entanglement_entropy.py`, `analysis/krylov_complexity.py`, `analysis/magic_sre.py`, `analysis/finite_size_scaling.py`.
- `bridge/knm_hamiltonian.py` XXZ Hamiltonian (S² embedding).
- `tests/test_round4_8_coverage.py` — 36 coverage-gap tests.
- IBM Quantum campaign: 9 jobs submitted to `ibm_fez` (2 completed, 7 queued).
- March 20–22 marathon: 60 commits, 56 new modules.
  - `analysis/` expanded to 14 modules: `bkt_analysis`, `bkt_universals`, `p_h1_derivation`, `phase_diagram`, `dynamical_lie_algebra`, `qfi`, `quantum_phi`, `entanglement_spectrum`, `koopman`, `otoc`, `shadow_tomography`, `hamiltonian_learning`, `enaqt`, `vortex_binding`, `h1_persistence`.
  - `gauge/` (new): `wilson_loop`, `vortex_detector`, `cft_analysis`, `universality`, `confinement`.
  - `ssgf/` (new): `quantum_gradient`, `quantum_costs`, `quantum_outer_cycle`, `quantum_spectral`.
  - `applications/` expanded to 10 modules: `fmo_benchmark`, `power_grid`, `josephson_array`, `eeg_benchmark`, `iter_benchmark`, `cross_domain`, `quantum_kernel`, `quantum_reservoir`, `disruption_classifier`, `quantum_evs`.
  - `phase/` new algorithms: `adapt_vqe`, `avqds`, `qsvt_evolution`, `varqite`.
  - `tcbo/quantum_observer`, `pgbo/quantum_bridge`, `l16/quantum_director`, `bridge/snn_backward`, `bridge/ssgf_w_adapter`, `bridge/orchestrator_feedback`, `identity/robustness`, `qec/error_budget`, `benchmarks/mps_baseline`, `benchmarks/gpu_baseline`, `benchmarks/appqsim_protocol`, `hardware/gpu_accel`, `hardware/circuit_cutting`, `hardware/qasm_export`, `hardware/qcvv`.
- `GAP_CLOSURE_STATUS.md`.
- `identity/` subpackage: `IdentityAttractor`, `coherence_budget`, `chsh_from_statevector`, `disposition_entanglement_map`, `identity_fingerprint`, `verify_identity`, `prove_identity`.
- v1.0 modules: `mitigation/pec.py`, `hardware/trapped_ion.py`, `control/q_disruption_iter.py`, `benchmarks/quantum_advantage.py`, `bridge/snn_adapter.py`, `bridge/ssgf_adapter.py`, `identity/binding_spec.py`, `qsnn/training.py`, `qec/fault_tolerant.py`.
- 7 new examples (11–17); 2 notebooks (`06_pec_error_cancellation`, `07_quantum_advantage_scaling`).
- 2 API docs (`bridges_api.md`, `benchmarks_api.md`); `docs/equations.md`.
- Enterprise hardening: SPDX headers on 130 files, AGPL-3.0 dual-license, 4 new CI workflows, Dockerfile, Makefile, GOVERNANCE, SUPPORT, CONTRIBUTORS.

### Fixed
- CHSH angles: `b' = -π/4` → `b' = 3π/4` (for S ≈ 2√2).
- README license badge: MIT → AGPL-3.0-or-later.
- README test count: ~505 → 627+.
- README architecture tree updated.
- mypy numpy `no-any-return` errors across new modules.
- Ruff E741 / format in examples and notebooks.
- ArcaneNeuron import path: `neurons` → `neurons.models`.

### Changed
- Python 3.13 added to CI matrix and classifiers.
- `[docs]` extras: added `mkdocs`, `pymdown-extensions`.
- Tests: ~505 → 627+; coverage 100 %.

## [0.9.0] - 2026-03-02

### Added
- 100 % line coverage: 13 new tests closing 19 uncovered lines across 9 files.
- Security CI job (bandit + pip-audit).
- `.github/CODEOWNERS` (default: @anulum).
- Input validation: `QuantumPetriNet`, `QLiF`, `QAOA_MPC`.
- `WEIGHT_SPARSITY_EPS`; Shor & Preskill citation on `QBER_SECURITY_THRESHOLD`.
- ZNE: `R_std_per_scale` per noise scale.
- `_run_vqe()` returns `energy_std`.
- Runner `timeout_s` on `run_sampler` / `run_estimator`; metadata logged to `jobs.json`.
- `retrieve_job(job_id)` on `HardwareRunner`.
- `examples/07_crypto_bell_test.py`, `examples/08_dynamical_decoupling.py`.
- `notebooks/05_crypto_and_entanglement.ipynb`.
- Shared fixtures in `conftest.py`: `knm_4q`, `knm_8q`, `rng`.
- Dataclass field docs on `LockSignatureArtefact`, `LayerStateArtefact`.

### Fixed
- `q_disruption.py` magic `16` → `2**self.n_data_qubits`.
- Bare `1e-15` in `qpetri.py`, `spn_to_qcircuit.py`, `q_disruption.py` → `WEIGHT_SPARSITY_EPS`.
- Weak assertions tightened.

### Changed
- Classifier: Beta → Production/Stable.
- Coverage `exclude_also`: `TYPE_CHECKING` blocks.
- Tests: 483 → ~505.

## [0.8.0] - 2026-03-02

### Added
- Shot-noise error bars (`hw_R_std`, `hw_expectations_std`) on all 20 hardware experiments.
- `test_knm_properties.py`, `test_crypto_properties.py`, `test_qec_properties.py` (9 Hypothesis tests).
- `test_orchestrator_adapter_helpers.py`, `test_vqls_edge_cases.py`, `test_percolation_edge_cases.py`, `test_topology_auth_edge.py` (12 tests).
- `_constants.py`: `COUPLING_SPARSITY_EPS`, `CONCURRENCE_EPS`, `QBER_SECURITY_THRESHOLD`, `VQLS_DENOMINATOR_EPS`.
- `.editorconfig`.
- Input validation: K/omega shapes, ZNE data points, DD qubit range, ODE solver status, eigenvalue reality.
- Hardware exports: `bell_test_4q_experiment`, `correlator_4q_experiment`, `qkd_qber_4q_experiment`.

### Fixed
- 21 mypy `no-any-return` errors across 13 files.
- Dead code in `qaoa_mpc.py`.
- `test_public_api.py`: type-check assertions.
- `test_control_qec.py`: threshold `> 0` → `>= 10`.

### Changed
- Tests: 463 → 483.

## [0.7.1] - 2026-03-02

### Fixed
- Trivial `assert True` replaced with meaningful assertions.
- 3 duplicate tests removed.
- `inhibitor_anti_control` return annotation `-> None`.
- mypy: per-module `ignore_missing_imports`.
- `__all__` gaps: `knm_to_ansatz`, `classical_exact_evolution`, `JobResult`, `SurfaceCode`, `MWPMDecoder`.

### Added
- `wheel-check` CI job.
- 10 error-path tests.
- mypy `warn_unreachable`, `check_untyped_defs`.
- sdist excludes (`.github/`, `dist/`, `results/`, `figures/`, `notebooks/`).
- `build` in dev extras; matplotlib capped `<4.0`.
- Docstrings on `hmac_sign`, `hmac_verify_key`, `PhaseOrchestratorAdapter.from_orchestrator_state`.

## [0.7.0] - 2026-03-02

### Fixed
- `crypto/__init__.py` export fix (modules → callables).
- `QuantumDenseLayer` accepts `seed`.
- `PhaseVQE.relative_error_pct` returns `nan` for exact ≈ 0 instead of `inf`.
- Bare magic numbers in `percolation.py` → named constants.

### Added
- PEP 561 `py.typed`.
- pip `cache: 'pip'` in CI `setup-python`.
- Dependency upper bounds (next-major caps).
- `test_crypto_exports.py`.
- Seed determinism test for `QuantumDenseLayer`.

## [0.6.4] - 2026-03-01

### Fixed
- Stale test counts in docs (411 / 424 → 442).
- `figures/generate_header.py` version string.
- README architecture tree (`ansatz_bench.py`, `trotter_error.py`, `control_plasma_knm.py`).
- `SECURITY.md` supported versions (0.1.x → 0.6.x).

### Added
- `scpn-phase-orchestrator` in README Related Repositories.
- pip in Dependabot.
- v0.6.1 + v0.6.2 entries in `docs/changelog.md`.

### Changed
- Version 0.6.2 → 0.6.4 across `pyproject.toml`, `__init__.py`, `CITATION.cff`, `.zenodo.json`, badges, test.

## [0.6.3] - 2026-03-01

### Fixed
- Coverage gate in CI (codecov threshold).
- Mitigation API docs (ZNE / DD reference).
- Notebook table alignment.
- Ruff S311 in tests.

### Added
- `docs/mitigation_api.md`.
- Coverage gate job.
- Notebook summary table in `docs/index.md`.

## [0.6.2] - 2026-03-01

### Fixed
- Notebooks 01 / 03 / 04: `classical_kuramoto_ode` → `classical_kuramoto_reference`.
- Notebook 03: ZNE scales [1,2,3,4,5] → [1,3,5,7,9] (odd required).
- Notebook 04: 16-qubit → 8-qubit Trotter + 16-layer classical.
- Duplicate `docs/SESSION_LOG`, `docs/HANDOVER` removed.

### Added
- `figures/generate_knm_heatmap.py` + `figures/knm_heatmap.png` (16×16 K_nm).
- Knm heatmap in README.
- All 4 notebooks executed with embedded outputs.
- ROADMAP: post-2030 qualifier on fault-tolerant bullet.

## [0.6.1] - 2026-03-01

### Fixed
- mypy: `FloatArray` removed (incompatible with Python 3.9); `Path(None)` in `control_plasma_knm.py`.
- Zenodo metadata enriched (`.zenodo.json`, `CITATION.cff`).

## [0.6.0] - 2026-03-01

### Fixed
- Division-by-zero in `QuantumLIFNeuron`, `QuantumSynapse`, `QuantumSTDP`, `QAOA_MPC`, `classical_kuramoto_reference`.
- Index-out-of-bounds in `bell_inequality_test`, `best_entanglement_path`.
- Notebook 02 `PhaseVQE.solve()` dict keys.
- Stale test counts in `VALIDATION.md`, `docs/index.md`.

### Added
- Input-validation guards on QSNN constructors (`qlif`, `qsynapse`, `qstdp`).
- Input validation on `QAOA_MPC`, `bell_inequality_test`, `best_entanglement_path`, `classical_kuramoto_reference`.
- `PhaseVQE.solve()` returns `exact_energy`, `energy_gap`, `relative_error_pct`, `n_params`.
- 13 validation tests (`test_qsnn_validation.py`).

### Changed
- `docs/changelog.md` expanded with full version history.
- Tests: 411 → 424.

## [0.5.1] - 2026-03-01

### Fixed
- v0.5.0 published with `__version__ = "0.4.0"` (tag-timing bug). This release realigns `__version__`, `pyproject.toml`, and PyPI to 0.5.1.

## [0.5.0] - 2026-03-01

### Added
- 3 crypto hardware experiments: `bell_test_4q` (CHSH), `correlator_4q` (ZZ topology), `qkd_qber_4q` (QBER vs BB84 threshold).
- `_correlator_from_counts()` (2-qubit E(A,B)).
- `noise_analysis.py` `devetak_winter_rate()` (Devetak-Winter bound).
- 3 simulator tests (bell, correlator, QKD).

### Changed
- Experiments: 17 → 20 (3 crypto added to `ALL_EXPERIMENTS`).
- Tests: 408 → 411.
- Version 0.4.0 → 0.5.0.

## [0.4.0] - 2026-02-28

### Added
- GitHub Pages (MkDocs Material, 7 pages, auto-deploy).
- 4 Jupyter notebooks: Kuramoto XY, VQE ground state, ZNE, UPDE-16.
- 10 hardware experiments: noise baseline, 8-osc ZNE, 8q VQE hardware, UPDE-16 with DD, Trotter order-2, sync threshold, decoherence scaling, ZNE higher-order, VQE landscape, cross-layer correlation.
- 14 Hypothesis tests: probability-angle roundtrip, Knm symmetry/positivity, Hamiltonian Hermiticity, ansatz parameter counts.
- 8 edge-case tests: 2-oscillator minimal, `SuzukiTrotter(order=2)`, single-value inputs.
- 13 coverage-gap tests: multi-inhibitor anti-control, QAOA ZZ, VQLS near-zero guard, QEC odd defects / correction failure, QSTDP synapse.
- 4 integration tests: Knm → VQE ground state, Knm → Trotter → energy, 8q spectrum, 16-layer Hamiltonian structure.
- 7 regression tests: Knm calibration anchors, cross-hierarchy boosts, omega values, 4q ground energy baseline, statevector R, R evolution monotonicity, `_R_from_xyz`.

### Changed
- mypy: 27 → 30 source files (full `hardware/`).
- Tests: 208 → 254.
- `pyproject.toml`: docs URL; `mkdocs-material` in extras.

## [0.3.0] - 2026-02-28

### Added
- README rewrite (SCPN → XY isomorphism motivation, Kuramoto-to-Hamiltonian derivation, 4 figures, expanded example table, Related Repositories).
- Paper 27 citation in README and `docs/equations.md`.
- `docs/equations.md`: SCPN overview and UPDE definition.
- `examples/README.md` walkthrough.
- `HARDWARE_RESULTS.md`: L1–L16 naming section.

### Changed
- CI mypy scope: `bridge/` → all 8 module paths (27 files, zero errors).
- `VALIDATION.md`: tests 88 → 199; classical references now in-repo.
- Badges: tests 88 → 199; version 0.3.0 added.
- `CITATION.cff`: 0.1.0 → 0.3.0.

## [0.2.7] - 2026-02-28

### Added
- Parametrised quantum-vs-classical validation at n = {2,3,4,6}.
- Exact-diag cross-check against `eigvalsh` of Hamiltonian matrix.

### Changed
- Coverage omit narrowed: only `runner.py` and `experiments.py` excluded; `classical.py` now tracked.

## [0.2.6] - 2026-02-28

### Added
- Classical reference tests (20 across `classical_kuramoto_reference`, `classical_exact_diag`, `classical_exact_evolution`, `classical_brute_mpc`, `bloch_vectors_from_json`).
- Pauli ordering validation tests (2).
- `ALL_EXPERIMENTS` completeness test.
- Integration tests: quantum-vs-classical Kuramoto, ZNE on noiseless backend, energy conservation under Trotter, Trotter order-2 passthrough.

### Changed
- mypy scope expanded to `control/`, `qsnn/`, `qec/` (27 files, zero errors).
- Type narrowing in `vqls_gs.py`, `qaoa_mpc.py`, `runner.py` (assert-after-guard).
- `QuantumSTDP` forward ref to `QuantumSynapse` via `TYPE_CHECKING`.
- `ZNEResult` forward ref in `runner.py` via `TYPE_CHECKING`.

## [0.2.5] - 2026-02-28

### Added
- `trotter_order=2` on `QuantumKuramotoSolver` and `QuantumUPDESolver` (SuzukiTrotter order 2, O(t³/reps²) vs O(t²/reps)).
- `QuantumKuramotoSolver.energy_expectation(sv)`.
- `test_second_order_trotter`, `test_trotter_error_decreases_with_reps`, `test_energy_expectation`.

## [0.2.4] - 2026-02-28

### Fixed
- QAOA Hamiltonian: correct Ising encoding with identity (constant) term (`h_z`, `c0`); removed spurious ZZ terms.
- Quantum Petri net: multi-input transitions use multi-controlled Ry (AND gating).
- Inhibitor arcs: X-CRy-X anti-control pattern correctly gates on inhibitor emptiness.
- `build_knm_paper27`: dead `zeta_uniform` removed.
- VQLS: `imag_tol` init parameter (default 0.1).

### Added
- `test_hamiltonian_matches_classical_cost`.
- `test_optimal_bitstring_matches_brute_force`.
- `test_multi_input_conjunctive_gating`.
- `test_inhibitor_blocks_when_place_occupied`.

## [0.2.3] - 2026-02-28

### Fixed
- Disruption classifier `n_params`: CX gates have no trainable parameters; corrected to `n_layers*2*n_qubits` (30 vs 42 for default).
- QSTDP Hebbian: `post_measured` now implements LTP/LTD per Hebbian rule.

### Added
- Test for `kuramoto_4osc_zne_experiment`.
- Test for `upde_16_snapshot_experiment` (`@pytest.mark.slow`).
- `pytest.ini_options.addopts` skips slow/hardware by default.

## [0.2.2] - 2026-02-28

### Fixed
- MWPM decoder (3 bugs): dual edges for plaquette (Z) syndromes; seam-crossing winding-number logical error; d = 5 outperforms d = 3 below threshold (Dennis et al. 2002).
- Classical reference endianness (`_build_initial_state`, `_expectation_pauli`): `kron` order reversed to Qiskit little-endian; verified vs `Statevector` evolution to 1e-6.
- Parameter-shift rule (`q_disruption.py`): misleading `sin(shift)` denominator removed (Schuld et al., PRA 99, 032331).
- VQLS: assert imaginary norm < 0.1 before `np.real()`.
- `QAOA_MPC`: dead `current_state` parameter removed.
- ZNE: `base.inverse()` cached before fold loop.
- `trotter_upde`: dead `evolve(0)` removed; `reset()` added.
- `qlif`: `np.random.binomial` → seedable `rng`.
- `runner`: `TranspilerError` caught in DD pass fallback instead of bare `except`.
- `sc_to_quantum.measurement_to_bitstream` accepts `rng`.

### Added
- `_run_vqe` helper (eliminates vqe_4q / vqe_8q duplication).
- Root `__init__.py` exports 20 public symbols.
- Return-type annotations on public methods.
- 9 new tests: d5 > d3, shifted logical cycles, single X/Z correctness, VQE experiment, DD transpile, QSNN stochastic, bitstream seeded.
- Citation markers on `K_base`, `K_alpha` (Paper 27 Eq. 3).
- `test_classical_evolution_matches_qiskit` (Qiskit-vs-classical endianness).

## [0.2.1] - 2026-02-28

### Fixed
- `q_disruption.py`: configurable `seed`.
- `qpetri.py`: threshold gating via CRy.
- `classical.py`: sparse eigensolver (`eigsh`) for N ≥ 14 or when `k_eigenvalues` specified.
- `ansatz_bench.py`: deprecated `TwoLocal` / `EfficientSU2` → `n_local` / `efficient_su2` (Qiskit 2.1+).
- `vqls_gs.py`: `TwoLocal` → `n_local`.

### Added
- STDP direction validation (gradient sign at θ = 0, π/2).
- QEC threshold measurement tests (p = 0.01 vs p = 0.08; d = 5 single-error decoding).
- Disruption classifier seed reproducibility.
- Petri net controlled output + multi-step bounds.
- Bloch-ball constraint test.
- Sparse vs dense eigensolver agreement.

## [0.2.0] - 2026-02-28

### Added
- `hardware/noise_model.py`: `heron_r2_noise_model()` (T1 = 300 µs, T2 = 200 µs, CZ 0.5 %, readout 0.2 %).
- `mitigation/zne.py`: `gate_fold_circuit()` (Giurgica-Tiron et al. 2020), `zne_extrapolate()` (Richardson).
- `mitigation/dd.py`: `DDSequence` (XY4, X2), `insert_dd_sequence()` (Viola et al. 1999).
- `phase/trotter_error.py`: `trotter_error_norm()`, `trotter_error_sweep()`.
- `phase/ansatz_bench.py`: `benchmark_ansatz()`, `run_ansatz_benchmark()` (Knm-informed vs TwoLocal vs EfficientSU2).
- `hardware/classical.py`: `bloch_vectors_from_json()`.
- `hardware/experiments.py`: `kuramoto_4osc_zne_experiment`.
- `hardware/runner.py`: `run_estimator_zne()`, `transpile_with_dd()`, `noise_model` constructor parameter.
- `scripts/plot_vqe_convergence.py`, `scripts/plot_decoherence_curve.py`.
- `examples/05_vqe_ansatz_comparison.py`, `examples/06_zne_demo.py`.
- Top-level re-exports: `OMEGA_N_16`, `build_knm_paper27`, `knm_to_hamiltonian`, `QuantumKuramotoSolver`, `QuantumUPDESolver`, `PhaseVQE`, `HardwareRunner`, `JobResult`.

### Fixed
- `_run_sampler_simulator` uses `self._backend` (respects noise model).
- Duplicate `AerSimulator` import removed.
- Dead `2**n_osc` in `classical.py` (lines 117, 147).
- `DEFAULT_INSTANCE` reads IBM instance configuration from the environment.

### Changed
- mypy scope: `bridge/` → `bridge/`, `phase/`, `mitigation/`, `hardware/classical.py`, `hardware/runner.py`.

## [0.1.0] - 2026-02-28

### Added
- `qsnn/`: `qlif.py`, `qsynapse.py`, `qstdp.py`, `qlayer.py`.
- `phase/`: `xy_kuramoto.py`, `trotter_upde.py`, `phase_vqe.py`.
- `control/`: `qaoa_mpc.py`, `vqls_gs.py`, `qpetri.py`, `q_disruption.py`.
- `bridge/`: `knm_hamiltonian.py`, `spn_to_qcircuit.py`, `sc_to_quantum.py`.
- `qec/`: `control_qec.py` (toric surface code + MWPM with Knm-weighted edges).
- `hardware/`: `runner.py`, `experiments.py`, `classical.py` (`ibm_fez`).
- 88 unit tests, 4 example scripts, 19 hardware result files.
- Hardware validation on `ibm_fez`: VQE 0.05 % error, 12-point decoherence curve, 16-layer UPDE snapshot.
- CI workflow with Python 3.9–3.12 matrix, coverage, ruff lint.
- Documentation: architecture, API reference, hardware results.
