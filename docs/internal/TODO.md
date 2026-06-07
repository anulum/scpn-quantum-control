# Internal TODO

Private execution queue for SCPN-QUANTUM-CONTROL. Keep this file internal; do not promote planning debt into public docs.

Canonical rule:

- This is the only active internal TODO for this repository.
- Specialised internal plans, gap registers, checklists, session logs, and handovers are supporting records only.
- Every active item in a specialised record must be copied here or explicitly linked from here.
- Supporting records must stay under ignored internal paths and must not use competing `TODO` filenames.
- Public roadmaps and published evidence pages describe product direction or historical artefacts; they are not the active task queue.

Specialised internal registers linked from this queue:

- Differentiable programming state-of-art gap register: `docs/internal/differentiable_programming/2026-06-02_state_of_art_gap_register.md`.
- MIF-core downstream quantum gap register: `docs/internal/cross_repo_requirements/2026-06-03_mif_core_quantum_gap_register.md`.
- Visibility and registration campaign gap register: `docs/internal/strategy/2026-05-06_visibility_campaign_gap_register.md`.
- Grant, publication, operations, strategy, and topological optimisation records under `docs/internal/` are specialised supporting records; copy any active task that still matters into this file.
- Timestamped `.coordination/sessions/` and `.coordination/handovers/` files are immutable historical logs, not active task queues.
- Historical task fragments consolidated into this queue: `.coordination/archive/consolidated_task_fragments/`.

## Coverage gap inventory (as of 2026-06-07)

- Source used: `coverage.xml` (current file timestamp: 2026-05-22).
- Source root: `src/scpn_quantum_control`
- Threshold used for this inventory: 100% file-level line coverage.
- Files below 100%: `665`
  - `below_threshold`: `149`
  - `missing_from_report`: `516`
- Missing-from-report files are modules that existed in source but were not exercised by the covered test run.

Full table (all files below 100%): `docs/internal/coverage_below_100_files_2026-06-07.md`

## 2026-06-05 - Differentiable programming world-class promotion queue

Status as of 2026-06-05:

- Local framework parity passes across `SCPN`, `JAX`, `PyTorch`, `TensorFlow`, and `PennyLane`.
- Registered Phase-QNode execution exists.
- Parameter-shift gradients exist.
- Compiler lowering metadata exists.
- Medium model-training evidence exists.
- Benchmark harness exists, but current benchmark evidence is still `functional_non_isolated`.

Promotion rule:

- Do not make a public world-leading or state-of-the-art claim until every claim has implementation, tests, benchmark evidence, docs, and known-gap records.
- Treat current status as a SOTA-candidate evidence lane until isolated benchmark evidence and external comparisons are complete.

### P0: Promotion blockers

- [x] Make the TensorFlow/PyTorch overlay reproducible.
  - [x] Document the ext4-backed framework overlay install profile.
  - [x] Add a CI-equivalent CPU-wheel install path.
  - [x] Ensure parity does not depend on hidden workstation state.
- [ ] Produce true `isolated_affinity` benchmark evidence.
  - [ ] Reserve a CPU set.
  - [ ] Record low host load before and after runs.
  - [ ] Record governor and frequency metadata.
  - [ ] Confirm no heavy concurrent jobs.
  - [ ] Commit small raw JSON/CSV evidence or generated summaries from ignored raw artifacts.
- [x] Add external benchmark comparisons.
  - [x] Compare against JAX native AD.
  - [x] Compare against PyTorch `torch.func`.
  - [x] Compare against TensorFlow `GradientTape`.
  - [x] Compare against PennyLane QNodes.
  - [x] Compare against optional Enzyme/compiler AD with hard-gap classification when tooling is absent.
  - [x] Measure value accuracy, gradient error, runtime, memory, batching behavior, and failure classification.
- [x] Create a formal claim ledger.
  - [x] Map every claim to implementation.
  - [x] Map every claim to tests.
  - [x] Map every claim to benchmark evidence.
  - [x] Map every claim to docs.
  - [x] Map every claim to known gaps.

### P1: Core differentiable engine

- [ ] Expand Phase-QNode circuit support.
  - [ ] Add multi-qubit registered templates.
  - [ ] Add arbitrary registered-depth circuits.
  - [ ] Add more controlled gates and decompositions.
  - [ ] Add density-matrix and noisy-channel routes.
  - [ ] Add strict support reports for every blocked path.
- [ ] Expand observable support.
  - [ ] Add dense Hermitian observables.
  - [ ] Add larger-scale sparse Hamiltonians.
  - [ ] Add covariance observables.
  - [ ] Add Fisher information.
  - [ ] Add quantum Fisher information.
  - [ ] Add natural-gradient metrics.
- [ ] Add higher-order differentiation.
  - [ ] Add native `jvp`.
  - [ ] Add native `vjp`.
  - [ ] Add native `jacfwd`.
  - [ ] Add native `jacrev`.
  - [ ] Add Hessians for deterministic local simulators.
  - [ ] Add Hessian-vector products.
  - [ ] Add complex and Wirtinger derivative contracts.
- [ ] Add stochastic and finite-shot gradient estimators.
  - [ ] Add parameter-shift with shot variance.
  - [ ] Add SPSA.
  - [ ] Add score-function estimators where mathematically valid.
  - [ ] Add confidence intervals and failure policies.

### P2: Native framework integration

- [ ] Build the JAX-native path.
  - [ ] Add `custom_vjp` or `custom_jvp`.
  - [ ] Add `jit` compatibility.
  - [ ] Add `vmap` compatibility.
  - [ ] Add `pmap` or sharding support where meaningful.
  - [ ] Add PyTree parameter support.
- [ ] Build the PyTorch-native path.
  - [ ] Add `torch.autograd.Function`.
  - [ ] Add `torch.func.grad`, `vmap`, and `jacrev` compatibility.
  - [ ] Add `torch.compile` compatibility.
  - [ ] Add module and layer wrappers.
- [ ] Build the TensorFlow-native path.
  - [ ] Add `GradientTape` compatibility.
  - [ ] Add `tf.function` compatibility.
  - [ ] Add XLA-compatible bounded kernels.
  - [ ] Add Keras layer wrappers.
- [ ] Mature the PennyLane bridge.
  - [ ] Add round-trip QNode conversion.
  - [ ] Add gradient-method parity.
  - [ ] Add device metadata and shot-policy mapping.

### P3: Compiler and runtime

- [ ] Move MLIR lowering from metadata to executable lowering.
  - [ ] Add real dialect operations.
  - [ ] Add shape and type verifier.
  - [ ] Add lowering tests.
  - [ ] Block interpreter fallback from being reported as compiled success.
- [ ] Add LLVM/Enzyme-backed AD experiments.
  - [ ] Use Enzyme for Rust, C, C++, or MLIR-compatible differentiable kernels where practical.
  - [ ] Add correctness checks against native parameter-shift and finite-difference references.
  - [ ] Add performance evidence against Python-level execution.
- [ ] Add Rust/PyO3 parity.
  - [ ] Implement Rust kernels for supported promoted paths.
  - [ ] Add Python/Rust result parity tests.
  - [ ] Add Rust benchmark path for every promoted primitive.
- [ ] Add GPU and accelerator paths.
  - [ ] Promote CPU first.
  - [ ] Add CUDA or ROCm where hardware exists.
  - [ ] Record explicit device metadata.
  - [ ] Prevent silent CPU fallback in benchmark claims.

### P4: Model and scientific evidence

- [ ] Expand training suites.
  - [ ] Add QNN suites.
  - [ ] Add QGNN suites.
  - [ ] Add QSNN suites.
  - [ ] Add Kuramoto-XY suites.
  - [ ] Add open-system control suites.
  - [ ] Add inverse-coupling recovery suites.
  - [ ] Add multi-seed convergence.
  - [ ] Add loss landscapes.
  - [ ] Add gradient agreement evidence.
  - [ ] Add unsuitable-scenario records.
- [ ] Add domain benchmark datasets.
  - [ ] Add synthetic exact-answer cases.
  - [ ] Add published physics/control cases.
  - [ ] Preserve cited mathematical formulations without simplifying away required terms.
- [ ] Add optimizer comparisons.
  - [ ] Add SGD.
  - [ ] Add Adam.
  - [ ] Add L-BFGS.
  - [ ] Add natural gradient.
  - [ ] Add SPSA.
  - [ ] Add CMA-ES or another derivative-free baseline.
  - [ ] Record runtime and convergence evidence.

### P5: Reliability, UX, and release

- [ ] Build a unified differentiable API.
  - [ ] Add one namespace for value, gradient, Jacobian, Hessian, compile, benchmark, and support reports.
  - [ ] Stabilize the JSON evidence schema.
- [ ] Improve diagnostics.
  - [ ] Add "why cannot this differentiate?" reports.
  - [ ] Add suggested alternatives.
  - [ ] Add dependency, device, and backend matrices.
- [ ] Add tutorials and examples.
  - [ ] Add minimal QNode example.
  - [ ] Add framework-native examples.
  - [ ] Add compiler example.
  - [ ] Add training example.
  - [ ] Add benchmark reproduction example.
- [ ] Expand CI and reproducibility.
  - [ ] Cover Python 3.10 through 3.13.
  - [ ] Add CPU sparse and full lanes.
  - [ ] Add optional GPU lane.
  - [ ] Add scheduled benchmark artifacts.
  - [ ] Always enforce the module-specific test audit.
- [ ] Build an external validation package.
  - [ ] Add comparison paper or technical report.
  - [ ] Add reproducible artifact bundle.
  - [ ] Add exact environment lockfiles.
  - [ ] Add public claim table.

Recommended next lane:

- Start with native framework transforms plus isolated benchmark evidence plus executable compiler lowering.
- This is the shortest path from SOTA-candidate to a defensible world-class claim.

## Differentiable Program AD state-of-art closure queue

Supporting detail: `docs/internal/differentiable_programming/2026-06-02_state_of_art_gap_register.md`.

Status as of 2026-05-25:

- Current program AD is exact for supported executed Python/NumPy operations through derivative-carrying scalar/array values.
- Current program AD is not yet full state-of-the-art arbitrary Python AD.
- Unsupported derivative-losing behavior must continue to fail closed; no finite-difference fallback may be hidden under whole/program AD names.

Required to reach state-of-art / best-in-class:

1. Program IR completeness
   - 2026-05-25 update: program AD results now emit deterministic `ProgramADEffectIR` metadata with versioned SSA values, ordered effects, mutation/alias edges, runtime branch regions, source control regions, and stable serialization for supported executed programs.
   - Remaining: replace metadata capture with a complete bytecode/source compiler frontend, not only runtime operator graph evidence.
   - Represent branches, loops, phi nodes, array views, mutation, alias edges, and effect ordering.
   - Add phi nodes across non-executed branches, full alias lattice, array view identity, round-trip parser, and round-trip tests.

2. Reverse-mode whole-program AD
   - 2026-05-25 update: supported scalar program AD traces now attach `ProgramADAdjointResult` from reverse-mode adjoint replay over captured IR nodes, with parity tests against forward program AD and fail-closed unsupported-op reporting.
   - Implement reverse-mode adjoint generation over the program IR.
   - Remaining: replace replay metadata with reverse-mode adjoint generation over the stabilized program IR.
   - Support loops with tape/checkpoint semantics beyond unrolled executed scalar traces.
   - Support branch adjoints for non-executed path semantics and documented nondifferentiable boundary behavior.
   - Add gradient parity against forward program AD for supported programs.

3. Alias/effect analysis
   - Implement alias sets for arrays, slices/views, lists, object attributes, and local variable rebinding.
   - Track mutation effects with versioned values and fail closed on unknown aliasing.
   - Add mutation semantics tests for array views, list aliases, slice writes, and loop-carried state.

4. NumPy semantics expansion
   - 2026-05-25 update: primitive transform registry entries now carry derivative, batching, lowering, shape, dtype, nondifferentiability, and effect metadata per primitive identity.
   - 2026-05-25 update: primitive registry now exposes fail-closed accessors for shape rules, dtype rules, nondifferentiability policy, and effect classification, and partial batching/lowering updates preserve that contract metadata.
   - 2026-05-25 update: program AD now supports exact bounded decompositions for `inner`, `outer`, `trace`, `diag`, `tensordot` axes 0/1 selected one-axis contractions, and explicit `einsum` dot/outer/matmul/trace/diag forms; `sort`/`argsort` fail closed pending explicit nondifferentiability policy.
   - 2026-05-25 update: program AD binary ufuncs, comparisons, `where`, and helper multiplication now share NumPy-compatible broadcasting for supported trace-array ranks, with adjoint parity tests and fail-closed incompatible-shape diagnostics.
   - 2026-05-25 update: program AD rank-N arrays now support axis reductions for `sum`/`mean`, reversed-axis `.T`, explicit `np.transpose(..., axes=...)`, adjoint parity tests, and invalid-axis fail-closed diagnostics.
   - Remaining: replace ad hoc function coverage with registry-dispatched program AD operations.
   - Add broadcasting beyond scalar broadcast.
   - Add rank-N reshape/index/reduction semantics.
   - Add `einsum`, `tensordot`, `outer`, `inner`, `diag`, `trace`, `sort` fail-closed semantics, and supported linear algebra primitives where mathematically differentiable.

5. Python semantics expansion
   - Add support or explicit fail-closed diagnostics for closures, default arguments, kwargs, comprehensions, dataclass/object attributes, recursion, generators, context managers, exceptions, and decorators.
   - Add bytecode/source tests for each accepted or rejected construct.

6. Higher-order transform algebra
   - Prove composition for `grad`, `jacfwd`, `jacrev`, `hessian`, `jvp`, `vjp`, `vmap`, custom rules, and program AD.
   - Add nesting tests for program AD under `vmap` and transform combinations.

7. Polyglot/compiler chain
   - 2026-05-25 update: compiler AD transform plans now export primitive shape-rule, dtype-rule, nondifferentiability-policy, and effect metadata into deterministic MLIR/interchange status records.
   - Lower stabilized program AD IR to MLIR dialect operations.
   - Add Rust-side IR structs/interpreter if Rust primitive surfaces are present or planned.
   - Add LLVM/JIT lowering only when executable differentiated kernels are actually generated and runtime-verified.
   - Keep Rust/LLVM status blocked until real execution exists.

8. Numerical and physics-grade robustness
   - 2026-05-25 update: program AD selection primitives now fail closed at nondifferentiable `maximum`/`minimum` ties and `clip` boundary points in forward tracing, not only adjoint replay.
   - 2026-05-25 update: ordering predicates used for program AD branches and `where` conditions now fail closed at equality boundaries instead of selecting an arbitrary active path.
   - Add nondifferentiability policy for abs/max/min/clip/where boundary points.
   - Add conditioning diagnostics for norm/linear algebra primitives.
   - Add finite-value, dtype, shape, and trainable-mask invariants at every transform boundary.

9. Benchmarks and conformance
   - 2026-05-25 update: added deterministic differentiable-programming conformance benchmark rows for loop-heavy, mutation-heavy, matrix-heavy, and transform-nesting program AD cases against analytic references, with explicit non-performance claim boundaries.
   - Add behavioural conformance tests, not bucket tests.
   - Add targeted benchmark cases for loop-heavy, mutation-heavy, and matrix-heavy program AD.
   - Compare forward/reverse/program AD outputs against analytic references and trusted finite-difference diagnostics only as diagnostics, never as the implementation.

10. Documentation truthfulness
   - Public docs must state exact supported semantics and fail-closed boundaries.
   - Capability manifests must be regenerated after every public API/capability change.
   - Do not call planning, interchange, metadata, or finite-difference paths whole-program AD.

## Publication DOI closure queue after Schneider/Paredes thread

Status as of 2026-05-25:

- Dr. Johannes Schneider wrote on 2026-05-25 that he cannot pursue quantum-computing publications without an authority from the field and that unofficial collaboration is not possible.
- Miguel Paredes wrote on 2026-05-24 that he will not participate as an author because his contribution would be marginal.
- The collaboration path should therefore be treated as closed unless they later reopen it.
- The intended role of external collaborators was content verification and possible academic guarantor support, not mandatory ownership of the work.
- The remaining publication route is independent archival publication: Zenodo first, then copies on Academia.edu.

## SCPN-MIF-CORE downstream quantum gap queue

Supporting detail: `docs/internal/cross_repo_requirements/2026-06-03_mif_core_quantum_gap_register.md`.

Required modules from the downstream MIF compatibility review:

- [ ] QUA-C.1: QRNG streaming harness with NIST SP 800-22 health checks.
- [ ] QUA-C.2: NIST-approved post-quantum trigger signer.
- [ ] QUA-C.6: Sub-microsecond realtime runtime tracker.
- [ ] QUA-C.3: QAOA-MPC FRC pulsed-shot cost function.
- [ ] QUA-C.4: Pulse-to-UltraScale+ HLS code generation.
- [ ] QUA-C.5: NV-centre 20 T magnetometry calibration, hardware-blocked.

## Historical consolidated task fragments

The following ignored local files are no longer active task queues. They remain
only as provenance for already-consolidated work:

- `.coordination/archive/consolidated_task_fragments/2026-04-06_new_modules.md`
- `.coordination/archive/consolidated_task_fragments/2026-04-07_strategic_tweaks.md`
- `.coordination/archive/consolidated_task_fragments/2026-04-29_high_impact_execution.md`
- `.coordination/archive/consolidated_task_fragments/coverage_gap_ledger.md`

Required publication sequence:

1. Paper inventory
   - Enumerate every publishable manuscript and its current source/PDF pair.
   - Start from `paper/submissions/` and `paper/submissions_joss/`.
   - Confirm which manuscripts are ready for archival publication and which still need claim-boundary edits.

2. Authorship and acknowledgements
   - Keep authorship limited to actual intellectual contribution.
   - Do not add Dr. Schneider or Miguel Paredes as authors unless they explicitly approve and contribute materially.
   - If appropriate, acknowledge prior discussion only in neutral wording and only after checking that the acknowledgement is acceptable.

3. Zenodo DOI reservation
   - Reserve one Zenodo DOI per paper package before public announcement.
   - Capture concept DOI, version DOI, deposition ID, title, authors, version, upload files, SHA-256 checksums, and publication date.
   - Keep repository software DOI separate from manuscript DOIs unless a paper is explicitly a software archive companion.

4. Paper metadata and references
   - Update each manuscript with its own DOI once reserved.
   - Add DOI references to sibling papers only where there is a real dependency or cited result.
   - Update BibTeX files, CITATION surfaces, and publication-planning notes consistently.
   - Do not cite a DOI before it is reserved or published.

5. Archive contents
   - Each Zenodo package must include the PDF, source archive or TeX sources, bibliography, reproducibility manifest, and evidence pointers.
   - Include raw-data archives only when the paper depends on them and the package size/provenance rules are satisfied.
   - Record SHA-256 hashes for every uploaded artifact.

6. Academia.edu copies
   - Upload only after Zenodo publication or DOI reservation is stable.
   - Academia copies must point back to the Zenodo DOI as the canonical archive.
   - Do not treat Academia as the archival record.

7. Repository updates after DOI allocation
   - Update paper references, README publication links where applicable, docs publication surfaces, and generated capability/publication manifests if they include paper metadata.
   - Run the repository's publication/documentation checks before committing public DOI edits.

8. Claim hygiene
   - Preserve the current claim boundaries: source-bounded, hardware-bounded, and validation-bounded statements only.
   - Do not imply journal acceptance, peer review, institutional endorsement, or collaborator endorsement from Zenodo publication.
   - State that Zenodo is the archival publication venue available for independent release.

- 2026-05-25: Program AD NumPy extreme reductions now support np.max/np.min with strict unique-selector adjoints and fail-closed tie boundaries; remaining selection work includes documented differentiable policies for non-strict selection/sort primitives.

- 2026-05-25: Program AD static gather coverage now includes np.take and TraceADArray.take with exact repeated-index adjoint accumulation and fail-closed dynamic-index/mode boundaries.

- 2026-05-25: Program AD index-valued selection boundaries now explicitly fail closed for argmax/argmin function and method forms pending a primitive nondifferentiability policy.

- 2026-05-25: Program AD product reductions now support np.prod and TraceADArray.prod with exact product-rule adjoints, including single-zero factors.

- 2026-05-25: Program AD variance and standard-deviation reductions now support np.var/np.std and method forms with exact differentiable composition and fail-closed ddof validation.

- 2026-05-25: Program AD cumulative sums now support np.cumsum and TraceADArray.cumsum with exact prefix-adjoint semantics for flat and axis-specific arrays.

- 2026-05-25: Program AD cumulative products now support np.cumprod and TraceADArray.cumprod with exact prefix product adjoints, including single-zero prefixes.

- 2026-05-25: Program AD finite differences now support np.diff with repeated axis-specific linear adjoints and fail-closed boundary-extension validation.

- 2026-05-25: Program AD like-constructors now support zeros_like, ones_like, and full_like with explicit derivative-zero constants and fail-closed shape overrides.

- 2026-05-25: Program AD broadcast_to now preserves broadcasted derivative paths and rejects subclass propagation through subok.

- 2026-05-25: Program AD basic indexing now supports static integer, slice, ellipsis, and newaxis selectors while failing closed on advanced indexing.

- 2026-05-25: Program AD reverse adjoint replay now supports static setitem effects whose scalar dataflow is represented in the captured IR.

- 2026-05-25: Primitive registry now exposes unified PrimitiveContract lookups spanning derivative, batching, lowering, shape, dtype, nondifferentiability, and effect contracts.

- 2026-05-25: Transform algebra behavioural tests now bind nested grad, vmap, jacfwd, jacrev, hessian, JVP, and VJP against analytic references.

- 2026-05-25: Executable compiler AD kernels now expose verified scalar-output gradient execution through MLIR-runtime VJP cotangent-one semantics while LLVM/JIT remains fail-closed.

- 2026-05-25: Executable compiler AD kernels now emit deterministic LLVM-style scalar-gradient IR provenance for verified MLIR-runtime scalar gradients while native LLVM/JIT remains fail-closed.

- 2026-05-25: Program AD now supports exact np.reciprocal forward tangents and reverse adjoint replay with zero-input singular boundaries fail-closed.

- 2026-05-25: Primitive registry now exposes fail-closed complete contracts requiring derivative, batching, lowering, shape, dtype, nondifferentiability, effect, and lowering metadata facets for compiler/vectorization consumers.

- 2026-05-25: Transform algebra now exposes canonical jvp/value_and_jvp and vjp/value_and_vjp names over validated finite-difference JVP/VJP backends with nested vmap/jacobian/hessian behavioural tests.

- 2026-05-25: Program AD now supports exact np.log1p and np.expm1 forward tangents plus reverse adjoint replay, with log1p domain boundaries fail-closed.

- 2026-05-25: Program AD now supports exact np.tan forward tangents and reverse adjoint replay, with cosine-zero tangent singularities fail-closed.

- 2026-05-25: Program AD now supports exact np.arcsin and np.arccos forward tangents plus reverse adjoint replay, with inverse-trig singular and invalid domains fail-closed.

- 2026-05-25: Program AD shape-only array transforms now support exact squeeze and expand-dims semantics with reverse adjoint parity and fail-closed invalid axes.

- 2026-05-25: Program AD rank-N axis permutation coverage now supports exact np.swapaxes, TraceADArray.swapaxes, and np.moveaxis semantics with reverse adjoint parity and fail-closed invalid axes.

- 2026-05-25: Program AD static roll coverage now supports exact np.roll flattened and axis-specific permutation semantics with reverse adjoint parity and fail-closed invalid shift/axis contracts.

- 2026-05-25: Program AD flip-family coverage now supports exact np.flip, np.flipud, and np.fliplr permutation semantics with reverse adjoint parity and fail-closed invalid axis/rank contracts.

- 2026-05-25: Program AD rotation coverage now supports exact np.rot90 static-axis permutation semantics with reverse adjoint parity and fail-closed invalid k/axis contracts.

- 2026-05-25: Program AD repeat coverage now supports exact np.repeat and TraceADArray.repeat semantics with reverse adjoint accumulation and fail-closed invalid repeat/axis contracts.
- 2026-05-25: Program AD tiling coverage now supports exact np.tile semantics with repeated-source adjoint accumulation and fail-closed invalid static repetition contracts.
- 2026-05-25: Program AD rank-lift coverage now supports exact np.atleast_1d, np.atleast_2d, and np.atleast_3d semantics for single and multi-input arrays with reverse adjoint parity.
- 2026-05-25: Program AD reshape coverage now supports one NumPy-compatible inferred -1 dimension with exact shape-only adjoint preservation and fail-closed ambiguous or size-losing shapes.
- 2026-05-25: Program AD real linear algebra coverage now supports exact np.vdot flattened inner-product semantics with reverse adjoint parity and fail-closed size mismatches.
- 2026-05-25: Program AD real linear algebra coverage now supports exact np.linalg.det determinant expansion for square rank-2 matrices with cofactor-gradient parity and fail-closed invalid matrix contracts.
- 2026-05-25: Program AD real linear algebra coverage now supports exact np.linalg.inv adjugate/determinant expansion for nonsingular square rank-2 matrices with inverse-differential parity and fail-closed invalid matrix contracts.
- 2026-05-25: Program AD real linear algebra coverage now supports exact np.linalg.solve through inverse-backed scalar AD expansion for vector and matrix right-hand sides with implicit linear-system differential parity.
- 2026-05-25: Program AD real linear algebra coverage now supports exact np.linalg.matrix_power for static integer powers on square rank-2 matrices, including negative powers through inverse expansion and fail-closed invalid contracts.

## 2026-05-25 - Differentiable programming hardening follow-up
- Continue closing program AD array/linalg parity after `np.linalg.multi_dot`: explicit spectral-operation policies (`eig`, `eigh`, `svd`, `pinv`), broader advanced indexing contracts, and registry-dispatched linalg primitive metadata.
- Keep compiler-backed AD honest: current native MLIR/LLVM/JIT path remains provenance/tiny scalar runtime only, not general executable differentiated kernels.

## 2026-05-25 - Differentiable programming spectral policy follow-up
- Spectral linalg (`eig`, `eigh`, `eigvals`, `eigvalsh`, `svd`, `pinv`) now fails closed with explicit primitive-policy wording; next step is implementing registered spectral primitive rules with degeneracy, multiplicity, and nondifferentiability semantics before enabling derivatives.
- Keep SVD/eigen compiler lowering blocked until native MLIR/LLVM kernels can verify the same branch and multiplicity policy at runtime.

## 2026-05-25 - Differentiable programming registry bridge follow-up
- Program AD linalg primitives (`det`, `inv`, `solve`, `matrix_power`, `multi_dot`) now resolve identity/policy/effect contracts before trace execution.
- Next step: replace trace-only linalg contract placeholders with real primitive-specific derivative, batching, shape, dtype, MLIR/LLVM lowering, and Rust parity rules as those backends become executable.

## 2026-05-25 - Differentiable programming linalg batching follow-up
- Program AD linalg primitive contracts now include deterministic NumPy batching rules for mapped scalar/array outputs.
- Next step: specialize batching contracts per primitive where shape semantics need tighter validation, then add executable lowering rules only after MLIR/LLVM kernels can verify the same contracts.

## 2026-05-25 - Differentiable programming linalg shape/dtype follow-up
- Program AD linalg primitive contracts now expose primitive-specific shape rules for det, inv, solve, matrix_power, and multi_dot plus dtype validation.
- Next step: add external-baseline linalg AD conformance benchmarks and then replace trace-only derivative placeholders with true primitive-specific derivative rules where feasible.

## 2026-05-25 - Differentiable programming linalg benchmark follow-up
- Differentiable-programming conformance benchmarks now include a linalg primitive row covering det, inv, solve, matrix_power, and multi_dot against closed-form analytic derivatives.
- Next step: add optional external-backend comparison rows where dependencies are available, while keeping claim boundaries diagnostic and non-performance unless a controlled performance harness is used.

## 2026-05-25 - Differentiable programming external reference benchmark follow-up
- Optional JAX external-reference conformance rows now exist for supported linalg program AD when the backend is installed; unavailable backends return an empty suite instead of pretending coverage.
- Next step: add more optional external-reference rows for transform nesting and loop-heavy program AD, then consider controlled performance benchmarks separately from correctness conformance.

## 2026-05-25 - Differentiable programming expanded external references follow-up
- Optional JAX external-reference coverage now includes loop-heavy, linalg, and transform-nesting program AD rows.
- Next step: add controlled performance benchmark harnesses separately from correctness conformance, or begin primitive-specific linalg derivative kernels.

## 2026-05-25 - Differentiable programming linalg direct derivative follow-up
- Program AD linalg registry contracts now expose direct JVP kernels for det, inv, and vector-RHS solve where the flat-vector signature is unambiguous.
- Matrix_power and multi_dot remain trace-dispatch contracts until registry signatures can safely encode static powers and operand sequences without ambiguity.

## 2026-05-25 - Differentiable programming static argument registry follow-up
- Primitive contracts now include static-argument rules, and program AD linalg identities declare no-static, static-power, or static-operand-sequence contracts.
- Next step: use static-argument rules to add direct derivative factories for matrix_power and multi_dot, then map the same signatures into MLIR lowering metadata.

## 2026-05-25 - Differentiable programming static linalg derivative factory follow-up
- Static-argument registry contracts now have direct derivative factories for fixed matrix_power powers and fixed multi_dot operand signatures.
- Next step: map these static signatures into MLIR lowering metadata and add executable compiler verification for the direct linalg kernels.

## 2026-05-25 - Differentiable programming static linalg compiler metadata follow-up
- Compiler AD transform planning now carries static-argument-rule presence and deterministic lowering metadata into MLIR-style interchange.
- Program AD linalg matrix_power and multi_dot contracts now expose static derivative factory names and static MLIR signature metadata while native LLVM/JIT lowering remains fail-closed.
- Next step: attach executable MLIR-runtime lowering rules for concrete static matrix_power and multi_dot signatures and verify kernel outputs.

## 2026-05-25 - Differentiable programming static linalg executable lowering follow-up
- Added executable MLIR-runtime lowering-rule factories for concrete static matrix_power and multi_dot signatures.
- Kernels are verified against direct derivative factories for value and JVP outputs while native LLVM/JIT lowering remains fail-closed.
- Next step: thread concrete static linalg lowering registration into higher-level compiler AD pipelines and add multi-shape executable conformance coverage.
