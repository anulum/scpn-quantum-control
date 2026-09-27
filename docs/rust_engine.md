# Rust Acceleration Engine

`scpn_quantum_engine` is an optional PyO3 extension module that accelerates
hot-path computations. All Python modules transparently fall back to pure
Python/NumPy when the Rust engine is not installed.

## Installation

```bash
cd scpn_quantum_engine
pip install maturin
maturin develop --release
```

Requires Rust toolchain (rustup) and a C compiler for PyO3. The engine wheel
declares `scpn-quantum-control>=1.1.0` as a runtime dependency. Hamiltonian
calls require its shared execution-memory policy; install both builds from
the same source checkout during development. Installed-wheel tests exercise
this boundary outside the checkout and do not substitute source imports for
a matching installed control build.

## Architecture

- **PyO3 0.29** — Python bindings
- **rayon 1.10** — data parallelism (PEC sampling, MPC, OTOC time loop, GUESS batch extrapolation, hypergeometric envelope, ICI mixing angle)
- **ndarray 0.16** — N-dimensional arrays (real and complex via `num-complex`)
- **numpy 0.25** — zero-copy array exchange with Python

All functions accept split real/imaginary arrays for complex data (no complex128
across the FFI boundary). Python wrappers handle the conversion transparently.

**FFI boundary hardening (v0.9.5):** Every exported `#[pyfunction]` returns
`PyResult<T>` and validates its inputs via the helpers in `validation.rs`
(`validate_n`, `validate_positive`, `validate_range`, `validate_finite`,
`validate_flat_square`, `validate_statevec_len`, `validate_domain_range`).
Pure Rust inner functions are kept separate so the algorithms can be
unit-tested without a Python interpreter.

## Studio WASM verifier kernel

`scpn_quantum_engine/studio_wasm_kernel` is intentionally separate from the
PyO3 extension crate. It has no Python or NumPy dependency and builds to
`wasm32-unknown-unknown`:

```bash
cargo build --release --target wasm32-unknown-unknown \
  --manifest-path scpn_quantum_engine/studio_wasm_kernel/Cargo.toml
```

The exported `scpn_xy_compile_digest` ABI consumes the canonical
little-endian `studio.xy-compile-recompute.v1` byte payload and writes a
32-byte SHA-256 digest over the structural XY compile terms. This is the
bit-exact recompute path for compile claims only; it does not execute QPU jobs
or grade continuous simulator values.

**Static FFI safety audit (2026-07-03):** `tools/audit_rust_ffi_safety.py`
inventories the Rust `src/*.rs` boundary before binding expansion. The current
committed artefact,
`data/rust_ffi_safety/rust_ffi_safety_audit_2026-07-03.json`, reports:

- 177 exported `#[pyfunction]` boundaries.
- 1 `#[pymodule]` initializer.
- 0 unregistered PyO3 functions.
- 0 `unsafe` occurrences.
- 0 `extern "C"` declarations.

Run the gate with:

```bash
PYTHONPATH=src ./.venv/bin/python tools/audit_rust_ffi_safety.py \
  --crate-root scpn_quantum_engine
```

The audit is intentionally fail-closed: introducing any literal Rust `unsafe`
token or any unregistered `#[pyfunction]` changes the status to `fail`. Its
claim boundary is static source inventory only; it does not replace Miri,
sanitizer, fuzzing, or formal memory-safety evidence if unsafe Rust is ever
introduced.

**Static execution-mode audit (2026-07-03):**
`tools/audit_rust_kernel_execution.py` records whether each Rust PyO3 kernel is
currently tagged as `scalar_or_unknown`, `ndarray_dot`, `rayon_threaded`, or
`explicit_simd` before any performance promotion. The current committed
artefact,
`data/rust_kernel_execution/rust_kernel_execution_audit_2026-07-03.json`,
reports:

- 172 PyO3 kernel records.
- 19 `rayon_threaded` records.
- 1 `ndarray_dot` record.
- 0 `explicit_simd` records.
- 152 `scalar_or_unknown` records.
- 0 performance-claim-eligible records.

Run the gate with:

```bash
PYTHONPATH=src ./.venv/bin/python tools/audit_rust_kernel_execution.py \
  --crate-root scpn_quantum_engine
```

This is static source evidence, not a benchmark. Existing speedup tables below
are historical local regression evidence unless a row is explicitly tied to a
separate `isolated_affinity` benchmark artefact with CPU affinity, host-load,
governor/frequency, runner labels, and heavy-job metadata.

## Functions

The Rust crate exports 177 PyO3 bindings across 85 Rust source files (the
execution-mode audit above tracks 172 of them as compute-kernel records; the
remainder are metadata/validation surfaces). They are organised below by topic.

### Classical Kuramoto

| Function | Description | Complexity |
|----------|-------------|------------|
| `kuramoto_euler(theta0, omega, K, dt, n_steps)` | Single Euler integration run | O(n_steps × n²) |
| `kuramoto_trajectory(theta0, omega, K, dt, n_steps)` | Trajectory with R(t) at each step | O(n_steps × n²) |
| `higher_order_kuramoto_trajectory(theta0, omega, K, hyperedges, hyper_weights, dt, n_steps)` | Pairwise plus anchored triadic Kuramoto trajectory | O(n_steps × (n² + n_edges)) |
| `monitored_kuramoto_trajectory(theta0, omega, K, target_r, monitor_gain, measurement_strength, dt, n_steps)` | Monitored order-parameter feedback trajectory | O(n_steps × n²) |
| `pt_symmetric_kuramoto_trajectory(theta0, omega, K, gain_loss, dt, n_steps)` | Balanced gain/loss complex Kuramoto trajectory | O(n_steps × n²) |
| `kuramoto_witness_candidate_features(theta0, omega, K, candidates, dt, n_steps)` | Batch features for Bayesian/bandit witness discovery candidates | O(n_candidates × n_steps × n²) |
| `order_parameter(theta)` | Classical Kuramoto R from phase array | O(n) |
| `build_knm(n, k_base, alpha)` | Paper 27 coupling matrix with anchors | O(n²) |

### Hamiltonian Construction

| Function | Description | Complexity |
|----------|-------------|------------|
| `build_xy_hamiltonian_dense(K_flat, omega, n)` | Dense XY Hamiltonian via bitwise flip-flop | O(2^n × n²) |
| `build_sparse_xy_hamiltonian(K_flat, omega, n)` | Sparse COO triplets for XY Hamiltonian | O(2^n × n²) |

Both entries reject overflowing dimensions and byte counts before native output
allocation. They require the installed control package's memory policy, charge
output storage through its shared reservation and inherit an active owner's
deadline/cancellation. Sparse storage declares the actual diagonal plus admitted
nonzero-pair triplet count. Allocation failures become Python memory errors;
nonfinite constructed values refuse. Caller inputs must remain unchanged during
the operation. Native charges are additional to enclosing declarations, so the
budget can be more conservative than output bytes alone. The installed-engine
boundary and independent matrix cases are in `tests/test_hamiltonian_native.py`;
exact-source build, parity and benchmark gates still determine qualification.

### Symmetry

| Function | Description | Complexity |
|----------|-------------|------------|
| `magnetisation_labels(n)` | Total magnetisation M for all 2^N basis states via hardware popcount | O(2^n) |

Constructs $H = -\sum_{i<j} K_{ij}(X_iX_j + Y_iY_j) - \sum_i \omega_i Z_i$ directly in the
computational basis without Qiskit. The XY flip-flop interaction gives nonzero matrix element
$H_{k, k \oplus \text{mask}_{ij}} = -2K_{ij}$ when bits $i$ and $j$ differ. 10-50× faster
than `knm_to_hamiltonian(...).to_matrix()` for n ≤ 10.

Returns flat real array (XY Hamiltonian is real in the computational basis).

### Quantum State Analysis

| Function | Description | Complexity |
|----------|-------------|------------|
| `state_order_param_sparse(psi_re, psi_im, n_osc)` | Quantum R from statevector via bitwise Pauli | O(n × 2^n) |
| `order_param_from_statevector(psi_re, psi_im, n)` | Kuramoto R from state vector (MCWF inner loop) | O(n × 2^n) |
| `expectation_pauli_fast(psi_re, psi_im, n, qubit, pauli)` | Single-qubit Pauli expectation | O(2^n) |
| `all_xy_expectations(psi_re, psi_im, n_osc)` | Batch X,Y expectations for all qubits | O(n × 2^n) |

`all_xy_expectations` returns `(exp_x[n], exp_y[n])` in a single FFI call,
avoiding 2n individual calls to `expectation_pauli_fast`.

### Phase-QNode Differential Kernels

| Function | Description | Complexity |
|----------|-------------|------------|
| `phase_qnode_fubini_study_metric_rust(state_re, state_im, derivatives_re, derivatives_im)` | Pure-state Fubini-Study metric, QFI, and derivative norms from split complex state derivatives | O(p²d) |
| `phase_qnode_computational_basis_fisher_rust(state_re, state_im, derivatives_re, derivatives_im, min_probability)` | Exact computational-basis classical Fisher matrix from state and probability derivatives | O(p²d) |
| `phase_qnode_vector_jvp_rust(jacobian, tangent)` | Dense vector-output JVP contraction | O(mp) |
| `phase_qnode_vector_vjp_rust(jacobian, cotangent)` | Dense vector-output VJP contraction | O(mp) |
| `phase_qnode_hessian_vector_product_rust(hessian, vector)` | Dense Hessian-vector contraction | O(p²) |
| `phase_qnode_vector_hessian_tensor_rust(hessian_tensor, symmetry_tolerance=1e-12)` | Validate and symmetrise materialised vector-output Hessian tensors | O(kp²) |
| `phase_qnode_complex_derivative_contract_rust()` | Rust-visible real-only complex/W boundary metadata | O(1) |

The metric kernels consume already materialised statevector derivative
evidence: split real and imaginary state amplitudes plus split real and
imaginary parameter-derivative rows. They do not execute circuits and do not
claim finite-shot, density-matrix, noisy-channel, provider, hardware, or
optimal measurement metrics. The directional kernels are the Rust parity layer
for the promoted deterministic local Phase-QNode JVP, VJP, and Hessian-vector
product surfaces. The tensor kernel gives the vector-output Hessian route a
Rust-visible validation and parity surface without claiming Rust execution of
arbitrary Python objectives.

`cargo bench --bench hot_paths` includes the
`phase_qnode_metric_and_transform_kernels` group for these inner kernels. Any
result captured without the benchmark-isolation metadata required by the
project benchmark policy is local regression evidence only.

### Program AD Metadata and Replay

| Function | Description | Complexity |
|----------|-------------|------------|
| `program_ad_effect_ir_metadata_summary(serialization)` | Validate and summarize Python-emitted `program_ad_effect_ir.v1` metadata | O(n) |
| `program_ad_effect_ir_interpret_forward(serialization, inputs)` | Execute bounded opcode-bearing scalar/static-interpolation/static-signal/static-stencil/static-cumulative/static-linalg Program AD IR forward replay | O(n) |
| `program_ad_effect_ir_interpret_value_and_gradient(serialization, inputs)` | Execute bounded scalar/static-linalg including static vector- and matrix-RHS solve nodes, elementwise-array, static-structural, static source-map, static-reduction, compact interpolation, compact signal, compact stencil, compact cumulative, and inert assignment/expression alias metadata value plus reverse-gradient replay for supported IR rows | O(n) |
| `program_ad_registry_metadata_mirror(snapshot)` | Validate the Python registry-dispatch coverage snapshot and return family/facet counts plus conservative Rust replay overlap | O(n) |

The three Program AD metadata/forward/value-and-gradient PyO3 entries admit
UTF-8 source and numeric input copies through the installed control package
before Rust extraction. Inputs must be plain list/tuple/range or a plain
one-dimensional real numeric ndarray; opaque iterators refuse. The input owner
inherits cancellation/deadline and disposes on conversion or replay failure.
`tests/test_native_replay_admission.py` exercises the installed entries and
refusal/retry cases. This input boundary does not yet qualify internal primitive
workspaces, parser container peaks, in-kernel cancellation or returned JSON
retention; pure Rust/WASM callers retain their independent replay contract.

The registry mirror is metadata-only. It validates the 118-primitive Python
registry snapshot shape and reports overlap with the already bounded Rust
scalar/static-linalg plus compact interpolation, compact signal, compact stencil, compact cumulative, and
elementwise/static-structural replay; it does not
promote executable registry coverage, dynamic array semantics, LLVM/JIT
lowering, provider execution, hardware execution, or performance evidence.
`scpn_quantum_engine/tests/program_ad_panic_boundary.rs` adds a deterministic
panic-boundary corpus for malformed JSON, missing schema fields, unsafe alias
metadata, unsupported opcodes, non-finite inputs, and malformed compact signal
and cumulative metadata. The corpus checks the public Rust forward and
value+gradient APIs fail closed. `scpn_quantum_engine/fuzz/fuzz_targets/program_ad_ir.rs`
adds a `cargo-fuzz` target over the same public parser, forward replay, and
value+gradient replay APIs, with seed corpus entries under
`scpn_quantum_engine/fuzz/corpus/program_ad_ir/`.

Static `index_map:` replay reserves metadata, forward values and reverse
contributions through fallible Rust vector reservations before filling them.
Capacity and allocator refusals return an effect-specific replay error; repeated
source slots still accumulate cotangents and constant slots contribute none.
The shared replay crate also serves Studio WASM, so these reservations do not
depend on Python. They do not establish a host memory ceiling or prevent an
operating system from terminating a process under memory pressure. Public
weighted-gradient and malformed-map recovery cases are in
`scpn_quantum_engine/tests/program_ad_static_source_map.rs`; constrained-memory
qualification and exact-source native/WASM validation remain required.

Bounded pseudoinverse replay checks source/output and both square projector
sizes against native byte addressability before their allocation. Metadata uses
fixed field storage; copies, rank-one output, projectors, transposes, matrix
products and cotangents use fallible reservations. Owned checkpoints cover Gram
accumulation, matrix products and forward/reverse buffer traversal. The existing
rank-one, N×2 and 2×N support and cutoff policy remain unchanged; finite output
and adjoint validation refuse overflowing contributions. Public independent
rank-one and rectangular/square differential and cancellation recovery cases
are in `scpn_quantum_engine/tests/program_ad_pinv_memory.rs`; native build,
numerical parity and full aggregate host-memory qualification remain required.

Compact `multi_dot` replay checks operand sums and every inferred intermediate
shape against native byte addressability. Metadata and numeric buffers use
fallible reservations; operands stream into the existing left-associated chain
without cloning the first operand or retaining every operand copy. Reverse
replay reuses a zeroed variation buffer per operand instead of cloning all
inputs per differentiated element. Owned checkpoints cover parsing, copies,
multiply/dot loops and reverse variations; the existing basis VJP, accumulation
order and scalar dot signed-zero identity remain unchanged. Independent
matrix/vector-chain and cancellation recovery cases are in
`scpn_quantum_engine/tests/program_ad_multi_dot_memory.rs`; native build,
numerical parity and full aggregate host-memory qualification remain required.

Static-grid trapezoidal replay uses fallible output, cotangent, grid and rank
buffers. Metadata streams field pairs, resolves the axis before comparing grid
length, and reserves only a matching axis/full-shape grid. Flat source indices
are computed with checked arithmetic directly from the reduced index, avoiding
per-segment source-index allocations. Owned checkpoints cover conversion,
validation and every forward/reverse segment. The existing signed and
nonmonotonic grid-width semantics remain unchanged. Public independent `dx`,
axis-grid and full-grid value/gradient and cancellation recovery cases are in
`scpn_quantum_engine/tests/program_ad_trapezoid_memory.rs`; native build,
numerical parity and full host-memory qualification remain required.

Compact `diag` and `diagflat` replay validates source bytes and the declared
square output size including the signed offset before replaying its selected
scalar identity. Opcode fields and `diag` rank metadata use fixed storage;
reverse contributions use a fallible single-entry reservation. Diagonal
extraction computes selection length directly, without traversing all rows.
Owned checkpoints cover metadata work and contribution formation. Public
offset, empty-selection, overflow and cancellation recovery cases are in
`scpn_quantum_engine/tests/program_ad_diagonal_memory.rs`; these compact nodes
receive one selected scalar and do not materialise a dense matrix. Native build,
numerical parity and full host-memory qualification remain required.

Singular-value replay checks dense and retained matrix byte addressability
before comparing flattened operands. It builds column-major input in a fallibly
reserved buffer, transfers that buffer to `nalgebra`, and takes ownership of the
returned singular-value storage without cloning it. Opcode fields use fixed
storage; conversion, vector validation, spectral-gap checks and reverse
contributions observe owned checkpoints. Checkpoints bracket the `nalgebra`
decomposition, whose internal allocations remain infallible and whose inner
iteration does not invoke this callback. The solver has a finite inherited
iteration ceiling and returns a refusal on exhaustion. Replay admission declares
the flattened source, owned matrix, U/VT storage, spectrum, optional reverse
contribution and nalgebra 0.35 bidiagonal/copy/work vectors before numeric replay.
Layout validation shares the kernel parser and rejects malformed dimensions,
selected spectrum index and operand count before the memory callback. The sum
conservatively includes work from all phases; allocator bookkeeping and hard
interruption during an opaque call remain separately unqualified. Finite spectrum
and vector validation precedes descending in-place ordering, whose comparisons
and paired vector swaps observe owned checkpoints. Public rectangular/square independent oracles
and observed outer-phase cancellation recovery cases are in
`scpn_quantum_engine/tests/program_ad_svd_memory.rs`; native build, numerical
parity and constrained-memory qualification remain required.

The bounded 2×2 spectral replay keeps opcode fields in fixed storage and uses
fallible reservations for reverse contributions. Owned checkpoints cover
metadata parsing, matrix admission, eigenbasis construction and contribution
formation; non-finite cotangents refuse before reverse scatter. Eigenvalue
ordering, eigenvector gauge and the existing unsupported-spectrum boundaries
remain unchanged. Public independent eigenpair, cancellation and malformed
metadata recovery cases are in
`scpn_quantum_engine/tests/program_ad_spectral_memory.rs`; native build,
numerical parity and constrained-memory qualification remain required.

Compact static-gradient replay reserves shape, coordinate, index and cotangent
buffers fallibly, checks dense byte addressability and flat-index arithmetic,
and observes the owned checkpoint while parsing and traversing those buffers.
Coordinate count is compared with the admitted axis before coordinate allocation.
The two or three stencil coefficients use fixed storage, and source indices are
reused instead of cloned per coefficient. First-order right edges support two
coordinate samples without accessing a third point. Independent scalar,
nonuniform/decreasing-coordinate, multi-axis and cancellation recovery cases are
in `scpn_quantum_engine/tests/program_ad_stencil_memory.rs`; native build,
numerical parity and constrained-memory qualification remain required.

Compact convolution and correlation replay stream term indices directly into
forward accumulation and reverse scatter, observing the owned checkpoint at
each term. Checked operand sizes and output-window arithmetic precede indexing;
cotangent buffers use fallible replay reservations and finite-output validation.
Fixed metadata field storage avoids allocating a token vector. Public full,
same and valid mode cases, asymmetric operands, signed zero and cancellation
recovery are in `scpn_quantum_engine/tests/program_ad_signal_memory.rs`. Native
build, numerical parity and constrained-memory qualification remain required.

Compact interpolation replay checks combined sample/grid input sizes before
comparing them and reserves grid and cotangent buffers through fallible replay
helpers. Fixed metadata field storage avoids a token-vector allocation; grid
parsing, validation, binary search and cotangent filling observe the owned replay
checkpoint. Grid knots remain unsupported, and endpoint/static boundary
semantics stay unchanged. Public lifecycle and malformed-size recovery cases are
in `scpn_quantum_engine/tests/program_ad_interpolation_memory.rs`; native build,
numerical parity and constrained-memory qualification remain required.

Native matrix-power replay also uses fallible numeric and retained-power
reservations. Matrix and augmented inverse dimensions use checked products,
metadata parsing uses fixed field storage, and reverse replay moves retained
matrices rather than allocating clones. Its product/inverse arithmetic remains
unchanged. Public replay cases in
`scpn_quantum_engine/tests/program_ad_matrix_power_memory.rs` cover independent
power/inverse differentials and malformed/native-overflow metadata with retry.
Host ceilings, cooperative native lifecycle enforcement and constrained-memory
qualification still require separate evidence; fallible allocation does not
prevent operating-system termination under memory pressure.

Shaped native value-and-gradient replay also checks float64 byte addressability
for each shape and the aggregate flattened parameter count before input
materialisation. Parameter aggregation uses checked arithmetic without a
separate counts vector. Filled reverse-reduction values reserve both shape and
numeric storage fallibly. Public cases in
`scpn_quantum_engine/tests/program_ad_parameter_admission.rs` cover oversized
individual and aggregate metadata, native product overflow, scalar compatibility
and independent reduction/product gradients. These are addressability and
allocator-refusal boundaries; full workspace/host policy qualification remains
required.

The shaped replay metadata map borrows names and shape slices from its owning
IR after a fallible map reservation. Parameter copies, owned target shapes,
parameter labels and source symbols use fallible reservations; the value map
reserves for its effect count before evaluation. Labels retain UTF-8 names and
flattened parameter order. Public parameter-admission cases also cover missing
shape metadata, duplicate-name last-shape behavior and Unicode labels. Borrowed
metadata remains within the IR lifetime. Other primitive clones, parser peaks
and process-level storage/lifecycle qualification remain separate requirements.

The shared IR parser validates JSON without retaining a generic DOM, then reads
borrowed raw fields into fallibly reserved typed records. Unknown fields are
still validated, and the last duplicate key wins before schema decoding.
Escaped strings and positional records retain their existing schema semantics.
JSON grammar is checked through borrowed raw tokens. Numbers are checked for
finite range through Rust core's fixed-buffer conversion, while typed integer
and boolean fields use their own primitive parsers. Type guards prevent malformed
metadata from entering serde's optional allocating float conversion. A quote-aware
depth check preserves the default JSON nesting limit before raw parsing.
Its separate `with_replay_metadata_admission` policy admits conservative serde
byte scratch, owned strings and vector growth before allocation. PyO3 installs
this policy alongside numeric admission and charges both cumulatively to the
same input reservation. Nested metadata policies inherit parent refusal and
restore on return or unwind. Standalone replay checks addressability without
establishing host capacity. Serde scratch and allocator internals remain
third-party operations; admitted declarations do not guarantee OOM immunity.

Effect ordering admits both its indexed and returned reference buffers before
allocation. Replay symbol copies, indexed parameter-label strings and returned
label-vector headers also use the separate metadata policy. Parameter-target
records admit their complete flattened capacity before numeric replay admission;
the reserved vector then refuses growth beyond that capacity. Source symbol
lengths and runtime type sizes determine these declarations. They remain a
conservative cumulative bound rather than measured peak memory; later label
payloads, map internals and complete lifetime qualification remain separate.

Reverse adjoint-map growth and key copies also use fallible reservations.
Creating a zero adjoint propagates filled-storage errors through `Result`;
it does not assume that validated shape metadata guarantees allocator success.

Owned numeric replay operands and reverse cotangents now copy their shape and
value buffers through fallible reservations. The numeric value type no longer
provides an infallible `Clone`, and operand lists reserve before collecting
owned copies. The public parameter-admission tests include repeated broadcast
and reverse reduction with an independent scalar value and gradient oracle.
Scalar construction, scalar operand gathers, seed-adjoint storage and returned
gradient/label buffers also reserve fallibly. Full primitive workspace and
host/lifecycle qualification remain open; this does not establish an allocator peak.

Structural and reverse-reduction shape copies, including reshape cotangent
values, also reserve before copying. Scaling and binary/ternary elementwise
outputs, common broadcast/transpose outputs and concatenate/stack output
capacity use fallible reservations. Stack reserves its inserted dimension and
checks rank arithmetic. The public parameter-admission corpus includes composed
stack/concatenate/transpose and signed/quotient reverse-gradient oracles.
Coordinate scratch, axis-removal, reduction zero buffers and reverse
contribution/result containers also reserve fallibly. Transpose reverses its
owned coordinates in place. Effect ordering uses original row position as a
secondary key, preserving stable ordering with an in-place sort and explicit
fallible buffers. Branch and region metadata maps/sets reserve before insertion.
These source changes do not establish full process memory peaks.

Determinant, inverse and solve replay check square/RHS element counts and
float64 byte addressability before operand-count comparison or workspace
construction. Solve checks the combined matrix/RHS input size as well. Opcode
field parsing uses fixed storage. Inverse and determinant validate matrix length;
closed-form inverse copies, general inverse work buffers and reverse solve
storage reserve fallibly. The public `program_ad_linalg_admission` corpus
covers overflow/malformed metadata with recovery and independent composed
objective values/gradients for determinant, inverse and vector/matrix RHS solve.
Serde parsing, all kernel workspaces and aggregate host/process memory still
require separate qualification.

Three further targets extend the fuzz surface to the remaining
highest-exposure input boundaries (THREAT_MODEL B8):
`studio_kuramoto_input` (the browser-facing studio WASM kernel byte
parsers `parse_kuramoto_input` / `parse_compile_input` plus bounded
simulate/digest replay), `ml_dsa_ntt` (the pure-Rust NTT/INTT cores over
the full `i64` coefficient domain, asserting the forward/inverse pair is
a bijection on `[0, q)`), and `knm_validators` (the shared
`validation::check_*` guards against independent predicates, plus a
bounded `build_knm_inner` replay). Seed corpora live under
`scpn_quantum_engine/fuzz/corpus/<target>/`.

Focused fuzz-harness build checks are run from the Rust crate directory:

```bash
cargo +nightly fuzz check
```

That check is build reliability evidence only. Sustained coverage-guided fuzz
campaign artifacts, Miri, sanitizer, registry, LLVM/JIT, provider, hardware,
and performance promotion claims remain blocked.

### Stochastic Gradient Kernels

| Function | Description | Complexity |
|----------|-------------|------------|
| `parameter_shift_gradient_uncertainty_rust(plus_values, minus_values, plus_variances, minus_variances, plus_shots, minus_shots, coefficients, trainable, confidence_z=1.959963984540054)` | Validate and propagate materialised finite-shot parameter-shift uncertainty into gradient, standard error, diagonal covariance, and confidence radius | O(tp) |
| `spsa_gradient_rust(plus_values, minus_values, perturbations, plus_variances, minus_variances, plus_shots, minus_shots, trainable, perturbation_radius, confidence_z=1.959963984540054)` | Validate and propagate materialised SPSA probe records into gradient, standard error, diagonal covariance, and confidence radius | O(rp) |
| `score_function_gradient_rust(rewards, score_vectors, trainable, baseline=0.0, confidence_z=1.959963984540054)` | Validate materialised rewards and likelihood-ratio score vectors, then return gradient, empirical standard error, covariance, and confidence radius | O(sp²) |
| `gradient_confidence_interval_rust(gradient, standard_error, trainable, confidence_z=1.959963984540054, max_standard_error=None, max_confidence_radius=None)` | Validate materialised stochastic-gradient uncertainty, return lower/upper confidence bounds, and fail closed when active trainable parameters exceed policy thresholds | O(p) |

These kernels mirror the core Python finite-shot uncertainty primitives for
already materialised shifted expectation, SPSA probe, or score-function sample
records. They validate finite shifted means, rewards, score vectors,
non-negative variances, positive integer shot counts, finite rule coefficients,
SPSA perturbations, trainable-mask width, finite baselines, and positive
confidence radius scaling, but they return only numeric parity arrays. Python
`StochasticGradientResult` owns the `ParameterShiftSampleRecord` evidence
envelope, claim boundary, confidence interval, failure-policy status, and
`hardware_execution=False` contract. The confidence-interval kernel also
validates materialised gradients and standard errors, rejects all-false
trainable masks, and returns machine-readable failure reasons for exceeded
standard-error or confidence-radius thresholds. These kernels do not execute
provider callbacks,
allocate shots, submit hardware jobs, infer sampler score vectors, or create
claim-ledger evidence by themselves.

The `phase_qnode_metric_and_transform_kernels` benchmark group includes
`parameter_shift_uncertainty`, `spsa_gradient`, `score_function_gradient`, and
`gradient_confidence_interval`; without isolation metadata, those timings remain
`functional_non_isolated` regression evidence only.

### Error Mitigation

| Function | Description | Complexity |
|----------|-------------|------------|
| `pec_coefficients(gate_error_rate)` | PEC quasi-probability coefficients [q_I, q_X, q_Y, q_Z] | O(1) |
| `pec_sample_parallel(gate_error_rate, n_gates, n_samples, base_exp_z, seed)` | Parallel PEC Monte Carlo (rayon) | O(n_samples × n_gates) |

### Dynamical Lie Algebra

| Function | Description | Complexity |
|----------|-------------|------------|
| `dla_dimension(generators_flat, dim, n_generators, max_iter, max_dim, tol)` | DLA dimension via commutator closure (rayon) | O(dim³ × basis²) |
| `dla_protected_memory_mask(n_logical, code_distance, target_parity)` | Dense fixed-parity repetition-code memory mask | O(2^(n_logical·code_distance)) |
| `dla_protected_memory_metrics(probabilities, n_logical, code_distance, target_parity)` | Protected, code, target-parity, opposite-parity, and total probability weights | O(2^(n_logical·code_distance)) |
| `dla_protected_trajectory_metrics(probabilities, n_logical, code_distance, target_parity)` | Batch protected-memory metrics for scar and memory trajectories | O(T·2^(n_logical·code_distance)) |

### Biological Surface Code

| Function | Description | Complexity |
|----------|-------------|------------|
| `biological_decode_z_errors(edge_u, edge_v, edge_weight, n_nodes, syndrome_x)` | Weighted shortest-path + exact MWPM correction on biological coupling graph edges | O(D·(E log V) + D²2^D), D = number of defects |

### Monte Carlo

| Function | Description | Complexity |
|----------|-------------|------------|
| `mc_xy_simulate(K_flat, n, temperature, n_thermalize, n_measure, seed)` | Metropolis XY model on arbitrary coupling graph | O((n_therm + n_meas) × n²) |

### Operator Lanczos

| Function | Description | Complexity |
|----------|-------------|------------|
| `lanczos_b_coefficients(H_re, H_im, O_re, O_im, dim, max_steps, tol)` | Lanczos b-coefficients for $\mathcal{L}=[H,\cdot]$ | O(max_steps × dim³) |

Computes the operator Lanczos iteration on the Liouvillian superoperator.
Each step performs a complex matrix commutator $[H, O] = HO - OH$ (two dense
matrix multiplies) plus Hilbert-Schmidt inner product for orthogonalisation.

Uses `num_complex::Complex<f64>` with ndarray generic dot. For dim ≤ 256
(8 qubits), Rust avoids Python per-step overhead (5-10× speedup). For dim ≥ 1024,
numpy+BLAS via the Python fallback may be comparable.

### OTOC

| Function | Description | Complexity |
|----------|-------------|------------|
| `otoc_from_eigendecomp(eigenvalues, eigvecs_re, eigvecs_im, W_re, W_im, V_re, V_im, psi_re, psi_im, times, dim)` | Parallel OTOC via eigendecomposition (rayon) | O(dim³) + O(n_times × dim²) |

Computes $F(t) = \text{Re}\langle\psi| W^\dagger(t) V^\dagger W(t) V |\psi\rangle$ where
$W(t) = e^{iHt} W e^{-iHt}$.

Instead of calling `scipy.linalg.expm` twice per time point ($O(d^3)$ Padé each),
diagonalises $H$ once (done in Python via `numpy.linalg.eigh`) and computes
$W(t)_{ij} = e^{i(E_i - E_j)t} W^{\text{eig}}_{ij}$ — a phase rotation that is $O(d^2)$.
Time points are parallelised with rayon.

### Model Predictive Control

| Function | Description | Complexity |
|----------|-------------|------------|
| `brute_mpc(B_flat, target, dim, horizon)` | Brute-force binary MPC, cost `sum_t \|\|u_t·(B·1) − target\|\|²` (rayon parallel) | O(2^horizon × horizon × dim) |

### Symmetry-Decay ZNE (GUESS)

Symmetry-guided zero-noise extrapolation following Oliva del Moral *et al.*,
arXiv:2603.13060. Uses the conserved total magnetisation of $H_{XY}$ as the
guide observable and extrapolates target observables via the learned
exponential decay $\langle S \rangle_g = \langle S \rangle_{\text{ideal}} \, e^{-\alpha(g-1)}$.

| Function | Description | Complexity |
|----------|-------------|------------|
| `fit_symmetry_decay(s_ideal, noisy_values, noise_scales)` | Least-squares fit of $\alpha$ from log-transformed ratios | O(N) |
| `guess_extrapolate_batch(target_noisy, symmetry_noisy, s_ideal, alpha)` | Apply $(\lvert S_{\text{ideal}}/S_{\text{noisy}}\rvert)^\alpha$ correction in parallel via rayon | O(N) |

### DynQ Quality Scoring

Topology-agnostic qubit placement (Liu *et al.*, arXiv:2601.19635) uses Louvain
community detection on a calibration-weighted QPU graph. The scoring step is
Rust-accelerated for large devices.

| Function | Description | Complexity |
|----------|-------------|------------|
| `score_regions_batch(gate_errors_flat, n_qubits, region_offsets, region_qubits)` | Per-region connectivity, fidelity, and composite quality (rayon) | O(R × k²) |

### Pulse Shaping

PMP-optimal ICI sequences (Liu *et al.*, 2023) and the unified
$(\alpha,\beta)$-hypergeometric pulse family (Ventura Meinersen *et al.*,
arXiv:2504.08031). All three functions are Rust-accelerated for production
pulse-schedule construction.

| Function | Description | Complexity |
|----------|-------------|------------|
| `hypergeometric_envelope_batch(times, alpha, beta, gamma_width)` | $\Omega(t)/\Omega_0 = \mathrm{sech}(\gamma t)\cdot{}_2F_1$ via Gauss series + rayon | O(N × series_terms) |
| `ici_mixing_angle_batch(times, t_total, theta_jump)` | Three-segment PMP-optimal $\theta(t)$ via rayon | O(N) |
| `ici_three_level_evolution_batch(times, omega_p, omega_s, gamma)` | Forward-Euler integration of the 3×3 complex density matrix under $H + \mathcal{L}_\text{decay}$ | O(N × 9) |

Verified parity vs the Python reference implementation: max absolute
difference $4.97 \times 10^{-14}$ for $n_\text{points} = 500$.

## Measured Benchmarks

Dense XY-Hamiltonian construction is measured by a reproducible, gated harness;
the numbers, methodology, side-by-side CI vs declared-hardware comparison, and
reproduction steps live in the **[Native Speedup Benchmark](native_speedup_benchmark.md)**
page. The declared-hardware baseline (i5-11600K, pinned, warm-up + repeats,
parity-checked) is committed at `benchmarks/baselines/native_speedup.json`:

| System | Rust kernel p50 | Qiskit p50 | Speedup (p50) |
|--------|------|--------|---------|
| L=4 (16×16) | 2.79 µs | 269.5 µs | **96.5×** |
| L=8 (256×256) | 23.1 µs | 779.0 µs | **33.7×** |
| L=10 (1024×1024) | 635.3 µs | 2131.3 µs | **3.35×** |
| L=12 (4096×4096) | 42.2 ms | 93.0 ms | **2.20×** |

These are a **local regression guard, not a published claim**
(`production_claim_allowed: false`) — the ratios are environment-dependent. The
earlier "5401×" headline was a cold-start artefact (an un-warmed Qiskit
first-call). The production `knm_to_dense_matrix` wrapper additionally casts
float64 to complex128 (a downstream cost excluded from the kernel comparison).

> **Caveat on the secondary micro-benchmarks below (OTOC / Lanczos / sparse /
> popcount / order-parameter).** Unlike the four rows above, these are
> **illustrative single-machine micro-benchmarks, not committed regression
> guards** — treat the ratios as order-of-magnitude, not measured claims. Some
> compare *different algorithms*, not just Rust vs Python: the OTOC "264×", for
> instance, is a single Rust eigendecomposition against a 60-call `scipy.expm`
> loop, so the ratio bundles an **algorithmic** advantage (reuse one
> decomposition) with the language speedup and is not a like-for-like kernel
> comparison. Only the `native_speedup.json` rows (96.5×/33.7×/3.35×/2.20×) are
> warm-up-controlled, repeat-measured, and parity-checked.

### OTOC (30 time points)

| System | Rust eigendecomp + rayon | scipy.expm loop (60 calls) | Speedup |
|--------|------|--------|---------|
| n=4 (16 dim) | 0.3 ms | 74.7 ms | **264×** |
| n=6 (64 dim) | 47.9 ms | 5662.5 ms | **118×** |

### Lanczos (50 steps)

| System | Rust complex commutator | numpy commutator loop | Speedup |
|--------|------|--------|---------|
| n=3 (8 dim) | 0.05 ms | 1.3 ms | **27×** |
| n=4 (16 dim) | 0.5 ms | 4.8 ms | **10×** |

### Batch Pauli Expectations

| System | `all_xy_expectations` (1 call) | 2n × `expectation_pauli_fast` | Speedup |
|--------|------|--------|---------|
| n=4 | 3.2 µs | 19.6 µs | **6.2×** |
| n=6 | 2.5 µs | 10.4 µs | **4.2×** |
| n=8 | 6.2 µs | 15.6 µs | **2.5×** |

## Python Integration Pattern

All modules use the same pattern for transparent Rust/Python fallback:

```python
try:
    import scpn_quantum_engine as _engine
    result = _engine.some_function(...)
except (ImportError, AttributeError):
    # Pure Python/NumPy fallback
    result = python_implementation(...)
```

For Hamiltonian construction, use `knm_to_dense_matrix` from `bridge.knm_hamiltonian`
which encapsulates this pattern.

## New Functions (March 2026)

### `build_sparse_xy_hamiltonian` — 80× faster sparse construction

Returns COO triplets `(rows, cols, vals)` for `scipy.sparse.csc_matrix`.
Same bitwise flip-flop as `build_xy_hamiltonian_dense` but outputs sparse format.
Eliminates the Python `for k in range(2^n)` bottleneck.

```python
import scpn_quantum_engine as eng
rows, cols, vals = eng.build_sparse_xy_hamiltonian(K.ravel(), omega, n)
H = scipy.sparse.csc_matrix((vals, (rows, cols)), shape=(2**n, 2**n))
```

**Wired into:** `bridge/sparse_hamiltonian.py`
**Measured:** 0.024 ms (Rust) vs 1.9 ms (Python) at n=8 → **80×**

### `magnetisation_labels` — 97× faster popcount

Returns array of magnetisation $M$ for all $2^N$ basis states using hardware
`count_ones()` instruction. $M = N - 2 \times \text{popcount}(k)$.

```python
labels = eng.magnetisation_labels(n)
# labels[k] = total magnetisation of basis state |k⟩
```

**Wired into:** `analysis/magnetisation_sectors.py::basis_by_magnetisation()`
**Measured:** 0.001 ms (Rust) vs 0.11 ms (Python) at n=8 → **97×**

### `order_param_from_statevector` — 851× faster order parameter

Computes Kuramoto $R$ from complex state vector via bitwise Pauli
expectations. Critical inner loop in MCWF trajectories.

```python
R = eng.order_param_from_statevector(psi.real, psi.imag, n)
```

**Wired into:** `phase/tensor_jump.py::_order_param_vec()`
**Measured:** 0.008 ms (Rust) vs 6.47 ms (Python) at n=8 → **851×**

## Benchmarks: New Functions (2026-03-30)

Linux, Python 3.12, Rust release build, Xeon E5-2670 v2.

### Sparse Hamiltonian Construction

| System | Rust `build_sparse_xy_hamiltonian` | Python loop | Speedup |
|--------|------|--------|---------|
| n=8 (256×256) | 0.024 ms | 1.9 ms | **80×** |

### Magnetisation Labels

| System | Rust `magnetisation_labels` | Python popcount | Speedup |
|--------|------|--------|---------|
| n=8 (256 states) | 0.001 ms | 0.11 ms | **97×** |

### Order Parameter from Statevector

| System | Rust `order_param_from_statevector` | Python Pauli loop | Speedup |
|--------|------|--------|---------|
| n=8 (256 dim) | 0.008 ms | 6.47 ms | **851×** |

## New Functions (April 2026)

### `correlation_matrix_xy` — rayon-parallel XY correlation matrix

Computes $C_{ij} = \langle X_iX_j + Y_iY_j \rangle$ for all qubit pairs from a
statevector via bitwise operators. Parallelised over pairs with rayon.

The XY flip-flop interaction is nonzero only when bits $i$ and $j$ differ:
$\langle XX + YY \rangle = 2 \sum_{k: b_i \oplus b_j = 1} \text{Re}(\psi^*_k \psi_{k \oplus \text{mask}})$.

```python
C = eng.correlation_matrix_xy(psi.real, psi.imag, n_osc)
# C[i,j] = <XX_ij + YY_ij>, symmetric, zero diagonal
```

**Wired into:** `qsnn/dynamic_coupling.py::DynamicCouplingEngine._measure_correlation_matrix()`
**Measured:** 3.7 ms (Rust) vs 10.7 ms (Qiskit) at n=3 → **2.9×** (scales with $O(n^2 \cdot 2^n)$)

### `lindblad_jump_ops_coo` — Lindblad jump operator COO data

Builds all jump operators as COO triplets in a single pass. Each operator $L_k$
flips $|...1_i...0_j...\rangle \to |...0_i...1_j...\rangle$ (excitation transfer).
Returns `(rows, cols, op_starts, n_ops)` where `op_starts[k]` marks the first
entry belonging to operator $k$.

```python
rows, cols, starts, n_ops = eng.lindblad_jump_ops_coo(K.ravel(), n, threshold)
```

**Wired into:** `phase/lindblad_engine.py::LindbladSyncEngine._build_jump_operators_sparse()`
**Measured:** 0.008 ms (Rust) vs 0.1 ms (Python) at n=3 → **12×**

### `lindblad_anti_hermitian_diag` — anti-Hermitian diagonal sum

Computes the diagonal of $\sum_k L_k^\dagger L_k$ for the effective non-Hermitian
Hamiltonian in quantum trajectory evolution. Each entry counts the number of
active jump channels that can fire from that basis state.

```python
diag = eng.lindblad_anti_hermitian_diag(K.ravel(), n, threshold)
```

**Wired into:** `phase/lindblad_engine.py::LindbladSyncEngine._build_anti_hermitian_sum()`

### `parity_filter_mask` — Z2 parity classification (rayon)

Classifies bitstrings by popcount parity using hardware `count_ones()`.
Returns boolean mask for each bitstring matching the expected parity sector.

```python
mask = eng.parity_filter_mask(bitstring_ints, expected_parity)
```

**Wired into:** `mitigation/symmetry_verification.py::parity_postselect()`

## Benchmarks: New Functions (2026-04-04)

Linux, Python 3.12, Rust release build, i5-11600K.

### XY Correlation Matrix

| System | Rust `correlation_matrix_xy` | Qiskit SparsePauliOp loop | Speedup |
|--------|------|--------|---------|
| n=3 (8 dim) | 3.7 ms | 10.7 ms | **2.9×** |
| n=4 (16 dim) | 3.1 ms | — | — |
| n=8 (256 dim) | 1.7 ms | — | — |

### Lindblad Jump Operators

| System | Rust `lindblad_jump_ops_coo` | Python loop | Speedup |
|--------|------|--------|---------|
| n=3 (8 dim) | 0.008 ms | 0.1 ms | **12×** |
| n=5 (32 dim) | 0.05 ms | — | — |
| n=7 (128 dim) | 0.09 ms | — | — |

## Python ↔ Rust Wiring Diagram

Which Python module calls which Rust function:

```
bridge/knm_hamiltonian.py
  └── build_xy_hamiltonian_dense()    → 96.5× at L=4, parity by L=12

bridge/sparse_hamiltonian.py
  └── build_sparse_xy_hamiltonian()   → 80× speedup

hardware/fast_classical.py
  └── build_sparse_xy_hamiltonian()   → 80× (Hamiltonian construction)

analysis/magnetisation_sectors.py
  └── magnetisation_labels()          → 97× speedup

phase/tensor_jump.py
  └── order_param_from_statevector()  → 851× speedup

phase/lindblad_engine.py
  ├── lindblad_jump_ops_coo()         → 12× speedup
  └── lindblad_anti_hermitian_diag()

phase/quantum_kuramoto.py
  ├── state_order_param_sparse()
  ├── expectation_pauli_fast()
  └── all_xy_expectations()           → 6.2× speedup

qsnn/dynamic_coupling.py
  └── correlation_matrix_xy()         → 2.9× speedup

mitigation/symmetry_verification.py
  └── parity_filter_mask()

analysis/otoc.py
  └── otoc_from_eigendecomp()         → 264× speedup

analysis/krylov.py
  └── lanczos_b_coefficients()        → 27× speedup

mitigation/pec.py
  ├── pec_coefficients()
  └── pec_sample_parallel()

analysis/dla.py
  └── dla_dimension()

phase/classical_kuramoto.py
  ├── kuramoto_euler()
  ├── kuramoto_trajectory()
  ├── order_parameter()
  └── build_knm()

analysis/monte_carlo.py
  └── mc_xy_simulate()

control/mpc.py
  └── brute_mpc()

mitigation/symmetry_decay.py            (GUESS — Oliva del Moral 2026)
  ├── fit_symmetry_decay()              → least-squares α fit
  └── guess_extrapolate_batch()         → batch correction (rayon)

hardware/qubit_mapper.py                (DynQ — Liu 2026)
  └── score_regions_batch()             → region quality scoring (rayon)

phase/pulse_shaping.py                  (ICI + (α,β)-hypergeometric)
  ├── hypergeometric_envelope_batch()   → 44×   speedup vs scipy ₂F₁ loop
  ├── ici_mixing_angle_batch()          → trivial speedup, parity-checked
  └── ici_three_level_evolution_batch() → 1665× speedup vs Python forward-Euler
```

## Benchmarks: New Functions (April 2026)

### Hypergeometric envelope (10,000 time points)

| Implementation | Time | Speedup |
|----------------|-----:|--------:|
| Python (`scipy.special.hyp2f1` loop) | 114.5 ms | 1× |
| Rust (`hypergeometric_envelope_batch`, custom ₂F₁ series + rayon) | 2.6 ms | **44×** |

### ICI three-level evolution (2,000 time points)

| Implementation | Time | Speedup |
|----------------|-----:|--------:|
| Python (forward-Euler over 3×3 complex density matrix) | 68.30 ms | 1× |
| Rust (`ici_three_level_evolution_batch`, fixed-size local arrays) | 0.04 ms | **1,665×** |

Verified parity (Rust vs Python): max absolute difference
$4.97 \times 10^{-14}$ for $n_\text{points} = 500$. The difference is at
machine precision, confirming the Rust implementation reproduces the
reference numerical algorithm bit-for-bit.


Compact cumulative replay declares native element bytes before reserving shape,
metadata-field, prefix-index, difference-term and cotangent buffers. Reservations
are fallible, and shape copies use the same guarded allocation path. Difference
coordinates and flattened indices use checked arithmetic. Binomial coefficients
cancel exact integer factors before multiplication; coefficients outside the
native integer range refuse instead of wrapping or panicking. Source and result
finite checks remain required. Mid-kernel cooperative lifecycle propagation and
measured native allocation peaks require separate runtime qualification.


Owned replay checkpoints are thread-local and inherit parent policies. The PyO3
boundary installs the active Python reservation as the native callback and returns
its original cancellation/deadline/ownership exception after disposing the input
scope. Shared replay buffer reservations, effect dispatch/reverse iteration and
cumulative metadata/numeric loops observe that callback. Input and final-gradient finite checks operate in bounded chunks; SSA, operand, alias, branch-region and phi-path validation also checkpoint during traversal. Branch maps reserve capacity fallibly before inserting rows, including the per-region phi count. Standalone Rust/WASM
callers can install an owned callback; no callback means no requested interruption
policy. Opaque vendor operations still require checks at their own boundaries and
are not forcibly stopped mid-call.

Retained numeric replay declares every effect's float64 primal storage before
creating its value map. Gradient replay also declares a matching adjoint set and
the flattened parameter-gradient buffer. Checked sums and dtype multiplication
precede admission; numeric gradient replay checks unary and binary metadata
against source and broadcast shapes. The PyO3 boundary presents these declarations to its existing process
reservation in addition to input conversion storage, preserving the original
memory-refusal exception and disposing the scope afterward. Nested requests
retain cumulative declarations until the outer owner exits; this conservative
charge is not a measured live-memory peak. Standalone Rust/WASM callers can
install a memory-admission policy. Without it, addressability checks do not
constitute host-budget admission. Multi-dot and matrix-power replay now declare
numeric kernel workspaces from the same metadata parser used by their kernels.
Multi-dot follows the existing left-associated chain and declares source copies,
intermediate products and reverse basis buffers. Matrix-power declares forward
accumulators, inverse storage and reverse prefix powers, including power-list
headers. Gradient preflight refuses unaddressable prefix storage before forward
numeric work. Sequential effects share the maximum declared kernel workspace;
retained primal and adjoint storage remain separate charges. Bounded pseudoinverse
replay also uses its canonical layout checks to declare source copies, numeric
transposes, reverse terms and both projector squares. The declaration accounts
for identity, product and subtraction output coexisting during projector
construction; on very wide matrices this can exceed the later retained reverse
buffers. Indexed pseudoinverse outputs still require a scalar objective consumer.
Determinant, inverse and solve declarations reuse the existing opcode parsers,
including small closed-form versus general Gaussian/LU allocation paths. Solve
reverse replay also declares all RHS solution entries; its peak is the larger
of inverse construction and retained inverse/solution storage. Malformed
workspace metadata returns an unsupported forward result before numeric-map
construction; an installed memory-policy refusal retains its error boundary.
Compact convolution and correlation also declare flattened source copies and
reverse left/right contribution storage using the same source-count and output
window checks as their kernels. Pair indices remain streamed, so this boundary
does not materialize a pair table or an entire output array for a scalar opcode.
Interpolation preflight validates static grid labels without allocating the grid,
using the same layout parser as numeric replay. It declares flattened source,
materialized grid and optional reverse contribution buffers before numeric-map
construction. Increasing-grid, finite-boundary and output-index checks run in
this metadata pass; actual grid materialization follows admission. Knot refusal
and interpolation formulas retain their existing contract.
Compact diag/diagflat opcodes declare their selected scalar source and optional
scalar reverse contribution. Their canonical metadata checks still reject
unaddressable source/constructed shapes and invalid offsets or selections before
numeric replay. These opcodes do not allocate the full logical matrix; dense
construction remains the responsibility of its original array/trace owner.
Bounded 2x2 eigvalsh/eigvals/eig/eigh replay declares the flattened source and
optional four-entry reverse contribution. Eigenpair algebra retains fixed stack
arrays; output metadata and exact 2x2 arity are checked by existing parsers before
numeric replay. Numeric symmetry, distinct-real-spectrum and eigenbasis checks
remain at the original numerical owners. This declaration does not qualify
opaque SVD solver allocation or broaden the supported spectral boundary.
Compact cumulative preflight streams the opcode fields and source shape without
allocating a token table or shape vector. The shared layout parser checks source
addressability/count, axis, difference order and selected output before numeric
replay. It declares the flattened source, optional reverse contribution, shape
and coordinate vectors, and the selected prefix-index or difference-term buffer.
Metadata field ordering remains flexible. Numeric prefix products and checked
binomial differences retain their existing algorithms; some malformed metadata
now refuses in preflight rather than after source materialization.
Static-grid trapezoid preflight also receives the existing borrowed SSA shapes.
Its canonical metadata parser validates `dx`, `x` and `xfull` labels and lengths
without materializing grid values, and checks the selected axis and reduced
target shape. The declaration includes source/output copies, grid, coordinates,
reverse cotangent and contribution, and conservative adjoint-accumulation
buffers. Actual segment arithmetic and finite-width rejection stay at the
numeric kernel. Descending, nonmonotone and zero-width grids remain supported.
Scalar-only forward replay remains unsupported for ranked trapezoid nodes;
this declaration qualifies its existing shaped value/gradient adapter only.
Product and corrected variance/standard-deviation admission use the existing
axis and correction parsers with borrowed SSA source/target shapes. The actual
moment kernel and admission share the group-count/correction denominator rule;
invalid denominators refuse before constructing reduction groups. All-axis
reductions declare the source/cotangent/contribution and conservative adjoint
accumulation buffers. Axis reductions additionally declare numeric group entries,
outer Vec headers, current group value/VJP buffers and live coordinates. Moment
forward grouping has its own source-sized buffer and header declaration.
Single-zero product gradients and numerical standard-deviation singularity
refusals retain their original rules. Scalar-only forward replay remains
unsupported; these declarations cover the existing numeric value/gradient path.
Selector and order-statistic reductions reuse their actual axis/q parser before
numeric replay. They declare indexed source groups, optional outer Vec headers,
source/contribution/cotangent copies, live shape/coordinate buffers and sorting
scratch. Value validation and index ordering use separate scratch vectors; the
selected VJP reserves two index/value slots after ordering finishes. Admission
uses the larger scratch requirement and the conservative reverse accumulation
bound, without performing a sort or materializing groups. Strict-order refusal
and quantile/percentile interpolation keep their original numerical contract.
Compact stencil admission streams the source shape and spacing coordinates
without materializing either vector. Its shared layout parser checks source
bytes/count, axis, edge-order sample requirements, finite nonzero scalar or
strictly monotone coordinate spacing, and selected output before replay.
Forward and reverse declarations cover the flattened source, shape and target
coordinate vectors, optional spacing grid and optional reverse contribution.
The two/three coefficient entries stay in fixed stack storage. Their finite
numerical checks and gradient formulas remain at the original kernel; subnormal
spacing can therefore refuse numerically after admission. Both public forward
and value/gradient entry points retain their supported stencil boundary.
These declarations are conservative source-derived bounds, not measured
allocation peaks. Other kernel families, metadata-map capacity, vendor workspaces and returned JSON
retention still need additional allocation qualification. Boundary and recovery
tests are authored in `program_ad_multi_dot_memory.rs`,
`program_ad_matrix_power_memory.rs`, `program_ad_pinv_memory.rs`,
`program_ad_linalg_admission.rs`, `program_ad_signal_memory.rs`,
`program_ad_interpolation_memory.rs`, `program_ad_diagonal_memory.rs` and
`program_ad_spectral_memory.rs`, `program_ad_cumulative.rs` and
`program_ad_trapezoid_memory.rs`, `program_ad_reduction_memory.rs` and
`program_ad_order_statistic_memory.rs` and `program_ad_stencil_memory.rs`; remote
native validation remains pending.


Reentrant PyO3 replay inherits the active Python exception owner as well as its
checkpoint policy. A child's inherited callback failure retains the original
exception instance for both child and parent until the outermost scope ends;
child cleanup cannot consume or replace that failure. Independent subsequent
calls start with a fresh exception owner.

Native matrix-power replay checks the active lifecycle owner while building identity and retained powers, multiplying and transposing matrices, accumulating reverse contributions, and eliminating inverse workspaces. Zero-buffer initialization also checks periodically after fallible allocation. Cancellation releases these local buffers through normal Rust unwinding; independent replay can start with a fresh owner.

Native shaped replay checks the active lifecycle owner while copying numerical buffers in bounded chunks, validating finite entries and unary domains, filling adjoints, scanning zero cotangents, and evaluating elementwise unary operations and whole-array sum/mean reductions. Sum/mean retain the original sequential accumulation order and signed-zero identity. Independent calls restore their own lifecycle policy after refusal.

Native reverse replay reserves derivative buffers fallibly and checks lifecycle ownership during derivative/domain evaluation, adjoint validation and accumulation. Structural buffer initialization uses the same fallible, periodically checked fill path as adjoints. Broadcast, transpose, concatenation, stacking, axis reduction, and their reverse scatter/index loops also observe the active owner; cancellation returns through normal buffer disposal before an independent retry.

Native static source maps count metadata entries with checked arithmetic and lifecycle checks before reserving parsed entries. A declared target or cotangent size mismatch refuses before this allocation. Parsing, forward gathering, reverse buffer initialization and repeated-index scatter observe the active lifecycle owner. Constant entries retain their floating-point value and contribute no source cotangent.

Native scalar determinant, inverse and solve replay checks lifecycle ownership while preparing Gaussian/LU workspaces, selecting and swapping pivots, normalizing and eliminating rows, validating the inverse and accumulating the solution. Workspace initialization uses fallible, periodically checked buffers. Reverse linalg adapters share the checked operand reservation path instead of allocating through iterator collection. These changes retain the numerical formulas and pivot order.

Native inverse/determinant reverse contributions and solve adjoints check lifecycle ownership while traversing matrix and RHS entries. Forward solve and reverse solve workspace construction use one checked sequential dot-product kernel, retaining the original signed-zero identity. Multiple RHS columns and pivot-swapped matrices use the same cancellation and recovery path.

Native product reductions reserve result, group, index and adjoint buffers fallibly after checked byte sizing. Group buffers reserve the declared axis length. Parsing/index traversal, multiplication, zero counting, group collection and reverse scatter observe lifecycle ownership. Flat-index offsets use checked multiplication and addition. The existing gradient boundary remains one zero per reduction group; two or more zeros refuse and independent replay can recover.

Native variance/std replay reserves group, result, index and adjoint buffers fallibly and checks lifecycle ownership during centered moments, group construction and reverse scatter. Static metadata parses field pairs without allocating a field list; index offsets use checked arithmetic. Sequential mean/centered sums retain their original floating-point identities. Invalid correction denominators and zero-variance standard-deviation gradients retain their refusal boundaries.

Native order statistics reserve group, selection and cotangent buffers fallibly. Static metadata parses without a token list and flat offsets use checked arithmetic. In-place heapsort checks lifecycle ownership while building and draining the heap, with no sort scratch allocation. Forward and gradient replay use this checked ordering path for effects as well, sorting by the declared ordering key and original row position so equal keys retain their original replay order. Interpolation and source scatter preserve the existing strict-order selector contract; tied groups refuse and independent replay can recover.


PyO3 replay captures the input-copy charge once, then adds cumulative native
workspace declarations to that fixed baseline. Reentrant inherited callbacks
retain each declaration without recharging earlier totals. The binding routes
ordinary conversion errors and Rust unwinds through the Python input scope's
exit; a caught unwind becomes a runtime error and does not invoke a fallback.
Allocator aborts and interruptions inside external solvers remain unqualified.
Public reentrant accounting fixtures await native execution; this source change
alone is not panic-path or constrained-memory acceptance evidence.


Native metadata, forward and gradient JSON results use two streaming passes over
the same immutable result. The first counts UTF-8 bytes without a result-sized
buffer; admission charges those bytes before fallible exact-capacity reservation.
The second writes within that admitted bound and checks the final byte count.
Both passes observe lifecycle checkpoints. UTF-8 storage transfers into the Rust
string without copying. The binding also charges a conservative Python Unicode
header/payload declaration before fallible conversion while its input scope is
still active. Caller-retained output lifetimes remain unqualified.


Python Unicode conversion uses `PyString::from_bytes`, which reports allocation
errors rather than the infallible constructor's panic. Runtime-observed empty,
ASCII, Latin-1, BMP and non-BMP string sizes bound the header, and four bytes per
UTF-8 input byte conservatively bound payload width. Admission precedes conversion
and checks cancellation again before returning. Native binding wrappers return
owned Python strings; Python callers still receive the same JSON `str`. The
reservation ends at return and does not charge objects later retained by callers.


Binary retained-shape admission compares borrowed operand and target dimensions
axis by axis using the numeric kernel's same broadcast rule. It does not allocate
an inferred shape vector before numeric memory admission. Incompatible operand
shapes still refuse before target mismatch; ranked/singleton/scalar broadcasting
and the numeric materialization order retain their existing contract. Public
ranked value/gradient, malformed-target and recovery fixtures await execution.


WASM Program AD input decoding validates the complete borrowed envelope, UTF-8,
input arity, exact byte count and finite inputs before owned source/numeric
copies. Those copies and the combined value/gradient output reserve fallibly.
The public replay FFI bounds its input envelope before raw-slice construction
and validates the exact output byte count before copying IR or running the
numerical interpreter. Rejected output lengths leave the sink unchanged. Existing
status codes and numerical replay are retained; full linear-memory budget and
external-solver allocation/iteration qualification remain required.

Elementwise replay validates unary and binary operand counts before retained
workspace admission on both the forward and value-plus-gradient public paths.
Malformed arity preserves the existing refusal diagnostic and does not invoke
the owned numeric memory callback or materialize retained numerical values.

Singular-value replay uses the same nalgebra implicit-shift decomposition with
its original tolerance and a finite per-call ceiling of 10,000 iterations.
`with_replay_solver_iterations` permits a positive tighter ceiling; nested
owners inherit the minimum and return or unwind restores their parent. Exhaustion
refuses the replay. This is a safety policy, not a measured wall-clock deadline.
The spectrum and vectors are checked before allocation-free descending reordering,
so a non-finite spectrum never enters nalgebra's NaN-panicking sort path. Vector
columns/rows follow the same permutation. The workspace declaration includes
nalgebra 0.35's bidiagonal coefficients, copied off-diagonal and phase work vectors.
Allocator bookkeeping, vendor allocation failure and hard interruption during an
opaque solver call still require their separate host/process qualification.

The browser bindings retain the existing kernel ABI. Kuramoto checks the
kernel-reported oscillator and step ceilings before encoding; unknown limits
refuse. Its codec validates representable byte/header counts and finite fields
before allocating the payload and streams the fields without a combined copy.
Both bindings own guest buffers immediately after each successful allocation,
attempt both releases even if one release throws, and refuse allocation,
execution or cleanup traps. Complete output windows and finite results are
required; retained results copy values out before freeing guest buffers.
These declared shape limits do not establish a total browser heap quota or
worker-disposal proof. The real-WASM frontend regressions run in the Studio CI
category; they remain unexecuted locally under the hooks-only validation policy.

Replay validation declares its branch, region and phi hash tables before fallible
reservation. Shape, scalar value, numeric value and adjoint maps use the same
checked table bound. The adjoint map reserves its complete target capacity once;
new keys cannot grow beyond that admitted capacity. These are conservative table
layout declarations for the maintained Rust/WASM targets, based on the standard
library hash-table bucket and control-byte layout, rather than allocator peak
measurements. Public branch and gradient regressions exercise refusal before
numeric admission and recovery with unchanged mathematical results.

The browser replay kernel applies a 64 MiB product ceiling to cumulative declared
storage for each call. Owned input/output copies, parser metadata, replay tables
and numerical requests share that charge; an explicit Rust caller may tighten
it with `replay_value_and_gradient_with_memory_budget`. This is a policy ceiling,
not observed browser capacity or a memory reservation. Parent admission remains
binding, allocation failures refuse, and a refused FFI call leaves output bytes
unchanged. Independent subsequent calls start with a fresh charge.

Shaped elementwise, broadcast, reshape, transpose, concatenation, stacking,
source-map and sum/mean adapters declare copied operands, broadcast values,
cotangents, contributions, shape/coordinate vectors and operand containers
before numerical replay. The conservative bound covers the live div/pow reverse
chain and structural split/scatter storage. Static source maps include their
actual tagged-entry type width. Scalar-only stack arithmetic has no such shaped
workspace. These declarations preserve kernel equations and execution order.
