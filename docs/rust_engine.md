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

Requires Rust toolchain (rustup) and a C compiler for PyO3.

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
32-byte SHA-256 digest over the structural XY compile terms. This is the WS-1
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

The Rust crate exports 177 PyO3 bindings across 65 Rust source files (the
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

Focused fuzz-harness build checks are run from the Rust crate directory:

```bash
cargo +nightly fuzz check program_ad_ir
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
| `brute_mpc(B_flat, target, dim, horizon)` | Brute-force binary MPC (rayon parallel) | O(2^horizon × horizon) |

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
  └── build_xy_hamiltonian_dense()    → ~111× at L=4, parity by L=12

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
