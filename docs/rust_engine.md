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

- **PyO3 0.25** — Python bindings
- **rayon 1.10** — data parallelism (PEC sampling, MPC, OTOC time loop)
- **ndarray 0.16** — N-dimensional arrays (real and complex via `num-complex`)
- **numpy 0.25** — zero-copy array exchange with Python

All functions accept split real/imaginary arrays for complex data (no complex128
across the FFI boundary). Python wrappers handle the conversion transparently.

## Functions (15)

### Classical Kuramoto

| Function | Description | Complexity |
|----------|-------------|------------|
| `kuramoto_euler(theta0, omega, K, dt, n_steps)` | Single Euler integration run | O(n_steps × n²) |
| `kuramoto_trajectory(theta0, omega, K, dt, n_steps)` | Trajectory with R(t) at each step | O(n_steps × n²) |
| `order_parameter(theta)` | Classical Kuramoto R from phase array | O(n) |
| `build_knm(n, k_base, alpha)` | Paper 27 coupling matrix with anchors | O(n²) |

### Hamiltonian Construction

| Function | Description | Complexity |
|----------|-------------|------------|
| `build_xy_hamiltonian_dense(K_flat, omega, n)` | Dense XY Hamiltonian via bitwise flip-flop | O(2^n × n²) |

Constructs $H = -\sum_{i<j} K_{ij}(X_iX_j + Y_iY_j) - \sum_i \omega_i Z_i$ directly in the
computational basis without Qiskit. The XY flip-flop interaction gives nonzero matrix element
$H_{k, k \oplus \text{mask}_{ij}} = -2K_{ij}$ when bits $i$ and $j$ differ. 10-50× faster
than `knm_to_hamiltonian(...).to_matrix()` for n ≤ 10.

Returns flat real array (XY Hamiltonian is real in the computational basis).

### Quantum State Analysis

| Function | Description | Complexity |
|----------|-------------|------------|
| `state_order_param_sparse(psi_re, psi_im, n_osc)` | Quantum R from statevector via bitwise Pauli | O(n × 2^n) |
| `expectation_pauli_fast(psi_re, psi_im, n, qubit, pauli)` | Single-qubit Pauli expectation | O(2^n) |
| `all_xy_expectations(psi_re, psi_im, n_osc)` | Batch X,Y expectations for all qubits | O(n × 2^n) |

`all_xy_expectations` returns `(exp_x[n], exp_y[n])` in a single FFI call,
avoiding 2n individual calls to `expectation_pauli_fast`.

### Error Mitigation

| Function | Description | Complexity |
|----------|-------------|------------|
| `pec_coefficients(gate_error_rate)` | PEC quasi-probability coefficients [q_I, q_X, q_Y, q_Z] | O(1) |
| `pec_sample_parallel(gate_error_rate, n_gates, n_samples, base_exp_z, seed)` | Parallel PEC Monte Carlo (rayon) | O(n_samples × n_gates) |

### Dynamical Lie Algebra

| Function | Description | Complexity |
|----------|-------------|------------|
| `dla_dimension(generators_flat, dim, n_generators, max_iter, max_dim, tol)` | DLA dimension via commutator closure (rayon) | O(dim³ × basis²) |

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

## Measured Benchmarks (2026-03-28)

Windows 11, Python 3.12, Rust release build, i7 CPU.

### Hamiltonian Construction

| System | Rust `build_xy_hamiltonian_dense` | Qiskit `SparsePauliOp.to_matrix()` | Speedup |
|--------|------|--------|---------|
| n=4 (16×16) | 0.004 ms | 20.9 ms | **5401×** |
| n=6 (64×64) | 0.008 ms | 37.1 ms | **4589×** |
| n=8 (256×256) | 0.399 ms | 63.0 ms | **158×** |

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
