# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Mitigation API Reference

# Mitigation API Reference

The `mitigation` package provides six error mitigation techniques for
noise reduction on NISQ hardware. Each technique exploits a different
aspect of the noise: ZNE extrapolates through noise amplification,
PEC cancels errors via quasi-probability sampling, CPDR trains a
regression model on near-Clifford circuits, DD suppresses decoherence
on idle qubits, symmetry verification exploits the XY Hamiltonian's
Z2 parity conservation, and Mitiq integration provides production-grade
implementations.

6 modules, 17 public symbols.

## Architecture

```
Raw Noisy Circuit
    │
    ├── ZNE: gate_fold_circuit() → zne_extrapolate()
    │         Amplify noise → Richardson extrapolation → zero-noise estimate
    │
    ├── PEC: pauli_twirl_decompose() → pec_sample()
    │         Quasi-probability decomposition → Monte Carlo sampling
    │
    ├── CPDR: generate_training_circuits() → cpdr_mitigate()
    │          Near-Clifford training → linear regression → corrected value
    │
    ├── DD: insert_dd_sequence()
    │        XY4/X2/CPMG pulses on idle qubits → decoherence suppression
    │
    ├── Symmetry: parity_postselect() / symmetry_expand()
    │              Z₂ conservation → discard/redistribute wrong-parity shots
    │
    └── Mitiq: zne_mitigated_expectation() / ddd_mitigated_expectation()
               Production Mitiq wrappers → battle-tested implementations
```

## Compound Mitigation Strategy

The modules are designed to be composed. The recommended compound
mitigation pipeline for Kuramoto-XY circuits:

1. **DD** at transpiler level (idle qubit protection)
2. **Symmetry verification** at measurement level (free error detection)
3. **CPDR** or **ZNE** at estimation level (value correction)

CPDR is preferred over ZNE for deep circuits (> 50 layers) because it
does not require depth amplification. ZNE is simpler and better for
shallow circuits (< 30 layers).

---

## Module Reference

### 1. `zne` — Zero-Noise Extrapolation

Reference: Giurgica-Tiron et al., "Digital zero noise extrapolation
for quantum error mitigation", IEEE QCE 2020.

#### Principle

Amplify the circuit's effective noise by folding the unitary:
`G → G (G†G)^k`, then measure expectation at multiple noise levels
and extrapolate to zero noise via polynomial fit.

#### `gate_fold_circuit(circuit, scale)`

Global unitary folding. `scale` must be an odd positive integer:
- scale=1: original circuit (no folding)
- scale=3: `G G† G` (one fold, 3x noise)
- scale=5: `G G† G G† G` (two folds, 5x noise)

Measurements are stripped before folding and re-appended. The inverse
`G†` is computed via `circuit.inverse()`.

```python
from scpn_quantum_control.mitigation import gate_fold_circuit

folded_3 = gate_fold_circuit(circuit, scale=3)
folded_5 = gate_fold_circuit(circuit, scale=5)
```

Raises `ValueError` if scale is not a positive odd integer.

#### `zne_extrapolate(noise_scales, expectation_values, order=1)`

Richardson extrapolation to zero noise. Fits polynomial of degree `order`
through (scale, expectation) data points and evaluates at scale=0.

```python
from scpn_quantum_control.mitigation import zne_extrapolate

result = zne_extrapolate(
    noise_scales=[1, 3, 5],
    expectation_values=[0.85, 0.72, 0.61],
    order=1,
)
print(result.zero_noise_estimate)  # ≈ 0.91
print(result.fit_residual)         # L2 residual
```

#### `ZNEResult`

| Field | Type | Description |
|-------|------|-------------|
| `noise_scales` | list[int] | Folding factors used |
| `expectation_values` | list[float] | Measured values at each scale |
| `zero_noise_estimate` | float | Extrapolated zero-noise value |
| `fit_residual` | float | RMS of polynomial fit residual |

---

### 2. `pec` — Probabilistic Error Cancellation

Reference: Temme et al., "Error Mitigation for Short-Depth Quantum
Circuits", PRL 119, 180509 (2017).

#### Principle

Decompose the ideal (noiseless) operation as a quasi-probability
distribution over noisy implementable operations. Sample from this
distribution with signed weights to cancel the error on average.

For a single-qubit depolarizing channel with error rate p:
```
I_ideal = q_I · I + q_X · X + q_Y · Y + q_Z · Z
where:
  q_I   = 1 + 3p/(4-4p)
  q_XYZ = -p/(4-4p)
```

The negative coefficients mean some samples are subtracted. The
sampling overhead (variance increase) is gamma = sum(|q_i|) per gate,
and gamma^n_gates total.

#### `pauli_twirl_decompose(gate_error_rate, n_qubits=1)`

Returns [q_I, q_X, q_Y, q_Z] quasi-probability coefficients.

```python
from scpn_quantum_control.mitigation import pauli_twirl_decompose

coeffs = pauli_twirl_decompose(0.01)
# array([ 1.00757576, -0.00252525, -0.00252525, -0.00252525])
```

Currently single-qubit only. Raises `NotImplementedError` for n_qubits > 1.

**Rust acceleration**: `scpn_quantum_engine.pec_coefficients()` produces
identical coefficients. Parity verified in `test_rust_path_benchmarks.py`.

#### `pec_sample(circuit, gate_error_rate, n_samples, observable_qubit=0, rng=None)`

Monte Carlo PEC estimation of `<Z>` on observable_qubit:

1. For each sample: insert random Pauli corrections after each gate,
   sampled from the quasi-probability distribution
2. Simulate the corrected circuit
3. Accumulate `gamma_total * sign * <Z>` over all samples
4. Average = mitigated expectation

```python
from scpn_quantum_control.mitigation import pec_sample

result = pec_sample(circuit, gate_error_rate=0.01, n_samples=1000)
print(result.mitigated_value)
print(result.overhead)  # gamma^n_gates
```

#### `PECResult`

| Field | Type | Description |
|-------|------|-------------|
| `mitigated_value` | float | PEC-corrected expectation |
| `overhead` | float | Total sampling overhead gamma^n_gates |
| `n_samples` | int | Number of Monte Carlo samples |
| `sign_distribution` | list[float] | Per-sample signs (for diagnostics) |

---

### 3. `cpdr` — Clifford Perturbation Data Regression

Reference: Zhang et al., "Clifford Perturbation Data Regression for
Quantum Error Mitigation", arXiv:2412.09518 (Dec 2024).

#### Principle

Generate near-Clifford training circuits by snapping non-Clifford
rotation gates to their nearest Clifford angle (multiples of pi/2)
and adding small perturbations. Since near-Clifford circuits can be
simulated efficiently, compute their ideal expectation values. Run the
same circuits on the noisy backend. Fit a linear regression
`noisy = slope * ideal + intercept`. Invert the regression to correct
the target circuit's noisy output.

CPDR outperforms ZNE on IBM Eagle/Heron for deep circuits because it
does not require circuit depth amplification. The training circuits
have the same depth as the target — they just have different angles.

#### `generate_training_circuits(target_circuit, n_training=20, perturbation_scale=0.1, seed=42)`

Creates n_training near-Clifford variants:
1. Find all rotation gates (RZ, RY, RX, P, U1, RXX, RYY, RZZ)
2. For each rotation: snap to nearest Clifford angle, add N(0, sigma) perturbation
3. Re-append measurements

Clifford angles: {0, pi/2, pi, 3*pi/2, 2*pi}.

#### `compute_ideal_values(circuits, observable_qubits=None)`

Compute ideal `<Z>` via statevector simulation. For near-Clifford circuits
this is exact (though the implementation uses full statevector, not
Clifford-specific simulation).

#### `compute_noisy_values_from_counts(counts_list, n_qubits, observable_qubits=None)`

Extract `<Z>` from hardware measurement counts. Handles arbitrary
bitstring formats.

#### `fit_regression(ideal_values, noisy_values)`

Linear regression: `noisy = slope * ideal + intercept`.
Returns (slope, intercept, R^2).

#### `cpdr_mitigate(raw_noisy_value, ideal_training, noisy_training)`

Apply CPDR correction: `corrected = (raw - intercept) / slope`.

```python
from scpn_quantum_control.mitigation import cpdr_mitigate

result = cpdr_mitigate(
    raw_noisy_value=0.72,
    ideal_training=[0.9, 0.85, 0.88, ...],
    noisy_training=[0.73, 0.70, 0.71, ...],
)
print(result.mitigated_value)
print(result.regression_r_squared)
```

#### `cpdr_full_pipeline(target_circuit, target_counts, run_on_backend, ...)`

End-to-end: generate training → simulate ideal → run noisy → regress → correct.
The `run_on_backend` callable takes a list of circuits and returns counts.

#### `CPDRResult`

| Field | Type | Description |
|-------|------|-------------|
| `raw_value` | float | Uncorrected noisy measurement |
| `mitigated_value` | float | CPDR-corrected value |
| `n_training_circuits` | int | Number of training circuits used |
| `regression_r_squared` | float | R^2 of the regression fit |
| `regression_slope` | float | Slope of noisy vs ideal |
| `regression_intercept` | float | Intercept |

---

### 4. `dd` — Dynamical Decoupling

Reference: Viola et al., PRL 82, 2417 (1999).

#### Principle

Insert pi-pulse sequences on idle qubits to refocus low-frequency noise.
The pulse sequence averages out the qubit-environment coupling to first
order, extending the effective T2.

#### `DDSequence`

Enum with three supported sequences:

| Sequence | Gates | Reference | Best For |
|----------|-------|-----------|----------|
| `XY4` | X-Y-X-Y | Viola et al. 1999 | General dephasing + amplitude noise |
| `X2` | X-X | Spin echo | Pure dephasing |
| `CPMG` | Y-X-Y-X | Meiboom & Gill 1958 | Correlated noise |

#### `insert_dd_sequence(circuit, idle_qubits, sequence=DDSequence.XY4)`

Appends DD pulses to the specified idle qubits after existing gates.

```python
from scpn_quantum_control.mitigation import DDSequence, insert_dd_sequence

protected = insert_dd_sequence(circuit, idle_qubits=[2, 3], sequence=DDSequence.XY4)
```

For transpiler-level insertion (respecting circuit timing), use
`HardwareRunner.transpile_with_dd()` which integrates with Qiskit's
`PadDynamicalDecoupling` pass.

---

### 5. `symmetry_verification` — Z2 Parity Conservation

Reference: Bonet-Monroig et al., "Low-cost error mitigation by symmetry
verification", Phys. Rev. A 98, 062339 (2018).

#### Principle

The XY Hamiltonian conserves total Z2 parity: [H_XY, P] = 0 where
P = tensor(Z_i). This is proven by the DLA decomposition:
DLA(N) = su(2^(N-1)) + su(2^(N-1)), where the two sectors are the
even- and odd-parity subspaces.

Consequence: any measurement outcome whose parity differs from the
initial state's parity is guaranteed to be a hardware error. This
provides error detection at zero circuit overhead.

Two strategies:
1. **Post-selection**: discard wrong-parity outcomes (cleaner, loses shots)
2. **Symmetry expansion**: flip LSB to correct parity (preserves shot count, small bias)

#### `bitstring_parity(bitstring)`

Returns 0 (even) or 1 (odd) = count of '1' bits mod 2.

#### `initial_state_parity(omega)`

Computes the dominant parity sector of |psi_0> = tensor(Ry(omega_i)|0>).
For small angles (omega_i << pi), state is near |00...0> → even parity.

#### `parity_postselect(counts, expected_parity)`

Filters measurement counts by parity. Returns `SymmetryVerificationResult`:

| Field | Type | Description |
|-------|------|-------------|
| `raw_counts` | dict | Original counts |
| `verified_counts` | dict | Parity-correct counts |
| `rejected_counts` | dict | Parity-violating counts |
| `raw_shots` | int | Total shots |
| `verified_shots` | int | Accepted shots |
| `rejected_shots` | int | Rejected shots |
| `rejection_rate` | float | Fraction rejected |
| `expected_parity` | int | 0 or 1 |

```python
from scpn_quantum_control.mitigation import parity_postselect

result = parity_postselect(counts, expected_parity=0)
print(f"Rejection rate: {result.rejection_rate:.1%}")
# On noiseless simulator: 0%. On hardware: typically 5-30%.
```

#### `symmetry_expand(counts, expected_parity)`

Redistributes wrong-parity outcomes by flipping the least-significant
bit. Preserves total shot count at the cost of small bias.

#### `parity_verified_expectation(counts, n_qubits, expected_parity)`

Computes per-qubit `<Z>` from parity-verified counts only.
Returns `(exp_vals, std_vals, rejection_rate)`.

#### `parity_verified_R(z_counts, x_counts, y_counts, n_qubits, expected_parity)`

Computes order parameter R with compound symmetry mitigation:
- Z-basis: parity post-selection (strict)
- X-basis: symmetry expansion (relaxed, basis rotation breaks direct parity)
- Y-basis: symmetry expansion

Returns dict with `R_raw`, `R_verified`, `improvement`, `improvement_pct`.

---

### 6. `mitiq_integration` — Production Wrappers

Reference: LaRose et al., Quantum 6, 774 (2022).

Wraps Mitiq's battle-tested ZNE and DDD implementations. Mitiq handles
circuit folding, noise scaling, and extrapolation internally.

Requires: `pip install mitiq`

#### `is_mitiq_available()`

Returns True if Mitiq is installed. All other functions raise `ImportError`
if Mitiq is missing.

#### `zne_mitigated_expectation(circuit, executor=None, scale_factors=None, shots=8192)`

Run Mitiq ZNE with Richardson extrapolation:

```python
from scpn_quantum_control.mitigation.mitiq_integration import zne_mitigated_expectation

value = zne_mitigated_expectation(circuit, scale_factors=[1, 3, 5])
```

Default executor: AerSimulator with parity expectation.

#### `ddd_mitigated_expectation(circuit, executor=None, shots=8192)`

Run Mitiq DDD (Digital Dynamical Decoupling) with XX rule:

```python
from scpn_quantum_control.mitigation.mitiq_integration import ddd_mitigated_expectation

value = ddd_mitigated_expectation(circuit)
```

---

## Comparison of Techniques

| Technique | Circuit Overhead | Shot Overhead | Best For |
|-----------|-----------------|---------------|----------|
| ZNE | 3-5x depth amplification | None | Shallow circuits, simple noise |
| PEC | None | gamma^n_gates (exponential) | Short circuits, high precision |
| CPDR | None | n_training runs | Deep circuits, coherent errors |
| DD | Extra gates on idle qubits | None | Idle-time decoherence |
| Symmetry | None | Shot loss from rejection | XY Hamiltonian circuits |
| Mitiq ZNE | Same as ZNE | None | Production systems |

## Rust Acceleration

| Function | Module | Speedup | Parity |
|----------|--------|---------|--------|
| `pec_coefficients` | `pec` | exact | verified to machine precision |

The PEC quasi-probability decomposition has a Rust path in
`scpn_quantum_engine.pec_coefficients()`. Other mitigation modules
operate on circuits and counts, where Rust acceleration is not applicable.

## Dependencies

| Module | External | Required? |
|--------|----------|-----------|
| `zne` | qiskit | Yes |
| `pec` | qiskit | Yes |
| `cpdr` | qiskit | Yes |
| `dd` | qiskit | Yes |
| `symmetry_verification` | numpy | Yes |
| `mitiq_integration` | mitiq, qiskit-aer | Optional |

## Testing

42 tests across 7 test files:

- `test_zne.py` — Folding correctness, extrapolation accuracy, edge cases
- `test_pec.py` — Coefficient validation, sampling convergence, overhead scaling
- `test_cpdr.py` — Training generation, regression fit, pipeline roundtrip
- `test_dd.py` — Sequence insertion, qubit validation, gate counts
- `test_symmetry_verification.py` — Parity detection, post-selection, expansion
- `test_mitiq_integration.py` — Mitiq wrapper correctness (skipif not installed)
- `test_pipeline_wiring_performance.py` — End-to-end mitigation pipeline benchmarks

## Pipeline Performance

Measured on ML350 Gen8 (128 GB RAM, Xeon E5-2620v2):

| Operation | System | Wall Time |
|-----------|--------|-----------|
| `gate_fold_circuit` (scale=5) | 4 qubits, depth 20 | 0.2 ms |
| `zne_extrapolate` (3 points, linear) | — | 0.01 ms |
| `pauli_twirl_decompose` (p=0.01) | 1 qubit | 0.01 ms |
| `pec_sample` (1000 samples) | 2 qubits | 450 ms |
| `generate_training_circuits` (20) | 4 qubits | 3 ms |
| `cpdr_mitigate` | — | 0.05 ms |
| `insert_dd_sequence` (XY4) | 4 idle qubits | 0.1 ms |
| `parity_postselect` (10k shots) | 4 qubits | 0.5 ms |
| `symmetry_expand` (10k shots) | 4 qubits | 0.6 ms |
