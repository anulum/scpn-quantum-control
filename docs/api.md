# API Reference

## bridge

### `knm_hamiltonian`

```python
knm_to_hamiltonian(K: np.ndarray, omega: np.ndarray) -> SparsePauliOp
```
Convert Knm coupling matrix and natural frequencies to XY Hamiltonian.
K[i,j] -> J_ij(X_iX_j + Y_iY_j), omega_i -> h_i * Z_i.

```python
knm_to_ansatz(K: np.ndarray, reps: int = 2) -> QuantumCircuit
```
Build physics-informed ansatz: CZ entanglement only between pairs where K[i,j] > threshold.

```python
OMEGA_N_16: np.ndarray  # 16 canonical frequencies (rad/s)
build_knm_paper27() -> np.ndarray  # 16x16 coupling matrix
```

### `sc_to_quantum`

```python
probability_to_angle(p: float) -> float  # p -> 2*arcsin(sqrt(p))
angle_to_probability(theta: float) -> float  # theta -> sin^2(theta/2)
bitstream_to_statevector(bits: np.ndarray) -> np.ndarray
measurement_to_bitstream(counts: dict, length: int) -> np.ndarray
```

### `spn_to_qcircuit`

```python
spn_to_circuit(W_in, W_out, thresholds) -> QuantumCircuit
```

## qsnn

### `qlif.QuantumLIFNeuron`

```python
QuantumLIFNeuron(v_rest=-70.0, v_threshold=-55.0, tau_mem=10.0, dt=1.0, n_shots=100)
    .step(input_current: float) -> int  # 0 or 1
    .get_circuit() -> QuantumCircuit
    .reset()
```

### `qsynapse.QuantumSynapse`

```python
QuantumSynapse(weight: float, w_min: float = 0.0, w_max: float = 1.0)
    .apply(circuit, pre_qubit, post_qubit)
    .effective_weight() -> float
```

### `qstdp.QuantumSTDP`

```python
QuantumSTDP(learning_rate: float = 0.01, shift: float = pi/2)
    .update(synapse, pre_measured, post_measured)
```

### `qlayer.QuantumDenseLayer`

```python
QuantumDenseLayer(n_neurons, n_inputs, weights=None, neuron_params=None)
    .forward(input_values: np.ndarray) -> np.ndarray  # spike array
```

## phase

### `xy_kuramoto.QuantumKuramotoSolver`

```python
QuantumKuramotoSolver(n_oscillators, K_coupling, omega_natural, backend=None)
    .build_hamiltonian() -> SparsePauliOp
    .evolve(time, trotter_steps=1) -> QuantumCircuit
    .measure_order_parameter(statevector) -> tuple[float, float]  # (R, psi)
    .run(t_max, dt, shots=10000) -> dict  # R_trajectory, phases
```

### `trotter_upde.QuantumUPDESolver`

```python
QuantumUPDESolver(n_layers=16, knm=None, omega=None)
    .build_hamiltonian() -> SparsePauliOp
    .step(dt, shots=10000) -> dict  # per_layer_R, global_R
    .run(n_steps, dt, shots=10000) -> dict
```

### `phase_vqe.PhaseVQE`

```python
PhaseVQE(hamiltonian: SparsePauliOp, ansatz_reps=2, knm=None)
    .build_ansatz() -> QuantumCircuit
    .solve(optimizer="COBYLA", maxiter=200) -> dict  # energy, params
```

## control

### `qaoa_mpc.QAOA_MPC`

```python
QAOA_MPC(B_matrix, target_state, horizon, p_layers=2)
    .build_cost_hamiltonian() -> SparsePauliOp
    .optimize() -> np.ndarray  # action sequence
```

### `vqls_gs.VQLS_GradShafranov`

```python
VQLS_GradShafranov(grid_size=4, source_profile=None)
    .discretize() -> tuple[np.ndarray, np.ndarray]  # (A, b)
    .build_ansatz(n_qubits, reps=2) -> QuantumCircuit
    .solve() -> np.ndarray  # psi profile
```

### `qpetri.QuantumPetriNet`

```python
QuantumPetriNet(n_places, n_transitions, W_in, W_out, thresholds=None)
    .encode_marking(marking) -> QuantumCircuit
    .step(marking, shots=1000) -> np.ndarray  # new marking
```

### `q_disruption.QuantumDisruptionClassifier`

```python
QuantumDisruptionClassifier(n_features=11, n_layers=3)
    .predict(features: np.ndarray) -> float  # risk score [0, 1]
    .train(X, y, epochs=10, lr=0.01)
```

## qec

### `control_qec.ControlQEC`

```python
ControlQEC(distance=3)
    .protect_signal(circuit) -> QuantumCircuit
    .decode_syndrome(syndrome) -> np.ndarray  # correction
```

## hardware

### `runner`

```python
run_experiment(experiment_name, backend_name="ibm_fez", shots=10000, **kwargs) -> dict
```

### `experiments`

```python
build_kuramoto_circuits(n_qubits, t_max, dt, trotter_reps=1) -> list[QuantumCircuit]
build_vqe_circuit(n_qubits, params, knm=None) -> QuantumCircuit
build_qaoa_circuit(cost_hamiltonian, p_layers, params) -> QuantumCircuit
```

### `classical`

```python
classical_kuramoto(K, omega, t_max, dt) -> dict  # R_trajectory, phases
```
