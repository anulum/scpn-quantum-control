# Physics-First Kuramoto-XY Tutorial

This tutorial starts from the physics object: a network of coupled oscillators.
It does not require IBM credentials, SCPN layer constants, or a hardware account.
The goal is to move from `K_nm` and `omega` to a Hamiltonian, a Trotter circuit,
and the Kuramoto order parameter.

## 1. Start With Oscillators

The classical Kuramoto model tracks phases $\theta_i$:

$$
\frac{d\theta_i}{dt} =
\omega_i + \sum_j K_{ij}\sin(\theta_j - \theta_i).
$$

- `omega[i]` is the natural frequency of oscillator `i`;
- `K_nm[i, j]` is the coupling from oscillator `j` to oscillator `i`;
- stronger coupling pulls phases into alignment;
- the order parameter `R` measures collective phase alignment.

The gate-model mapping used here expects a symmetric pair-coupling matrix. Domain
pipelines should convert source data into this canonical form before calling the
compiler facade.

## 2. Build a Minimal Problem

```python
import numpy as np

from scpn_quantum_control import build_kuramoto_problem

K_nm = np.array(
    [
        [0.0, 0.7, 0.0, 0.2],
        [0.7, 0.0, 0.5, 0.0],
        [0.0, 0.5, 0.0, 0.4],
        [0.2, 0.0, 0.4, 0.0],
    ],
    dtype=float,
)
omega = np.array([0.9, 1.1, 0.8, 1.2], dtype=float)

problem = build_kuramoto_problem(
    K_nm,
    omega,
    metadata={"domain": "physics-first-tutorial", "source": "inline-example"},
)

print(problem.n_oscillators)
print(problem.to_metadata())
```

`build_kuramoto_problem()` copies the arrays, sets the diagonal to zero, rejects
non-finite values, rejects asymmetric coupling, and stores serialisable metadata
for later result artifacts.

## 3. Compile the XY Hamiltonian

The quantum Hamiltonian is:

$$
H = -\sum_{i<j} K_{ij}(X_iX_j + Y_iY_j) - \sum_i \omega_i Z_i.
$$

```python
from scpn_quantum_control import compile_hamiltonian

hamiltonian = compile_hamiltonian(problem)
print(hamiltonian.num_qubits)
print(len(hamiltonian))
print(hamiltonian.paulis.to_labels()[:6])
```

This sparse Hamiltonian is the common object used by simulator, VQE, witness,
and hardware workflows.

## 4. Compile a Trotter Circuit

```python
from scpn_quantum_control import compile_trotter_circuit

circuit = compile_trotter_circuit(
    problem,
    time=0.25,
    trotter_steps=3,
    trotter_order=1,
)

print(circuit.num_qubits)
print(circuit.depth())
```

The circuit is hardware-oriented but still local. Submitting it to a QPU is a
separate step through the hardware runners and should only happen once the input
artifact and budget are explicit.

## 5. Measure the Order Parameter

For a first local check, prepare the equatorial product state
$|+\rangle^{\otimes N}$ and measure the order parameter:

```python
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from scpn_quantum_control import measure_order_parameter

initial = QuantumCircuit(problem.n_oscillators)
initial.h(range(problem.n_oscillators))
state = Statevector.from_instruction(initial)

R, psi = measure_order_parameter(problem, state)
print(f"R={R:.3f}, psi={psi:.3f}")
```

`R` close to one means the qubit phases are aligned. `R` close to zero means the
single-qubit phase expectations cancel.

## 6. Where SCPN Enters

SCPN-specific layers are not required for the compiler. They are one benchmark
family that supplies structured `K_nm` and `omega` values. A domain workflow can
instead supply:

- a power-grid admittance or adjacency matrix;
- an EEG phase-locking-value matrix;
- a plasma mode-coupling matrix;
- a connectome-derived coupling artifact;
- a synthetic test matrix explicitly labelled as synthetic.

The handoff contract is the same in every case: produce audited `K_nm`, `omega`,
and metadata first; then call `build_kuramoto_problem()`.

## 7. Next Steps

- Use [Kuramoto Core Facade](kuramoto_core_facade.md) for the stable public API.
- Use [Kuramoto Handbook](kuramoto_handbook.md) for the full generated facade
  inventory, model-family map, benchmark evidence, and claim boundaries.
- Use [QPU Data Artifact](qpu_data_artifact.md) when the source data comes from
  another repository or experiment.
- Use [Hardware Status Ledger](hardware_status_ledger.md) to decide whether a
  result is simulator-only, hardware-measured, mitigated, or noise-limited.
