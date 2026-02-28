# Architecture

## Module Dependency Graph

```
bridge/
├── knm_hamiltonian.py   (standalone - canonical Knm data)
├── sc_to_quantum.py     (standalone - angle/probability conversion)
└── spn_to_qcircuit.py   (uses sc_to_quantum)

qsnn/
├── qlif.py              (standalone)
├── qsynapse.py          (standalone)
├── qstdp.py             (uses qsynapse)
└── qlayer.py            (uses qlif, qsynapse)

phase/
├── xy_kuramoto.py       (uses knm_hamiltonian)
├── trotter_upde.py      (uses knm_hamiltonian)
└── phase_vqe.py         (uses knm_hamiltonian)

control/
├── qaoa_mpc.py          (uses knm_hamiltonian)
├── vqls_gs.py           (standalone)
├── qpetri.py            (uses spn_to_qcircuit)
└── q_disruption.py      (uses sc_to_quantum)

qec/
└── control_qec.py       (standalone)

hardware/
├── runner.py            (uses xy_kuramoto, phase_vqe, qaoa_mpc)
├── experiments.py       (uses knm_hamiltonian)
└── classical.py         (standalone - numpy ODE reference)
```

## Classical-to-Quantum Mapping

Each module maps a classical SCPN computation to its quantum analog:

| Classical (SCPN) | Quantum (this repo) | Mapping |
|-------------------|---------------------|---------|
| Stochastic LIF membrane potential | Ry(theta) rotation angle | theta = pi * (v - v_rest) / (v_threshold - v_rest) |
| Bitstream AND-gate synapse | CRy(theta_w) controlled rotation | P(out) = P(pre) * sin^2(theta_w/2) |
| STDP correlation learning | Parameter-shift gradient rule | dw = lr * pre * d<Z>/d(theta) |
| Kuramoto ODE (dtheta/dt) | XY Hamiltonian Trotter evolution | H = -K_ij(XX + YY) - omega_i Z_i |
| 16-layer UPDE coupling | 16-qubit spin chain | Knm -> J_ij entangling gates |
| MPC quadratic cost | QAOA Ising Hamiltonian | ||state - target||^2 -> ZZ + Z terms |
| Grad-Shafranov PDE | VQLS linear system | Laplacian A, source b -> A|x> ~ |b> |
| SPN token probability | Qubit amplitude | p -> amplitude encoding |
| Disruption feature vector | Amplitude-encoded state | 11-D -> 16-D zero-padded |

## Hardware Execution Model

```
Classical optimizer loop (COBYLA/SPSA)
  |
  v
Build QuantumCircuit (parameterized)
  |
  v
Transpile to native gates (CZ, RZ, SX, X on Heron r2)
  |
  v
Submit via qiskit-ibm-runtime SamplerV2
  |
  v
Parse bit-string counts -> expectation values
  |
  v
Compute order parameter R from <X>, <Y>, <Z>
```

Circuit depth after transpilation determines which decoherence regime applies:
- depth < 150: publishable accuracy (< 10% error)
- depth 150-400: usable with error mitigation
- depth > 400: qualitative results only

## Data Flow: Knm -> Hamiltonian -> Circuit -> Measurement -> R

```python
from scpn_quantum_control.bridge.knm_hamiltonian import (
    OMEGA_N_16, build_knm_paper27, knm_to_hamiltonian,
)
from scpn_quantum_control.phase.xy_kuramoto import QuantumKuramotoSolver

K = build_knm_paper27()
omega = OMEGA_N_16[:4]
solver = QuantumKuramotoSolver(4, K[:4, :4], omega)
result = solver.run(t_max=0.4, dt=0.1)
# result["R_trajectory"] -> [0.80, 0.78, 0.76, 0.73]
```
