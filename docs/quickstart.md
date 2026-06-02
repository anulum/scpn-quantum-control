# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Quickstart

# Quickstart

All examples run on the local AerSimulator — no IBM credentials needed.

## What You Are About to Run

The first path uses a small oscillator network and the Kuramoto-XY mapping:

1. build a coupling matrix `K_nm`;
2. build natural frequencies `omega`;
3. compile the matching XY Hamiltonian;
4. Trotterise the evolution locally;
5. read the synchronisation order parameter `R(t)`.

Use this page when you want a working run in minutes. Use
[Onboarding](onboarding.md) first if you need the business, application, and
claim-boundary overview.

## 1. Kuramoto dynamics (4 oscillators)

```python
from scpn_quantum_control.bridge import OMEGA_N_16, build_knm_paper27
from scpn_quantum_control.phase import QuantumKuramotoSolver

K = build_knm_paper27(L=4)
omega = OMEGA_N_16[:4]

solver = QuantumKuramotoSolver(4, K, omega)
result = solver.run(t_max=0.5, dt=0.1, trotter_per_step=2)

for t, R in zip(result["times"], result["R"]):
    print(f"  t={t:.1f}: R={R:.4f}")
```

The Kuramoto order parameter R measures phase synchronization: R=1 means
all oscillators are in phase, R=0 means incoherent.

## 2. VQE ground state

```python
from scpn_quantum_control.phase import PhaseVQE
from scpn_quantum_control.bridge import build_knm_paper27, OMEGA_N_16

K = build_knm_paper27(L=4)
omega = OMEGA_N_16[:4]

vqe = PhaseVQE(K, omega, ansatz_reps=2)
sol = vqe.solve(optimizer="COBYLA", maxiter=200)
print(f"VQE energy:   {sol['ground_energy']:.6f}")
print(f"Exact energy: {sol['exact_energy']:.6f}")
print(f"Error:        {sol['energy_gap']:.6f}")
```

The repository contains a legacy 4-qubit hardware row with 0.05% VQE
ground-state error. Cite it only through the hardware ledger and artefact path,
not as a broad hardware-validation claim.

## 3. Run a hardware experiment on simulator

```python
from scpn_quantum_control.hardware import HardwareRunner
from scpn_quantum_control.hardware.experiments import kuramoto_4osc_experiment

runner = HardwareRunner(use_simulator=True)
runner.connect()

result = kuramoto_4osc_experiment(runner, shots=10000, n_time_steps=4, dt=0.1)
print(f"hw_R:  {result['hw_R']}")
print(f"exact: {result['classical_R']}")
```

## 4. ZNE error mitigation

```python
from scpn_quantum_control.hardware import HardwareRunner
from scpn_quantum_control.hardware.experiments import kuramoto_4osc_zne_experiment

runner = HardwareRunner(use_simulator=True)
runner.connect()

result = kuramoto_4osc_zne_experiment(runner, shots=10000, scales=[1, 3, 5])
print(f"R at scale 1: {result['R_per_scale'][0]:.4f}")
print(f"R at scale 5: {result['R_per_scale'][2]:.4f}")
print(f"ZNE R(0):     {result['zne_R']:.4f}")
print(f"Exact R:      {result['classical_R']:.4f}")
```

ZNE (zero-noise extrapolation) runs the same circuit at increasing noise
levels, then fits a polynomial to extrapolate to zero noise.

## 5. Full 16-layer UPDE

```python
from scpn_quantum_control.phase import QuantumUPDESolver

solver = QuantumUPDESolver()  # uses canonical SCPN parameters
result = solver.step(dt=0.05)
print(f"R_global: {result['R_global']:.4f}")
```

## 6. Crypto Bell test on simulator

```python
from scpn_quantum_control.hardware import HardwareRunner
from scpn_quantum_control.hardware.experiments import bell_test_4q_experiment

runner = HardwareRunner(use_simulator=True)
runner.connect()

result = bell_test_4q_experiment(runner, shots=10000, maxiter=100)
print(f"S_hw:  {result['S_hw']:.4f}")
print(f"S_sim: {result['S_sim']:.4f}")
print(f"Violates classical (S>2): {result['violates_classical_hw']}")
```

The Bell test prepares the VQE ground state of H(K_nm), measures in 4 basis
combinations (ZZ, ZX, XZ, XX), and checks whether the CHSH S-value exceeds
the classical bound of 2.

## Available experiments

20 pre-built experiments in `ALL_EXPERIMENTS`:

```python
from scpn_quantum_control.hardware.experiments import ALL_EXPERIMENTS
for name in sorted(ALL_EXPERIMENTS):
    print(name)
```

See [Experiment Roadmap](EXPERIMENT_ROADMAP.md) for the full plan.

## Differentiable primitive smoke path

Supported compiler-AD primitives can be exercised without QPU access. This is
useful when optimisation code needs an executable gradient-bearing kernel:

```python
import numpy as np

from scpn_quantum_control import (
    CompilerADExecutableConfig,
    CustomDerivativeRule,
    compile_matrix_matrix_product_ad_to_native_llvm_jit,
)

rule = CustomDerivativeRule(
    name="matmul_demo",
    value=lambda values: values,
    derivative=lambda values, tangent: tangent,
)
kernel = compile_matrix_matrix_product_ad_to_native_llvm_jit(
    rule,
    dimension=2,
    sample_values=np.array([1.0, -2.0, 0.5, 3.0, 4.0, -1.0, 2.0, 0.25]),
    config=CompilerADExecutableConfig(backend="native_llvm_jit"),
)
print(kernel.value(np.array([1.0, -2.0, 0.5, 3.0, 4.0, -1.0, 2.0, 0.25])))
```

This lane is intentionally bounded: supported primitive kernels execute; a
general arbitrary-program MLIR/LLVM AD compiler remains an open engineering
frontier.

Program AD execution is registry-gated. Supported traced NumPy primitives must
resolve through a primitive identity with derivative, batching, shape, dtype,
static-argument, policy/effect, and lowering-provenance metadata before the
operator-intercepted program path executes. Smooth primitives may omit
nondifferentiable-boundary metadata; primitives that declare a boundary must
also declare the boundary policy as fail-closed.

Reverse-mode program gradients are available through
`program_adjoint_grad(...)` and `program_adjoint_value_and_grad(...)`. These
functions execute program capture, then require supported adjoint replay over
the captured scalar IR. Unsupported replay operations fail closed; the API does
not substitute finite differences or claim a general arbitrary-Python
MLIR/LLVM compiler.

Supported captured scalar program traces can also be promoted to an executable
replay kernel with `compile_whole_program_ad_trace_to_executable(...)`. The
kernel carries deterministic MLIR provenance, checks parameter shape and
branch/signature stability on every replay, and returns reverse-adjoint
gradients. The same kernel also supports `batch_value_and_grad(...)`,
`batch_value(...)`, and `batch_gradient(...)` for two-dimensional batches whose
rows preserve the compiled branch/signature contract. It remains a bounded
supported-trace executable path, not an arbitrary source compiler or native
LLVM/JIT implementation for all Python.

Supported scalar traces over arithmetic, expanded elementary functions
(`sin`, `cos`, `tan`, `tanh`, `exp`, `expm1`, `log`, `log1p`, `sqrt`,
`arcsin`, `arccos`, `reciprocal`, `square`, and nonzero `abs`), and stable
executed branch paths can be lowered further with
`compile_whole_program_ad_trace_to_native_llvm_jit(...)`. That path emits
deterministic MLIR provenance plus executable LLVM/JIT value, gradient, JVP,
VJP, compiled batched value/gradient kernels, and compiled batched JVP/VJP
kernels. Use `analyse_whole_program_ad_native_lowering(...)` on the captured
`WholeProgramADResult` when a service needs to inspect the native boundary
before compilation; the report lists lowerable operations, unsupported
operations, control-flow evidence, effect kinds, and the exact fail-closed
reason. Repeated identical compilations reuse a verified process-local native
compile cache keyed by deterministic trace and LLVM provenance; use
`native_whole_program_ad_compile_cache_stats()` and
`clear_native_whole_program_ad_compile_cache()` for long-running service
diagnostics and explicit invalidation. Strict scalar `np.where`, `maximum`,
`minimum`, and `clip` selection operations lower to native ordered compare/select kernels
and still replay the trace at runtime to reject equality ties before returning
an undefined selection or clipping adjoint. Runtime rows that change the compiled
branch/signature, cross unsupported primitive domains, hit
nondifferentiable boundaries, use unsupported operations, loop/control joins, or
shape changes fail closed and should use the replay executable until a native
lowering rule exists.

## GUESS error mitigation in 5 lines (added April 2026)

For any XY Hamiltonian run on hardware, the conserved total magnetisation
$\sum Z_i$ is a free guide observable. GUESS (Oliva del Moral *et al.*,
arXiv:2603.13060) uses its decay under noise to correct the target:

```python
from scpn_quantum_control.mitigation.symmetry_decay import (
    learn_symmetry_decay, guess_extrapolate, xy_magnetisation_ideal,
)

s_ideal = xy_magnetisation_ideal(n_qubits=4, initial_state="ground")  # = +4
model = learn_symmetry_decay(s_ideal,
                             noisy_symmetry_values=[3.92, 3.65, 3.10],
                             noise_scales=[1, 3, 5])
mitigated = guess_extrapolate(target_noisy_value=0.45,
                              symmetry_noisy_value=3.92,
                              decay_model=model).mitigated_value
```

See [`symmetry_decay_guess.md`](symmetry_decay_guess.md) for the full
theory, the Phase 1 ibm_kingston worked example, and a comparison with
generic Mitiq ZNE.

## Running examples

```bash
python examples/01_qlif_demo.py           # Quantum LIF neuron
python examples/02_kuramoto_xy_demo.py    # Kuramoto XY dynamics
python examples/05_vqe_ansatz_comparison.py  # Ansatz benchmark
python examples/06_zne_demo.py            # ZNE demo
```
