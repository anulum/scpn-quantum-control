# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Example Gallery

# Example Gallery

Ten task-shaped entry points, ordered from the fastest no-credential run to the
hardware-evidence and replay surfaces. Each entry states what it does, when to
reach for it, and the fastest safe command. Every command here runs on the
statevector simulator or on committed artefacts; none needs IBM Quantum
credentials.

Install the development extra once:

```bash
pip install -e ".[dev]"
```

The examples are onboarding aids. Reusable logic lives in `src/`, `scripts/`,
committed fixtures, and release gates — see [Onboarding](onboarding.md) for the
claim boundaries that separate simulation, method verification, hardware
evidence, and commercial readiness.

## 1. Quick simulator run

**What:** build a 4-oscillator XY Hamiltonian from `K_nm` and evolve it on the
statevector simulator, printing the order parameter at each step.
**When:** your first run, to confirm the install and see the core mapping.

```bash
python examples/02_kuramoto_xy_demo.py
```

## 2. Kuramoto-XY compile

**What:** turn a coupling matrix and frequency vector into a Trotter circuit
through the stable facade.
**When:** you want the circuit object to inspect, transpile, or run elsewhere.

```python
import numpy as np
from scpn_quantum_control import KuramotoProblem, build_knm_paper27, compile_trotter_circuit

problem = KuramotoProblem(K_nm=build_knm_paper27(L=4), omega=np.linspace(0.1, 0.4, 4))
circuit = compile_trotter_circuit(problem, time=1.0, trotter_steps=5)
print(circuit.num_qubits, circuit.depth())
```

See the [Kuramoto Core Facade](kuramoto_core_facade.md).

## 3. Synchronisation witness extraction

**What:** build synchronisation witness operators and read their expectation on
an evolved state.
**When:** you need an observable that certifies phase locking rather than a raw
amplitude.

```bash
python examples/19_sync_witness_operator.py
```

## 4. Order parameter

**What:** evolve the system and read the Kuramoto order parameter `R(t)` through
the solver.
**When:** you care about the synchronisation trajectory, not the full state.

```python
import numpy as np
from scpn_quantum_control import build_knm_paper27
from scpn_quantum_control.phase import QuantumKuramotoSolver

solver = QuantumKuramotoSolver(4, build_knm_paper27(L=4), np.linspace(0.1, 0.4, 4))
result = solver.run(t_max=1.0, dt=0.2, trotter_per_step=3)
print(result["R"])
```

## 5. Rust-accelerated Hamiltonian path

**What:** assemble the XY Hamiltonian; the optional Rust engine accelerates the
build and falls back to NumPy when the engine wheel is absent.
**When:** you build many Hamiltonians or larger systems and want the fast path.

```python
import numpy as np
from scpn_quantum_control import build_knm_paper27, knm_to_hamiltonian

hamiltonian = knm_to_hamiltonian(build_knm_paper27(L=4), np.linspace(0.1, 0.4, 4))
print(len(hamiltonian))
```

See [Pipeline Performance](pipeline_performance.md) for the measured backend
ordering.

## 6. Hardware-pack verification

**What:** verify the integrity and promotion status of committed hardware result
packs through the packaged CLI.
**When:** you want to confirm which hardware claims are artefact-backed before
quoting them.

```bash
scpn-verify-hardware-packs --help
```

See [Hardware Result Packs](hardware_result_packs.md).

## 7. Classical baseline comparison

**What:** compare the exact reference, the SciPy ODE baseline, and the
statevector Trotter route, and emit a reproducible artefact with documented
failure modes.
**When:** you need an honest classical-vs-quantum reference with a fixed claim
boundary.

```bash
python examples/09_classical_vs_quantum_benchmark.py \
    --artifact data/classical_quantum_comparison/reproducible_comparison_n8.json
```

For the system sizes this comparison can run (`n <= 16`) the classical exact
route is faster and exact; the quantum route is an evidence and falsification
surface, not a generic speed-up. See [Classical Baselines](classical_baselines.md).

## 8. Differentiable parameter-shift

**What:** walk the unified differentiable API, including parameter-shift
gradients with fail-closed framework boundaries.
**When:** you want gradients of a circuit observable for optimisation or training.

```bash
python examples/23_differentiable_api_workflow.py
```

See the [Differentiable API](differentiable_api.md).

## 9. Provider / HAL dry-run

**What:** exercise the provider hardware-abstraction layer without submitting a
job, recording capability and fail-closed boundaries.
**When:** you want to check provider readiness before any credentialled run.

```bash
scpn-provider-smoke --help
```

See [QPU Provider Readiness](qpu_provider_readiness.md).

## 10. Evidence-ledger replay

**What:** replay the preregistered `K_nm` downstream experiment gate against its
committed contract.
**When:** you want to reproduce a ledger-promoted result from committed inputs.

```bash
scpn-bench paper0-knm-preregistered-replay-gate
```

See the [Hardware Status Ledger](hardware_status_ledger.md).
