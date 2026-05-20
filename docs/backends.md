# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Backends & Plugin Registry Documentation

# Backends & Plugin Registry

Two modules for runtime backend management:

1. **Backend dispatch** (`backend_dispatch.py`) — switch between numpy,
   JAX, and PyTorch for array operations
2. **Plugin registry** (`hardware/plugin_registry.py`) — legacy runner
   registration for direct adapter construction
3. **Provider-neutral backend registry** (`hardware/backends.py`) —
   production routing descriptors for IBM Runtime, local Aer, Cirq,
   Amazon Braket, PennyLane, analogue, and hybrid compiler paths
4. **Hardware abstraction layer** (`hardware/hal.py`) — provider-neutral
   workload, job, result, approval, and adapter protocol across cloud and
   simulator routes

---

## Part 1: Backend Dispatch

`scpn_quantum_control.backend_dispatch`

Runtime array backend selection, inspired by TensorCircuit's
`tc.set_backend()`. All array operations in downstream code use the
selected backend.

### API Reference

```python
from scpn_quantum_control.backend_dispatch import (
    set_backend,
    get_backend,
    get_array_module,
    to_numpy,
    from_numpy,
    available_backends,
)
```

| Function | Signature | Description |
|----------|-----------|-------------|
| `set_backend(name)` | `str → None` | Set active backend: `"numpy"`, `"jax"`, `"torch"` |
| `get_backend()` | `() → str` | Current backend name |
| `get_array_module()` | `() → module` | Active array module (`np`, `jnp`, or `torch`) |
| `to_numpy(arr)` | `Any → ndarray` | Convert any backend array to numpy |
| `from_numpy(arr)` | `ndarray → Any` | Convert numpy to current backend |
| `available_backends()` | `() → list[str]` | List installed backends |

### Example

```python
from scpn_quantum_control.backend_dispatch import (
    set_backend, get_backend, available_backends,
    get_array_module, to_numpy, from_numpy
)
import numpy as np

# Check what's available
print(available_backends())  # ['numpy', 'jax', 'torch'] (if installed)

# Default is numpy
assert get_backend() == "numpy"

# Switch to JAX
set_backend("jax")
xp = get_array_module()  # jax.numpy
arr = from_numpy(np.array([1.0, 2.0, 3.0]))
print(type(arr))  # jaxlib.xla_extension.ArrayImpl

# Convert back
arr_np = to_numpy(arr)
print(type(arr_np))  # numpy.ndarray

# Switch to PyTorch
set_backend("torch")
xp = get_array_module()  # torch
arr_t = from_numpy(np.array([1.0, 2.0]))
print(type(arr_t))  # torch.Tensor

# Reset to numpy
set_backend("numpy")
```

---

## Part 2: Plugin Registry

`scpn_quantum_control.hardware.plugin_registry`

Extensible plugin architecture for quantum hardware backends. Register
and discover backends at runtime without hard-coding imports.

Inspired by OpenFermion's plugin system (Google Quantum AI).

### Built-In Backends

The registry includes lazy loaders for three backends:

| Backend | Package | Provides |
|---------|---------|----------|
| `qiskit` | `qiskit` | Trotter circuits, IBM execution |
| `pennylane` | `pennylane` | Differentiable circuits and VQE value/gradient adapter |
| `cirq` | `cirq-core` | Google Quantum circuits |

These are loaded on first access — no import cost if unused.

### API Reference

```python
from scpn_quantum_control.hardware.plugin_registry import registry
```

#### `PluginRegistry` Methods

| Method | Signature | Description |
|--------|-----------|-------------|
| `list_backends()` | `() → list[str]` | All registered + lazy-loadable names |
| `available_backends()` | `() → list[str]` | Only importable backends |
| `is_available(name)` | `str → bool` | Check if backend is installed |
| `get_runner(name, K, omega, **kw)` | `(str, ndarray, ndarray, ...) → Runner` | Get instantiated runner |
| `register(name)` | `str → decorator` | Decorator for custom backends |
| `register_class(name, cls)` | `(str, type) → None` | Programmatic registration |

#### Runner Interface

Runners returned by `get_runner` implement:

```python
class Runner:
    def __init__(self, K, omega, **kwargs): ...
    def run_trotter(self, t: float, reps: int) -> dict: ...
    def run_vqe(self, **kwargs) -> dict: ...  # optional
```

### Example: Using Built-In Backends

```python
from scpn_quantum_control.hardware.plugin_registry import registry
import numpy as np

n = 4
K = 0.45 * np.exp(-0.3 * np.abs(np.subtract.outer(range(n), range(n))))
np.fill_diagonal(K, 0.0)
omega = np.linspace(0.8, 1.2, n)

# List available backends
print(registry.available_backends())

# Use Qiskit backend
if registry.is_available("qiskit"):
    runner = registry.get_runner("qiskit", K, omega)
    result = runner.run_trotter(t=0.1, reps=5)
    print(f"Qiskit circuit depth: {result['depth']}")
```

### Example: Custom Backend

```python
from scpn_quantum_control.hardware.plugin_registry import registry

@registry.register("my_simulator")
class MySimulator:
    def __init__(self, K, omega, **kwargs):
        self.K = K
        self.omega = omega

    def run_trotter(self, t=0.1, reps=5):
        # Custom simulation logic
        return {"energy": -1.23, "method": "my_simulator"}

# Now usable via registry
runner = registry.get_runner("my_simulator", K, omega)
result = runner.run_trotter(t=0.1, reps=5)
```

---

## Part 3: Provider-Neutral Quantum Backends

`scpn_quantum_control.hardware.backends`

The production registry exposes a single capability contract across the
hardware and simulator surface. Registry lookup is deliberately
non-authenticating and non-submitting: it imports at most the local SDK
module needed for availability checks and never reads credentials, opens
network sessions, or queues QPU jobs.

### Built-In Production Descriptors

| Backend | Provider | Execution mode | Submission policy |
|---------|----------|----------------|-------------------|
| `qiskit_ibm` | IBM Quantum | cloud QPU | approval required |
| `qiskit_aer` | local Qiskit Aer | local simulator | no submission |
| `cirq` | Google Cirq | local simulator/export | no submission |
| `braket` | Amazon Braket | cloud QPU or managed simulator | approval required |
| `pennylane` | PennyLane | adapter router | provider plugin decides |
| `analog_kuramoto` | internal compiler | analogue programme compiler | no registry-time submission |
| `hybrid_digital_analog` | internal compiler | hybrid compiler | no registry-time submission |

Cloud descriptors advertise that a submission interface exists, but they
also set `submit_requires_approval=True`. Production execution code must
pass through the explicit hardware approval scheduler before any live IBM
or AWS work is attempted.

### Descriptor API

```python
from scpn_quantum_control.hardware import describe_backend, list_quantum_backends

ibm = describe_backend("qiskit_ibm")
assert ibm.provider == "ibm_quantum"
assert ibm.can_submit is True
assert ibm.submit_requires_approval is True

local = describe_backend("qiskit_aer")
assert local.can_simulate is True
assert local.can_submit is False

for descriptor in list_quantum_backends():
    print(descriptor.name, descriptor.execution_mode, descriptor.available)
```

Every descriptor records:

| Field | Meaning |
|-------|---------|
| `name` | registry key used by routing code |
| `provider` | provider namespace, e.g. `ibm_quantum`, `local_qiskit_aer`, `aws_braket` |
| `execution_mode` | local simulator, cloud QPU, managed simulator, or adapter router |
| `sdk_package` | Python package expected for the route |
| `adapter_module` | repository module that owns execution or export |
| `available` | import-time availability, without credentials or network calls |
| `can_simulate` / `can_submit` | whether the descriptor exposes simulation or live submission semantics |
| `submit_requires_approval` | mandatory cloud-job approval flag |
| `supports_*` | shot, statevector, mid-circuit, and pulse capability flags |
| `capabilities` / `workloads` | stable machine-readable routing tags |

Legacy third-party plugins that only implement `name` and
`is_available()` are still accepted. `describe_backend()` gives them a
conservative descriptor with `can_submit=False` and
`submit_requires_approval=True` until they implement a real
`descriptor()` method returning `QuantumBackendDescriptor`.

---

## Part 4: Hardware Abstraction Layer

`scpn_quantum_control.hardware.hal`

The HAL is the execution contract above provider descriptors. It decouples SCPN
workloads from provider SDKs by using immutable metadata profiles and injected
adapter objects. Constructing the HAL is offline and metadata-only: it does not
import cloud SDKs, inspect credentials, open network sessions, or submit jobs.

### Built-In HAL Profiles

Built-in route families include:

| Backend id | Provider | Broker | Modality |
|------------|----------|--------|----------|
| `ibm_quantum` | IBM | direct | superconducting gate model |
| `ionq_cloud` | IonQ | direct | trapped-ion gate model |
| `aws_braket_ionq` | IonQ | AWS Braket | trapped-ion gate model |
| `aws_braket_iqm` | IQM | AWS Braket | superconducting gate model |
| `aws_braket_quera` | QuEra | AWS Braket | neutral-atom analogue |
| `aws_braket_rigetti` | Rigetti | AWS Braket | superconducting gate model |
| `aws_braket_aqt` | AQT | AWS Braket | trapped-ion gate model |
| `aws_braket_sv1` | AWS | AWS Braket | managed statevector simulator |
| `aws_braket_dm1` | AWS | AWS Braket | managed density-matrix simulator |
| `aws_braket_tn1` | AWS | AWS Braket | managed tensor-network simulator |
| `azure_quantum_quantinuum` | Quantinuum | Azure Quantum | trapped-ion gate model |
| `azure_quantum_quantinuum_emulator` | Quantinuum | Azure Quantum | trapped-ion emulator |
| `azure_quantum_ionq` | IonQ | Azure Quantum | trapped-ion gate model |
| `azure_quantum_ionq_simulator` | IonQ | Azure Quantum | managed gate-model simulator |
| `azure_quantum_rigetti` | Rigetti | Azure Quantum | superconducting gate model |
| `azure_quantum_rigetti_qvm` | Rigetti | Azure Quantum | managed QVM simulator |
| `azure_quantum_pasqal` | Pasqal | Azure Quantum | neutral-atom analogue |
| `azure_quantum_pasqal_emulator` | Pasqal | Azure Quantum | neutral-atom emulator |
| `azure_quantum_qci_preview` | Quantum Circuits | Azure Quantum | private-preview superconducting route |
| `quantinuum_cloud` | Quantinuum | direct | trapped-ion gate model |
| `rigetti_qcs` | Rigetti | direct | superconducting gate model |
| `quera_bloqade` | QuEra | direct | neutral-atom analogue |
| `iqm_cloud` | IQM | direct | superconducting gate model |
| `pasqal_cloud` | Pasqal | direct | neutral-atom analogue |
| `oqc_cloud` | OQC | direct | superconducting gate model |
| `qbraid_ionq` | IonQ | qBraid | trapped-ion gate model |
| `quandela_cloud` | Quandela | direct | photonic gate model |
| `dwave_leap` | D-Wave | direct | quantum annealing |
| `local_statevector` | SCPN | local | deterministic simulator |
| `local_braket_sv` | AWS Braket SDK | local | statevector simulator |
| `local_braket_dm` | AWS Braket SDK | local | density-matrix simulator |
| `local_braket_ahs` | AWS Braket SDK | local | analogue Hamiltonian simulator |
| `local_qiskit_aer` | Qiskit Aer | local | simulator |
| `local_cirq` | Cirq | local | simulator |
| `local_pennylane` | PennyLane | local | simulator |

All cloud profiles set `submit_requires_approval=True`. A cloud workload fails
closed unless the application has registered a concrete adapter and supplied an
approval token for that submission. Provider credentials, queue selection,
region policy, pricing, and detailed target selection belong inside the
provider adapter, not the HAL registry.

### HAL API

```python
from scpn_quantum_control.hardware import (
    HardwareAbstractionLayer,
    LocalDeterministicSimulator,
    QuantumWorkload,
)

hal = HardwareAbstractionLayer.with_builtin_profiles()
hal.register_backend(LocalDeterministicSimulator(hal.profile("local_statevector")))

job = hal.submit(
    "local_statevector",
    QuantumWorkload(
        workload_id="demo",
        ir_format="mlir",
        program="module {}",
        n_qubits=4,
        shots=1024,
    ),
)
result = hal.result(job)
```

Injected adapters implement `QuantumBackend.submit`, `status`, `result`, and
`cancel`. The HAL validates workload id, IR format, qubit limits, shot count,
backend registration, and approval before delegating.

---

## Comparison

| Feature | Backend Dispatch | Plugin Registry | HAL | TensorCircuit |
|---------|-----------------|-----------------|-----|---------------|
| Array backend switching | Yes | No | No | Yes |
| Hardware backend registry | No | Yes | Yes | No |
| Provider-neutral execution contract | No | No | Yes | No |
| Cloud approval gate | No | Partial | Yes | No |
| Custom backends | No | Yes (decorator) | Yes (protocol adapter) | No |
| Lazy loading | N/A | Yes | Metadata-only | No |
| JAX support | Yes | Via backends | Via adapter | Yes |
| PyTorch support | Yes | Via backends | Via adapter | Yes |

---

## References

1. Zhang, S.-X. *et al.* "TensorCircuit: An open-source cloud-oriented
   quantum computing platform." arXiv:2205.10091 (2022).
2. McClean, J. R. *et al.* "OpenFermion: The electronic structure package
   for quantum computers." *Quantum Sci. Technol.* **5**, 034014 (2020).

---

## See Also

- [Backend Selector](backend_selector.md) — auto-select based on system size
- [Multi-Platform Export](multi_platform.md) — circuit export to QASM/Quil/Cirq
- [GPU Batch VQE](gpu.md) — PyTorch GPU acceleration
