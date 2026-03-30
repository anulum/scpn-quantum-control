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
2. **Plugin registry** (`hardware/plugin_registry.py`) — register and
   discover quantum hardware backends at runtime

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
| `pennylane` | `pennylane` | Differentiable circuits |
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

## Comparison

| Feature | Backend Dispatch | Plugin Registry | TensorCircuit |
|---------|-----------------|-----------------|---------------|
| Array backend switching | Yes | No | Yes |
| Hardware backend registry | No | Yes | No |
| Custom backends | No | Yes (decorator) | No |
| Lazy loading | N/A | Yes | No |
| JAX support | Yes | Via backends | Yes |
| PyTorch support | Yes | Via backends | Yes |

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
