# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Backend Selector Documentation

# Automatic Backend Selection

`scpn_quantum_control.phase.backend_selector`

Auto-selects the best simulation backend based on system size, available
RAM, installed packages, and whether open-system dynamics are needed.
Two modes: recommendation-only (`recommend_backend`) and auto-execute
(`auto_solve`).

---

## Decision Tree

| System size | Open system? | quimb? | Backend selected |
|:-----------:|:------------:|:------:|-----------------|
| $n \leq 14$ | No | — | `exact_diag` (numpy `eigh`) |
| $n = 15$–$16$ | No | — | `sector_ed` (Z₂ parity sectors) |
| $n = 17$–$64$ | No | Yes | `mps_dmrg` (quimb DMRG) |
| $n = 17$–$64$ | No | No | `sparse_eigsh` (ARPACK) |
| $n > 64$ | No | — | `hardware` (IBM/cloud) |
| $n \leq 12$ | Yes | — | `lindblad_scipy` |
| $n = 13$–$16$ | Yes | — | `mcwf` (quantum jumps) |

The thresholds are conservative defaults tuned for a 32 GB workstation.
With more RAM, `exact_diag` extends to larger $n$.

---

## API Reference

```python
from scpn_quantum_control.phase.backend_selector import (
    recommend_backend,
    auto_solve,
)
```

### `recommend_backend`

```python
rec = recommend_backend(
    n: int,                       # number of oscillators
    ram_gb: float = 32.0,         # available RAM
    has_quimb: bool = False,      # quimb installed?
    has_gpu: bool = False,        # GPU available?
    want_open_system: bool = False,  # Lindblad dynamics?
) -> dict
```

**Returns:**

```python
{
    "backend": str,       # backend name (see table above)
    "reason": str,        # human-readable explanation
    "memory_mb": float,   # estimated memory requirement
    "feasible": bool,     # True if it fits in available RAM
}
```

### `auto_solve`

```python
result = auto_solve(
    K: np.ndarray,
    omega: np.ndarray,
    ram_gb: float = 32.0,
    want_open_system: bool = False,
    gamma_amp: float = 0.0,
    gamma_deph: float = 0.0,
    t_max: float = 1.0,
    dt: float = 0.1,
) -> dict
```

Selects the backend via `recommend_backend`, runs the appropriate solver,
and returns the result.

**Returns:**

```python
{
    "backend_used": str,      # which backend was selected
    "result": dict,           # solver output (keys depend on backend)
    "recommendation": dict,   # output of recommend_backend
}
```

---

## Tutorial

### Recommendation Only

```python
from scpn_quantum_control.phase.backend_selector import recommend_backend

# Small system
rec = recommend_backend(n=8)
print(f"{rec['backend']}: {rec['reason']}")
# → exact_diag: System fits in dense matrix (1 MB)

# Medium system
rec = recommend_backend(n=16, ram_gb=32.0)
print(f"{rec['backend']}: {rec['reason']}")
# → sector_ed: Z₂ parity reduces memory from 32 GB to 8 GB

# Large system with quimb
rec = recommend_backend(n=32, has_quimb=True)
print(f"{rec['backend']}: {rec['reason']}")
# → mps_dmrg: MPS/DMRG scales to n=32-64

# Open system
rec = recommend_backend(n=8, want_open_system=True)
print(f"{rec['backend']}: {rec['reason']}")
# → lindblad_scipy: Full Lindblad feasible for n≤12
```

### Auto-Solve

```python
import numpy as np
from scpn_quantum_control.phase.backend_selector import auto_solve

n = 8
K = 0.45 * np.exp(-0.3 * np.abs(np.subtract.outer(range(n), range(n))))
np.fill_diagonal(K, 0.0)
omega = np.linspace(0.8, 1.2, n)

# Closed-system
result = auto_solve(K, omega)
print(f"Backend: {result['backend_used']}")
print(f"Ground energy: {result['result']['ground_energy']:.6f}")

# Open-system
result_open = auto_solve(K, omega, want_open_system=True,
                          gamma_amp=0.05, t_max=2.0)
print(f"Backend: {result_open['backend_used']}")
```

---

## Comparison

| Feature | This module | Maestro (Qoro) | TensorCircuit |
|---------|-------------|----------------|---------------|
| Auto-selection | Yes | Yes | No |
| Backends | ED, sparse, MPS, Lindblad, MCWF | Various | numpy/JAX/torch |
| Hamiltonian | Kuramoto-XY | Generic | Generic |
| Cloud dispatch | `hardware` (placeholder) | Full cloud API | No |

---

## References

1. Maestro (Qoro). "Unified quantum simulation platform."
   arXiv:2512.04216 (2025).

---

## See Also

- [Backends & Dispatch](backends.md) — numpy/JAX/torch array dispatch
- [Symmetry Sectors](symmetry.md) — Z₂ and U(1) used by `sector_ed`
- [Tensor Networks](tensor_networks.md) — MPS/DMRG used by `mps_dmrg`
- [Lindblad Solver](lindblad.md) — used by `lindblad_scipy`
