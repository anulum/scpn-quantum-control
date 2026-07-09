# Sparse Kuramoto CPU Path

`oscillatools` exposes an explicit sparse CPU route for large classical
Kuramoto networks:

- `SparseKuramotoCoupling`
- `sparse_coupling_from_scipy`
- `ring_sparse_coupling`
- `sparse_networked_kuramoto_force`
- `sparse_kuramoto_euler_trajectory`
- `sparse_kuramoto_rk4_trajectory`

The dense `networked_kuramoto_force` and dense Euler/RK4 routes remain unchanged.
They are still the small dense-matrix correctness and multi-language dispatch
surfaces. The sparse route accepts SciPy sparse coupling matrices or canonical
COO edge records and evaluates
`F_i = sum_j K_ij sin(theta_j - theta_i)` over stored edges only.

## Example

```python
import numpy as np
from scipy import sparse

from oscillatools import (
    sparse_coupling_from_scipy,
    sparse_kuramoto_rk4_trajectory,
    sparse_networked_kuramoto_force,
)

theta0 = np.linspace(0.0, 2.0 * np.pi, 100_000, endpoint=False)
omega = np.zeros(theta0.shape[0])
coupling = sparse.diags(
    diagonals=[0.05, 0.05],
    offsets=[-1, 1],
    shape=(theta0.shape[0], theta0.shape[0]),
    format="csr",
)
sparse_coupling = sparse_coupling_from_scipy(coupling)

force = sparse_networked_kuramoto_force(theta0, sparse_coupling)
trajectory = sparse_kuramoto_rk4_trajectory(theta0, omega, sparse_coupling, dt=0.01, n_steps=4)
```

## Evidence

The committed scaling artifact is
`docs/benchmarks/sparse_kuramoto_cpu.json`, generated with:

```bash
python scripts/bench_sparse_kuramoto_cpu.py --samples 3 --n-steps 1 --sizes 1000 10000 100000 1000000
```

Measured on 2026-07-09 with Python 3.12.13, NumPy 2.2.6, and SciPy 1.15.3:

| N | Stored edges | Force median | Euler median | RK4 median |
|---:|---:|---:|---:|---:|
| 1,000 | 2,000 | 0.040 ms | 0.063 ms | 0.214 ms |
| 10,000 | 20,000 | 0.262 ms | 1.023 ms | 2.140 ms |
| 100,000 | 200,000 | 14.236 ms | 15.552 ms | 49.873 ms |
| 1,000,000 | 2,000,000 | 81.760 ms | 141.238 ms | 566.206 ms |

This is sparse classical CPU scaling evidence for ring-network Kuramoto force
and fixed-step integrators. It is not quantum hardware evidence and not a broad
performance claim.
