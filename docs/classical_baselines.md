# Classical Baselines

This page records the supported classical baseline surfaces for Kuramoto-XY
workflows. Each baseline returns a `ClassicalBaselineRun` envelope with the
backend name, availability flag, elapsed wall time, time grid, order-parameter
trajectory, and metadata needed for provenance.

## Baseline Matrix

| Baseline | Dependency | Purpose | Availability behaviour |
| --- | --- | --- | --- |
| SciPy ODE | Runtime dependency | Classical Kuramoto phase dynamics via `solve_ivp(RK45)`. | Always available. |
| QuTiP Lindblad | `[opensys]` or `[xvalidate]` extra | Independent density-matrix open-system reference via `qutip.mesolve`. | Returns `available=False` when QuTiP is absent. |
| MPS TEBD | `[tensor]` extra | Tensor-network time evolution through the existing quimb TEBD backend. | Returns `available=False` when quimb is absent. |

## Quick Use

```python
import numpy as np

from scpn_quantum_control.benchmarks.classical_baselines import (
    available_baselines,
    run_documented_classical_baselines,
)

K = np.array(
    [
        [0.0, 0.4, 0.0],
        [0.4, 0.0, 0.3],
        [0.0, 0.3, 0.0],
    ]
)
omega = np.array([0.8, 1.0, 1.2])

print(available_baselines())
runs = run_documented_classical_baselines(K, omega, t_max=0.5, dt=0.1)

for name, run in runs.items():
    if run.available:
        print(name, run.backend, run.r_final)
    else:
        print(name, run.unavailable_reason)
```

## SciPy ODE

`scipy_ode_baseline` integrates the classical Kuramoto equations:

```text
d theta_i / dt = omega_i + sum_j K_ij sin(theta_j - theta_i)
```

This is the baseline for classical phase locking. It is not a quantum
Hamiltonian simulation and should be labelled as a classical ODE reference in
reports.

## QuTiP Lindblad

`qutip_lindblad_baseline` builds an independent QuTiP XY Hamiltonian and evolves
the initial product state under amplitude-damping collapse operators. Use it for
small open-system cross-checks where density-matrix scaling is acceptable.

The function does not fabricate a result when QuTiP is missing. It returns a
`ClassicalBaselineRun` with `available=False` and an unavailable reason of
`qutip missing`.

## MPS TEBD

`mps_tebd_baseline` wraps `phase.mps_evolution.tebd_evolution`. The quimb local
Hamiltonian path uses nearest-neighbour terms, matching the existing MPS module
contract. Use it to document whether a tensor-network baseline is available and
what bond dimensions the run reached.

The wrapper explicitly enables nearest-neighbour truncation for this
diagnostic path and records `coupling_scope` plus `omitted_coupling_l1`
in metadata. Direct calls to `tebd_evolution` reject non-adjacent
couplings unless `allow_long_range_truncation=True` is passed.

The function returns `available=False` with `unavailable_reason="quimb missing"`
when the `[tensor]` extra is absent.

## Reproducible head-to-head comparison artifact

`run_reproducible_kuramoto_comparison` composes the classical exact reference,
the SciPy ODE baseline, and the statevector Trotter route into a single
serialisable record so an example can emit a reproducible artifact instead of
printing transient numbers. It does not reimplement any solver.

For `n <= 16`, the exact route is the reference; the ODE and quantum rows carry
their final order-parameter error against it. For `n > 16`, the artifact becomes
a scalable classical baseline: the SciPy ODE row is the reference, while the
exact and statevector Trotter rows are marked `available=False` with an
unavailable reason. The order-parameter values and their errors are RNG-free
and repeat byte-for-byte across runs and machines for identical inputs. The
recorded `seed` governs only the optional seeded random-phase mode; the default
initial condition is derived deterministically from `omega`.

Wall-clock `elapsed_ms` is recorded for context but is advisory and
machine-dependent, so it is excluded from the reproducible-quantity set. Each
artifact also embeds the documented `failure_modes` and a `claim_boundary`
statement: for statevector-scale sizes (`n <= 16`) the classical exact route is
faster and exact, so the record states **no quantum advantage**; above that
boundary, unavailable statevector rows prevent an implicit speed-up claim.

```python
from scpn_quantum_control import run_reproducible_kuramoto_comparison

comparison = run_reproducible_kuramoto_comparison(8, t_max=1.0, dt=0.1, seed=42)
artifact = comparison.to_dict()
```

For larger classical baselines:

```python
comparison = run_reproducible_kuramoto_comparison(20, t_max=0.2, dt=0.1, seed=42)
assert comparison.reference_method == "classical_ode"
assert comparison.row("quantum_trotter").available is False
```

Example 09 emits this artifact on demand:

The partitioned circuit-cutting planner covers larger synthetic Kuramoto-XY
networks without relaxing the dense statevector boundary. For example,
`circuit_cutting_plan(build_knm_paper27(L=128), max_partition_size=16)` returns
eight 16-oscillator partitions and reports the classical reconstruction
overhead. Multi-partition energies stay labelled as partition-local diagnostics;
the full-system dense energy is not claimed.

```bash
python examples/09_classical_vs_quantum_benchmark.py \
    --artifact data/classical_quantum_comparison/reproducible_comparison_n8.json
```

A committed reference artifact lives at
`data/classical_quantum_comparison/reproducible_comparison_n8.json`.
