# Automated Kuramoto Witness Discovery

`scpn_quantum_control.analysis.witness_discovery` runs a replayable search over
Kuramoto control candidates and scores each candidate with synchronisation
witnesses. It is intended for small simulator sweeps and QPU pre-screening:
find the parameter regions where correlation and Fiedler witnesses fire before
spending hardware shots.

The loop combines two proposal mechanisms:

- an RBF Bayesian surrogate with upper-confidence-bound acquisition,
- a bandit-style local exploration policy around the best current candidate.

Trajectory feature extraction uses the Rust PyO3 kernel
`kuramoto_witness_candidate_features` when available, with a NumPy reference path
for parity and portability.

## Search Space

Each candidate is

\[
c = (s_K, s_\omega, b_\theta),
\]

where:

- \(s_K\) scales the coupling matrix,
- \(s_\omega\) scales natural frequencies,
- \(b_\theta\) shifts the initial phase vector.

For each candidate, the kernel integrates

\[
\dot\theta_i = s_\omega\omega_i +
\sum_j s_KK_{ij}\sin(\theta_j-\theta_i)
\]

and returns final \(R\), mean pairwise phase correlation, and final phases. The
Python scorer then builds the correlation matrix
\(\cos(\theta_i-\theta_j)\), evaluates the existing synchronisation witnesses,
and ranks candidates by a weighted witness objective.

## API

```python
import numpy as np

from scpn_quantum_control.analysis import (
    WitnessDiscoverySpec,
    discover_kuramoto_witnesses,
)

K = np.array(
    [
        [0.0, 0.5, 0.2, 0.0],
        [0.5, 0.0, 0.4, 0.1],
        [0.2, 0.4, 0.0, 0.3],
        [0.0, 0.1, 0.3, 0.0],
    ],
    dtype=np.float64,
)
omega = np.array([0.0, 0.35, 0.7, 1.05], dtype=np.float64)
theta0 = np.array([0.0, 0.7, 1.4, 2.8], dtype=np.float64)

spec = WitnessDiscoverySpec(
    dt=0.025,
    n_steps=48,
    n_initial=8,
    n_iterations=4,
    batch_size=3,
    pool_size=32,
    seed=20260429,
    correlation_threshold=0.25,
    fiedler_threshold=0.2,
)

result = discover_kuramoto_witnesses(K, omega, theta0=theta0, spec=spec)
print(result.best.candidate.to_metadata())
print(result.best.score, result.best.final_r)
```

For fixed candidate batches, use `score_witness_candidates(...)`. The
`RLDiscoveryAgent` compatibility class now delegates to this real discovery
loop and fails loudly when no `K_nm`/`omega` problem is configured. The wrapper
does not accept unwired compatibility settings: `runner` must be `None`,
`observables` must remain `["correlation", "fiedler"]`, `reward_function` must
be `"witness_score"`, and `n_episodes` must be positive.

## Result Fields

`WitnessDiscoveryResult` contains:

| Field | Description |
|-------|-------------|
| `evaluations` | Full scored candidate trace. |
| `best` | Highest-scoring `WitnessDiscoveryEvaluation`. |
| `backend` | Rust or NumPy feature backend used by the trace. |
| `ranked(limit)` | Candidate ranking by descending score. |
| `to_json()` | Replay metadata for result archiving. |

Each evaluation records the candidate, score, final \(R\), mean correlation,
Fiedler value, witness margins, proposal source, acquisition value, and backend.

## Validation

`tests/test_witness_discovery.py` covers:

- fixed candidate scoring through real witness objects,
- deterministic end-to-end search,
- presence of initial, Bayesian, and bandit proposal sources,
- Rust/NumPy feature parity,
- RBF surrogate behaviour,
- JSON serialisation,
- configured and unconfigured `RLDiscoveryAgent` behaviour,
- shape/range validation before search.

## Benchmark

See [Pipeline Performance](pipeline_performance.md) for command provenance. On
the ASRock H510 Pro BTC+ / i5-11600K / Ubuntu 24.04.4 machine, a 20-evaluation
4-oscillator discovery loop with 48 Euler steps per candidate runs in 10.145 ms
using the Rust PyO3 feature kernel.
