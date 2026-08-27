# Quantum Reservoir Computing and Classical Surrogates

This page defines the bounded BL-45 quantum reservoir computing (QRC), matched
classical baseline, and differentiable classical-surrogate surfaces.

## Production Surfaces

The QRC surface preserves the existing reservoir and adds certificate layers:

- `scpn_quantum_control.applications.quantum_reservoir` maps classical inputs
  through Kuramoto-XY Hamiltonian evolution and Pauli expectation features, then
  fits a ridge readout. Exact-statevector allocation is budget checked, oversized
  input vectors are refused, and Pauli labels are generated without enumerating
  the full `4**n` string space.
- `scpn_quantum_control.applications.qrc_baseline` compares QRC with a
  deterministic ESN at equal feature count. The held-out comparison continues
  ESN state from training into validation instead of resetting it.
- `scpn_quantum_control.applications.quantum_reservoir_product` creates disjoint
  synthetic forecast/classification certificates and weighted exact Pauli-feature
  objectives.
- `scpn_quantum_control.surrogates` fits a Gaussian radial-basis surrogate,
  exposes its analytic input gradient, rejects train/validation leakage, and
  certifies held-out values and gradients against the exact local objective.
- `scpn_quantum_control.analysis.qrc_phase_detector` uses exact dense
  ground-state Pauli features as a small-system phase-detector reference.

## Held-out QRC / ESN certificates

```python
import numpy as np

from scpn_quantum_control.applications import (
    ReservoirTaskKind,
    certify_reservoir_training,
    generate_synthetic_reservoir_task,
)

K = np.array([[0.0, 0.65], [0.65, 0.0]])
dataset = generate_synthetic_reservoir_task(
    ReservoirTaskKind.CLASSIFICATION,
    n_train=18,
    n_validation=8,
    seed=4139971,
)

certificate = certify_reservoir_training(
    dataset,
    K,
    omega=np.array([0.15, -0.1]),
    alpha=0.1,
    max_weight=1,
    t=0.8,
    seed=4139971,
)
```

The committed deterministic evidence uses 18 training and 8 validation rows
for each synthetic task. Both systems have six readout features:

| Synthetic task | QRC validation MSE | ESN validation MSE | Lower MSE |
|---|---:|---:|---|
| Nonlinear classification | 0.0696474284 | 0.0913905636 | QRC |
| One-step forecast | 0.890831726 | 0.0195529047 | ESN |

The result is intentionally mixed. It establishes two working held-out
certificate paths, not a general performance result. The forecast row is also a
direct negative control against presenting QRC as the default winner.

## Classical Baseline

`classical_esn_feature_matrix` implements a deterministic ESN reference with:

- seeded input and recurrent weights;
- recurrent matrix rescaled to the requested spectral radius;
- leaky state update;
- ridge readout through `classical_esn_ridge_regression`.

Callers can request a deliberately unmatched `reservoir_size`, but BL-45
evidence requires equal feature count and reports the capacity match explicitly.

## Differentiable classical surrogate

The BL-45 surrogate is a regularised Gaussian radial-basis model. Fitting stores
SHA-256 identities for every training row and the target vector. Certification
rejects any validation row that overlaps the training set.

On the frozen two-parameter weighted-Pauli objective, 25 training points and 16
disjoint validation points produced:

| Gate | Frozen threshold | Observed | Result |
|---|---:|---:|---|
| Held-out RMSE | <= 0.01 | 0.000422869640 | pass |
| Held-out maximum absolute error | <= 0.025 | 0.000791764885 | pass |
| Held-out R-squared | >= 0.98 | 0.999994691 | pass |
| Analytic-gradient maximum error | <= 0.02 | 0.000893768362 | pass |

The gradient reference uses central differences of the exact local statevector
objective at four disjoint points. It is not an analytic quantum gradient or a
hardware-gradient result.

## Exact-validated co-design proposal

`propose_and_validate_surrogate_step(...)` converts the surrogate gradient into
a norm-bounded, unapplied `ControllerProposal`, then evaluates both current and
candidate parameters through the caller's exact local objective. The frozen
evidence candidate improved the exact objective from `0.0123150159` to
`-0.0490886919`. The acceptance flag records that exact local observation only;
the function does not emit a BL-33 safety decision or apply an update.

Regenerate and byte-check the evidence with:

```bash
PYTHONPATH=src python scripts/run_quantum_reservoir_surrogate_evidence.py
PYTHONPATH=src python scripts/run_quantum_reservoir_surrogate_evidence.py --check
```

Committed custody:

- `data/quantum_reservoir_surrogates/quantum_reservoir_evidence.json`
- `data/quantum_reservoir_surrogates/quantum_reservoir_evidence.md`
- content digest `8b555933e6ec7f9b2ee3c885379ef87af11e1002ce4a2d119fa83f26507de41c`

## Scientific basis

- [Fujii and Nakajima (2017)](https://doi.org/10.1103/PhysRevApplied.8.024030)
  motivates fixed quantum dynamics with a trained classical readout.
- [Schreiber, Eisert, and Meyer (2023)](https://doi.org/10.1103/PhysRevLett.131.100803)
  defines classical surrogates through bounded reproduction of quantum-model
  input-output relations and treats them as a natural honesty baseline.
- [O'Leary et al. (2025)](https://doi.org/10.1038/s42005-025-02423-4)
  motivates radial-basis proposals followed by a true quantum-objective query.

These papers motivate the architecture. They do not validate this repository's
specific fidelity thresholds, tasks, or proposal policy.

This is functional evidence for wiring and bounded task behaviour. It is not an
isolated-core production benchmark.

## Explicit Boundaries

- The QRC feature map is exact-statevector and small-system bounded.
- The phase detector is an exact dense reference, not a scalable reservoir
  simulator.
- The ESN baseline is a deterministic NumPy reference comparator, not an
  accelerated service path.
- BL-37 now supplies a bounded simulation-only multimodal schema and classical
  forecasting product, but no QRC-to-BL-37 adapter was added in BL-45. No real
  clinical, grid, or plasma data is admitted by either product.
- Differentiable notebook curriculum expansion is outside the reservoir
  evidence scope and is not represented as complete.
- No hardware QRC, provider execution, unseen-domain generalisation, closed-loop
  control, optimisation advantage, publication, or deployment claim is made.

## Related Surfaces

- [`Analysis API`](analysis_api.md)
- [`API Overview`](api.md#applications)
- [`Pipeline Performance`](pipeline_performance.md#quantum-reservoir-computing)
