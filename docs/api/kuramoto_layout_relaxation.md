# Kuramoto layout-relaxation API

`scpn_quantum_control.hardware.kuramoto_layout_relaxation` studies whether an
annealed Sinkhorn relaxation can improve on the repository's discrete layout
search at a matched budget of true-cost evaluations. The surface remains
research-labelled: its committed comparison found no consistent gain, so the
discrete optimiser remains the recommended route.

## Relaxation and rounding

`sinkhorn_normalise(logits, n_iterations)` alternates row and column
normalisation in log space and returns a numerically doubly-stochastic matrix.
`swap_distance_surrogate(P, K, distances)` then prices expected coupling-graph
distance under the relaxed placement. The optimiser differentiates that
surrogate, applies the gradient to placement logits, and rounds each annealing
step with a Hungarian assignment.

Rounded candidates are evaluated by `kuramoto_layout_cost`, not by the
surrogate. Consequently the comparison retains routed depth, product-formula
error, calibrated fidelity, and the configured weights of the discrete-search
objective.

## Configuration and result

`SinkhornRelaxationConfig` binds the temperature endpoints, annealing and
gradient step counts, learning rate, Sinkhorn iterations, true-cost budget,
seed, cost weights, evolution time, repetitions, and formula order. Invalid
temperatures, non-positive counts or rates, and malformed cost controls fail
before search execution.

`RelaxationSearchResult` records the best rounded layout and its complete
`LayoutCost`, the number of distinct true-cost evaluations, the surrogate
trajectory, and the research label. Both records provide JSON-ready mappings.

```python
from scpn_quantum_control.hardware import (
    SinkhornRelaxationConfig,
    relax_kuramoto_layout,
)

result = relax_kuramoto_layout(
    coupling,
    frequencies,
    coupling_map,
    physical_qubits=(0, 1, 2, 3, 4),
    mean_gate_fidelity=0.99,
    config=SinkhornRelaxationConfig(
        seed=7,
        n_anneal_steps=8,
        max_true_cost_evaluations=8,
    ),
)
print(result.best_layout, result.best_cost.total)
```

## Evidence boundary

The surrogate is a differentiable search guide, not a hardware measurement or
the comparison metric. A fixed seed makes the current NumPy search
deterministic, while custom depth providers may carry their own provenance and
reproducibility requirements. The result does not establish quantum advantage,
hardware success, or promotion readiness; those require separate approved
evidence.

The preregistration and measured no-gain outcome are recorded in
[`layout_relaxation_preregistration.md`](../layout_relaxation_preregistration.md).
The surrounding mapper and discrete-search contract is documented in
[`dynq_qubit_mapping.md`](../dynq_qubit_mapping.md).

## API reference

::: scpn_quantum_control.hardware.kuramoto_layout_relaxation
    options:
      show_root_heading: true
      members_order: source
