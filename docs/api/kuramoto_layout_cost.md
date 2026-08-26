# Kuramoto-XY layout-cost API

`scpn_quantum_control.hardware.kuramoto_layout_cost` assigns one comparable
objective to an injective logical-to-physical qubit placement. It composes
post-routing depth, a product-formula error bound, and calibrated mean gate
infidelity without introducing a second compiler, error model, or mapper.

## Objective and records

For layout `l`, coupling matrix `K`, frequencies `omega`, and target topology,
the objective is

`w_depth * routed_depth + w_error * trotter_error + w_infidelity * (1 - fidelity)`.

`CostWeights` requires finite non-negative weights and rejects the all-zero
case. `LayoutCost` retains the total plus every unweighted value and weighted
term; `to_dict()` produces a JSON-ready mapping without discarding provenance
needed to interpret the objective.

## Cost construction

`kuramoto_layout_cost(...)` validates a square coupling matrix, matching
frequency vector, distinct non-negative physical indices, fidelity in `[0, 1]`,
positive finite evolution time, and positive repetition count. Invalid inputs
fail before compilation or routing.

The default `routed_layout_depth` compiles the existing XY product-formula
circuit and routes it with Qiskit against the supplied coupling map and initial
layout. Callers may inject a `DepthProvider` when they already own a measured or
cached routing result. Injection changes only the depth source; validation,
Trotter error, fidelity pricing, and weighted assembly remain identical.

```python
from qiskit.transpiler import CouplingMap
from scpn_quantum_control.hardware import CostWeights, kuramoto_layout_cost

cost = kuramoto_layout_cost(
    layout=(0, 1, 2, 3),
    K=coupling,
    omega=frequencies,
    coupling_map=CouplingMap([[0, 1], [1, 2], [2, 3]]),
    mean_gate_fidelity=0.99,
    weights=CostWeights(depth=1.0, trotter_error=50.0, infidelity=10.0),
)
print(cost.total, cost.routed_depth)
```

## Reproducibility and evidence boundary

Qiskit routing can be stochastic. `routed_layout_depth` therefore accepts
`seed_transpiler`; callers constructing a reproducible landscape must bind it.
The routed depth is a compiler measurement for the supplied target and seed.
The product-formula error is an analytic bound, and the fidelity term is a
calibration-priced model. The combined objective is suitable for comparing
placements under one fixed contract; it is not hardware success probability or
evidence of quantum advantage.

`dynq_mean_gate_fidelity(result)` extracts the selected execution-region
fidelity from a validated `QubitMappingResult`; it does not recompute or promote
the mapper's calibration evidence.

## API reference

::: scpn_quantum_control.hardware.kuramoto_layout_cost
    options:
      show_root_heading: true
      members_order: source
