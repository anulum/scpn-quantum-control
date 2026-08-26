# Circuit-cutting product boundary

`scpn_quantum_control.circuit_cutting_product` freezes and governs the existing
circuit-cutting planner and partition-local simulator. It is a local,
no-submit product surface; it does not claim general circuit reconstruction or
large-N hardware execution.

## Resource certificate

```python
from scpn_quantum_control.bridge.knm_hamiltonian import build_knm_paper27
from scpn_quantum_control.circuit_cutting_product import (
    build_cutting_resource_certificate,
)

certificate = build_cutting_resource_certificate(
    build_knm_paper27(L=4),
    max_partition_size=4,
    shots_per_fragment=256,
)
assert certificate.feasible
assert certificate.fragment_evaluations == 1
assert certificate.estimated_total_shots == 256
```

The planner uses the ambient cut count and computes
`fragment_evaluations = 4**n_cuts` and
`estimated_total_shots = fragment_evaluations * shots_per_fragment`. The
selected hardware-safe execution policy supplies the per-fragment and
total-shot ceilings. A
non-finite `4**n_cuts` estimate, an over-budget plan, an oversized partition,
or a would-submit request is refused.

## Synthetic reconstruction certificate

```python
from scpn_quantum_control.circuit_cutting_product import (
    certify_synthetic_reconstruction,
)

evidence = certify_synthetic_reconstruction(
    observable_id="R_global",
    exact_value=0.8,
    reconstructed_value=0.79,
    declared_error_bound=0.02,
)
assert evidence.within_bound
assert evidence.synthetic_only
assert not evidence.hardware_result
```

The caller supplies the synthetic exact and reconstructed observables plus the
bound. The certificate computes and checks the absolute error; it does not
manufacture a reconstruction or convert simulation data into hardware evidence.

## Fail-closed boundaries

- The existing runner evolves partitions independently. With multiple
  partitions, its energy is `partition_local_sum`, and omitted
  cross-partition coupling is reported. It is not full-system energy.
- Partition-local energy requires explicit caller acceptance.
- Dense large-N all-to-all plans usually have prohibitive or non-finite
  `4**n_cuts` overhead and are refused, not promoted as feasible.
- Large-system benchmark-family registration remains optional and is not
  implied by this product.
- No product path submits a provider job or claims hardware advantage.
