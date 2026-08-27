# Fault-tolerant resource product

`scpn_quantum_control.fault_tolerant_resource_product` turns the existing QEC
resource primitives into one deterministic, digest-bound planning report for a
bounded Kuramoto/XY request. It does not build another decoder or surface-code
circuit.

Serialized reports use schema `fault_tolerant_resource_product.v2`. Stored
reports with stale or unknown schema identifiers are refused by the immutable
product contract.

The request pins oscillator count, evolution time, target precision, coupling
density, Trotter steps, assumed physical error rate, syndrome-cycle duration,
NISQ shots, and maximum odd code distance. The estimator then:

1. splits target precision equally across Trotter, logical-failure, and
   rotation-synthesis budgets;
2. selects the smallest odd distance whose phenomenological logical-error
   opportunities satisfy a conservative union bound;
3. reuses the existing `2d²-1` rotated-patch and `2d-1` repetition-scaffold
   register counts;
4. counts local and pairwise arbitrary Z-axis rotations from the declared
   coupling density; and
5. applies Selinger's conservative `ceil(10 + 4 log2(1/epsilon))` Clifford+T
   Z-rotation estimate after allocating synthesis precision across all counted
   rotations.

```python
from scpn_quantum_control.fault_tolerant_resource_product import (
    SyncProblemResourceRequest,
    build_fault_tolerant_resource_product,
    render_ft_resource_markdown,
)

request = SyncProblemResourceRequest(
    n_oscillators=4,
    evolution_time=1.0,
    target_precision=0.01,
    coupling_density=0.5,
    trotter_steps=8,
)
product = build_fault_tolerant_resource_product(request)
assert product.estimate.hardware_availability_claim_allowed is False
markdown = render_ft_resource_markdown(product)
```

## Regime boundaries

The report keeps six regimes non-equivalent: classical reference, NISQ
sampling, bit-flip-only repetition scaffold, unmeasured surface-code-shaped
scaffold, bounded analog-feasibility dependency, and the fault-tolerant
planning model.
Register size never means a named device can execute the workload. The reported
syndrome time is a floor for the assumed number of syndrome cycles; it excludes
decoding, routing, feed-forward, magic-state factories, and logical-gate
schedules, so it is not total runtime.

The Trotter allocation is a budget, not proof that a particular Hamiltonian and
step count attain it. The surface-code logical-rate expression is a repository
phenomenological ansatz. The existing `SurfaceCodeUPDE` operations remain
structural proxies with no measured syndrome, decoder, or validated logical
RZ/RZZ semantics.

## Verified primary-source pins

- [Horsman et al., *Surface code quantum computing by lattice surgery*](https://arxiv.org/abs/1111.4022)
- [Fowler et al., *Surface codes: Towards practical large-scale quantum computation*](https://arxiv.org/abs/1208.0928)
- [Selinger, *Efficient Clifford+T approximation of single-qubit operators*](https://arxiv.org/abs/1212.6253)
- [Google Quantum AI et al., *Quantum error correction below the surface code threshold*](https://arxiv.org/abs/2408.13687)

These pins support architecture/scaling context and the declared synthesis
estimate. They do not turn the product's simplified assumptions into a
hardware-calibrated resource compiler.

## Claim boundary

Conservative future-resource planning only. No available fault-tolerant
hardware, validated logical RZ/RZZ, decoder integration, magic-state factory,
total runtime, target-precision attainment, or fault-tolerant execution is
claimed. The previous notebook's named-device register-size comparison is not
an execution-feasibility result; this document is its claim-bounded successor.
