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

## Decoder evidence, code structure and overhead scenarios

Four things stay separate, so that no single number reads as more than it is.

- **Code structure.** `enumerate_single_error_syndromes(distance)` flips each
  data qubit of an admitted toric code (distance 3 or 5) once. It records which
  vertex checks fire and whether the existing minimum-weight perfect-matching
  decoder restores a trivial logical class. Every single flip fires exactly the
  two vertices its edge touches and is corrected.
- **Decoder Monte Carlo.** `run_toric_decoder_experiment(distance, p, shots,
  seed)` samples independent bit flips at rate `p` with one round of perfect
  syndrome extraction, decodes every shot and counts logical failures. The same
  seed repeats the sample.
- **Calibrated confidence.** `logical_failure_confidence(failures, trials,
  sample_model=...)` gives an exact two-sided Clopper–Pearson interval, stated
  only for the sample model it names. Zero failures in `n` trials still give a
  positive upper bound, `1 - (α/2)^(1/n)`. A normal approximation would
  collapse to zero width there.
- **Logical-operation meaning.** `LOGICAL_OPERATION_SEMANTICS` records the
  status of each operation:
  - logical RZ and RZZ are counted from the synthesis estimate but not
    validated;
  - syndrome extraction is modelled as ideal;
  - magic-state distillation and lattice-surgery routing are not modelled.
- **Future resources.** `estimate_overhead_scenario(request, assumptions)`
  adds the routing factor, the factories (count, size and cycles per T state)
  and the decoder latency that you state to the base estimate. The base
  estimate leaves these out (`EXCLUDED_FROM_BASE_ESTIMATE`).

  `build_qec_claim_qualification` combines several scenarios, the operation
  meanings and an optional decoder experiment. It reports the spread of the
  scenarios as the scaling uncertainty. The record refers to the base report by
  its payload digest and leaves the base report unchanged.

```python
from scpn_quantum_control.fault_tolerant_resource_product import (
    OverheadAssumptions,
    SyncProblemResourceRequest,
    build_qec_claim_qualification,
    run_toric_decoder_experiment,
)

request = SyncProblemResourceRequest(
    n_oscillators=4,
    evolution_time=1.0,
    target_precision=0.01,
    coupling_density=0.5,
    trotter_steps=8,
)
experiment = run_toric_decoder_experiment(3, 0.001, 300, seed=7)
assert experiment.confidence.failures == 0 and experiment.confidence.upper > 0.0
record = build_qec_claim_qualification(
    request,
    (
        OverheadAssumptions("compact", 1.5, 2, 1000, 10, 1e-6),
        OverheadAssumptions("wide", 3.0, 8, 1000, 10, 1e-6),
    ),
    decoder_experiment=experiment,
)
low, high = record.total_physical_qubit_range
assert low < high and record.device_available_claim_allowed is False
```

Overhead scenarios are space and time floors under assumptions you state. They
are not runtimes. Neither a scenario nor a qualification record can claim that
hardware is available or that it runs fault-tolerantly. A decoder experiment is
a software sample under the stated error model. It is not a measured reduction
in logical error.

The Studio QEC-readiness bundle applies the same exact interval to each
committed decoder aggregate, recomputed from that row's failures and trials.
The artefact stores normal-approximation intervals, which read `[0, 0]` for
rows with no failures, so the bundle does not carry them.

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
