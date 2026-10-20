# JAX NQS baseline product

`scpn_quantum_control.jax_nqs_baseline_product` is the exact-reference evidence
layer for the ambient JAX restricted-Boltzmann-machine runner. It is deliberately
a small-system research baseline.

## What it establishes

For a validated symmetric finite coupling matrix and matching frequency vector,
the product:

1. restricts the evidence run to `2 <= N <= 6`;
2. runs the existing real-valued RBM with exact enumeration and JAX automatic
   differentiation;
3. computes the exact dense ground energy for the same Hamiltonian;
4. reports the variational gap, relative error, energy-history diagnostics, and
   declared-tolerance result;
5. records the JAX version, backend, device kind, and X64 configuration;
6. binds the observation to canonical JSON with SHA-256; and
7. attaches the default no-advantage certificate.

The exact-reference comparison is the result. An optimiser reaching the
declared tolerance on one instance is not a general accuracy guarantee.

## Usage

```python
import numpy as np

from scpn_quantum_control.jax_nqs_baseline_product import (
    JAXNQSBaselineSpec,
    run_jax_nqs_baseline,
    write_jax_nqs_baseline_evidence,
)

K = np.array(((0.0, 1.0), (1.0, 0.0)))
omega = np.array((-0.2, 0.2))
spec = JAXNQSBaselineSpec.from_arrays(
    K,
    omega,
    n_hidden=4,
    learning_rate=0.03,
    n_iterations=200,
    seed=7,
    relative_error_tolerance=0.2,
)
product = run_jax_nqs_baseline(spec)
write_jax_nqs_baseline_evidence(product, "evidence.json", "evidence.md")
```

JAX is optional. If it is absent, `run_jax_nqs_baseline` raises `ImportError`;
it does not silently substitute the NumPy finite-difference implementation.

## Contracts

`JAXNQSBaselineSpec` copies array inputs into immutable tuples and rejects:

- non-square, asymmetric, or non-finite coupling matrices;
- mismatched or non-finite frequency vectors;
- non-positive hidden width or learning rate;
- negative seeds;
- iteration counts outside the bounded `[1, 5000]` interval;
- invalid error/slack tolerances or dense-memory budgets; and
- systems outside the product evidence limit.

The underlying ambient runner still has its historical `N <= 12` API cap. That
larger cap is exponential exact enumeration, not scalable sampled VMC. This
product uses the stricter `N <= 6` limit so every claim remains directly
checkable against exact diagonalisation.

The Hamiltonian builder consumes only pair terms with `i < j`. Coupling-matrix
diagonal entries are therefore retained in request provenance but explicitly
reported as unused; they are neither rejected nor silently interpreted as
self-interactions.

## Evidence interpretation

The variational principle gives an upper-bound check for a normalised trial
state. The product allows only the explicitly recorded numerical slack when
testing that relation. It separately reports:

- `absolute_gap = |E_RBM - E_exact|`;
- `relative_error = absolute_gap / max(|E_exact|, epsilon)`;
- whether the final energy decreased from the first recorded step; and
- whether the requested relative-error tolerance was met.

JAX defaults to 32-bit values when `jax_enable_x64` is false. The product records
that setting because precision is part of interpreting the comparison; it does
not mutate global JAX configuration. JAX documents `grad` as a transformation
of scalar-valued functions and `jit` as XLA compilation with an upfront compile
and caching boundary:

- [JAX automatic differentiation](https://docs.jax.dev/en/latest/automatic-differentiation.html)
- [JAX JIT compilation](https://docs.jax.dev/en/latest/jit-compilation.html)
- [JAX default dtypes and X64](https://docs.jax.dev/en/latest/default_dtypes.html)

The RBM form follows the neural quantum-state variational approach introduced
by Carleo and Troyer. Their results concern selected spin models and do not
establish universal convergence for this repository's implementation:

- [Carleo and Troyer, *Solving the Quantum Many-Body Problem with Artificial
  Neural Networks*](https://arxiv.org/abs/1606.02318)

## Claim boundary

This product does not establish:

- sampled or stochastic VMC;
- scaling beyond exact enumeration;
- parity with NetKet or another external NQS framework;
- a NumPy-versus-JAX speedup;
- category leadership or promotion of the `jax_native_transforms` scorecard;
- hardware/provider execution; or
- quantum or classical performance advantage.

The JSON record hard-codes those promotion flags to `false`, and construction
fails if they are changed.

## Frozen-state qualification

The additive public qualify_jax_nqs_baseline(spec, held_out=...) returns a
VariationalBaselineQualification companion. Its baseline is the unchanged
jax_nqs_baseline_product.v2 record; the companion uses
variational_baseline_qualification.v1. The companion digest covers its full
JSON payload, including the original baseline and observed costs.

The qualifier performs two actual native training runs with the same request
and seed. It requires the entire loss history to reproduce exactly within that
process and binds the replay parameters by shape, dtype and value. It then
evaluates energy and the lowest-reference-subspace projector through the public
JAX energy function. An independently reconstructed NumPy RBM state checks both
expectations within 1e-5 + 1e-4 * abs(reference). Observed parameter dtypes and
both tolerance constants are recorded. The original runner builds its training
Hamiltonian and spin configurations as float32; enabling global JAX X64 does
not make every intermediate float64.

~~~python
from scpn_quantum_control.jax_nqs_baseline_product import qualify_jax_nqs_baseline

qualified = qualify_jax_nqs_baseline(
    spec, held_out={"opposite_field": (K, -omega)}
)
print(qualified.training.reference_subspace_overlap)
print(qualified.errors)
print(qualified.to_dict()["evidence_sha256"])
~~~

Held-out Hamiltonians are copied, validated and frozen before training. Up to
eight distinct effective Hamiltonians with the same visible shape are admitted;
the label training is reserved. Changes only to the unused coupling diagonal
do not create a new problem. Evaluation never updates the fitted parameters.
This measures transfer of a fixed state between Hamiltonians; it does not
establish statistical generalisation to an unseen dataset. Qualification admits
at most 4096 RBM parameters and retains the original two-through-six-spin cap.

The squared overlap is with all eigenvectors within
1e-10 + 1e-8 * abs(E0) of the dense minimum. The dimension and energy window
are recorded. For a degenerate ground space, this is a subspace projection,
rather than fidelity with a particular eigenvector.

## Error attribution and limitations

Exact enumeration has zero Monte Carlo sampling standard error. That statement
does not remove finite precision, representation bias or optimisation error.

The qualifier separates representation bias from optimisation excess only for
two analytically identifiable effective Hamiltonian families:

- A diagonal Hamiltonian admits a computational-basis ground state in the closure
  of the real RBM family; its ansatz energy infimum equals the exact minimum.
- For two spins with zero field, write c = H[1, 2] = -2*K[0, 1].
  Positivity of the real RBM amplitudes gives ansatz infimum min(c, 0),
  while the exact ground energy is -abs(c). Negative pair coupling therefore
  has representation bias 2*abs(K[0, 1]): its antisymmetric ground state
  requires a relative sign unavailable to this positive-amplitude ansatz.

These are infima in the closure; finite parameters need not attain them.
Other problems retain None for the infimum, representation bias and
optimisation excess. Their combined signed energy excess is still measured.
Round-off residuals are retained without clipping. A decreasing loss alone
does not qualify state overlap, a global optimum or an advantage claim.

The required runtime tests compare the sign-obstructed two-spin fixture with
an actual Aer matrix-product-state preparation of the antisymmetric ground
state. This bounded reference is not a many-body crossover benchmark.

Wall times separately record the original baseline run, seeded replay and
frozen evaluation, including compilation and reference work. Explicit loss,
gradient-transform and observable-call counts disclose the doubled training
budget; exhaustive configurations are not Monte Carlo samples. These functional
timings make no isolated speed or scaling claim. Run the complete file example
with PYTHONPATH=src:oscillatools/src python examples/variational_baseline_qualification.py.
