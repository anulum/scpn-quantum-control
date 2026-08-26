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
