<!--
SPDX-License-Identifier: AGPL-3.0-or-later
Commercial license available
© Concepts 1996–2026 Miroslav Šotek. All rights reserved.
© Code 2020–2026 Miroslav Šotek. All rights reserved.
ORCID: 0009-0009-3560-0851
Contact: www.anulum.li | protoscience@anulum.li
SCPN Quantum Control — Choosing a gradient path
-->

# Choosing a gradient path

This is the front door for the differentiable-programming surface: a fast route
from *what you are differentiating* to *which entry point to call*. It is a guide,
not a promotion — see [Claim status](#claim-status) before you quote any number.

## Start here — one canonical entry point

Import the canonical namespace and call it like a JAX-style transform library:

```python
from scpn_quantum_control import diff  # alias: scpn.diff

value, gradient = diff.value_and_grad(objective, parameters)
```

`diff` exposes `grad`, `value_and_grad`, `jacfwd`, `jacrev`, `jacobian`,
`hessian`, `jvp`, `vjp`, `vmap`, and `jit_or_explain`. It wraps the supported
local routes and returns a **fail-closed diagnostic** (not a silent finite-difference
substitute) for any unsupported JIT, provider, hardware, or performance route.
The full namespace map is the [Differentiable API](differentiable_api.md); the
executable capability rows are the [Differentiable Support Matrix](differentiable_support_matrix.md).

## 60-second decision table

| What you are differentiating | Use | Why |
|---|---|---|
| A control objective with **many parameters → one scalar cost** (e.g. a long control campaign) | `diff.value_and_grad` / `diff.jacrev` (reverse mode) | Reverse mode costs one backward pass regardless of parameter count. |
| A **few parameters → many outputs** | `diff.jacfwd` (forward mode) | Forward mode costs one pass per input, so it wins when inputs are few. |
| **Quantum-circuit / variational** parameters (gates, VQE, phase-QNN) | parameter-shift — `diff.value_and_grad(..., method="parameter_shift")` or `phase.param_shift` | Exact circuit gradients from the parameter-shift rule; the default method. |
| Gradients **through the Kuramoto simulator or MPC controller** | the oscillatools JAX-adjoint family `diff_kuramoto_{euler,rk4,dopri,adaptive,delayed,inertial,noisy}` and `diff_kuramoto_mpc_kkt` | Exact reverse-mode adjoints of the integrator/controller, each witnessed against JAX autodiff. |
| **Verifying** another path's gradient | `method="finite_difference"` or `method="complex_step"` | Independent numerical references; complex-step avoids subtractive cancellation for real-analytic objectives. |
| A **whole-program** trace (research) | `method="whole_program"` (Program AD) | Traces the program's primitives; see the claim status below — this lane is research-tier. |

The lower-level dispatcher `differentiable_canonical_api.value_and_grad(objective,
values, *, method=...)` accepts exactly these methods: `parameter_shift` (default),
`finite_difference`, `complex_step`, `forward_mode`, `reverse_mode`, `whole_program`.
An unknown method raises rather than guessing.

## Claim status

The differentiable surface is deliberately **fail-closed and non-promotional**.
In the claim ledger (`data/differentiable_phase_qnode/claim_ledger.md`) every row is
currently `bounded_candidate` — **no row is `promoted`**. The promotion vocabulary is:

- **`promoted`** — production-supported; requires passing isolated-benchmark and
  external-comparison artefacts. *No differentiable-programming row holds this status yet.*
- **`bounded_candidate`** — implemented and locally exercised, but not yet promoted;
  the current state of the whole lane.
- **`hard_gap`** — a required dependency or artefact is missing (e.g. an unconfigured
  Enzyme/LLVM runner).
- **`blocked`** — explicitly out of scope until artefacts exist: PyTorch dynamic-shape
  and fullgraph compile, AOTAutograd/export persistence, and any provider/QPU
  hardware-gradient execution.

Practical consequence: pick a path from the table above for *local, bounded* work,
pair it with the support matrix and tests for your exact primitive/backend/shape,
and do not quote provider, hardware, compiler-execution, or performance advantage
until the corresponding ledger row is promoted with evidence.

## Where to go next

- [Differentiable API](differentiable_api.md) — the full public namespace map.
- [Differentiable Support Matrix](differentiable_support_matrix.md) — executable
  capability rows and fail-closed boundaries.
- [Differentiable Reviewer Evidence](differentiable_reviewer_evidence.md) — scoped
  reproduction commands and open-gap pointers.
