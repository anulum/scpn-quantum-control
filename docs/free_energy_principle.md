# Free Energy Principle (FEP) on Quantum Substrate

## 1. Mathematical Formalism

### Variational Free Energy

The Free Energy Principle (Friston 2010) defines variational free energy as:

$$F = \mathbb{E}_q[\log q(z) - \log p(z, x)] = D_\text{KL}[q(z) \| p(z|x)] - \log p(x)$$

Since $D_\text{KL} \geq 0$, free energy upper-bounds surprise: $F \geq -\log p(x)$.
Minimising $F$ thus minimises surprise (self-information of observations).

For Gaussian distributions:

$$q(z) = \mathcal{N}(\mu, \Sigma), \quad p(z) = \mathcal{N}(0, \Pi^{-1})$$

the implemented reduced objective decomposes as:

$$F = \underbrace{\frac{1}{2}\left(\mu^\top \Pi \mu + \text{tr}(\Pi\Sigma) - \log|\Sigma| - \log|\Pi| - n\right)}_\text{complexity (KL to prior)} + \underbrace{\frac{1}{2}(x - g(\mu))^\top \Gamma (x - g(\mu))}_\text{accuracy (prediction error)}$$

where $\Pi$ is prior precision, $\Gamma$ is sensory precision, and $g(\mu)$
is the generative model. Here the public API computes prior KL plus quadratic
prediction error. It omits likelihood normalisation, including
`-0.5 * logdet(sensory_precision)`, and evaluates the model at the belief mean.
Consequently `elbo = -F` and `surprise_bound = F` retain their compatibility
field names but do not establish a normalised evidence bound for arbitrary
models or comparisons across sensory precisions. The abstract identity above
describes the probabilistic formalism, not an additional runtime guarantee.

### KL Divergence Between Gaussians

$$D_\text{KL}[\mathcal{N}(\mu_q, \Sigma_q) \| \mathcal{N}(\mu_p, \Sigma_p)] = \frac{1}{2}\left(\text{tr}(\Sigma_p^{-1}\Sigma_q) + (\mu_p - \mu_q)^\top \Sigma_p^{-1}(\mu_p - \mu_q) - n + \log\frac{|\Sigma_p|}{|\Sigma_q|}\right)$$

This is exact (not an approximation) for multivariate Gaussians. The
implementation uses Cholesky factors, triangular solves and the full covariance
log-determinants. It neither forms an explicit covariance inverse for the KL
calculation nor replaces a full covariance with its diagonal.

### Free Energy Gradient (Belief Dynamics)

For symmetric Gaussian precision weights, the gradient drives belief updates
via gradient descent. In the implementation, the prior is regularised by
`PRECISION_RIDGE`; the equations below use this effective precision:

$$\frac{d\mu}{dt} = -\frac{\partial F}{\partial \mu} = -\Pi\mu + J^\top \Gamma(x - g(\mu))$$

where $J = \partial g / \partial \mu$ is the Jacobian of the generative model.
With the identity generative model ($g(\mu) = \mu$, $J = I$):

$$\frac{\partial F}{\partial \mu} = \Pi\mu - \Gamma(x - \mu)$$

At the MAP point: $\mu_\text{MAP} = (\Pi + \Gamma)^{-1}\Gamma x$. For
$\Pi = \Gamma = I$: $\mu_\text{MAP} = x/2$.

### SCPN Mapping

The FEP maps onto the SCPN hierarchy as follows:

| FEP Concept | SCPN Entity | Module |
|-------------|-------------|--------|
| Belief mean $\mu$ | Oscillator phases $\theta_i$ | `phase/phase_vqe.py` |
| Prior precision $\Pi$ | K_nm coupling matrix, when positive definite | `bridge/knm_hamiltonian.py` |
| Sensory precision $\Gamma$ | Measurement confidence | Identity (simulation) |
| Generative model $g$ | Forward prediction | Coupling-weighted mean |
| Prediction error $\varepsilon$ | Phase mismatch | Hierarchical PC |
| Free energy $F$ | Total prediction cost | `variational_free_energy()` |

### ELBO Interpretation

For a fully specified normalised probabilistic model, the formal evidence lower
bound is the negative variational free energy. The existing result fields
`elbo` and `surprise_bound` instead contain the negative and positive reduced
objective defined above. Their names do not qualify model evidence, a posterior
approximation or a physical oscillator interpretation. For fixed model, sensory
weights and covariance, the mean-gradient objective remains explicitly evaluable.

### MAP Point Derivation

Setting $\partial F / \partial \mu = 0$ with identity generative model:

$$\Pi\mu - \Gamma(x - \mu) = 0$$
$$(\Pi + \Gamma)\mu = \Gamma x$$
$$\mu_\text{MAP} = (\Pi + \Gamma)^{-1}\Gamma x$$

For $\Pi = K_\text{nm}$ and $\Gamma = I$:

$$\mu_\text{MAP} = (K_\text{nm} + I)^{-1} x$$

This is a regularised least-squares solution: the K_nm prior pulls the
beliefs toward zero (prior mean), while observations pull toward $x$.
Stronger coupling (larger K_nm eigenvalues) means stronger regularisation
— beliefs deviate less from the prior.

### Predictive Coding Algorithm

```
Algorithm: Hierarchical Predictive Coding on SCPN

Input: observations x, beliefs μ, coupling K, prior precision Π, lr, n_steps
Output: proposed beliefs μ, reduced objective trajectory F[t]

1. For t = 1 to n_steps:
   a. For each layer i:
      Π_i ← Σ_j K[i,j]                    (local precision)
      x̂_i ← Σ_j K[i,j] μ_j / Π_i         (coupling-weighted prediction)
      ε_i ← Π_i (x_i − x̂_i)              (precision-weighted error)

   b. Compute gradient:
      ∂F/∂μ = Π_reg μ − Γ(x − μ)          (Rust-accelerated)

   c. Update beliefs:
      μ ← μ − lr × ∂F/∂μ

   d. Record F[t] = complexity(μ) + accuracy(μ, x)

2. Return μ, F
```

The returned update is a gradient step on the declared reduced objective.
A convex objective alone does not guarantee convergence for an arbitrary fixed
learning rate. The API does not certify convergence or control stability, and
it returns a new belief array rather than applying a physical action.

### Hierarchical Predictive Coding

For each SCPN layer $i$, the prediction from connected layers is:

$$\hat{x}_i = \frac{\sum_j K_{ij} \mu_j}{\sum_j K_{ij}}$$

The precision-weighted prediction error is:

$$\varepsilon_i = \Pi_i (x_i - \hat{x}_i), \quad \Pi_i = \sum_j K_{ij}$$

Layers with stronger K_nm coupling (higher precision) produce larger
prediction errors — they are more "surprised" by mismatches. This maps
to the predictive coding hierarchy (Friston 2005, Bastos et al. 2012)
where higher layers predict lower layers' activity.

## 2. Theoretical Context

### Why FEP in the SCPN?

This module supplies Gaussian belief objectives alongside the oscillator models.
Interpreting phases as beliefs and choosing a coupling-derived prior precision
are modelling decisions. The numerical APIs qualify the specified Gaussian and
quadratic contracts; they do not establish an identity between oscillator
synchronisation, the UPDE and probabilistic inference.

### Relationship to Kuramoto Models

The implemented identity-model belief gradient is a linear quadratic expression
in the belief mean. The Kuramoto solvers have their own phase, frequency and
coupling equations. No general nonlinear or linearised Kuramoto/FEP equivalence
is established by the functions or conformance tests on this page. In
particular, a coupling matrix is not automatically a valid Gaussian precision;
choose a separate positive-definite prior when required.

### Precision-Weighted Message Passing

In biological predictive coding (Bastos et al. 2012), cortical layers
communicate via two message types:

| Message | Direction | Content | SCPN Equivalent |
|---------|-----------|---------|-----------------|
| Top-down prediction | Higher → Lower | $\hat{x}_i = g(\mu_{i+1})$ | K_nm-weighted mean |
| Bottom-up error | Lower → Higher | $\varepsilon_i = \Pi_i(x_i - \hat{x}_i)$ | Precision-weighted mismatch |

The precision $\Pi_i = \sum_j K_{ij}$ determines how much weight each
layer's error signal carries. Layers with strong coupling (high precision)
dominate the belief update — their errors are treated as more informative.

This maps to cortical anatomy where layer-specific precision corresponds
to neuromodulatory gain (dopamine, acetylcholine). In the SCPN, the
analogous modulator is the K_nm coupling strength.

### Connection to Active Inference

Active inference extends the FEP to include action: the system not only
updates beliefs but also acts on the world to reduce prediction errors.
In the SCPN context, this maps to:
- **Perception:** updating phase beliefs from quantum measurements
- **Action:** modulating coupling strengths (DynamicCouplingEngine)

This module implements the perception side. The action side is handled
by `qsnn/dynamic_coupling.py` (Hebbian learning from XY correlations).

### What This Module Does NOT Claim

- It does NOT claim that the brain literally performs quantum FEP
- It does NOT derive the FEP from quantum mechanics
- The mapping SCPN → FEP is a formal analogy, not a proof of identity
- The identity generative model ($g(\mu) = \mu$) is a simplification —
  the real SCPN generative model would be the full UPDE

### Relationship to Other SCPN Modules

| Module | FEP Role | Connection |
|--------|----------|------------|
| `phase/phase_vqe.py` | Provides oscillator phases (beliefs) | VQE output → FEP input |
| `bridge/knm_hamiltonian.py` | Provides prior precision | K_nm → Π matrix |
| `qsnn/dynamic_coupling.py` | Active inference (action) | Hebbian update ← XY correlations |
| `analysis/sync_witness.py` | Measures synchronisation | Sync order ↔ low free energy |
| `qec/multiscale_qec.py` | Error correction | QEC protects belief states |

These links describe possible module compositions. Local objective agreement
does not qualify biological inference, physical actuation or a shared
probabilistic interpretation of every linked solver.

## 3. Pipeline Position

```
bridge/knm_hamiltonian.py → K_nm (prior precision Π)
        ↓
fep/variational_free_energy.py
  ├─ _complexity_term() → KL[q || prior]
  ├─ _accuracy_term() → prediction error energy
  ├─ variational_free_energy() → FreeEnergyResult
  ├─ free_energy_gradient() → ∂F/∂μ (Rust-accelerated)
  └─ evidence_lower_bound() → ELBO = −F
        ↓
fep/predictive_coding.py
  ├─ hierarchical_prediction_error() → ε_i (Rust-accelerated)
  └─ predictive_coding_step() → PredictiveCodingResult
        ↓
phase/phase_vqe.py → updated beliefs → circuit parameters
```

**Inputs:** oscillator phases (beliefs), observed phases, K_nm coupling.

**Outputs:** free energy decomposition, gradient, updated beliefs.

## 4. Features

- **Belief objective:** full Gaussian prior KL plus quadratic prediction error
- **KL divergence:** multivariate Gaussian KL via Cholesky solves and log-determinants
- **Compatibility fields:** `elbo = -F` and `surprise_bound = F` for the reduced objective
- **Free energy gradient:** analytical $\partial F / \partial \mu$ for gradient descent belief update
- **Hierarchical predictive coding:** coupling-weighted predictions across SCPN layers
- **Predictive coding step:** single gradient descent update with error tracking
- **Rust acceleration:** identity-model gradient and prediction error use the optional native engine
- **Custom generative models:** supports arbitrary $g(\mu)$ and Jacobian $J(\mu)$ with Python fallback
- **Convergence tracking:** prediction error norm monitored across PC iterations

## 5. Usage Examples

### Compute Free Energy

```python
import numpy as np
from scpn_quantum_control.bridge.knm_hamiltonian import build_knm_paper27
from scpn_quantum_control.fep.variational_free_energy import variational_free_energy

K = build_knm_paper27()
n = K.shape[0]  # 16

mu = np.zeros(n)  # initial belief: all phases zero
sigma = 0.1 * np.eye(n)  # belief covariance
prior_precision = np.eye(n)  # explicit valid prior; separate from coupling K
x = np.random.default_rng(42).standard_normal(n) * 0.1  # observations

result = variational_free_energy(mu, sigma, x, prior_precision)
print(f"F = {result.free_energy:.4f}")
print(f"  complexity = {result.complexity:.4f}")
print(f"  accuracy = {result.accuracy:.4f}")
print(f"  ELBO = {result.elbo:.4f}")
```

### Predictive Coding Convergence

```python
from scpn_quantum_control.fep.predictive_coding import predictive_coding_step

K = build_knm_paper27(L=4)
x = np.array([0.5, 0.3, -0.2, 0.1])
beliefs = np.zeros(4)
prior_precision = np.eye(4)

for step in range(100):
    result = predictive_coding_step(
        x, beliefs, K, learning_rate=0.001, prior_precision=prior_precision
    )
    beliefs = result.beliefs
    if step % 20 == 0:
        print(f"Step {step}: F={result.free_energy:.4f}, "
              f"||ε||={result.total_error_norm:.4f}")
```

### Custom Generative Model

```python
from scpn_quantum_control.fep.variational_free_energy import free_energy_gradient

# Nonlinear generative model: g(μ) = sin(μ)
def g(mu):
    return np.sin(mu)

def J(mu):
    return np.diag(np.cos(mu))

grad = free_energy_gradient(
    mu=np.zeros(4), sigma=np.eye(4),
    x_observed=np.ones(4) * 0.5, K_precision=np.eye(4),
    generative_fn=g, generative_jac=J,
)
```

### Gradient Descent on Free Energy

```python
from scpn_quantum_control.fep.variational_free_energy import (
    variational_free_energy, free_energy_gradient,
)

K = build_knm_paper27()
n = K.shape[0]
mu = np.zeros(n)
sigma = 0.1 * np.eye(n)
prior_precision = np.eye(n)
x = np.random.default_rng(42).standard_normal(n) * 0.1

# Manual gradient descent (equivalent to predictive_coding_step)
lr = 0.001
for step in range(200):
    grad = free_energy_gradient(mu, sigma, x, prior_precision)
    mu = mu - lr * grad

result = variational_free_energy(mu, sigma, x, prior_precision)
print(f"Final F = {result.free_energy:.6f}")
print(f"Final ||grad|| = {np.linalg.norm(grad):.6f}")
```

### KL Divergence Verification

```python
from scpn_quantum_control.fep.variational_free_energy import kl_divergence_gaussian

# KL[q || p] = 0 when q = p
mu = np.array([1.0, 2.0, 3.0])
sigma = np.eye(3) * 0.5
kl = kl_divergence_gaussian(mu, sigma, mu, sigma)
assert abs(kl) < 1e-10, "KL[q||q] must be 0"

# KL is always non-negative (Gibbs' inequality)
rng = np.random.default_rng(42)
for _ in range(100):
    mu_q = rng.standard_normal(3)
    mu_p = rng.standard_normal(3)
    A = rng.standard_normal((3, 3))
    sigma_q = A @ A.T + 0.1 * np.eye(3)
    B = rng.standard_normal((3, 3))
    sigma_p = B @ B.T + 0.1 * np.eye(3)
    assert kl_divergence_gaussian(mu_q, sigma_q, mu_p, sigma_p) >= -1e-10
```

Both covariances must be finite, symmetric and positive definite, and the two
means must be finite and share their dimension. Anything else is rejected with a
`ValueError` naming the offending argument, rather than producing a number that
cannot be a divergence — a negative-definite covariance has no log-determinant
of the required sign, and silently taking its magnitude yields a negative "KL".

The divergence is evaluated through Cholesky factors of both covariances: no
explicit inverse is formed, and `log|Σ|` is taken as `2 Σ log L_ii`, which needs
no sign correction. Accuracy therefore follows the conditioning of the
covariances; `KL[p || p]` returns a residual bounded by machine epsilon times
the condition number rather than exactly zero, and that residual is never
clamped. Where a precision matrix is inverted to form a prior covariance, the
ridge added first is the documented constant `PRECISION_RIDGE`, and the
regularised matrix must still be positive definite.

## 6. Technical Reference

### Classes

#### `FreeEnergyResult`

| Field | Type | Description |
|-------|------|-------------|
| `free_energy` | `float` | $F = \text{complexity} + \text{accuracy}$ |
| `complexity` | `float` | $D_\text{KL}[q \| \text{prior}]$ |
| `accuracy` | `float` | $\frac{1}{2}(x - g(\mu))^\top \Gamma (x - g(\mu))$ |
| `elbo` | `float` | $-F$ |
| `surprise_bound` | `float` | Compatibility alias for reduced $F$; no additional bound qualification |

#### `PredictiveCodingResult`

| Field | Type | Description |
|-------|------|-------------|
| `prediction_errors` | `ndarray` | $\varepsilon_i$ per layer |
| `beliefs` | `ndarray` | Updated $\mu_i$ |
| `free_energy` | `float` | $F$ after update |
| `total_error_norm` | `float` | $\|\varepsilon\|$ |

### Functions

#### `variational_free_energy(mu, sigma, x_observed, K_precision, sensory_precision, generative_fn)`

Compute $F$ with full decomposition. Returns `FreeEnergyResult`.

#### `kl_divergence_gaussian(mu_q, sigma_q, mu_p, sigma_p)`

Exact KL divergence between multivariate Gaussians.

#### `free_energy_gradient(mu, sigma, x_observed, K_precision, sensory_precision, generative_fn, generative_jac)`

Analytical gradient $\partial F / \partial \mu$. Rust-accelerated for identity
generative model. A custom model and its Jacobian are one contract and must be
supplied together; see below.

#### `hierarchical_prediction_error(observations, beliefs, K)`

Precision-weighted prediction errors across SCPN layers.
Rust-accelerated.

#### `predictive_coding_step(observations, beliefs, K, learning_rate, sigma, prior_precision)`

Single gradient descent step. Returns `PredictiveCodingResult`.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `observations` | `ndarray` | required | Measured phases $x_i$ |
| `beliefs` | `ndarray` | required | Current beliefs $\mu_i$ |
| `K` | `ndarray` | required | K_nm coupling, used for the prediction errors |
| `learning_rate` | `float` | `0.01` | Gradient step size |
| `sigma` | `ndarray \| None` | `0.1 × I` | Belief covariance |
| `prior_precision` | `ndarray \| None` | `K` | Prior precision $\Pi$, positive definite |

The coupling matrix and the prior precision are separate objects. A K_nm
matrix with a zero diagonal is a valid coupling — a layer is not coupled to
itself — but it is not a positive-definite Gaussian precision. An algebraic
inverse, if one exists, need not be a valid covariance. Passing `K` for both roles is correct only when `K` is
positive definite; otherwise supply `prior_precision`, for example the graph
Laplacian of `K` with a positive shift, which is what the bundled FEP
application plugin uses.

#### `variational_free_energy` — Full Signature

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mu` | `ndarray` | required | Belief mean |
| `sigma` | `ndarray` | required | Belief covariance ($n × n$) |
| `x_observed` | `ndarray` | required | Observed data |
| `K_precision` | `ndarray` | required | Prior precision (K_nm) |
| `sensory_precision` | `ndarray \| None` | `I` | Likelihood precision $\Gamma$ |
| `generative_fn` | `Callable \| None` | identity | Forward model $g(\mu)$, returning a finite length-$m$ vector |

#### `free_energy_gradient` — Full Signature

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mu` | `ndarray` | required | Belief mean |
| `sigma` | `ndarray` | required | Belief covariance |
| `x_observed` | `ndarray` | required | Observed data |
| `K_precision` | `ndarray` | required | Prior precision |
| `sensory_precision` | `ndarray \| None` | `I` | Likelihood precision |
| `generative_fn` | `Callable \| None` | identity | Forward model; requires `generative_jac` |
| `generative_jac` | `Callable \| None` | — | Jacobian $\partial g / \partial \mu$; requires `generative_fn` |

When both model arguments are `None`, the wrapper uses the Rust engine when
available and otherwise evaluates the Python identity-model gradient. Shape and
finiteness checks precede dispatch; the direct native exports additionally
require a nonempty belief vector.

A model and its Jacobian are supplied together or not at all. Neither is
inferred from the other: substituting the identity for an absent Jacobian
returns the gradient of a different model — for $g(\mu) = 2\mu$ at $\mu = 0.4$
with $x = 0$ and $K = \Gamma = I$ it returns $1.2$ where the derivative of $F$
is $2.0$ — and a Jacobian without its model differentiates the identity
prediction. Either half alone raises `ValueError` before the callables run, so
no belief state is updated from an incomplete contract.

Models need not be square. For $g: \mathbb{R}^n \to \mathbb{R}^m$ the Jacobian
is $(m, n)$, `x_observed` and the model output have length $m$, the sensory
precision is $(m, m)$, and the gradient has length $n$. The model must return a
finite one-dimensional vector of length $m$: a scalar or an $(m, 1)$ column is
rejected rather than broadcast against the observation.

`sigma` is accepted for symmetry with `variational_free_energy` and does not
enter the gradient; the terms of $F$ carrying the belief covariance are constant
in $\mu$.

### Rust Engine API

Available via `import scpn_quantum_engine`:

#### `free_energy_gradient_rust(mu, x_observed, k_precision, sensory_precision, ridge)`

Computes $\partial F / \partial \mu$ for identity generative model.
The four vector/matrix parameters are NumPy float64 arrays; `ridge` is a
finite scalar. Returns `ndarray[f64]`.

The dimension $n$ comes from `mu`. `x_observed` must have length $n$, and both
matrices must be exactly $n \times n$; all four arguments and `ridge` must be
finite. Violations raise `ValueError` naming the argument. Non-contiguous views
are read through their strides and are not rejected.

Symmetry is not required by the quadratic-gradient interface: it uses the
symmetric parts of both prior and sensory weights, matching the derivative of
the scalar quadratic forms. General weights are not thereby Gaussian
precisions. The energy export separately requires a valid regularised prior.

#### `hierarchical_prediction_error_rust(observations, beliefs, k)`

Computes precision-weighted prediction errors. Returns `ndarray[f64]`.

#### `variational_free_energy_rust(mu, x_observed, k_precision, sensory_precision, sigma_diag, ridge)`

Computes $(F, \text{complexity}, \text{accuracy})$ for isotropic
$\Sigma = \text{sigma_diag} I$. The scalar argument does not represent an
arbitrary diagonal or full covariance; the Python objective retains the
broader full-covariance interface.

Same length, shape and finiteness contract as the gradient export, plus
`sigma_diag` positive and finite. `k_precision` must additionally be symmetric
within $10^{-10}$ times the largest absolute entry of the regularised matrix,
matching the scale-aware Python covariance tolerance, and positive definite
once the ridge is added. The
symmetry check exists because the log-determinant is taken from a Cholesky
factor, which reads only the lower triangle: without it, an asymmetric matrix
returns the determinant of its symmetrised triangle instead of an error.

Both exports validate the whole contract before any indexing. Before this, an
undersized argument reached an ndarray bounds panic that crosses PyO3 as
`pyo3_runtime.PanicException`, a `BaseException` that ordinary Python error
handling does not catch, and a merely mis-shaped argument — a $2 \times 3$ read
as $2 \times 2$ — produced a silently wrong number. The contract is exercised
against a freshly built extension by the `fep-ffi-contract-quality` job in
`.github/workflows/ci-native-integration.yml`.

### Internal Functions

| Function | Module | Description |
|----------|--------|-------------|
| `_complexity_term(mu, sigma, K)` | `variational_free_energy.py` | KL divergence to prior |
| `_accuracy_term(mu, x, Γ, g)` | `variational_free_energy.py` | Prediction error energy |

### Numerical Considerations

- **Ridge regularisation:** the prior receives the declared $10^{-10} I$
  shift and must still be positive definite. The shift does not repair an
  arbitrary indefinite coupling matrix into a valid prior.
- **Log-determinant:** taken from a Cholesky factor as $2 \sum_i \log L_{ii}$,
  which needs no separate sign and cannot silently return the magnitude of a
  negative determinant.
- **Gradient at MAP:** residual $\|\nabla F\| \sim 10^{-10}$ due to ridge
  regularisation. This is not a bug — it is the price of numerical stability.
- **Convergence criterion:** PC step uses fixed learning rate. Adaptive
  step sizes (e.g., line search) would improve convergence but add
  complexity. The current implementation prioritises simplicity.
- **Custom generative models:** when `generative_fn` is not None, Rust
  engine is bypassed. The Python path supports arbitrary callables but
  has no current cross-route speed guarantee. The gradient requires the
  paired `generative_jac`; it is not approximated, because a numerical
  derivative would change the accuracy of the result without saying so.
  A caller who wants one builds it explicitly and passes it as the pair:
  `generative_jac=lambda mu: jacobian(g, mu)` with
  `from scpn_quantum_control.diff import jacobian`, whose return shape
  $(m, n)$ is already the one this contract requires. Against the analytic
  Jacobian of $g = \sin$ on three coordinates the two agree to $4×10^{-11}$.

## 7. Performance Benchmarks

Measured on Intel i5-11600K, Python 3.12, $n = 16$ oscillators.

| Function | Time | Engine |
|----------|------|--------|
| `variational_free_energy` | 88 μs | Python (numpy linalg) |
| `free_energy_gradient` | 110 μs | Rust |
| `hierarchical_prediction_error` | 40 μs | Rust |
| `predictive_coding_step` | 260 μs | Rust (inner) |
| `kl_divergence_gaussian` | ~50 μs | Python (numpy linalg) |

These figures predate the argument validation that `free_energy_gradient` and
`variational_free_energy` now perform before dispatch, and the Cholesky
factorisation that replaced the explicit inverse in `kl_divergence_gaussian`.
They have not been re-measured on an isolated host, so they are retained as the
last qualified measurement rather than refreshed with a loaded-workstation
number. The validation is deliberate cost: it is what keeps an inadmissible
argument from reaching either the Python or the native tier.

### Convergence Rate

The example above uses an explicit identity prior separately from its coupling
matrix and a fixed small learning rate. Its finite run illustrates belief
updates; it is not a general convergence or control-stability certificate.

### Scaling with System Size

| $n$ | `variational_free_energy` | `free_energy_gradient` | `predictive_coding_step` |
|-----|--------------------------|----------------------|--------------------------|
| 4 | ~20 μs | ~30 μs | ~60 μs |
| 8 | ~40 μs | ~50 μs | ~110 μs |
| 16 | ~88 μs | ~110 μs | ~260 μs |

Scaling is $O(n^2)$ for gradient (matrix-vector) and $O(n^3)$ for
`variational_free_energy` (Cholesky factorisation and solves).

### Test Coverage

The following owners cover this surface:

| Test file | Scope |
|-----------|-------|
| `tests/test_fep.py` | Free energy, KL contract, predictive coding, performance budgets |
| `tests/test_fep_generative_model_contract.py` | Generative model and Jacobian contract for the gradient |
| `tests/test_control_objective_conformance.py` | Full covariance, rectangular Jacobian, unchanged beliefs on refusal and actual native boundary comparisons |

`fep/predictive_coding.py` carries an exact 100% statement and branch coverage
gate, measured from `tests/test_fep.py` alone; see
`tools/predictive_coding_quality_gates.py`. `fep/variational_free_energy.py` is
guarded at 100% statement and branch coverage by the native FEP cohort, which
includes these conformance tests. See the
[control objective contract](differentiable_api.md#control-costs-and-gaussian-belief-objectives)
for the matching signed MPC comparison and proposal boundary.

## 8. Citations

1. Friston, K. "The free-energy principle: a unified brain theory?"
   *Nature Reviews Neuroscience* **11**, 127–138 (2010).
   DOI: 10.1038/nrn2787

2. Friston, K. "Life as we know it."
   *J. R. Soc. Interface* **10**, 20130475 (2013).
   DOI: 10.1098/rsif.2013.0475

3. Friston, K. "A theory of cortical responses."
   *Phil. Trans. R. Soc. B* **360**, 1249–1266 (2005).
   DOI: 10.1098/rstb.2005.1622

4. Bastos, A. M. et al. "Canonical microcircuits for predictive coding."
   *Neuron* **76**, 695–711 (2012).
   DOI: 10.1016/j.neuron.2012.10.038

5. Bogacz, R. "A tutorial on the free-energy framework for modelling
   perception and learning."
   *J. Math. Psych.* **76**, 198–211 (2017).
   DOI: 10.1016/j.jmp.2015.11.003

6. Buckley, C. L. et al. "The free energy principle for action and
   perception: a mathematical review."
   *Entropy* **19**, 318 (2017).
   DOI: 10.3390/e19070318

7. Fields, C. & Glazebrook, J. F. "A mosaic of Chu spaces and Channel
   Theory I: Category-theoretic concepts and tools."
   *J. Exp. Theor. Artif. Intell.* **34**, 1–38 (2022).
   DOI: 10.1080/0952813X.2019.1689909

8. Šotek, M. "God of the Math — The SCPN Master Publications."
   Zenodo (2025). DOI: 10.5281/zenodo.17419678
