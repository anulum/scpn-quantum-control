# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Free Energy Principle

# Free Energy Principle (FEP) on Quantum Substrate

## 1. Mathematical Formalism

### Variational Free Energy

The Free Energy Principle (Friston 2010) defines variational free energy as:

$$F = \mathbb{E}_q[\log q(z) - \log p(z, x)] = D_\text{KL}[q(z) \| p(z|x)] - \log p(x)$$

Since $D_\text{KL} \geq 0$, free energy upper-bounds surprise: $F \geq -\log p(x)$.
Minimising $F$ thus minimises surprise (self-information of observations).

For Gaussian distributions:

$$q(z) = \mathcal{N}(\mu, \Sigma), \quad p(z) = \mathcal{N}(0, \Pi^{-1})$$

the free energy decomposes as:

$$F = \underbrace{\frac{1}{2}\left(\mu^\top \Pi \mu + \text{tr}(\Pi\Sigma) - \log|\Sigma| - \log|\Pi| - n\right)}_\text{complexity (KL to prior)} + \underbrace{\frac{1}{2}(x - g(\mu))^\top \Gamma (x - g(\mu))}_\text{accuracy (prediction error)}$$

where $\Pi$ is prior precision, $\Gamma$ is sensory precision, and $g(\mu)$
is the generative model.

### KL Divergence Between Gaussians

$$D_\text{KL}[\mathcal{N}(\mu_q, \Sigma_q) \| \mathcal{N}(\mu_p, \Sigma_p)] = \frac{1}{2}\left(\text{tr}(\Sigma_p^{-1}\Sigma_q) + (\mu_p - \mu_q)^\top \Sigma_p^{-1}(\mu_p - \mu_q) - n + \log\frac{|\Sigma_p|}{|\Sigma_q|}\right)$$

This is exact (not an approximation) for multivariate Gaussians. The
implementation uses `numpy.linalg.inv` and `numpy.linalg.slogdet` for
numerical stability.

### Free Energy Gradient (Belief Dynamics)

The gradient drives belief updates via gradient descent:

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
| Prior precision $\Pi$ | K_nm coupling matrix | `bridge/knm_hamiltonian.py` |
| Sensory precision $\Gamma$ | Measurement confidence | Identity (simulation) |
| Generative model $g$ | Forward prediction | Coupling-weighted mean |
| Prediction error $\varepsilon$ | Phase mismatch | Hierarchical PC |
| Free energy $F$ | Total prediction cost | `variational_free_energy()` |

### ELBO Interpretation

The Evidence Lower BOund (ELBO) is $-F$:

$$\text{ELBO} = -F = \log p(x) - D_\text{KL}[q(z) \| p(z|x)]$$

Maximising ELBO is equivalent to:
1. Maximising the model evidence $p(x)$ — the model explains the data well
2. Minimising the posterior gap $D_\text{KL}[q \| p(z|x)]$ — the approximate
   posterior $q$ is close to the true posterior

In the SCPN context: high ELBO means the oscillator phases (beliefs)
are well-matched to the observed phases, and the K_nm prior is consistent
with the data.

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

Input: observations x, initial beliefs μ, coupling K, learning_rate lr, n_steps
Output: converged beliefs μ*, free energy trajectory F[t]

1. For t = 1 to n_steps:
   a. For each layer i:
      Π_i ← Σ_j K[i,j]                    (local precision)
      x̂_i ← Σ_j K[i,j] μ_j / Π_i         (coupling-weighted prediction)
      ε_i ← Π_i (x_i − x̂_i)              (precision-weighted error)

   b. Compute gradient:
      ∂F/∂μ = K_reg μ − Γ(x − μ)          (Rust-accelerated)

   c. Update beliefs:
      μ ← μ − lr × ∂F/∂μ

   d. Record F[t] = complexity(μ) + accuracy(μ, x)

2. Return μ, F
```

The algorithm is equivalent to gradient descent on the variational
free energy surface. Convergence is guaranteed for convex F (which
holds when K_nm is positive semi-definite and the generative model
is the identity).

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

The SCPN's third axiom (Teleological Optimisation) posits that
consciousness optimises a quantity across the hierarchy. The FEP provides
the mathematical framework: the UPDE dynamics (Kuramoto coupling) can be
reinterpreted as belief updating under the FEP — oscillator phases are
sufficient statistics, and synchronisation is free energy minimisation.

This is not a metaphor. The Kuramoto model with heterogeneous frequencies
and distance-dependent coupling IS a predictive coding scheme where:
- Coupling K_nm = precision (how much to trust the neighbour)
- Phase difference = prediction error
- Synchronisation = free energy minimum

### Kuramoto-FEP Equivalence

The Kuramoto model with coupling $K_{ij}$ and natural frequencies $\omega_i$:

$$\frac{d\theta_i}{dt} = \omega_i + \sum_j K_{ij} \sin(\theta_j - \theta_i)$$

can be rewritten as gradient descent on an energy function:

$$E = -\sum_{i<j} K_{ij} \cos(\theta_i - \theta_j) - \sum_i \omega_i \theta_i$$

This is equivalent to the FEP gradient descent $d\mu/dt = -\partial F / \partial \mu$
where:

- $\mu = \theta$ (phases are beliefs)
- The prior energy is $\frac{1}{2}\mu^\top K \mu$ (quadratic approximation
  of the cosine coupling near synchronisation)
- The accuracy term encodes the observation that each oscillator's phase
  should match its natural frequency: $\frac{1}{2}(\omega - \dot{\theta})^\top \Gamma (\omega - \dot{\theta})$

The linearised Kuramoto around the synchronised state IS the FEP gradient
with K_nm as prior precision. The nonlinear $\sin$ terms provide additional
structure not captured by the Gaussian FEP, but near synchronisation (small
phase differences), the equivalence is exact.

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

The FEP provides the theoretical glue: synchronisation (low F) is the
system's way of minimising surprise about its environment.

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

- **Variational free energy:** exact Gaussian computation with complexity/accuracy decomposition
- **KL divergence:** exact multivariate Gaussian KL via matrix inversion and log-determinant
- **Evidence lower bound (ELBO):** $-F$ for variational inference optimisation
- **Free energy gradient:** analytical $\partial F / \partial \mu$ for gradient descent belief update
- **Hierarchical predictive coding:** coupling-weighted predictions across SCPN layers
- **Predictive coding step:** single gradient descent update with error tracking
- **Rust acceleration:** gradient and prediction error use Rust engine (110 μs and 40 μs per call)
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
x = np.random.default_rng(42).standard_normal(n) * 0.1  # observations

result = variational_free_energy(mu, sigma, x, K)
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

for step in range(100):
    result = predictive_coding_step(x, beliefs, K, learning_rate=0.001)
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
x = np.random.default_rng(42).standard_normal(n) * 0.1

# Manual gradient descent (equivalent to predictive_coding_step)
lr = 0.001
for step in range(200):
    grad = free_energy_gradient(mu, sigma, x, K)
    mu = mu - lr * grad

result = variational_free_energy(mu, sigma, x, K)
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
for _ in range(100):
    mu_q = np.random.randn(3)
    mu_p = np.random.randn(3)
    A = np.random.randn(3, 3)
    sigma_q = A @ A.T + 0.1 * np.eye(3)
    B = np.random.randn(3, 3)
    sigma_p = B @ B.T + 0.1 * np.eye(3)
    assert kl_divergence_gaussian(mu_q, sigma_q, mu_p, sigma_p) >= -1e-10
```

## 6. Technical Reference

### Classes

#### `FreeEnergyResult`

| Field | Type | Description |
|-------|------|-------------|
| `free_energy` | `float` | $F = \text{complexity} + \text{accuracy}$ |
| `complexity` | `float` | $D_\text{KL}[q \| \text{prior}]$ |
| `accuracy` | `float` | $\frac{1}{2}(x - g(\mu))^\top \Gamma (x - g(\mu))$ |
| `elbo` | `float` | $-F$ |
| `surprise_bound` | `float` | $F$ (upper bound on $-\log p(x)$) |

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
generative model.

#### `hierarchical_prediction_error(observations, beliefs, K)`

Precision-weighted prediction errors across SCPN layers.
Rust-accelerated.

#### `predictive_coding_step(observations, beliefs, K, learning_rate, sigma)`

Single gradient descent step. Returns `PredictiveCodingResult`.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `observations` | `ndarray` | required | Measured phases $x_i$ |
| `beliefs` | `ndarray` | required | Current beliefs $\mu_i$ |
| `K` | `ndarray` | required | K_nm coupling (prior precision) |
| `learning_rate` | `float` | `0.01` | Gradient step size |
| `sigma` | `ndarray \| None` | `0.1 × I` | Belief covariance |

#### `variational_free_energy` — Full Signature

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mu` | `ndarray` | required | Belief mean |
| `sigma` | `ndarray` | required | Belief covariance ($n × n$) |
| `x_observed` | `ndarray` | required | Observed data |
| `K_precision` | `ndarray` | required | Prior precision (K_nm) |
| `sensory_precision` | `ndarray \| None` | `I` | Likelihood precision $\Gamma$ |
| `generative_fn` | `Callable \| None` | identity | Forward model $g(\mu)$ |

#### `free_energy_gradient` — Full Signature

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mu` | `ndarray` | required | Belief mean |
| `sigma` | `ndarray` | required | Belief covariance |
| `x_observed` | `ndarray` | required | Observed data |
| `K_precision` | `ndarray` | required | Prior precision |
| `sensory_precision` | `ndarray \| None` | `I` | Likelihood precision |
| `generative_fn` | `Callable \| None` | identity | Forward model |
| `generative_jac` | `Callable \| None` | `I` | Jacobian $\partial g / \partial \mu$ |

When `generative_fn` and `generative_jac` are both `None`, uses Rust engine.

### Rust Engine API

Available via `import scpn_quantum_engine`:

#### `free_energy_gradient_rust(mu, x_observed, k_precision, sensory_precision, ridge)`

Computes $\partial F / \partial \mu$ for identity generative model.
All parameters are numpy arrays. Returns `ndarray[f64]`.

#### `hierarchical_prediction_error_rust(observations, beliefs, k)`

Computes precision-weighted prediction errors. Returns `ndarray[f64]`.

#### `variational_free_energy_rust(mu, x_observed, k_precision, sensory_precision, sigma_diag, ridge)`

Computes $(F, \text{complexity}, \text{accuracy})$ tuple for diagonal $\Sigma$.

### Internal Functions

| Function | Module | Description |
|----------|--------|-------------|
| `_complexity_term(mu, sigma, K)` | `variational_free_energy.py` | KL divergence to prior |
| `_accuracy_term(mu, x, Γ, g)` | `variational_free_energy.py` | Prediction error energy |

### Numerical Considerations

- **Ridge regularisation:** K_nm + $10^{-10} I$ ensures invertibility.
  Without this, zero eigenvalues in K_nm cause singular matrix errors.
- **Log-determinant:** uses `numpy.linalg.slogdet` for numerical stability
  (avoids overflow from large determinants).
- **Gradient at MAP:** residual $\|\nabla F\| \sim 10^{-10}$ due to ridge
  regularisation. This is not a bug — it is the price of numerical stability.
- **Convergence criterion:** PC step uses fixed learning rate. Adaptive
  step sizes (e.g., line search) would improve convergence but add
  complexity. The current implementation prioritises simplicity.
- **Custom generative models:** when `generative_fn` is not None, Rust
  engine is bypassed. The Python path supports arbitrary callables but
  is ~10× slower for the gradient computation.

## 7. Performance Benchmarks

Measured on Intel i5-11600K, Python 3.12, $n = 16$ oscillators.

| Function | Time | Engine |
|----------|------|--------|
| `variational_free_energy` | 88 μs | Python (numpy linalg) |
| `free_energy_gradient` | 110 μs | Rust |
| `hierarchical_prediction_error` | 40 μs | Rust |
| `predictive_coding_step` | 260 μs | Rust (inner) |
| `kl_divergence_gaussian` | ~50 μs | Python (numpy linalg) |

### Convergence Rate

For $n = 4$, $\text{lr} = 0.001$, K = `build_knm_paper27(L=4)`, starting from $\mu = 0$:
- 50 steps: error norm decreases (verified in test)
- 200 steps: belief change $< 0.01$ per step (convergence verified)

### Scaling with System Size

| $n$ | `variational_free_energy` | `free_energy_gradient` | `predictive_coding_step` |
|-----|--------------------------|----------------------|--------------------------|
| 4 | ~20 μs | ~30 μs | ~60 μs |
| 8 | ~40 μs | ~50 μs | ~110 μs |
| 16 | ~88 μs | ~110 μs | ~260 μs |

Scaling is $O(n^2)$ for gradient (matrix-vector) and $O(n^3)$ for
`variational_free_energy` (matrix inversion in KL computation).

### Test Coverage

16 tests across 6 dimensions:
- Empty/null: 3 tests (identical distributions, zero observation, perfect prediction)
- Error handling: 2 tests (singular covariance, zero precision)
- Negative cases: 2 tests (KL ≥ 0 property, gradient direction)
- Pipeline integration: 4 tests (SCPN K_nm, PC convergence, ELBO consistency, imports)
- Roundtrip: 3 tests (F decomposition, MAP gradient, PC equilibrium)
- Performance: 2 tests (F computation, PC step budgets)

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
