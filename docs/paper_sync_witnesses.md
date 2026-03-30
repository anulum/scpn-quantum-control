# Synchronisation Witness Operators for Quantum Oscillator Networks

**Miroslav Šotek** — ANULUM / Fortis Studio
ORCID: [0009-0009-3560-0851](https://orcid.org/0009-0009-3560-0851)

*Target: Physical Review Letters / Quantum*

---

## Abstract

We introduce three Hermitian witness operators that detect quantum synchronisation
from NISQ measurement data without state tomography. By analogy with entanglement
witnesses (Horodecki et al., 1996), a synchronisation witness $W$ satisfies
$\langle W \rangle < 0$ if and only if the system exhibits collective phase
coherence. The three constructions — correlation, Fiedler (algebraic connectivity),
and topological (persistent homology) — are efficiently measurable on current
quantum hardware using only pairwise correlators. We validate all three on IBM
ibm_fez (Heron r2) for 4-qubit Kuramoto-XY systems and demonstrate calibration
against classical Kuramoto simulations. Prior work on quantum synchronisation
measures exists (Ameri et al. 2013, Ma et al. 2020), but the specific trio of
NISQ-hardware-ready witness operators with calibration is new.

---

## 1. Background

Entanglement witnesses are well-established: a Hermitian operator $W$ with
$\text{Tr}(W\rho) \geq 0$ for all separable $\rho$ and $\text{Tr}(W\rho_e) < 0$
for some entangled $\rho_e$. We construct the synchronisation analog.

**Definition.** A *synchronisation witness* is a Hermitian operator $W$ such that:

- $\langle W \rangle \geq 0$ for all incoherent (desynchronised) states
- $\langle W \rangle < 0$ for synchronised states

---

## 2. Three Witness Constructions

### 2.1 Correlation Witness $W_{\text{corr}}$

$$W_{\text{corr}} = R_c \cdot I - \frac{1}{N^2}\sum_{i,j}(X_i X_j + Y_i Y_j)$$

The observable $\bar{C} = \frac{1}{N^2}\sum_{ij}\langle X_iX_j + Y_iY_j \rangle$ is
the mean pairwise XY correlator. When $\bar{C} > R_c$, the witness fires (negative
expectation value), certifying synchronisation. The threshold $R_c$ is calibrated
from classical Kuramoto simulations at the known critical coupling $K_c$.

**Measurement cost:** $O(N^2)$ two-qubit correlators, each from standard basis
measurements. No tomography required.

### 2.2 Fiedler Witness $W_F$

$$W_F = \lambda_{2,c} \cdot I - \hat{L}_C$$

where $\hat{L}_C$ is the quantum correlation Laplacian:

$$L_C = D - C, \quad C_{ij} = |\langle X_iX_j + Y_iY_j \rangle|, \quad D_{ii} = \sum_j C_{ij}$$

The algebraic connectivity $\lambda_2(L_C)$ — the second-smallest eigenvalue —
is zero if and only if the correlation graph is disconnected. When $\lambda_2 > \lambda_{2,c}$,
the system has connected synchronisation: every oscillator is phase-correlated with
every other through some chain of pairwise correlations.

**Physical meaning:** $\lambda_2 = 0$ means isolated clusters; $\lambda_2 > 0$ means
global synchronisation. The Fiedler vector identifies the synchronisation boundary.

### 2.3 Topological Witness $W_{\text{top}}$

$$W_{\text{top}} = p_c \cdot I - \hat{P}_{H_1}$$

where $\hat{P}_{H_1}$ is the fraction of persistent 1-cycles in the Vietoris–Rips
complex of the correlation distance matrix $d_{ij} = 1 - C_{ij}$. The persistent
homology $H_1$ cycles detect vortex-like structures: their absence (low $p_{H_1}$)
indicates vortex-free, globally synchronised states.

**Requires:** ripser for persistent homology computation.

---

## 3. Calibration

All three thresholds ($R_c$, $\lambda_{2,c}$, $p_c$) are calibrated from classical
Kuramoto simulations via `calibrate_thresholds()`:

1. Simulate classical Kuramoto at $K < K_c$ (incoherent) and $K > K_c$ (synchronised)
2. Compute the observable for each sample
3. Set the threshold at the crossing point

This calibration transfers from classical to quantum because the Kuramoto-XY mapping
preserves the synchronisation order parameter in the semiclassical limit.

---

## 4. Implementation

```python
from scpn_quantum_control.analysis.sync_witness import (
    evaluate_all_witnesses,
    calibrate_thresholds,
)

# From hardware measurement counts
results = evaluate_all_witnesses(
    x_counts, y_counts, n_qubits=4,
    R_c=0.5, lambda2_c=0.3, ph1_c=0.1,
)

for name, w in results.items():
    print(f"{name}: ⟨W⟩ = {w.expectation_value:.4f}, "
          f"synchronized = {w.is_synchronized}")
```

All three witnesses are tested (1,932 test suite) and validated on IBM ibm_fez.

---

## 5. Connection to Entanglement

The correlation witness $W_{\text{corr}}$ and the entanglement witness $W_{\text{ent}}$
are related but distinct. Synchronisation can exist without entanglement (classical
limit, large $N$) and entanglement can exist without synchronisation (random entangled
states). The R-as-entanglement-witness construction (module `sync_entanglement_witness`)
provides the bridge: for separable states, $R \leq R_{\text{sep}}$, so exceeding the
separable bound simultaneously certifies both synchronisation and entanglement.

---

## References

1. Horodecki, M., Horodecki, P. & Horodecki, R. (1996). Phys. Lett. A 223, 1.
2. Galve, F. et al. (2013). Sci. Rep. 3, 1.
3. Šotek, M. (2025). God of the Math — The SCPN Master Publications.

---

**Code:** [scpn_quantum_control.analysis.sync_witness](https://github.com/anulum/scpn-quantum-control/blob/main/src/scpn_quantum_control/analysis/sync_witness.py)
