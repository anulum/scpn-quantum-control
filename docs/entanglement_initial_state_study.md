# Entangled initial-state coherence study

This is a bounded four-qubit simulation study of how four pure initial-state
families evolve under one frozen Kuramoto-XY Hamiltonian. It does **not** show
that entanglement lowers a synchronisation threshold. The model is closed and
unitary: it has no drive, dissipation, limit cycle, coupling scan, provider, or
hardware execution.

## Why the observable changed

The earlier implementation assigned each qubit a phase with
`atan2(<Y_i>, <X_i>)`. Bell, GHZ, and W states have vanishing local transverse
Bloch vectors in this model. `atan2(0, 0)` nevertheless returns zero, so the old
code assigned every site the same artificial phase and reported (R=1).

The corrected local complex amplitude is

\[
z_i = \langle X_i \rangle + i\langle Y_i \rangle .
\]

Local phase order is reported only when the total transverse visibility is
non-zero:

\[
V = \frac{1}{N}\sum_i |z_i|, \qquad
R_V = \frac{|\sum_i z_i|}{\sum_i |z_i|}.
\]

When \(\sum_i |z_i|\le 10^{-12}\), `phase_defined` is false and the
compatibility value `R` is stored as zero. A zero value in that case means
“unobservable local phase”, not antiphase locking.

The separate pair diagnostic is

\[
C_{XY} = \binom{N}{2}^{-1}\sum_{i<j}
2\left|\langle \sigma_i^+\sigma_j^-\rangle\right|,
\qquad 0\le C_{XY}\le 1.
\]

`C_XY` measures transverse exchange coherence. It is a custom finite-system
diagnostic, not a spontaneous-synchronisation certificate.

## Matched controls and attribution

Each pure state \(\rho\) is compared with its computational-basis-dephased
control

\[
\mathcal D(\rho)=\sum_x |x\rangle\langle x|\rho|x\rangle\langle x|.
\]

The pair has identical computational-basis populations and local
\(Z\)-marginals. It is not matched on Hamiltonian energy or every correlation,
so a difference identifies an off-diagonal-coherence contribution, not a
unique causal contribution from entanglement. The separable product state is
retained as an attribution control for precisely this reason.

Initial pure-state entanglement is described by the mean normalised one-qubit
linear entropy,

\[
\bar L = \frac{1}{N}\sum_i 2\left(1-\operatorname{Tr}\rho_i^2\right).
\]

This value is used only for the pure initial states. It is not applied as an
entanglement measure to the mixed dephased controls.

## Frozen evidence

The committed evidence uses the Paper-27 four-qubit coupling matrix,
`omega = [1.329, 2.61, 0.844, 1.52]`, \(t\in[0,2]\), and 20 exact time steps.
The table reports time-averaged \(C_{XY}\).

| Initial state | Initial \(\bar L\) | Pure state | Dephased control | Difference |
|---|---:|---:|---:|---:|
| Product | 0 | 0.419481647 | 0.154436520 | 0.265045127 |
| Bell pairs | 1 | 0.132115673 | 0.091096078 | 0.041019595 |
| GHZ | 1 | 0 | 0 | 0 |
| W | 0.75 | 0.366405312 | approximately 0 | 0.366405312 |

Bell-pair and W states differ from their dephased controls. GHZ is the
zero-difference negative control. The separable product state also differs from
its control, and by more than the Bell-pair row. The evidence therefore
classifies the result as an **initial-coherence observation that is not
entanglement-specific**.

The digest-bound JSON and Markdown records are in
`data/entanglement_sync_product/`. Reproduce them with:

```bash
python scripts/run_entanglement_sync_evidence.py
```

## API

```python
from scpn_quantum_control.analysis.entanglement_enhanced_sync import (
    compare_initial_states_with_dephased_controls,
)

comparisons = compare_initial_states_with_dephased_controls(
    K,
    omega,
    t_max=2.0,
    n_steps=20,
)
```

`simulate_sync_trajectory(...)` and `compare_all_initial_states(...)` expose
both observables and the phase-defined flags. The legacy
`entanglement_advantage(...)` name remains for compatibility, but returns only
a descriptive comparison with a governed no-advantage certificate. It no
longer reports a convergence speedup.

## Literature boundary

- Fiderer, Kuś, and Braun formulate qubit phase synchronisation relative to a
  fixed basis ([Phys. Rev. A 94, 032336](https://doi.org/10.1103/PhysRevA.94.032336)).
- Galve, Giorgi, and Zambrini review why quantum synchronisation measures are
  model-dependent and can disagree ([arXiv:1610.05060](https://arxiv.org/abs/1610.05060)).
- Roulet and Bruder connect phase locking and entanglement in a
  driven-dissipative spin-1 model
  ([Phys. Rev. Lett. 121, 063601](https://doi.org/10.1103/PhysRevLett.121.063601)).
  That result does not transfer automatically to this closed qubit model.

These sources motivate careful phase and correlation diagnostics; none
validates `C_XY` as a universal synchronisation measure or supports the former
lowered-critical-coupling claim.

## Claim boundary

The evidence is local deterministic simulation for one Hamiltonian and time
grid. It does not establish an entanglement-specific cause, lower critical
coupling, universal enhancement, quantum advantage, hardware fidelity, or
control authority.
