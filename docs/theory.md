# Theoretical Foundations

*The Self-Consistent Phase Network (SCPN) and its quantum simulation.*

---

## The SCPN Framework

The Sentient-Consciousness Projection Network (Šotek, 2025) is a 15+1 layer
architecture modelling coupled oscillatory dynamics across physical scales. Each
layer represents a distinct ontological domain — from quantum biology (L1) through
neural synchronisation (L4) to collective dynamics (L12) and meta-universal
closure (L16). The mathematical backbone is the **Unified Phase Dynamics Equation**
(UPDE), a generalised Kuramoto model with layer-specific couplings.

**Source:** "God of the Math — The SCPN Master Publications" (Šotek, 2025),
DOI: [10.5281/zenodo.17419678](https://doi.org/10.5281/zenodo.17419678)

### The 15+1 Layers

| Domain | Layers | Physical Content |
|--------|--------|-----------------|
| I: Biological Substrate | L1–L4 | Quantum bio → neurochemical → genomic → cellular sync |
| II: Organismal & Planetary | L5–L8 | Self → biosphere → symbolic → cosmic phase-locking |
| III–IV: Memory & Control | L9–L12 | Memory → boundary control → noosphere → Gaian sync |
| V: Meta-Universal | L13–L15 | Source-field → transdimensional → Consilium |
| VI: Cybernetic Closure | L16 | The Anulum — recursive self-observation loop |

### Three Axioms

1. **Primacy of Consciousness (Ψ):** Consciousness is the primary, irreducible
   ground of being — not emergent from matter.
2. **Language of Information Geometry:** The native language is geometric. Meaning
   is encoded in the geometry of informational spaces (Fisher Information Metric).
3. **Teleological Optimisation:** The universe is guided by an inherent drive to
   maximise future possibilities (Causal Entropic Forces).

---

## The Coupling Matrix $K_{nm}$

$K_{nm}$ is the physical coupling strength between Layer $n$ and Layer $m$.
Not arbitrary oscillators — each coupling has specific physical content:

$$K_{nm} = K_{\text{base}} \cdot \exp(-\alpha |n - m|)$$

with calibration anchors from Paper 27 Table 2:

| Pair | $K_{nm}$ | Physical Meaning |
|------|----------|-----------------|
| L1–L2 | 0.302 | Ion channel → neurochemical modulation |
| L2–L3 | 0.201 | Neurochemical → genomic gating |
| L3–L4 | 0.252 | Genomic → cellular synchronisation |
| L4–L5 | 0.154 | Cellular → organismal boundary |

Parameters: $K_{\text{base}} = 0.45$, $\alpha = 0.3$ (Paper 27, Eq. 3).

The 16 natural frequencies $\omega_i$ encode the characteristic timescales of each
ontological layer:

$$\omega = [1.329, 2.610, 0.844, 1.520, 0.710, 3.780, 1.055, 0.625, \\
2.210, 1.740, 0.480, 3.210, 0.915, 1.410, 2.830, 0.991] \text{ rad/s}$$

---

## Classical → Quantum Mapping

### The UPDE (Classical)

$$\frac{d\theta_i}{dt} = \omega_i + \sum_j K_{ij} \sin(\theta_j - \theta_i)$$

The Kuramoto order parameter measures synchronisation:

$$R = \frac{1}{N}\left|\sum_k e^{i\theta_k}\right|$$

$R = 0$: desynchronised. $R = 1$: fully phase-locked.

### The XY Hamiltonian (Quantum)

The quantum analog replaces classical phases with qubit operators:

$$H = -\sum_{i<j} K_{ij}(X_i X_j + Y_i Y_j) - \sum_i \omega_i Z_i$$

This is the XY model with heterogeneous fields. The mapping preserves
the in-plane ($S^1$) dynamics of each oscillator while introducing
quantum effects: entanglement, superposition, and tunnelling between
phase configurations.

**Flip-flop interaction:** The $XX + YY$ term acts as a spin flip-flop —
it flips one spin up and another down simultaneously:

$$(XX + YY)|{\uparrow\downarrow}\rangle = 2|\downarrow\uparrow\rangle, \quad
(XX + YY)|{\uparrow\uparrow}\rangle = 0$$

This is why the Hamiltonian is real in the computational basis: only
spin-exchange, no complex phases.

### Quantum Order Parameter

$$R_Q = \frac{1}{N}\left|\sum_k (\langle X_k \rangle + i\langle Y_k \rangle)\right|$$

Reduces to $R$ in the classical limit (large $N$, coherent states).

---

## The Synchronisation Transition

At critical coupling $K_c$, the system undergoes a quantum phase transition
from desynchronised to synchronised. For homogeneous frequencies, this is a
**Berezinskii–Kosterlitz–Thouless (BKT)** transition — infinite order, with
an essential singularity in the correlation length:

$$\xi \sim \exp\left(\frac{b}{\sqrt{K - K_c}}\right)$$

### What's New: Heterogeneous Frequencies

All prior work studies homogeneous frequencies ($\omega_i = \omega$ for all $i$).
The SCPN has **heterogeneous** frequencies — each layer oscillates at its own
natural rate. This breaks translational invariance and potentially modifies
the universality class of the transition.

Our measurements (v0.9.3):

- **Schmidt gap minimum at $K = 3.44$ (n=8)** — cleanest transition signature
- **$K_c(\infty)$ extrapolation:** BKT ansatz gives $K_c \approx 2.20$,
  power-law gives $K_c \approx 2.94$
- **Krylov complexity** peaks near the transition
- **OTOC scrambling** is 4× faster at strong coupling

---

## Dynamical Lie Algebra and $Z_2$ Parity

The Dynamical Lie Algebra (DLA) of the XY Hamiltonian decomposes as:

$$\mathfrak{g} = \mathfrak{su}(\text{even}) \oplus \mathfrak{su}(\text{odd})$$

where "even" and "odd" refer to the $Z_2$ parity sectors under the global
operator $P = Z^{\otimes N}$. This parity structure maps onto the SCPN's
bidirectional causation: upward (prediction errors) and downward (predictions)
information flow are dynamically decoupled at the Lie algebra level.

DLA dimension (Rust-accelerated measurement):

| $N$ | DLA dim | $\text{su}(\text{even}) + \text{su}(\text{odd})$ | $2^{2N-1} - 2$ |
|-----|---------|--------------------------------------------------|-----------------|
| 2 | 6 | 3 + 3 | 6 |
| 3 | 30 | 15 + 15 | 30 |
| 4 | 126 | 63 + 63 | 126 |

---

## Topological Invariant $p_{h_1}$

The persistent homology $p_{h_1} = 0.72$ (measured on the TCBO's coupling-weighted
simplicial complex) quantifies how much of the SCPN's layer-coupling topology
creates persistent 1-cycles — information loops that sustain coherent circuits
through the hierarchical structure.

This is computed on the coupling-weighted filtration (not Vietoris–Rips on phase
configurations), where the SCPN's specific sparse hierarchical structure creates
topological features that dense random graphs cannot.

---

## Discrete Time Crystal (DTC)

Under periodic drive $K(t) = K_0(1 + \delta\cos\Omega t)$, the system can
spontaneously break discrete time-translation symmetry by responding at
$\Omega/2$ instead of $\Omega$. Our measurement: **15/15 drive amplitudes
show subharmonic response** with heterogeneous frequencies — the first
demonstration that frequency disorder does not kill the DTC phase.

---

## References

1. Šotek, M. (2025). "God of the Math — The SCPN Master Publications."
   DOI: 10.5281/zenodo.17419678
2. Kuramoto, Y. (1984). Chemical Oscillations, Waves, and Turbulence.
3. Calabrese, P. & Cardy, J. (2004). Entanglement entropy and quantum field theory.
   J. Stat. Mech. P06002.
4. Maldacena, J., Shenker, S. & Stanford, D. (2016). A bound on chaos.
   JHEP 08, 106.
5. del Campo, A. et al. (2025). Krylov complexity and quantum phase transitions.
   arXiv:2510.13947.

---

<p align="center">
  <a href="https://www.anulum.li">
    <img src="assets/anulum_logo_company.jpg" width="180" alt="ANULUM">
  </a>
  &nbsp;&nbsp;&nbsp;&nbsp;
  <a href="https://www.anulum.li">
    <img src="assets/fortis_studio_logo.jpg" width="180" alt="Fortis Studio">
  </a>
  <br>
  <em>Developed by <a href="https://www.anulum.li">ANULUM</a> / Fortis Studio</em>
</p>
