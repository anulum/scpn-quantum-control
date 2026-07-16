# Quantum Simulation of Coupled-Oscillator Synchronisation on a 156-Qubit Superconducting Processor

**Miroslav Šotek**
ANULUM / Fortis Studio, Marbach SG, Switzerland
[protoscience@anulum.li](mailto:protoscience@anulum.li) |
ORCID: [0009-0009-3560-0851](https://orcid.org/0009-0009-3560-0851)

*Preprint — March 2026*

??? note "Cite this work"
    ```bibtex
    @software{sotek2026scpnqc,
      author = {Šotek, Miroslav},
      title = {scpn-quantum-control: Quantum Simulation of Coupled-Oscillator
               Synchronisation on a 156-Qubit Superconducting Processor},
      year = {2026},
      version = {0.9.5},
      url = {https://github.com/anulum/scpn-quantum-control},
      doi = {10.5281/zenodo.18821929}
    }
    ```

---

## Abstract

This legacy preprint records artifact-backed legacy hardware evidence for
Kuramoto-XY synchronisation workflows with heterogeneous natural frequencies on
IBM's ibm_fez (Heron r2, 156 qubits). It is retained as historical experiment
context and is not promoted as broad hardware validation. Using a
Rust-accelerated simulation pipeline (dense Hamiltonian construction ~111×
faster than Qiskit for small systems, falling to parity for large dense
builds), we compute entanglement entropy, Krylov complexity, OTOC
scrambling, and Floquet discrete time crystal diagnostics across the
synchronisation transition for systems of 2–16 qubits. Legacy hardware snapshots
include CHSH Bell inequality violation ($S = 2.165$), QKD bit error rate of 5.5%
(below the BB84 threshold of 11%), and a 16-qubit descriptive hardware snapshot
with visible per-qubit structure. We extract the critical coupling
$K_c(\infty) \approx 2.2$ via BKT finite-size scaling and report
simulator-backed DTC frequency-disorder diagnostics without claiming a promoted
hardware DTC result. A hardware-anchored scaling analysis places the exact
Hilbert-space simulation crossover at n≈11.6, while broad quantum advantage
remains open because the Rust Kuramoto ODE baseline stays faster through the
measured n≤16 range. All code, data, and 17 figures are open-source
(AGPL-3.0) at
[github.com/anulum/scpn-quantum-control](https://github.com/anulum/scpn-quantum-control).

---

## 1. Introduction

The Kuramoto model describes $N$ coupled oscillators with natural frequencies
$\omega_i$ and coupling matrix $K_{ij}$:

$$\frac{d\theta_i}{dt} = \omega_i + \sum_j K_{ij}\sin(\theta_j - \theta_i)$$

At critical coupling $K_c$, the system undergoes a synchronisation phase
transition characterised by the order parameter $R = \frac{1}{N}|\sum_k e^{i\theta_k}|$
jumping from zero to a finite value.

The quantum analogue maps this to the XY spin Hamiltonian:

$$H = -\sum_{i<j} K_{ij}(X_i X_j + Y_i Y_j) - \sum_i \omega_i Z_i$$

This mapping is exact: the $XX + YY$ flip-flop interaction preserves the in-plane
($S^1$) dynamics while introducing entanglement, superposition, and quantum
tunnelling between phase configurations.

**Prior work** on quantum simulation of the XY model uses homogeneous frequencies
($\omega_i = \omega$ for all $i$). This preserves translational invariance and
the BKT universality class is well-characterised. Theoretical quantum Kuramoto
models with heterogeneity exist (Pikovsky, Ha et al.). This repository reports
one legacy superconducting-processor workflow and preserves it with explicit
artifact-bound claim limits.

We study the heterogeneous case using parameters from the SCPN framework
(Šotek, 2025): 16 natural frequencies and a nearest-neighbour coupling matrix
$K_{nm} = K_{\text{base}} \cdot \exp(-\alpha|n-m|)$ with calibration anchors
from Paper 27. This coupling matrix encodes the interaction structure of a
15+1 layer oscillator hierarchy spanning quantum-to-macroscopic scales.

---

## 2. Methods

### 2.1 Hamiltonian Construction

The XY Hamiltonian is constructed directly in the computational basis via
bitwise flip-flop operations, bypassing Qiskit's SparsePauliOp:

$$H_{k, k \oplus \text{mask}_{ij}} = -2K_{ij} \quad \text{when } b_i(k) \neq b_j(k)$$

$$H_{kk} = -\sum_i \omega_i (1 - 2b_i(k))$$

This Rust implementation (PyO3) is **111×** faster than Qiskit SparsePauliOp
at $L=4$ and **39×** at $L=8$, falling to parity by $L=12$ (measured with warm-up,
Table 1; artefact `data/native_speedup/`).

### 2.2 Analysis Pipeline

| Module | Method | Rust Speedup |
|--------|--------|-------------|
| OTOC | Eigendecomposition + rayon parallel | 264× (n=4) |
| Krylov | Complex Lanczos commutator loop | 27× (n=3) |
| Entanglement | numpy eigh + SVD | Hamiltonian: 158× |
| Order parameter | Batch bitwise Pauli expectations | 6.2× (n=4) |

**Table 1.** Measured Rust vs Python/Qiskit/scipy speedups.
Windows 11, Python 3.12, Rust release build.

### 2.3 Hardware

The March 2026 `ibm_fez` material is retained as legacy artifact-backed
hardware evidence. Quote individual values only with their committed result
artifact or ledger row. Error mitigation via zero-noise extrapolation (ZNE)
with fold levels [1, 3, 5, 7, 9] and dynamical decoupling (X-X echo) were
tested on baseline circuits, but these rows do not establish broad advantage
or frontier validation.

---

## 3. Simulation Results

### 3.1 Entanglement at the Synchronisation Transition

![Entanglement vs coupling](figures/publication/fig1_entanglement_vs_coupling.png)

*Figure 1. Half-chain entanglement entropy $S(A)$ and Schmidt gap
$\Delta_S = \lambda_1 - \lambda_2$ across coupling strength for
$n = 2, 3, 4, 6, 8$ oscillators with heterogeneous frequencies.*

The Schmidt gap shows a sharp minimum at $K \approx 3.44$ for $n=8$
(Figure 8), marking the synchronisation transition. The entropy
saturates at different values per system size, consistent with the
Calabrese-Cardy scaling $S \sim (c/3)\ln L$ for a $c=1$ CFT.

![Transition zoom](figures/publication/fig8_transition_zoom.png)

*Figure 8. High-resolution (60-point) transition zoom for $n=6$ and $n=8$.
The $n=8$ Schmidt gap drops sharply at $K = 3.44$.*

### 3.2 Krylov Complexity

![Krylov](figures/publication/fig2_krylov_vs_coupling.png)

*Figure 2. Peak Krylov complexity and mean Lanczos coefficient $\langle b_n \rangle$
vs coupling. Mean $b$ grows linearly with $K$ (operator growth rate scales
with coupling strength).*

### 3.3 OTOC Information Scrambling

![OTOC](figures/publication/fig3_otoc_time_traces.png)

*Figure 3. OTOC $F(t)$ at sub-critical ($K=1$) and super-critical ($K=4$)
coupling for $n = 4, 6, 8$. Strong coupling scrambles 4× faster:
$t^* = 0.28$ (K=4) vs $t^* = 1.17$ (K=1) at $n=8$.*

### 3.4 Floquet Discrete Time Crystal

![Floquet DTC](figures/publication/fig9_floquet_dtc_n3456.png)

*Figure 9. Subharmonic ratio $P(\Omega/2)/P(\Omega)$ and mean $R$ vs
drive amplitude $\delta$ for $n=3, 4, 6$. All 15 amplitudes show finite-size
simulator DTC signatures above threshold. This is not a promoted hardware DTC
measurement.*

### 3.5 Finite-Size Scaling

![FSS](figures/publication/fig6_finite_size_scaling.png)

*Figure 6. Critical coupling $K_c(N)$ from spectral gap minimum.
BKT ansatz: $K_c(\infty) \approx 2.20$. Power-law: $K_c(\infty) \approx 2.94$.
Gap closes exponentially $N=4 \to 6$, consistent with BKT universality.*

### 3.6 Combined Overview

![Combined](figures/publication/fig7_combined_transition.png)

*Figure 7. Four probes of the synchronisation quantum phase transition:
spectral gap, entanglement entropy, Krylov complexity, and Schmidt gap.
All computed with Paper 27 heterogeneous frequencies.*

---

## 4. Hardware Results

### 4.1 Bell Test

![Hardware](figures/publication/fig10_ibm_hardware.png)

*Figure 10. IBM hardware results. (a) Per-qubit $\langle Z \rangle$ heatmap.
(b) 8-qubit expectations show coupling pattern. (c) QKD QBER: 5.5%.
(d) CHSH: $S = 2.165 > 2$.*

Two independent Bell pairs yield $S_{01} = 2.165 \pm 0.022$ and
$S_{23} = 2.188 \pm 0.021$, violating the classical limit of 2 at
$7.5\sigma$ and $8.9\sigma$ significance respectively.
\[corrected 2026-07-16: both pairs were previously stated as $>8\sigma$;
recomputation from the committed counts
(`scripts/recompute_chsh_bell_test.py`) attributes $7.54\sigma$ to the
lower pair and $8.94\sigma$ to the higher pair\]

### 4.2 QKD Viability

The quantum bit error rate in matched bases (ZZ: 5.5%, XX: 5.8%) is
well below the BB84 security threshold of 11%. Mismatched basis (ZX)
gives 93.9% error, confirming correct basis discrimination.

### 4.3 Error Characterisation

![Quantitative](figures/publication/fig13_quantitative_hw.png)

*Figure 13. (a) Per-qubit readout errors: Q2 best (0.65%), Q3 worst (3.55%).
(b) ZNE stability per qubit across fold levels 1–9.
(c) CHSH correlators with statistical error bars.*

### 4.4 ZNE Stability

Zero-noise extrapolation is remarkably stable: mean $\langle Z \rangle$
varies by $<2\%$ across fold levels 1–9 for both 4-qubit and 8-qubit
systems. Richardson extrapolation provides $<2\%$ correction to raw values,
indicating well-characterised noise on Heron r2.

### 4.5 16-Qubit UPDE

![Complete analysis](figures/publication/fig14_complete_analysis.png)

*Figure 14. (a) ZZ correlation matrix from CX entangling layer.
(b) Trotter order comparison. (c) 16-qubit per-qubit $\langle Z \rangle$:
alternating pattern across all 16 qubits — the Kuramoto coupling structure
is visible at full UPDE scale. (d) VQE 8-qubit energy landscape.*

13 of 16 qubits show $|\langle Z \rangle| > 0.3$ in this descriptive hardware
snapshot. This is retained as an artifact-backed observation of visible
per-qubit structure, not as proof that the full Kuramoto coupling structure
survives hardware noise at UPDE scale.

### 4.6 Ansatz Comparison

![Hardware suite](figures/publication/fig12_full_hardware_analysis.png)

*Figure 12. (a-f) Complete hardware experiment suite.*

The physics-informed Knm ansatz (CZ gates only between coupled pairs)
produces output entropy of 2.36 bits vs 3.46 (TwoLocal) and 3.39
(EfficientSU2) in the retained artifact. The Knm ansatz concentrates 42% of
probability in the top bitstring vs 20% for those comparator circuits; this is
a circuit-family-specific observation, not a backend-general outperformance
claim.

### 4.7 Exact-Simulation Crossover Boundary

![Quantum advantage crossover](figures/publication/fig17_quantum_advantage_crossover.png)

*Figure 17. Exact-simulation crossover anchored by completed ibm_fez
hardware runs. Red points and fit show exact diagonalisation wall time;
blue squares and dashed curve show hardware QPU budget estimates for
completed runs; green triangles show the Rust Kuramoto ODE baseline.*

The crossover at n≈11.6 applies only to exact Hilbert-space simulation of
the Kuramoto-XY Hamiltonian. It is not a broad computational-advantage
claim: observable-level Rust Kuramoto ODE baselines remain in the
millisecond regime through n=16, and the largest hardware circuits are
noise-limited.

---

### 4.8 Many-Body Localisation Diagnostic

![MBL](figures/publication/fig15_mbl_level_spacing.png)

*Figure 15. Level spacing ratio $\bar{r}$ vs coupling. Poisson ($\bar{r} = 0.386$):
integrable/MBL. GOE ($\bar{r} = 0.530$): chaotic/thermalising. At $n=8$,
the system **never reaches GOE** — MBL protection strengthens with system size.*

The level spacing ratio $\bar{r} = \langle\min(\delta_n, \delta_{n+1})/\max(\delta_n, \delta_{n+1})\rangle$
distinguishes integrable ($\bar{r} \approx 0.386$, Poisson) from chaotic
($\bar{r} \approx 0.530$, GOE) spectra. For the heterogeneous Kuramoto-XY:

- $n = 4$: crosses from Poisson to near-GOE at $K \approx 2.1$
- $n = 6$: stays mostly below GOE, chaos onset only at $K = 8.0$
- $n = 8$: **never reaches GOE** (max $\bar{r} = 0.43$)

As system size increases, the level spacing narrows toward Poisson. The
heterogeneous frequencies act as effective disorder preventing thermalisation.

**Cross-validation via eigenstate entanglement** (Figure 16) reveals a nuanced
picture: excited-state entanglement is 30–40% below the thermal (Page) expectation,
confirming non-ergodicity. However, the entanglement grows with $N$ (sub-volume
law, not area law), ruling out deep MBL. The correct characterisation is a
**non-ergodic regime** where heterogeneous frequencies protect the coupling
topology from thermal scrambling without producing true many-body localisation.

![Eigenstate entanglement](figures/publication/fig16_eigenstate_entanglement.png)

*Figure 16. Eigenstate entanglement ratio $S_{\text{excited}}/S_{\max}$ vs
system size. Our data (coloured) sits 30–40% below the thermal expectation
(black dashed). Non-ergodic but not deep MBL.*

To our knowledge, this is the first non-ergodicity diagnostic applied to
heterogeneous-frequency Kuramoto-XY systems. Level-spacing statistics
themselves are standard (Oganesyan & Huse, 2007).

---

## 5. Discussion

### Heterogeneous Frequencies and BKT Universality

Two independent tests confirm BKT universality is preserved:

1. **CFT central charge:** $c = 1.04$ at $n=8$ (BKT predicts $c=1$),
   measured from $S(l) = (c/3)\ln l + \text{const}$ at $K \approx K_c$.
2. **Spectral gap essential singularity:** $\Delta \sim \exp(-b/\sqrt{K-K_c})$
   fits with $R^2 > 0.96$ at $n = 4, 6, 8$.

The heterogeneous frequencies shift $K_c$ (from $\sim 2.8$ at $n=4$ to
$\sim 3.6$ at $n=8$) but do not change the universality class. The central
charge drifts upward at $n \geq 10$ ($c = 1.21$ at $n=10$, $c = 1.31$ at $n=12$),
which may indicate finite-size corrections or a genuine modification of the
CFT at larger scales.

### Non-Ergodicity and Identity Persistence

The Poisson level statistics (Figure 15) combined with sub-thermal eigenstate
entanglement (Figure 16) establish that the heterogeneous Kuramoto-XY system
occupies a non-ergodic regime. This is not deep MBL (which would require
area-law entanglement for all eigenstates), but a weaker form of ergodicity
breaking where the coupling topology is protected from thermal scrambling.

For the SCPN framework, this has direct implications: the network's frequency
disorder — a structural feature, not a defect — provides a natural mechanism
for identity persistence. Perturbations below the spectral gap $\Delta = E_1 - E_0$
cannot change the ground state, and the non-ergodic spectrum ensures that
thermal fluctuations do not explore the full Hilbert space.

### DTC Resilience to Frequency Disorder

The observation that all 15 drive amplitudes show subharmonic response with
heterogeneous frequencies contradicts the naive expectation that frequency
disorder destroys time-crystalline order. The non-ergodic spectrum (Section 4.7)
provides the mechanism: the heterogeneous frequencies prevent thermalisation,
which is the same physics that stabilises MBL-protected DTCs in disordered
spin chains. Our system is not deep MBL, but the non-ergodicity is sufficient
to protect the subharmonic response.

### Hardware Noise Budget

The 5.5% QBER and 94% state preparation fidelity in the retained legacy rows
indicate usable shallow-circuit performance for the tested Kuramoto-XY
circuits. The 16-qubit experiment remains noise-limited (depth $\leq$ 50 CX
gates). ZNE provides marginal improvement ($<2\%$), so the noise-budget
discussion remains descriptive rather than a backend-general coherence claim.

### Limitations

- No broad quantum advantage demonstrated. Classical observable-level ODE
  solvers outperform at $n \leq 16$; Figure 17 is an exact Hilbert-space
  simulation boundary only.
- Trotter error at $dt = 0.1$ is significant (Q1 sign flip at finer $dt$).
- DD (X-X echo) is marginally counterproductive for this Hamiltonian — the
  pulse sequence may not commute favourably with the XY interaction.
- The SCPN coupling matrix (Paper 27) is an unpublished model; the
  Kuramoto-to-XY mapping itself is standard physics.

---

## 6. Conclusion

We have preserved a legacy artifact-backed hardware and simulation study of
coupled-oscillator synchronisation with heterogeneous natural frequencies.
Promoted claims should be drawn only from committed raw-count artefacts,
analysis scripts, and ledger rows; the ibm_fez material remains descriptive
unless a later review promotes a specific result. The Rust-accelerated pipeline
enables ~111× faster small-system Hamiltonian construction (parity at large dense
builds) and 264× faster OTOC computation
vs the measured standard-tool baselines.

All code, data, and figures are open-source:

- **Code:** [github.com/anulum/scpn-quantum-control](https://github.com/anulum/scpn-quantum-control) (v0.10.0, AGPL-3.0)
- **Results:** `results/publication_scans_2026-03-27.json`, `results/ibm_hardware_2026-03-{18,28}/`
- **Docs:** [anulum.github.io/scpn-quantum-control](https://anulum.github.io/scpn-quantum-control)

---

## Data Availability

All simulation data (59 KB JSON), hardware results (22 IBM Quantum jobs),
analysis code (154 Python modules + 885-line Rust engine), and publication
figures (14 PNG + PDF) are available at the GitHub repository under AGPL-3.0.

---

## References

1. Šotek, M. (2025). God of the Math — The SCPN Master Publications. DOI: 10.5281/zenodo.17419678
2. Kuramoto, Y. (1984). Chemical Oscillations, Waves, and Turbulence. Springer.
3. Calabrese, P. & Cardy, J. (2004). Entanglement entropy and quantum field theory. JSTAT P06002.
4. Maldacena, J., Shenker, S. & Stanford, D. (2016). A bound on chaos. JHEP 08, 106.
5. del Campo, A. et al. (2025). Krylov complexity and quantum phase transitions. arXiv:2510.13947.
6. Berezinskii, V. L. (1972). Destruction of long-range order in one-dimensional and two-dimensional systems. JETP 34, 610.
7. Kosterlitz, J. M. & Thouless, D. J. (1973). Ordering, metastability and phase transitions. JPC 6, 1181.
8. IBM Quantum. ibm_fez backend specifications. quantum.cloud.ibm.com (2026).

---

<p align="center">
  <a href="https://www.anulum.li">
    <img src="assets/anulum_logo_company.jpg" height="70" alt="ANULUM">
  </a>
  &nbsp;&nbsp;&nbsp;&nbsp;
  <a href="https://www.anulum.li">
    <img src="assets/fortis_studio_logo.jpg" height="70" alt="Fortis Studio">
  </a>
  <br>
  <em>Developed by <a href="https://www.anulum.li">ANULUM</a> / Fortis Studio</em>
</p>
