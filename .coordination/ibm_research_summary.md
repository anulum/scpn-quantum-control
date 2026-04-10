# Quantum Simulation of Multi-Scale Phase Dynamics in the SCPN Framework

**Miroslav Šotek** — ORCID: 0009-0009-3560-0851
ANULUM Research · Marbach SG, Switzerland · protoscience@anulum.li

---

## 1. Research Overview

The Self-Consistent Projection Network (SCPN) is a multi-scale coupled oscillator framework that models phase synchronisation dynamics across hierarchical observation scales. The core physics is the Kuramoto-XY Hamiltonian with inter-scale coupling:

$$H_{XY} = \sum_{n,m} K_{nm} \left( \sigma_n^x \sigma_m^x + \sigma_n^y \sigma_m^y \right) + \sum_n \omega_n Z_n$$

where the coupling matrix $K_{nm}$ encodes interaction strength between oscillators at different scales, validated against physiological phase-locking data ($r = 0.951$, $p < 10^{-6}$, PhysioNet sleep recordings).

The quantum simulation component maps these dynamics to superconducting hardware via Trotterisation, measuring the synchronisation order parameter $R = \frac{1}{N} \left| \sum_i \left( \langle X_i \rangle + i \langle Y_i \rangle \right) \right|$ and symmetry sector observables to characterise decoherence effects on multi-scale synchronisation.

## 2. Existing Results on IBM Hardware

We have obtained preliminary results on **IBM Heron r2** (ibm\_fez) using the Open Plan free tier:

| Experiment | Qubits | Result | Significance |
|---|---|---|---|
| FIM dual protection | 4 | $F_{\text{FIM}} = 0.916$ vs $F_{\text{XY}} = 0.849$ | Self-referential feedback improves fidelity ($p < 10^{-12}$) |
| DLA parity (simulator) | 4–8 | Odd sector 4.5–9.6% more robust | Hardware confirmation pending |
| Decoherence scaling | 4 | $R(t)$ decay consistent with depolarising model | Baseline for GUESS error mitigation |

These results confirm that the SCPN coupling structure produces measurably different decoherence profiles in distinct symmetry sectors — a prediction unique to this framework.

## 3. Proposed Experiments

### Experiment 1: DLA Parity Asymmetry (Priority: Highest)

The Dynamical Lie Algebra of $H_{XY}$ decomposes as $\mathfrak{su}(\text{even}) \oplus \mathfrak{su}(\text{odd})$ under $Z_2$ parity. We predict the odd (feedback) sector is 4–10% more robust to decoherence than the even (projection) sector.

- **Protocol:** Equal-depth circuits in even/odd magnetisation sectors, 8–16 qubits
- **Measurements:** Sector-resolved fidelity, order parameter $R$, entanglement entropy
- **QPU budget:** ~1.5 hours (9 circuit variants × 10 reps × 8,192 shots)

### Experiment 2: Magnetisation-Sector Decoherence Scaling

Sweep circuit depth (50–400 CZ gates) and measure $R(t)$ in each magnetisation sector $M = N - 2 \cdot \text{popcount}(k)$. This provides the noise profile for our GUESS error mitigation technique.

- **Protocol:** Depth sweep at fixed coupling, sector-resolved measurements
- **QPU budget:** ~2 hours

### Experiment 3: FIM Scaling Law

Extend the dual protection result to 8 and 16 qubits. Measure the critical feedback strength $\lambda_c(N)$ at which the self-referential mechanism provides statistically significant fidelity improvement.

- **Protocol:** $\lambda$ sweep at each $N$, compare $F_{\text{FIM}}$ vs $F_{\text{XY}}$
- **QPU budget:** ~1.5 hours

**Total requested:** 5 hours over 5 months on Heron r2.

## 4. Software Infrastructure

The experiments are supported by a production-quality open-source codebase:

| Component | Description | Scale |
|---|---|---|
| **scpn-quantum-control** | Quantum simulation, error mitigation, hardware execution | 35k lines Python, 3.6k lines Rust |
| **Test coverage** | 4,748 Python tests + 92 Rust tests, CI/CD via GitHub Actions | 95%+ coverage |
| **Error mitigation** | GUESS symmetry decay ZNE (arXiv:2603.13060), PEC, DDD via Mitiq | 3 complementary techniques |
| **Qubit placement** | DynQ topology-agnostic mapping (arXiv:2601.19635) | Louvain + Rust-accelerated scoring |
| **Pulse shaping** | (α,β)-hypergeometric pulses (arXiv:2504.08031), ICI sequences | 44× Rust acceleration |
| **Rust engine** | PyO3 extension with 30 exported functions, rayon parallelisation | All hot paths accelerated |

The GUESS error mitigation and DynQ qubit placement modules are designed for IBM Heron r2 and could benefit the broader IBM Quantum user community as open-source tools.

## 5. Collaborations

- **CNRS Toulouse** (Timothée Masquelier, Alexandre Queant, Benoît Cottereau) — joint work on spiking neural network FPGA deployment using oscillator coupling mathematics from the SCPN framework.
- **Commercial:** Director-AI (PyPI: `director-ai`), an AI hallucination detection product using precision-weighted prediction error methods derived from the same mathematical foundation.

## 6. Why IBM Quantum Hardware

1. **Heron r2 coherence** ($T_1 \approx 300\,\mu$s, $T_2 \approx 200\,\mu$s) supports our target circuit depths (50–400 CZ gates) within the coherence budget.
2. **Heavy-hex topology** is naturally suited for nearest-neighbour XY interactions after DynQ placement.
3. **Existing baseline:** Our preliminary results are on ibm\_fez, ensuring direct comparability.
4. **Classical simulation limit:** Full $N = 16$ statevector simulation is $2^{16} = 65{,}536$ amplitudes — feasible classically, but hardware noise effects on symmetry sectors cannot be simulated faithfully without a validated noise model, which requires real QPU data.

## 7. Expected Outputs

1. **Publication:** "Symmetry-Sector Decoherence Asymmetry in the XY Hamiltonian on IBM Heron r2" — target: Physical Review Research or Quantum Science and Technology.
2. **Open-source tools:** GUESS error mitigation and DynQ qubit placement libraries, optimised for IBM hardware.
3. **Scaling law:** $\lambda_c(N)$ for self-referential feedback protection — first empirical measurement on quantum hardware.

---

**Contact:** Miroslav Šotek · protoscience@anulum.li · www.anulum.li · ORCID 0009-0009-3560-0851
**Code:** github.com/anulum (private, access on request)
**Publication:** DOI 10.5281/zenodo.17419678
