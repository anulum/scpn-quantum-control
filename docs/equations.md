# Key Equations

## Kuramoto → XY Hamiltonian

Classical Kuramoto (the UPDE core):

$$\frac{d\theta_i}{dt} = \omega_i + \sum_j K_{ij} \sin(\theta_j - \theta_i)$$

Quantum XY Hamiltonian:

$$H = -\sum_{i<j} K_{ij} (X_i X_j + Y_i Y_j) - \sum_i \omega_i Z_i$$

Time evolution via Lie-Trotter decomposition:

$$U(t) = e^{-iHt} \approx \left[e^{-iH_{XY}\Delta t}\, e^{-iH_Z\Delta t}\right]^{t/\Delta t}$$

Order parameter from qubit expectations:

$$R = \frac{1}{N} \left|\sum_i \left(\langle X_i \rangle + i\langle Y_i \rangle\right)\right|$$

## K_nm Canonical Parameters

Natural frequencies (16 layers, Paper 27 Table 1):

$$\boldsymbol{\omega} = [1.329,\; 1.255,\; 1.183,\; 1.114,\; 1.048,\; 0.985,\; 1.068,\; 1.148,\; 1.095,\; 1.028,\; 0.974,\; 0.929,\; 1.012,\; 0.962,\; 1.042,\; 0.991]$$

Coupling matrix (Paper 27, Eq. 3):

$$K_{nm} = K_{\text{base}} \cdot e^{-\alpha|n-m|}, \quad K_{\text{base}}=0.45,\; \alpha=0.3$$

Calibration anchors: $K_{1,2}=0.302$, $K_{2,3}=0.201$, $K_{3,4}=0.252$, $K_{4,5}=0.154$.

## UPDE (Unified Phase Dynamics Equation)

$$\frac{d\theta_n}{dt} = \omega_n + \sum_m K_{nm} \sin(\theta_m - \theta_n) + F_n(\boldsymbol{\theta}, t)$$

$F_n$ captures layer-specific forcing. In the quantum mapping, $F_n$ corresponds
to single-qubit Z rotations beyond the natural frequency term.

## Quantum LIF Neuron

Membrane dynamics:

$$v(t+1) = v(t) - \frac{\Delta t}{\tau}\bigl(v(t) - v_{\text{rest}}\bigr) + R \cdot I \cdot \Delta t$$

Rotation angle encoding:

$$\theta = \pi \cdot \text{clip}\!\left(\frac{v - v_{\text{rest}}}{v_{\text{thresh}} - v_{\text{rest}}},\; 0,\; 1\right)$$

Spike probability:

$$P(\text{spike}) = \sin^2\!\left(\frac{\theta}{2}\right)$$

## Quantum Synapse (CRy)

Weight-to-angle:

$$\theta_w = \pi \cdot \frac{w - w_{\min}}{w_{\max} - w_{\min}}$$

Effective transmission probability:

$$P(\text{post}\mid\text{pre}=1) = \sin^2\!\left(\frac{\theta_w}{2}\right)$$

## Parameter-Shift Rule

Gradient of expectation w.r.t. rotation angle:

$$\frac{\partial\langle Z \rangle}{\partial\theta} = \frac{\langle Z \rangle\!\left(\theta + \frac{\pi}{2}\right) - \langle Z \rangle\!\left(\theta - \frac{\pi}{2}\right)}{2}$$

STDP weight update:

$$\Delta w = \eta \cdot s_{\text{pre}} \cdot \frac{\partial\langle Z_{\text{post}}\rangle}{\partial\theta_w}$$

QSNN training uses the same rule for MSE loss gradient:

$$\nabla_{\theta_{ni}} \mathcal{L} = \frac{\mathcal{L}(\theta_{ni} + \pi/2) - \mathcal{L}(\theta_{ni} - \pi/2)}{2}$$

## QAOA Cost Hamiltonian

MPC quadratic cost → Ising:

$$C = \sum_t \|Bu(t) - x_{\text{target}}\|^2 = \sum_{ij} J_{ij} Z_i Z_j + \sum_i h_i Z_i + \text{const}$$

QAOA circuit:

$$|\gamma, \beta\rangle = \prod_{p=1}^{P} \left[e^{-i\beta_p H_{\text{mixer}}}\, e^{-i\gamma_p H_{\text{cost}}}\right] |+\rangle^{\otimes n}$$

## VQLS Cost Function

For linear system $Ax = b$:

$$C_{\text{VQLS}} = 1 - \frac{|\langle b|A|x\rangle|^2}{\langle x|A^\dagger A|x\rangle}$$

where $|x\rangle = U(\theta)|0\rangle$ is a variational ansatz.

## Probabilistic Error Cancellation (PEC)

Temme et al., PRL 119, 180509 (2017).

Quasi-probability decomposition of the inverse depolarizing channel $\mathcal{E}^{-1}$
into Pauli operations $\{I, X, Y, Z\}$:

$$q_I = 1 + \frac{3p}{4-4p}, \quad q_{X,Y,Z} = -\frac{p}{4-4p}$$

Sampling overhead (cost multiplier):

$$\gamma = \left(\sum_k |q_k|\right)^{n_{\text{gates}}}$$

Monte Carlo estimator:

$$\langle O \rangle_{\text{mitigated}} = \frac{1}{N} \sum_{s=1}^{N} \gamma \cdot \text{sgn}(s) \cdot \langle O \rangle_s$$

## Trapped-Ion Noise Model

Mølmer-Sørensen gate error model (QCCD architecture):

| Parameter | Value | Source |
|-----------|-------|--------|
| MS 2-qubit error | 0.5% | QCCD benchmarks |
| $T_1$ | 100 ms | Ion trap coherence |
| $T_2$ | 1 ms | Dephasing time |
| SQ gate time | 10 μs | Single-qubit |
| MS gate time | 200 μs | Two-qubit |

Noise composition per MS gate:

$$\mathcal{E}_{\text{MS}} = \mathcal{E}_{\text{depol}}(p=0.005) \circ \bigl(\mathcal{E}_{\text{relax}}(T_1, T_2, t_{\text{MS}}) \otimes \mathcal{E}_{\text{relax}}\bigr)$$

All-to-all connectivity (no SWAP overhead).

## ITER Disruption Feature Space

11 physics-based features (ITER Physics Basis, Nuclear Fusion 39, 1999):

| Feature | Symbol | Range | Units |
|---------|--------|-------|-------|
| Plasma current | $I_p$ | 0.5–17 | MA |
| Safety factor | $q_{95}$ | 1.5–8 | — |
| Internal inductance | $l_i$ | 0.5–2 | — |
| Greenwald fraction | $n_{\text{GW}}$ | 0–1.5 | — |
| Normalized beta | $\beta_N$ | 0–4 | — |
| Radiated power | $P_{\text{rad}}$ | 0–100 | MW |
| Locked mode | LM | 0–0.01 | T |
| Loop voltage | $V_{\text{loop}}$ | −2–5 | V |
| Stored energy | $W$ | 0–400 | MJ |
| Elongation | $\kappa$ | 1–2.2 | — |
| Current ramp | $dI_p/dt$ | −5–5 | MA/s |

Min-max normalization to $[0, 1]$:

$$\hat{x}_i = \text{clip}\!\left(\frac{x_i - x_{\min,i}}{x_{\max,i} - x_{\min,i}},\; 0,\; 1\right)$$

## Fault-Tolerant UPDE (Repetition Code)

Each oscillator encoded in $d$ physical data qubits + $(d-1)$ ancilla qubits.

Total physical qubits: $N_{\text{phys}} = n_{\text{osc}} \cdot (2d - 1)$

Encoding (repetition code, bit-flip protection):

$$|\psi_L\rangle = R_y(\theta)|0\rangle^{\otimes d} \xrightarrow{\text{CNOT fan-out}} \cos(\theta/2)|0\rangle^{\otimes d} + \sin(\theta/2)|1\rangle^{\otimes d}$$

Transversal coupling between logical qubits $i$ and $j$:

$$U_{\text{ZZ}}^{(L)} = \prod_{k=1}^{d} R_{ZZ}\!\left(\frac{K_{ij}\Delta t}{d},\; q_k^{(i)},\; q_k^{(j)}\right)$$

Syndrome extraction via adjacent parity checks on ancillae.

## Quantum Advantage Scaling

Classical cost (matrix exponential): $O(2^{2n})$ for $n$ qubits.

Quantum cost (Trotter): $O(n^2 \cdot r)$ gates per step, where $r$ = Trotter repetitions.

Crossover estimate via exponential fit:

$$t_c(n) = a_c \cdot e^{b_c n}, \quad t_q(n) = a_q \cdot e^{b_q n}$$

$$n_{\text{cross}} = \frac{\ln(a_q/a_c)}{b_c - b_q} \quad (b_c > b_q)$$

## SSGF Quantum Loop

SSGF geometry matrix $W$ (symmetric, non-negative, zero diagonal) maps
directly to the XY Hamiltonian via the same $K_{nm} \to H$ compiler.

Phase encoding into qubit XY-plane:

$$|q_i\rangle = R_y(\pi/2)\, R_z(\theta_i)\, |0\rangle = \frac{|0\rangle + e^{i\theta_i}|1\rangle}{\sqrt{2}}$$

Phase recovery: $\theta_i = \text{atan2}(\langle Y_i\rangle, \langle X_i\rangle)$

## Identity Binding Topology

6-layer, 18-oscillator Arcane Sapience identity spec:

| Layer | $\omega$ (rad/s) | Oscillators |
|-------|-----------------|-------------|
| working_style | 1.2 | 3 |
| reasoning | 2.1 | 3 |
| relationship | 0.8 | 3 |
| aesthetics | 1.5 | 3 |
| domain_knowledge | 3.0 | 3 |
| cross_project | 0.9 | 3 |

Coupling: $K_{\text{intra}} = 0.6$, $K_{\text{inter}} = 0.4 \cdot e^{-0.25|l_i - l_j|}$

Maps to 35 oscillators in scpn-phase-orchestrator's identity_coherence domainpack
via centroid projection (circular mean for phase roundtrip).
