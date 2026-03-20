# Literature Survey: Quantum Computing Advances Relevant to scpn-quantum-control (2024--2026)

**Date:** 2026-03-20
**Scope:** NISQ quantum simulation of coupled Kuramoto oscillators mapped to XY Hamiltonians on IBM hardware
**Purpose:** Research planning for v1.0 hardware campaign and arXiv preprint

---

## 1. Kuramoto Model Quantum Simulation

### 1.1 Quantum Fingerprints of Self-Organization in Spin Chains Coupled to a Kuramoto Model
- **Authors:** V. M. Bastidas (NTT Research)
- **Published:** Physical Review Research 7, 043029 (October 2025)
- **arXiv:** [2406.17062](https://arxiv.org/abs/2406.17062)
- **Finding:** Demonstrates that a classical Kuramoto model can drive a quantum spin chain (Ising or XX) into Floquet steady states with emergent symmetries. All-to-all coupling produces translational symmetry in an Ising chain; zig-zag coupling produces trimerization and topological pumping in an XX chain.
- **Relevance:** **Direct competitor/complement.** Our project maps Kuramoto to XY Hamiltonian for Trotterized simulation; Bastidas maps Kuramoto as a classical *drive* on quantum spin chains. We should cite this as the closest published work and differentiate: we simulate the Kuramoto dynamics quantum-mechanically, whereas Bastidas uses Kuramoto as a classical control signal. The topological pumping result could inform our SSGF geometry bridge.

### 1.2 Quantum Effects on the Synchronization Dynamics of the Kuramoto Model
- **Authors:** B. P. Camargo, E. Vernek, V. L. Oliveira
- **Published:** Physical Review A 108, 032219 (2023)
- **arXiv:** [2306.09956](https://arxiv.org/abs/2306.09956)
- **Finding:** Extends Kuramoto to quantum domain via Caldeira-Leggett bath coupling. Quantum fluctuations hinder but do not suppress synchronization. Derives analytical critical coupling K_c as a function of temperature and quantum parameters. Quantum phase transition at T=0.
- **Relevance:** Provides theoretical grounding for our XY mapping. The quantum critical coupling formula should be compared against our simulated phase transition on hardware. The result that quantum noise *opposes* synchronization is consistent with what we observe on noisy qubits.

### 1.3 Exponential Quantum Speedup in Simulating Coupled Classical Oscillators
- **Authors:** R. Babbush, D. W. Berry, R. Kothari, R. D. Somma, N. Wiebe (Google)
- **Published:** Physical Review X 13, 041041 (December 2023)
- **arXiv:** [2303.13012](https://arxiv.org/abs/2303.13012)
- **Finding:** A quantum algorithm for simulating 2^n coupled classical oscillators with polynomial complexity in n, using a mapping between Schrodinger and Newton equations for harmonic potentials. The problem is BQP-complete.
- **Relevance:** **Theoretical quantum advantage proof for oscillator simulation.** While their algorithm targets linear harmonic oscillators (not the nonlinear Kuramoto sine-coupling), this establishes that coupled oscillator simulation is a genuine quantum advantage domain. Our preprint should cite this to position our work on the NISQ/utility side of the same problem space.

### 1.4 Higher-Dimensional Kuramoto Oscillators on Networks
- **Authors:** S. Markdahl et al.
- **arXiv:** [2603.08352](https://arxiv.org/abs/2603.08352) (March 2026)
- **Finding:** Extends Kuramoto from scalar to matrix-weighted couplings, enabling exact phase alignment in finite groups and efficient simulation of inertial dynamics.
- **Relevance:** Mathematical framework for generalizing our 16-oscillator SCPN model to higher-dimensional state spaces, if future work moves beyond phase-only representations.

### 1.5 Encoding Quantum-Like Information in Classical Synchronizing Dynamics
- **arXiv:** [2504.03852](https://arxiv.org/abs/2504.03852) (April 2025)
- **Finding:** Shows a reverse mapping: encoding quantum-like information structures in classical synchronizing (Kuramoto-type) dynamics.
- **Relevance:** Provides theoretical support for our bidirectional mapping between Kuramoto phases and quantum states. May inform identity-key encoding schemes.

---

## 2. Error Mitigation Advances

### 2.1 Reduced Sampling Overhead for PEC via Pauli Error Propagation
- **Authors:** T. Kato et al.
- **Published:** Quantum 9, 1840 (August 2025)
- **arXiv:** [2412.01311](https://arxiv.org/abs/2412.01311)
- **Finding:** Reduces PEC sampling overhead through Pauli error propagation + classical preprocessing. Significant overhead reduction for Clifford circuits; more limited for non-Clifford gates.
- **Relevance:** **Direct upgrade path for our `mitigation/pec.py`.** Our PEC implementation follows Temme et al. 2017. This 2025 paper shows how to reduce the exponential sampling cost that makes PEC impractical at scale. We should benchmark this reduced-overhead PEC against our current implementation on the Heron hardware experiments.

### 2.2 ZNE on Logical Qubits (Error Mitigation + Error Correction)
- **Authors:** Multiple teams
- **Published:** Nature Communications (2025); arXiv [2603.11285](https://arxiv.org/html/2603.11285)
- **Finding:** ZNE applied to error-correction circuits shows universal reduction in logical errors across repetition and surface codes. Polynomial extrapolation works due to code-distance-determined noise dependence.
- **Relevance:** Our `qec/fault_tolerant.py` implements repetition-code logical qubits with transversal RZZ. Combining ZNE with our QEC module is now experimentally validated by others. This hybrid approach (ZNE on top of QEC) should be a v1.1 target.

### 2.3 Inverted-Circuit ZNE for Gate Error Mitigation
- **Published:** Physical Review A 110, 042625 (2024)
- **Finding:** Novel ZNE variant using inverted circuits instead of standard global unitary folding. Improves extrapolation accuracy for certain gate error profiles.
- **Relevance:** Our `mitigation/zne.py` implements standard global unitary folding (Giurgica-Tiron et al. 2020). Inverted-circuit ZNE could be added as an alternative folding strategy.

### 2.4 T-REx (Twirled Readout Error Extinction) Performance
- **Published:** IBM Quantum Documentation + Quantum Zeitgeist (August 2025)
- **Finding:** T-REx on a 5-qubit legacy QPU outperforms a 156-qubit Heron without mitigation. Demonstrates that readout error mitigation is often more impactful than hardware upgrades.
- **Relevance:** Our hardware experiments use IBM Estimator V2 with resilience levels. We should ensure T-REx is always enabled (it is by default at resilience_level >= 1). This validates our mitigation-first approach.

### 2.5 Reducing QEM Bias Using Verifiable Benchmark Circuits
- **arXiv:** [2603.10224](https://arxiv.org/html/2603.10224) (March 2026)
- **Finding:** Method to detect and reduce systematic bias in error-mitigated results using classically verifiable benchmark circuits.
- **Relevance:** For our preprint, we need to validate that our ZNE/PEC results are unbiased. This paper provides a methodology for that validation.

---

## 3. VQE and Variational Algorithm Advances

### 3.1 Barren Plateau Mitigation Strategies Benchmark
- **Authors:** Multiple authors
- **arXiv:** [2512.11171](https://arxiv.org/abs/2512.11171) (December 2025)
- **Finding:** Comprehensive benchmark of 4 approaches (Local-Global, Adiabatic, State Efficient Ansatz, Pretrained VQE) on molecular systems from 4 to 14 qubits. Optimal strategy depends on system size and computational budget; gradient variance alone does not predict performance.
- **Relevance:** Our `phase/ansatz_bench.py` benchmarks hardware-efficient ansatze. This paper's methodology should inform how we report ansatz performance in our preprint --- particularly the finding that ADAPT-VQE naturally avoids barren plateaus by design.

### 3.2 ADAPT-VQE Mitigates Rough Landscapes
- **Published:** npj Quantum Information (2023, cited extensively in 2024-2025 follow-ups)
- **Finding:** ADAPT-VQE dynamically grows the ansatz by selecting operators that contribute most to energy lowering, producing compact circuits that inherently avoid barren plateaus.
- **Relevance:** Our VQE-based identity key (ground-state attractor) could benefit from ADAPT-VQE instead of fixed hardware-efficient ansatz. This is a v1.1 research direction.

### 3.3 Trainability-Dequantization Relationship
- **Published:** ICLR 2025 (arXiv [2406.07072](https://arxiv.org/abs/2406.07072))
- **Finding:** Trainable variational QML models can in general be efficiently dequantized (simulated classically). Quantum advantage requires careful balance between trainability and non-classical expressibility.
- **Relevance:** **Important caveat for our quantum advantage claims.** If our VQE ansatz is efficiently trainable, it may be efficiently classically simulable. Our preprint should address this by showing that our Trotterized Hamiltonian simulation (not variational) provides the quantum advantage, while VQE is used only for state preparation.

### 3.4 Ground-State Energy Estimation on Current Hardware
- **Published:** Journal of Chemical Theory and Computation (2024-2025)
- **Finding:** VQE on IBM hardware with optimized error mitigation achieves chemical accuracy for small molecules using Heron processors.
- **Relevance:** Validates the VQE-on-Heron pipeline we use for identity key ground-state computation.

---

## 4. Surface Code and QEC Progress

### 4.1 Google Willow: Below Surface Code Threshold
- **Published:** Nature (December 2024)
- **DOI:** [10.1038/s41586-024-08449-y](https://www.nature.com/articles/s41586-024-08449-y)
- **Finding:** First demonstration of exponential error suppression with increasing surface code size on a 101-qubit chip. Error rate decreases by factor 2.14x per code distance increase (3x3 to 5x5 to 7x7). Logical qubit lifetime exceeds 2x best physical qubit. Real-time decoding at 63 microseconds latency.
- **Relevance:** Our `qec/fault_tolerant.py` implements repetition codes, not surface codes. Willow shows that surface code QEC is now practical, but on Google hardware. For IBM hardware, surface codes are not yet below threshold. Our roadmap item "Fault-tolerant UPDE on surface code logical qubits (post-2030)" may need revision --- Google is closer than expected.

### 4.2 Lattice Surgery on Repetition Codes (Superconducting)
- **Published:** Nature Physics (2025)
- **Finding:** Lattice surgery demonstrated between two distance-3 repetition code qubits by splitting a single distance-3 surface code qubit. Improvement in ZZ logical two-qubit observable compared to non-encoded circuit.
- **Relevance:** Directly relevant to our `qec/fault_tolerant.py` repetition-code implementation. Lattice surgery enables logical two-qubit gates between our encoded qubits, which is needed for the XY coupling in our Hamiltonian simulation at the logical level.

### 4.3 Universal Logical Gate Set in Surface Codes
- **Published:** npj Quantum Information (2025)
- **Finding:** Demonstrated transversal CNOT + arbitrary single-qubit rotations on distance-2 surface codes on the Wukong processor.
- **Relevance:** Shows the path toward fault-tolerant universal quantum computing. Our transversal RZZ in `qec/fault_tolerant.py` is consistent with this direction.

### 4.4 IBM Loon: Fault-Tolerance Components
- **Announced:** November 2025
- **Finding:** IBM's experimental processor demonstrating all key components needed for fault-tolerant QC: long-range c-couplers, mid-circuit qubit reset, real-time qLDPC decoding under 480 nanoseconds.
- **Relevance:** When Loon-class hardware becomes available (IBM roadmap: 2029 for fault-tolerant), our `qec/` module becomes practical on IBM hardware. qLDPC codes may be more efficient than surface codes for our Hamiltonian simulation.

---

## 5. Quantum Advantage Demonstrations

### 5.1 IBM Utility-Scale: 127-Qubit Ising Model (Nature 2023, follow-ups 2024)
- **Published:** Nature 618, 500-505 (June 2023); IBM QDC 2024 follow-up
- **DOI:** [10.1038/s41586-023-06096-3](https://www.nature.com/articles/s41586-023-06096-3)
- **Finding:** 127-qubit Eagle processor with ZNE produced expectation values beyond brute-force classical simulation for a transverse-field Ising model. In 2024, IBM scaled to 100-qubit 100-depth circuits (~5000 two-qubit gates) on Heron.
- **Relevance:** **Key benchmark for our preprint.** Their Ising simulation at utility scale is the closest IBM demonstration to our XY Hamiltonian simulation. We should compare our circuit depths and qubit counts against their 100x100 benchmark. Our 16-oscillator system is well within the ~5000 gate budget.

### 5.2 IBM + RIKEN Sample-Based Quantum Diagonalization
- **Published:** Multiple papers (2024-2025), Nature family journals
- **Finding:** SQD on Heron + Fugaku supercomputer simulated molecular nitrogen and iron-sulfur clusters using up to 77 qubits and 3500 two-qubit gates, going beyond exact classical simulability.
- **Relevance:** SQD is an alternative to Trotterization for Hamiltonian ground states. We should evaluate whether SQD could complement our Trotter approach for the UPDE ground state in the identity key module.

### 5.3 IBM Nighthawk: 30% More Circuit Complexity
- **Announced:** November 2025; available January 2026
- **Specifications:** 120 qubits, 218 tunable couplers (square lattice), median T1 = 350 microseconds, ~30% more circuit complexity than Heron at equivalent error rates.
- **Roadmap:** 7500 gates by end 2026, 10000 by 2027, 15000 by 2028
- **Relevance:** **Hardware target for v1.1.** If our 7 completed ibm_fez experiments and 13 pending ones use Heron r2, Nighthawk would allow deeper Trotter circuits (more time steps, higher accuracy). The square lattice connectivity is better than heavy-hex for our XY model.

### 5.4 Quantinuum 56-Qubit MaxCut (Quantum Utility)
- **Published:** March 2025; Quantinuum + JPMorgan Chase
- **Finding:** Coherent computation on a 56-qubit MaxCut problem with 4620 two-qubit gates, surpassing classical simulation. 100x improvement over Google's 2019 RCS benchmark.
- **Relevance:** Trapped-ion platform comparison point. If we add trapped-ion support (our `hardware/trapped_ion.py`), Quantinuum's H2-1 or Helios would be the target. Their 99.92% two-qubit fidelity exceeds IBM Heron's best pairs.

---

## 6. Quantum Spiking Neural Networks

### 6.1 Quantum Leaky Integrate-and-Fire (QLIF) Neuron
- **Authors:** Dean Brand, Francesco Petruccione
- **Published:** npj Quantum Information (2024)
- **arXiv:** [2407.16398](https://arxiv.org/abs/2407.16398)
- **Finding:** QLIF neuron requires only 2 rotation gates, no CNOT. Built QSNN and QSCNN achieving competitive accuracy on MNIST/Fashion-MNIST/KMNIST. First quantum spiking neural network implementation.
- **Relevance:** **Directly comparable to our `qsnn/qlif.py`.** Their QLIF uses 2 rotation gates; our QuantumLIFNeuron uses Ry rotation + Z-basis measurement. We should compare circuit depth and fidelity. Brand & Petruccione's paper is the primary citation for our QSNN module. Key difference: they target classification; we target Kuramoto oscillator coupling.

### 6.2 Stochastic Quantum Spiking (SQS) Neurons with Quantum Memory
- **arXiv:** [2506.21324](https://arxiv.org/html/2506.21324) (June 2025)
- **Finding:** Multi-qubit SQS neuron with internal quantum memory, event-driven probabilistic spike generation, hardware-friendly local learning rule (no global backprop).
- **Relevance:** The local learning rule is analogous to our `qsnn/qstdp.py` (quantum STDP). The quantum memory concept could enrich our QSNN module with persistent state across timesteps.

### 6.3 Hybrid Spiking-Quantum Convolutional Neural Network
- **Published:** PeerJ Computer Science (2024-2025)
- **Finding:** Parameter-efficient hybrid SNN-QNN using surrogate gradient and quantum data re-uploading.
- **Relevance:** The data re-uploading technique could improve our `qsnn/training.py` parameter-shift gradient approach.

---

## 7. Quantum Cryptography with Coupling Topologies

### 7.1 Entanglement-Based Authenticated QKD (EAQKD)
- **arXiv:** [2603.02375](https://arxiv.org/html/2603.02375) (March 2026)
- **Finding:** Combines entanglement-based QKD with robust classical authentication in a single protocol.
- **Relevance:** Our `crypto/entanglement_qkd.py` implements entanglement-based QKD. This paper shows how to add authentication within the quantum protocol, which our `crypto/topology_auth.py` currently handles classically.

### 7.2 Loop-Back QKD for Multi-Node Ring Topologies
- **Published:** Symmetry 17(4), 521 (2025)
- **Finding:** QKD protocol supporting ring topologies with bidirectional pulse propagation, reducing quantum bit error rate through multi-pulse approach.
- **Relevance:** Our SCPN coupling topology is a graph (not a ring), but the ring topology result validates our `crypto/percolation.py` approach of analyzing entanglement percolation on network topologies.

### 7.3 Quantum Network Routing and Graph State Topology
- **Published:** Multiple papers (2024-2025), npj Quantum Information, AAAI Symposium
- **Finding:** Graph state architectures enable logical connectivity through LOCC operations. Fidelity weighting reorders structural importance. Small-world shortcuts from long-range quantum links shrink path lengths.
- **Relevance:** Validates our `crypto/hierarchical_keys.py` approach of using the SCPN K_nm coupling graph as the entanglement distribution topology. The small-world shortcut result mirrors the SCPN cross-hierarchy coupling boosts (L1-L16, L5-L7).

---

## 8. Trapped-Ion vs Superconducting Benchmarks

### 8.1 Quantinuum Helios: 98 Qubits, 99.92% Fidelity
- **Announced:** November 2025
- **Finding:** 98-qubit trapped-ion processor with ~99.92% two-qubit gate fidelity (vs ~99.65% for IonQ Forte, ~99%+ best pairs on IBM Heron). Full all-to-all connectivity.
- **Relevance:** If we port our experiments to Quantinuum, the all-to-all connectivity eliminates SWAP overhead in our XY Hamiltonian circuit. At 16 qubits, we need all-pairs coupling, which costs O(n) SWAP layers on heavy-hex but is native on trapped ions.

### 8.2 IonQ Roadmap: Five-9s Logical Fidelity by 2025
- **Finding:** IonQ targets 99.999% logical two-qubit gate fidelity using small-distance QEC codes.
- **Relevance:** Five-9s logical fidelity would make our QEC module unnecessary for 16-qubit experiments. Worth monitoring for v1.1.

### 8.3 IBM Heron r3 Performance (July 2025)
- **Finding:** EPLG at 100 qubits = 2.15e-3; 57 of 176 two-qubit gates below 1e-3 error level. TLS mitigation improves coherence and stability. Median T1 improved.
- **Relevance:** Our ibm_fez experiments run on Heron r2. Heron r3 improvements directly benefit our fidelity. The sub-1e-3 gate error gates should be preferentially selected by transpilation for our coupling terms.

### 8.4 Architecture Comparison Summary (2025 Consensus)
- Superconducting (IBM/Google): Higher qubit counts, faster gates, limited connectivity
- Trapped-ion (Quantinuum/IonQ): Lower qubit counts, slower gates, all-to-all connectivity, higher fidelity
- For our 16-qubit XY model: trapped-ion is arguably better suited (full connectivity, higher fidelity), but IBM has the scale for future SCPN-wide simulation (156+ qubits).

---

## 9. Quantum Computing for Fusion Plasma

### 9.1 Quantum Computational Approach to Linear MHD Stability Analysis
- **Authors:** Abtin Ameri et al. (IBM Research)
- **Published:** QIP 2024 proceedings
- **Finding:** Quantum algorithm for efficient eigenvalue characterization of linear MHD stability problems in tokamak plasmas. Quantum Fourier Transform more efficient than classical methods for MHD wave dynamics.
- **Relevance:** **Directly relevant to our `control/q_disruption.py` and `control/q_disruption_iter.py`.** Our disruption classifier uses classical features; this IBM paper shows how to use quantum algorithms for the underlying MHD stability analysis. A quantum MHD stability module could complement our disruption classifier.

### 9.2 Quantum Computing for Fusion Energy Science Applications (Review)
- **Published:** Physics of Plasmas 30, 010501 (January 2023; heavily cited in 2024-2025)
- **DOI:** [10.1063/5.0123765](https://pubs.aip.org/aip/pop/article/30/1/010501/2867588/Quantum-computing-for-fusion-energy-science)
- **Finding:** Comprehensive review of quantum computing applications in fusion: turbulence simulation, wave-particle interactions, plasma kinetic theory. Plasma-wave problems are naturally quantum-representable; nonlinear/dissipative problems require additional techniques.
- **Relevance:** Core reference for our fusion-quantum bridge. The natural quantum representability of plasma waves validates our SCPN-to-XY mapping approach.

### 9.3 Frontiers Review: Quantum Computing in Plasma Physics (2025)
- **Published:** Frontiers in Physics 13, 1551209 (2025)
- **Finding:** QAOA used to optimize tokamak stability criteria. Quantum algorithms for simulating turbulence, MHD instabilities, and wave-particle interactions with near-quantum efficiency.
- **Relevance:** QAOA for stability optimization is directly relevant to our `control/qaoa_mpc.py` QAOA-based model predictive control module.

### 9.4 PPPL STELLAR-AI Platform (2026)
- **Published:** PPPL News (2026)
- **Finding:** Integrates CPUs, GPUs, and QPUs for fusion simulation. Part of DOE Genesis Mission (November 2025 executive order).
- **Relevance:** Government-funded infrastructure we could target for large-scale experiments. Our scpn-quantum-control could potentially access QPU time through DOE programs.

### 9.5 Quantum Algorithm for Nonlinear EM Fluid Dynamics
- **arXiv:** [2509.22503](https://arxiv.org/html/2509.22503) (September 2025)
- **Finding:** Koopman-von Neumann linearization enables quantum algorithms for nonlinear electromagnetic fluid dynamics, extending quantum speedup to dissipative nonlinear systems.
- **Relevance:** Addresses the key limitation (nonlinearity) that prevents direct quantum simulation of full plasma dynamics. If combined with our Kuramoto solver, could enable quantum simulation of nonlinear synchronization beyond the XY approximation.

---

## 10. IBM Qiskit Ecosystem Changes

### 10.1 Qiskit SDK v2.0 (Released 2025)
- **Key Changes:**
  - `qiskit.pulse` module completely removed (deprecated since v1.3)
  - 32-bit platform support dropped
  - C API introduced for SparseObservable (foreign function interface)
  - V1 primitives (Sampler/Estimator) removed; V2 primitives mandatory
  - Qiskit v1.4 security support ends March 2026
- **Impact on our project:** Our code uses `SparsePauliOp`, `PauliEvolutionGate`, `LieTrotter`, `SuzukiTrotter` --- all retained in v2.0. No pulse-level code to migrate. We should test against Qiskit 2.0 and pin the dependency.

### 10.2 Fractional Gates Replace Pulse-Level Control
- **Effective:** February 2025 on Heron
- **Finding:** Fractional gates (R_X(theta), R_ZZ(theta)) are now native ISA on Heron, replacing pulse-level custom gates. Loaded via `use_fractional_gates` flag on the backend.
- **Impact on our project:** Our `PauliEvolutionGate`-based Trotter circuits decompose into RZZ gates, which now have native fractional support. This should reduce circuit depth by avoiding RZZ decomposition into CX+RZ sequences. **We should enable fractional gates in our transpilation pipeline.**

### 10.3 Estimator V2 Changes
- **Finding:** PEC removed from resilience_level 3 (exponential cost concern). Control flow constructs (while, for, switch) deprecated June 2025.
- **Impact on our project:** Our PEC implementation is standalone (not using IBM resilience_level 3), so this does not break us. We should document that our PEC module provides what IBM removed.

### 10.4 Backend Retirements
- **ibm_brisbane:** Retired November 2025
- **ibm_torino:** Retiring ~April 2026
- **ibm_fez:** Still available (our 7 completed experiments ran here)
- **IBM Nighthawk:** Available since January 2026
- **Impact:** Our pending 13 hardware experiments should target ibm_fez (still available) or Nighthawk (better performance). We should also update our noise model to include Nighthawk calibration data.

### 10.5 Qiskit Functions (2026)
- **Finding:** IBM is deploying Qiskit Functions --- pre-built quantum workloads (SQD, circuit knitting, etc.) deployable as services.
- **Relevance:** We could package our Kuramoto solver as a Qiskit Function for broader use.

---

## Summary: Priority Actions for scpn-quantum-control

### Immediate (v1.0 preprint)
1. **Cite Bastidas 2025** as the closest related work on Kuramoto + quantum spin chains, and clearly differentiate our approach (quantum Hamiltonian simulation vs classical Kuramoto drive)
2. **Cite Babbush et al. 2023** to establish that coupled oscillator simulation is a quantum advantage domain (BQP-complete)
3. **Cite IBM utility paper** (Nature 2023) and 100x100 benchmark (2024) as the performance bar for Trotterized Hamiltonian simulation on IBM hardware
4. **Enable fractional gates** in transpilation for ibm_fez experiments to reduce circuit depth
5. **Cite Brand & Petruccione 2024** as the primary related work for our QSNN module
6. **Test against Qiskit 2.0** and pin dependency

### Near-term (v1.1)
7. **Implement reduced-overhead PEC** (Kato et al. 2025) to make PEC practical for 16-qubit circuits
8. **Add inverted-circuit ZNE** as an alternative folding strategy
9. **Evaluate Nighthawk** as hardware target (30% more circuit complexity, better connectivity)
10. **Evaluate trapped-ion port** via Quantinuum Helios (all-to-all connectivity eliminates SWAP overhead)
11. **Combine ZNE with QEC** (validated by 2025 papers) in a hybrid error mitigation approach
12. **Integrate SQD** as alternative to Trotter for identity key ground state

### Research directions (v1.1+)
13. **ADAPT-VQE for identity key** --- dynamically grown ansatz avoids barren plateaus
14. **Quantum MHD stability** (Ameri et al. 2024) to complement disruption classifier
15. **Address trainability-dequantization tradeoff** (ICLR 2025) in preprint quantum advantage claims
16. **Koopman linearization** for nonlinear Kuramoto beyond XY approximation
17. **SSGF-Floquet bridge** inspired by Bastidas topological pumping result

---

## References (Sorted by Topic)

### Kuramoto + Quantum
- Bastidas, Phys. Rev. Research 7, 043029 (2025). [arXiv:2406.17062](https://arxiv.org/abs/2406.17062)
- Camargo et al., Phys. Rev. A 108, 032219 (2023). [arXiv:2306.09956](https://arxiv.org/abs/2306.09956)
- Babbush et al., Phys. Rev. X 13, 041041 (2023). [arXiv:2303.13012](https://arxiv.org/abs/2303.13012)
- Markdahl et al. (2026). [arXiv:2603.08352](https://arxiv.org/abs/2603.08352)

### Error Mitigation
- Kato et al., Quantum 9, 1840 (2025). [arXiv:2412.01311](https://arxiv.org/abs/2412.01311)
- ZNE on logical qubits, Nature Comms (2025). [Link](https://www.nature.com/articles/s41467-025-67768-4)
- Infinite Distance Extrapolation (2026). [arXiv:2603.11285](https://arxiv.org/html/2603.11285)
- Inverted-circuit ZNE, Phys. Rev. A 110, 042625 (2024). [Link](https://link.aps.org/doi/10.1103/PhysRevA.110.042625)
- ZNE on silicon spin qubit, Phys. Rev. A (2025). [Link](https://link.aps.org/doi/10.1103/925y-b4s1)

### VQE and Variational
- Barren plateau strategies benchmark (2025). [arXiv:2512.11171](https://arxiv.org/abs/2512.11171)
- ADAPT-VQE mitigates rough landscapes, npj Quantum Info (2023). [Link](https://www.nature.com/articles/s41534-023-00681-0)
- Trainability-dequantization, ICLR 2025. [arXiv:2406.07072](https://arxiv.org/abs/2406.07072)
- ADAPT-VQE measurement overhead, Phys. Rev. Research (2025). [Link](https://doi.org/10.1103/t1hr-y7c8)

### Surface Code and QEC
- Google Willow, Nature (2024). [DOI:10.1038/s41586-024-08449-y](https://www.nature.com/articles/s41586-024-08449-y)
- Lattice surgery on repetition codes, Nature Physics (2025). [Link](https://www.nature.com/articles/s41567-025-03090-6)
- Universal gate set in surface codes, npj Quantum Info (2025). [Link](https://www.nature.com/articles/s41534-025-01118-6)
- Logical qubit teleportation, Science (2024). [Link](https://www.science.org/doi/10.1126/science.adp6016)

### Quantum Advantage
- IBM utility paper, Nature 618, 500 (2023). [DOI:10.1038/s41586-023-06096-3](https://www.nature.com/articles/s41586-023-06096-3)
- IBM QDC 2024 100x100 benchmark. [Blog](https://www.ibm.com/quantum/blog/qdc-2024)
- IBM Nighthawk/Loon (2025). [Newsroom](https://newsroom.ibm.com/2025-11-12-ibm-delivers-new-quantum-processors,-software,-and-algorithm-breakthroughs-on-path-to-advantage-and-fault-tolerance)
- Quantinuum 56-qubit MaxCut (2025). [Link](https://www.quantinuum.com/press-releases/quantinuum-launches-industry-first-trapped-ion-56-qubit-quantum-computer-that-challenges-the-worlds-best-supercomputers)

### Quantum Spiking Neural Networks
- Brand & Petruccione, npj Quantum Info (2024). [arXiv:2407.16398](https://arxiv.org/abs/2407.16398)
- SQS neurons with quantum memory (2025). [arXiv:2506.21324](https://arxiv.org/html/2506.21324)

### Quantum Cryptography
- EAQKD (2026). [arXiv:2603.02375](https://arxiv.org/html/2603.02375)
- Loop-Back QKD, Symmetry (2025). [Link](https://www.mdpi.com/2073-8994/17/4/521)
- Graph state network science, AAAI Symposium (2025). [Link](https://ojs.aaai.org/index.php/AAAI-SS/article/view/36900)

### Trapped-Ion vs Superconducting
- Quantinuum Helios architecture (2025). [Link](https://postquantum.com/quantum-computing/quantinuum-helios-architecture/)
- IonQ roadmap (2025). [Link](https://www.ionq.com/roadmap)
- Heron r3 benchmarks (2025). [Blog](https://www.ibm.com/quantum/blog/quantum-metric-layer-fidelity)

### Quantum + Fusion
- Ameri et al., IBM Research, QIP 2024. [Link](https://research.ibm.com/publications/a-quantum-computational-approach-to-linear-magnetohydrodynamic-stability-analysis)
- Joseph et al., Physics of Plasmas 30, 010501 (2023). [DOI:10.1063/5.0123765](https://pubs.aip.org/aip/pop/article/30/1/010501/2867588/Quantum-computing-for-fusion-energy-science)
- Yeter-Aydeniz et al., Frontiers in Physics 13 (2025). [Link](https://www.frontiersin.org/journals/physics/articles/10.3389/fphy.2025.1551209/full)
- PPPL STELLAR-AI (2026). [Link](https://www.pppl.gov/news/2026/pppl-launches-stellar-ai-platform-accelerate-fusion-energy-research)

### Qiskit Ecosystem
- Qiskit v2.0 migration guide. [Link](https://quantum.cloud.ibm.com/docs/en/guides/qiskit-2.0)
- Qiskit SDK 2.0 release notes. [Link](https://docs.quantum.ibm.com/api/qiskit/release-notes/2.0)
- Fractional gates migration. [Link](https://quantum.cloud.ibm.com/docs/en/guides/pulse-migration)
- IBM backend retirements. [Link](https://quantum.cloud.ibm.com/announcements/en/product-updates)

### Trotterization
- Optimal-order Trotter-Suzuki, QIP (2024). [arXiv:2405.01131](https://arxiv.org/html/2405.01131)
- Symmetric Trotterization, JKPS (2025). [arXiv:2603.07903](https://arxiv.org/abs/2603.07903)

### Floquet Engineering
- QHiFFS on Quantinuum, npj Quantum Info (2024). [Link](https://www.nature.com/articles/s41534-024-00866-1)
- Floquet SPT phases, Nature (2022, cited 2024-2025). [Link](https://www.nature.com/articles/s41586-022-04854-3)
