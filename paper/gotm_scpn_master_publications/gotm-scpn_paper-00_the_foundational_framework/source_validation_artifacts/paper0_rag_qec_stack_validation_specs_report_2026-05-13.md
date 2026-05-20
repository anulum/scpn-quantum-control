# Paper 0 RAG QEC Stack Specs

- Source span: P0R06530 - P0R06559
- Source records consumed: 30
- Coverage match: True
- Hardware status: simulator_only_no_provider_submission
- Claim boundary: source-bounded RAG QEC stack simulator contract; not empirical evidence

## Specs

### rag_qec_stack.insert_framing

RAG insertion records are promoted as source-bounded additions to the compressed Paper 0 architecture, with enhanced mathematics separated from critical layer mechanisms.

Formulae:

Mechanisms:
- RAG inserts are framed as insertion-ready additions for Paper 0 compression
- enhanced mathematical foundations and critical layer mechanisms are separated
- Layer 1 QEC stack is bounded to the insertion block

Null controls:
- missing-insertion-scope control must be rejected
- collapsed-block-boundary control must be rejected
- unbounded-paper0-claim control must be rejected

### rag_qec_stack.layer1_qec_hamiltonian

Layer 1 is represented by a source Hamiltonian split into microtubule lattice, stabiliser, and syndrome-detection terms.

Formulae:
- H_QEC = H_MT + H_stab + H_syndrome
- H_MT = -J_x sum_<ij> sigma_i^x sigma_j^x - J_z sum_i sigma_i^z
- H_stab = -J_s sum_p S_p - J_l sum_l L_l
- H_syndrome = -gamma_s sum_i (sigma_i^z tensor E_i)

Mechanisms:
- microtubule lattice contribution is represented by the H_MT term
- stabiliser contribution is represented by plaquette and logical stabilisers
- syndrome contribution couples sigma-z states to error-detection channels

Null controls:
- missing-syndrome-term control must be rejected
- non-finite-Hamiltonian-component control must be rejected
- collapsed-Hamiltonian-decomposition control must be rejected

### rag_qec_stack.gap_coherence_protection

The source claims a 1.64 eV gap, physiological 0.026 eV thermal scale, 400 fs versus 25 fs timescale comparison, and a threshold expression whose stated approximation requires explicit consistency warning.

Formulae:
- Delta E approximately 1.64 eV >> k_B T approximately 0.026 eV
- tau_coherence approximately hbar / Delta E approximately 400 fs
- tau_thermal approximately hbar / (k_B T) approximately 25 fs
- Protection Factor approximately 16x enhancement
- p_th = [1 - exp(-2 Delta E / k_B T)] / [1 + exp(-2 Delta E / k_B T)] approximately 10^(-14)

Mechanisms:
- energy gap is compared with physiological thermal scale
- protected coherence and thermal timescales are source-bounded estimates
- source protection factor is retained as approximately 16x
- source threshold approximation is not silently corrected

Null controls:
- non-positive-gap control must be rejected
- threshold-approximation-warning control must be emitted
- unsupported-spectroscopy-evidence control must be rejected

### rag_qec_stack.programmability_and_observable

Tubulin conformational states are promoted as classical control bits selecting topological operations, with an observable spectroscopic target near 1.64 eV under coherent versus anaesthetic states.

Formulae:

Mechanisms:
- tubulin conformational states act as classical control bits
- classical control bits select topological operations on the quantum substrate
- observable target is spectroscopic signature near 1.64 eV under coherent versus anaesthetic states

Null controls:
- missing-control-bit control must be rejected
- missing-topological-operation control must be rejected
- unsupported-spectroscopy-evidence control must be rejected
